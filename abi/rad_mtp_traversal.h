/* rad_mtp_traversal.h -- Traversal Verification (Weng et al., arXiv 2505.12398) of a DARTree draft
 * tree under sampling (--draft-verify traversal, needs --draft-coupling on). Plain code shared by
 * the host oracle, the device kernel and the tests. DESIGN.md next to this work has the argument;
 * the short form:
 *
 * Row-wise verification (rad_mtp_rrs_decide) decides each node on its own. Traversal verification
 * decides a whole ROOT-TO-LEAF PATH at once, bottom-up, and so can accept a longer prefix from the
 * same tree (for a chain it is Block Verification, never worse than token-level rejection sampling).
 *
 * State of a node v during the walk: p_v (its acceptance rate, a sequence-level ratio), the target
 * law M_b(.|v) restricted to what has not been rejected below it (r over the draft support, and a
 * scale `so` on the off-support candidates, as rad_mtp_rrs_decide), and the draft law M_s = Q of v
 * with the children already rejected removed.
 *   - A child c of v (draft order) enters with p_c = min(1, p_v * r(c)/qn(c)), qn = qr/sum(qr).
 *   - The walk descends to the first leaf and tests eta < p_leaf. Accept: the path is accepted and
 *     the final token is drawn from that node's current residual law r (the plain target if none of
 *     its children was rejected). Reject: the leaf is deleted and its parent v is updated:
 *         S    = sum_x [ p_v r(x) - qn(x) ]+            (off-support: p_v * so * pout)
 *         p_v' = S / (S + 1 - p_v)
 *         r'   = [ p_v r - qn ]+ / S
 *         qr(c) = 0 (siblings are drawn WITHOUT replacement: the remaining Q renormalises)
 *     and the walk continues with v's next child, or tests v itself once it has none.
 * Exactness (Theorem 3.3 of the paper, Lemma A.2 by induction over a node's descendants) needs
 * the tree's shape to be independent of the drawn tokens, which the slot-priority grower and the
 * q-rank pruning already guarantee for recursive rejection sampling, and siblings drawn from the
 * law the update assumes: Plackett-Luce without replacement (Gumbel-top-k) with the renormalising
 * update, which is exactly what --draft-coupling's trees draw (the i.i.d. variant, qr left alone,
 * is the same recursion but needs drafts that may repeat a token, which the wire cannot carry, so
 * it is not built). tests/mtp_rrs_test.cpp checks the walk and shows that three plausible mistakes
 * (token-level ratios, no (1 - p) term, with-replacement update for without-replacement drafts)
 * fail the same statistic.
 *
 * Wire: no new state. The walk reads the same RadMtpCoupleState rows as the row-wise verify and
 * writes the SAME plane the core already walks (ADV_ACCEPT: the first child whose token is the
 * sampler's token at the row): on the accepted path every row's token is its path child's token and
 * the last row's token is the final one; the other rows are not on the walk and keep the pick.
 */
#ifndef RAD_MTP_TRAVERSAL_H
#define RAD_MTP_TRAVERSAL_H

#include "rad_spechub.h"
#include "rad_mtp_rrs.h"

#include <type_traits>

#ifdef __cplusplus

#define RAD_MTP_TRAV_MAX_DEPTH 16
#define RAD_MTP_TRAV_KEY       0x54524156ull   /* "TRAV": the walk's acceptance uniforms */

/* Test-only switches; the engine passes the zero value. */
struct RadMtpTravOpt {
    int  ctl = 0;   /* negative controls: 1 token-level ratios, 2 no (1 - p) term, 3 q never
                       renormalised (the with-replacement update on without-replacement drafts) */
};

struct RadMtpTravOut {
    int     n = 0;                           /* nodes on the path, root first */
    int     node[RAD_MTP_TRAV_MAX_DEPTH + 2] = {};
    int32_t y = -1;                          /* the final token ... */
    bool    has_y = false;                   /* ... when the last node carries a valid state */
};

/* A tree whose candidate scans are done by a workgroup (the device walk, plugin/spechub/spechub.hip)
 * also provides
 *   void live(int v, const int32_t* sid, int cnt, RadTravLaw* out) const
 * which, for node v's candidate row, gathers the live candidates (id >= 0, value not -inf) in ascending
 * order, each one's FP64 weight exp(val - top) and its slot among `sid[0 .. cnt)` (cnt when outside),
 * and the row's validity and top. The ordered sums below then run over those stored values: the same
 * operations in the same order as the scan they replace, so the walk is the host's bit for bit. The
 * result is valid until the next live() call. A tree without it (the host's) scans the row itself. */
struct RadTravLaw {
    bool valid = false;               /* no id past the vocabulary, no NaN or +inf, a finite top */
    float top = -INFINITY;
    int   L = 0;
    const int32_t* ids = nullptr;     /* the row's ids; entry i is ids[lj[i]] */
    const short* lj = nullptr;
    const double* w = nullptr;
    const unsigned char* k = nullptr;
};
template <class T, class = void> struct RadTravLive : std::false_type {};
template <class T> struct RadTravLive<T, std::void_t<decltype(&T::live)>> : std::true_type {};

/* The tree the walk reads (a Tree type provides):
 *   const RadMtpCoupleState* state(int v)   the node's bound state, or null (node is a leaf)
 *   bool cand(int v, const int32_t** ids, const float** val, int64_t* n)   target candidates
 *   int  child(int v, int32_t token)        the node's first child row with that token, or -1
 *   int64_t vocab() */
struct RadMtpTravFrame {
    int     node, ci, nk;
    bool    ok;                 /* a valid state with a target law */
    int32_t ids[8];
    double  qr[8], r[8], so, pout, p;
    const int32_t* cid; const float* cval; int64_t cn; float top; double sum;
};

template <class Tree>
static inline RAD_RRS_HD bool rad_mtp_trav_push(const Tree& T, int v, double p,
                                                 RadMtpTravFrame* f) {
    f->node = v; f->ci = 0; f->nk = 0; f->ok = false; f->p = p; f->so = 0; f->pout = 0;
    const RadMtpCoupleState* sp = T.state(v);
    if (!sp) return true;
    RadMtpCoupleState s = *sp;
    int64_t n = 0;
    const int32_t* ids = nullptr; const float* val = nullptr;
    if (!T.cand(v, &ids, &val, &n)) return true;
    /* canonical order: the children first, in draft order (a chain's one child is x) */
    int nk = s.nkids >= 2 ? (s.nkids < s.count ? s.nkids : s.count) : 1;
    if (s.nkids < 2) {
        int k = 0;
        while (k < s.count && s.ids[k] != s.x) ++k;
        if (k >= s.count) return true;
        const int32_t ti = s.ids[0]; s.ids[0] = s.ids[k]; s.ids[k] = ti;
        const double  tq = s.q[0];   s.q[0]   = s.q[k];   s.q[k]   = tq;
    }
    float top = -INFINITY;
    double sum = 0, pq[8] = {}, wout = 0;
    if constexpr (RadTravLive<Tree>::value) {
        RadTravLaw lw;
        T.live(v, s.ids, s.count, &lw);
        if (!lw.valid) return true;
        top = lw.top;
        for (int i = 0; i < lw.L; ++i) {
            const double w = lw.w[i];
            sum += w;
            const int k = lw.k[i];
            if (k < s.count) pq[k] += w; else wout += w;
        }
    } else {
        for (int64_t j = 0; j < n; ++j) {
            if (ids[j] < 0) continue;
            if (ids[j] >= T.vocab() || val[j] != val[j] || val[j] == INFINITY) return true;
            if (val[j] > top) top = val[j];
        }
        if (top == -INFINITY) return true;
        for (int64_t j = 0; j < n; ++j) {
            if (ids[j] < 0 || val[j] == -INFINITY) continue;
            const double w = ::exp((double)val[j] - top);
            sum += w;
            int k = 0;
            while (k < s.count && ids[j] != s.ids[k]) ++k;
            if (k < s.count) pq[k] += w; else wout += w;
        }
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return true;
    for (int k = 0; k < 8; ++k) {
        f->ids[k] = k < s.count ? s.ids[k] : -1;
        f->qr[k] = k < s.count ? s.q[k] : 0;
        f->r[k] = k < s.count ? pq[k] / sum : 0;
    }
    f->pout = wout / sum;
    f->so = 1.0;
    f->nk = nk;
    f->ok = true;
    f->cid = ids; f->cval = val; f->cn = n; f->top = top; f->sum = sum;
    return true;
}

/* The final token from a frame's residual law (support slots r, off-support so * P). */
static inline RAD_RRS_HD bool rad_mtp_trav_final(const RadMtpTravFrame& f, double u, int32_t* out) {
    double total = f.so * f.pout;
    for (int k = 0; k < 8; ++k) total += f.r[k];
    if (!(total > 0) || !rad_spechub_finite(total)) return false;
    double c = u * total;
    int32_t last = -1;
    unsigned seen = 0;
    for (int64_t j = 0; j < f.cn; ++j) {
        if (f.cid[j] < 0 || f.cval[j] == -INFINITY) continue;
        int k = 0;
        while (k < 8 && f.ids[k] >= 0 && f.cid[j] != f.ids[k]) ++k;
        double m;
        if (k < 8 && f.ids[k] >= 0) {
            if (seen & (1u << k)) continue;
            seen |= 1u << k;
            m = f.r[k];
        } else {
            m = f.so * (::exp((double)f.cval[j] - f.top) / f.sum);
        }
        if (m > 0) { last = f.cid[j]; c -= m; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return last >= 0;
}

/* The same over a tree whose rows are scanned by a workgroup (RadTravLive): the frame's node is
 * gathered again (one live() call) and the candidates walked from the stored weights and slots. */
template <class Tree>
static inline RAD_RRS_HD bool rad_mtp_trav_final_live(const Tree& T, const RadMtpTravFrame& f,
                                                      double u, int32_t* out) {
    double total = f.so * f.pout;
    for (int k = 0; k < 8; ++k) total += f.r[k];
    if (!(total > 0) || !rad_spechub_finite(total)) return false;
    int cnt = 0;
    while (cnt < 8 && f.ids[cnt] >= 0) ++cnt;
    RadTravLaw lw;
    T.live(f.node, f.ids, cnt, &lw);
    double c = u * total;
    int32_t last = -1;
    unsigned seen = 0;
    for (int i = 0; i < lw.L; ++i) {
        const int k = lw.k[i];
        double m;
        if (k < cnt) {
            if (seen & (1u << k)) continue;
            seen |= 1u << k;
            m = f.r[k];
        } else {
            m = f.so * (lw.w[i] / f.sum);
        }
        if (m > 0) { last = lw.ids[lw.lj[i]]; c -= m; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return last >= 0;
}

/* One tree, root `root`: the accepted path and the final token. Returns false when nothing was
 * decided (an invalid root state: every row keeps the pick). `pos0` is the root row's output
 * position, the coordinate of the acceptance uniforms. */
template <class Tree>
static inline RAD_RRS_HD bool rad_mtp_trav_walk(const Tree& T, int root, uint64_t seed,
                                                 uint64_t pos0, RadMtpTravOut* out,
                                                 RadMtpTravOpt opt = RadMtpTravOpt{}) {
    RadMtpTravFrame fr[RAD_MTP_TRAV_MAX_DEPTH + 1];
    int top = 0, tests = 0;
    rad_mtp_trav_push(T, root, 1.0, &fr[0]);
    if (!fr[0].ok) return false;
    for (;;) {
        RadMtpTravFrame& f = fr[top];
        if (f.ci < f.nk) {
            const int i = f.ci;
            double sq = 0;
            for (int k = 0; k < 8; ++k) sq += f.qr[k];
            const double qx = f.qr[i] / sq;
            const int c = top < RAD_MTP_TRAV_MAX_DEPTH && qx > 0 ? T.child(f.node, f.ids[i]) : -1;
            if (c < 0) { f.nk = i; continue; }   /* a child the verify no longer carries: cut here */
            const double pbase = opt.ctl == 1 ? 1.0 : f.p;
            double pc = pbase * f.r[i] / qx;
            if (!(pc < 1.0)) pc = 1.0;
            rad_mtp_trav_push(T, c, pc, &fr[top + 1]);
            ++top;
            continue;
        }
        /* a leaf: the sequence-level test */
        const double u = rad_spechub_uniform(
            seed, pos0, RAD_MTP_TRAV_KEY ^ rad_mtp_rrs_mix((uint64_t)(++tests)));
        if (u < f.p) {
            out->n = top + 1;
            for (int t = 0; t <= top; ++t) out->node[t] = fr[t].node;
            out->has_y = false;
            if (f.ok) {
                const double ur = rad_spechub_uniform(seed, pos0 + (uint64_t)top, RAD_SAMPLE_KEY_RESID);
                if constexpr (RadTravLive<Tree>::value) out->has_y = rad_mtp_trav_final_live(T, f, ur, &out->y);
                else out->has_y = rad_mtp_trav_final(f, ur, &out->y);
            }
            return true;
        }
        if (top == 0) return false;   /* the root always accepts (p = 1); only roundoff gets here */
        --top;
        RadMtpTravFrame& g = fr[top];
        double sq = 0;
        for (int k = 0; k < 8; ++k) sq += g.qr[k];
        double S = g.p * g.so * g.pout, d[8];
        for (int k = 0; k < 8; ++k) {
            const double v = g.p * g.r[k] - g.qr[k] / sq;
            d[k] = v > 0 ? v : 0;
            S += d[k];
        }
        const int ci = g.ci;
        const double pold = g.p;
        if (!(S > 0) || !rad_spechub_finite(S)) {
            g.p = 0;   /* nothing of the target is left above the draft: this node cannot accept */
        } else {
            const double den = opt.ctl == 2 ? S : S + 1.0 - pold;
            g.p = S / den;
            for (int k = 0; k < 8; ++k) g.r[k] = d[k] / S;
            g.so = pold * g.so / S;
        }
        if (opt.ctl != 3) g.qr[ci] = 0;
        ++g.ci;
    }
}

#endif /* __cplusplus */
#endif /* RAD_MTP_TRAVERSAL_H */

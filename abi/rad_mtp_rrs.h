/* rad_mtp_rrs.h -- recursive rejection sampling over a draft TREE's siblings (--draft-coupling on
 * with --draft-tree on): the draft side (children sampled without replacement from Q) and the
 * verify side (the walk over a node's children against the target's P), as plain code the host
 * oracle, the device kernels and the engine's binding all share, so they cannot drift.
 * DESIGN-coupling.md (next to DESIGN.md of the dartree work) is the proof; the short form:
 *
 * Draft. A node whose head pass left the law Q (the FP64 T1 softmax of the reranked top 8,
 * rad_mtp_couple_build_q) drafts its children x_1, x_2, ... by Gumbel-top-k over log Q: add an
 * independent Gumbel to every log q_i and take the entries in descending order. That is sampling
 * without replacement (Plackett-Luce): x_1 ~ Q, x_2 ~ Q(. | not x_1), ... The first m of them are
 * the node's children, m chosen by anything that does not read the values x_j it applies to
 * (the stage-1 leaf count; the grower's best-first budget, whose slot priorities are the j-th
 * largest q of Q, never the drawn token).
 *
 * Verify (Sequoia, Chen et al. 2024, Alg. 1; SpecInfer's multi-step verification with the draft
 * law renormalised after each removal). P_1 = the target's law, q_1 = Q. For i = 1..m:
 *     accept x_i with probability min(1, P_i(x_i) / q_i(x_i));
 *     on rejection  P_{i+1} = norm(max(P_i - q_i, 0)),   q_{i+1} = q_i with x_i removed, renormalised.
 * All rejected: the token is a draw from P_{m+1}. Exactness (induction on m): after a rejection of
 * the first draft the continuation depends on x_1 only through q_2, and P_2 = norm(max(P - q, 0))
 * does not depend on x_1 at all, so by the hypothesis it emits ~ P_2 whatever x_1 was; then
 * P(emit y) = sum_x q(x) min(1, P(x)/q(x)) [x = y] + P(reject) P_2(y) = min(P, q)(y) +
 * max(P - q, 0)(y) = P(y). The emitted token is the accepted child's token, or a residual draw that
 * has no mass on any rejected child (P_{i+1}(x_i) = 0 after a rejection), so the core's rule "follow
 * the first child whose token is the sampler's token at the row" (ADV_ACCEPT) walks exactly the
 * decision made here and needs no change.
 *
 * Wire: the node's RadMtpCoupleState holds Q (ids, q), the children in draft order in ids[0 ..
 * nkids), x = ids[0]. A state with nkids <= 1 is the chain's (rad_mtp_couple_decide, bit for bit).
 */
#ifndef RAD_MTP_RRS_H
#define RAD_MTP_RRS_H

#include "rad_spechub.h"
#include "rad_mtp_tree.h"

#ifdef __cplusplus

#ifdef __HIPCC__
#define RAD_RRS_HD __host__ __device__
#else
#define RAD_RRS_HD
#endif

#define RAD_MTP_RRS_KEY_TREE   0x4d54524545474d42ull   /* the draft's Gumbel stream */
#define RAD_MTP_RRS_MAX_WIDTH  16   /* the frontier rows a coupled grower pass handles */
#define RAD_MTP_RRS_KEY_ACCEPT 0x41435054ull

static inline RAD_RRS_HD uint64_t rad_mtp_rrs_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* The open-interval uniform of candidate `c` of draft node `node`: (seed, base) name the draft
 * stream (the request's seed and its pre-commit output length, which no later step reuses). */
static inline RAD_RRS_HD double rad_mtp_rrs_gumbel_u(uint64_t seed, uint64_t base,
                                                         uint64_t node, int c) {
    uint64_t z = rad_mtp_rrs_mix(seed ^ RAD_MTP_RRS_KEY_TREE);
    z = rad_mtp_rrs_mix(z ^ base);
    z = rad_mtp_rrs_mix(z + node * 16u + (uint64_t)c);
    return ((double)(z >> 11) + 0.5) * 0x1p-53;
}

/* Gumbel-top-k order of Q's support: order[0 .. s.count) are indices into s.ids, the draft order
 * (descending of log q + Gumbel, ties to the lower index). Draws one uniform per support entry from
 * the state's own draw, node `node`. */
static inline RAD_RRS_HD void rad_mtp_rrs_order(const RadMtpCoupleState& s, uint64_t node,
                                                    int* order) {
    double key[8];
    for (int i = 0; i < s.count; ++i) {
        const double u = rad_mtp_rrs_gumbel_u(s.draw.seed, s.draw.base, node, i);
        key[i] = ::log(s.q[i]) - ::log(-::log(u));
        order[i] = i;
    }
    for (int i = 1; i < s.count; ++i) {   /* insertion sort, stable */
        const int o = order[i];
        int j = i;
        while (j > 0 && key[order[j - 1]] < key[o]) { order[j] = order[j - 1]; --j; }
        order[j] = o;
    }
}

/* Draw the node's children: reorder Q's support into the draft order (ids and q together), set
 * x = ids[0] and nkids = `nkids` (0 = none yet, the grower raises it as it creates children). */
static inline RAD_RRS_HD void rad_mtp_rrs_draw(RadMtpCoupleState* s, uint64_t node, int nkids) {
    int order[8];
    rad_mtp_rrs_order(*s, node, order);
    int32_t ids[8];
    double  q[8];
    for (int i = 0; i < s->count; ++i) { ids[i] = s->ids[order[i]]; q[i] = s->q[order[i]]; }
    for (int i = 0; i < s->count; ++i) { s->ids[i] = ids[i]; s->q[i] = q[i]; }
    s->x = s->ids[0];
    s->nkids = nkids < s->count ? nkids : s->count;
}

/* The SLOT priorities of a node's law: slot j carries the log of the j-th LARGEST q of Q
 * (descending, ties by position), -inf past the support. They are a function of Q alone, never of
 * which token was drawn into which slot, so a rule that reads only them (the grower's best-first
 * budget, the confidence pruning below) admits children by a quantity independent of the values
 * the children take (DESIGN-coupling.md 3.2, 3.5). `s` is read before the draw reorders it or
 * after (the multiset of q is the same). */
static inline RAD_RRS_HD void rad_mtp_rrs_slot_logq(const RadMtpCoupleState& s, float* ov) {
    double qs[8];
    const int n = s.count < 8 ? s.count : 8;
    for (int i = 0; i < n; ++i) qs[i] = s.q[i];
    for (int i = 1; i < n; ++i) {
        const double v = qs[i];
        int j = i;
        while (j > 0 && qs[j - 1] < v) { qs[j] = qs[j - 1]; --j; }
        qs[j] = v;
    }
    for (int c = 0; c < 8; ++c) ov[c] = c < n ? (float)::log(qs[c]) : -INFINITY;
}

/* Confidence pruning of one node's drafted children under coupling: how many of the first `want`
 * slots (>= 1, <= count) the node keeps. The rule is the grower's (abi/rad_mtp_tree.h) applied to
 * the slot priorities, base 0: slot j is kept iff its probability (the log-softmax of the
 * priorities) >= prune_abs and q_(j) >= prune_rel * q_(0); slot 0 is always kept. Both are
 * non-increasing in j, so the kept slots are a PREFIX of the draft order; and they depend on Q only,
 * never on the drawn token's value, which is what keeps recursive rejection sampling exact. A
 * zeroed cfg keeps all `want`. */
static inline RAD_RRS_HD int rad_mtp_rrs_slot_keep_count(const RadMtpCoupleState& s,
                                                         const RadMtpTreeCfg* pc, int want) {
    if (want > s.count) want = s.count;
    if (want < 1 || !pc || !pc->prune) return want;
    float ov[8], lp[8], mx = 0.0f, lse = 0.0f;
    int ids[8];
    rad_mtp_rrs_slot_logq(s, ov);
    for (int c = 0; c < 8; ++c) ids[c] = c < s.count ? c : -1;
    const int best = rad_mtp_tree_row_logp(ids, ov, lp, &mx, &lse);
    if (best < 0) return 1;
    int m = 1;
    while (m < want && rad_mtp_tree_keep(pc, m, best, 0.0f, lp[m], ov[m], mx)) ++m;
    return m;
}

/* The longest prefix of the state's children that are the first `n` of `kid[]` (the tree's
 * children of this row, in node order, repeats dropped): what a verify may keep when a window cut
 * the tree short. 0 when even the first differs. */
static inline RAD_RRS_HD int rad_mtp_rrs_prefix(const RadMtpCoupleState& s, const int32_t* kid,
                                                    int n) {
    const int have = s.nkids >= 1 ? s.nkids : 1;
    int m = 0;
    while (m < have && m < n && m < s.count && s.ids[m] == kid[m]) ++m;
    return s.nkids >= 2 ? m : (n >= 1 && s.x == kid[0] ? 1 : 0);
}

/* The acceptance uniform of the i-th child at output position `pos`: the first is the chain's. */
static inline RAD_RRS_HD double rad_mtp_rrs_uaccept(uint64_t seed, uint64_t pos, int i) {
    return rad_spechub_uniform(seed, pos, i == 0 ? (uint64_t)RAD_SAMPLE_KEY_ACCEPT
                                                 : RAD_MTP_RRS_KEY_ACCEPT ^ rad_mtp_rrs_mix((uint64_t)i));
}

/* The verify of a node with nkids >= 2 children: the recursion above over the sampler's surviving
 * candidates (ids/val, n of them, FP64 P = softmax of val). Writes the emitted token to *out and
 * returns true, or false for an invalid target (the row keeps the ordinary token). */
static inline RAD_RRS_HD bool rad_mtp_rrs_decide(const RadMtpCoupleState& s, const int32_t* ids,
                                                     const float* val, int64_t n, int64_t vocab,
                                                     uint64_t seed, uint64_t pos, int32_t* out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    if (top == -INFINITY) return false;
    double sum = 0, pq[8] = {}, wout = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        const double w = ::exp((double)val[j] - top);
        sum += w;
        int k = 0;
        while (k < s.count && ids[j] != s.ids[k]) ++k;
        if (k < s.count) pq[k] += w; else wout += w;
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    /* P over Q's support (normalised), and the off-support mass's scale `so` (P_i(y) = so * w / sum
     * there). qr: the draft law with the drawn children removed. */
    double r[8], qr[8], so = 1.0;
    for (int k = 0; k < s.count; ++k) { r[k] = pq[k] / sum; qr[k] = s.q[k]; }
    const double pout = wout / sum;
    bool direct = false;
    for (int i = 0; i < s.nkids && !direct; ++i) {
        double sq = 0;
        for (int k = 0; k < s.count; ++k) sq += qr[k];
        if (!(sq > 0)) break;
        const double qx = qr[i] / sq;
        if (r[i] >= qx || rad_mtp_rrs_uaccept(seed, pos, i) * qx < r[i]) { *out = s.ids[i]; return true; }
        double z = so * pout;
        for (int k = 0; k < s.count; ++k) {
            const double d = r[k] - qr[k] / sq;
            r[k] = d > 0 ? d : 0;
            z += r[k];
        }
        if (!(z > 0) || !rad_spechub_finite(z)) { direct = true; break; }
        for (int k = 0; k < s.count; ++k) r[k] /= z;
        so /= z;
        qr[i] = 0;
    }
    const double u = rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID);
    double  c;
    int32_t last = -1;
    if (direct) {
        c = u * sum;
        for (int64_t j = 0; j < n; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = ::exp((double)val[j] - top);
                if (w > 0) { last = ids[j]; c -= w; if (c < 0) { *out = last; return true; } }
            }
        *out = last;
        return true;
    }
    double total = 0;
    for (int k = 0; k < s.count; ++k) total += r[k];
    total += so * pout;
    c = u * total;
    unsigned seen = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        int k = 0;
        while (k < s.count && ids[j] != s.ids[k]) ++k;
        double m;
        if (k < s.count) {
            if (seen & (1u << k)) continue;
            seen |= 1u << k;
            m = r[k];
        } else {
            m = so * (::exp((double)val[j] - top) / sum);
        }
        if (m > 0) { last = ids[j]; c -= m; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return last >= 0;
}

#endif /* __cplusplus */
#endif /* RAD_MTP_RRS_H */

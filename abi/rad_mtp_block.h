/* rad_mtp_block.h -- BLOCK VERIFICATION of a linear MTP draft chain (--draft-verify block).
 *
 * Sun, Mendlovic, Leviathan, Aharoni, Beirami, Ro, Suresh, "Block Verification Accelerates
 * Speculative Decoding" (arXiv 2403.10444): the chain's draft tokens X_1 .. X_g are accepted or
 * rejected JOINTLY. Token-wise speculative sampling (what --draft-coupling does at each row) is
 * optimal per token; block verification is optimal for the whole block: the expected number of
 * accepted tokens is never lower and the output law is the target's, exactly.
 *
 * With P_i = the target's law after X^i (the sampler's filtered candidate softmax at verify row i,
 * i = 0 .. g-1) and Q_{i+1} = the head's law the draft X_{i+1} was sampled from (the coupled state
 * of round i + 1, the FP64 softmax of the reranked top 8):
 *
 *     p_0 = 1,   p_i = min(1, p_{i-1} P_{i-1}(X_i) / Q_i(X_i))                  i = 1 .. g
 *     S_i = sum_x max(p_i P_i(x) - Q_{i+1}(x), 0)                                i = 0 .. g-1
 *     h_i = S_i / (S_i + 1 - p_i)   (i < g),    h_g = p_g
 *
 *     for i = g, g-1, ..., 0:  accept the block's first i tokens with probability h_i; the first
 *     success is tau. (h_0 = 1: it always stops.)
 *     tau = g:  X_1 .. X_g are emitted and the next token is the ordinary draw from P_g.
 *     tau < g:  X_1 .. X_tau are emitted, then Y ~ max(p_tau P_tau - Q_{tau+1}, 0) normalised.
 *
 * Exactness is checked in tests/mtp_block_test.cpp by exact enumeration of the draft tree of a toy
 * model (the law of the emitted tokens computed in closed form, to 1e-12) and by a chi-square of the
 * real device-shared code, with a negative control (the same rule with a wrong residual).
 *
 * Y may equal X_{tau+1}; the core then takes that draft as accepted (ADV_ACCEPT: "the draft is the
 * sampler's token at this row") and reads the next row, which keeps its ordinary pick: a draw from
 * the target's law given exactly the tokens emitted so far, so the extra tokens continue the same
 * autoregressive sample. The verify op therefore writes only rows 0 .. tau of the block and leaves
 * every later row alone.
 *
 * Relation to --draft-coupling: the DRAFT half is shared (x ~ Q at temperature 1 from the head's
 * top 8, coupled states with their FP64 Q); only the verify half differs. Maximal coupling decides
 * row k from P_k and Q_{k+1} alone, so it is a per-row rule the core's token comparison walks
 * unchanged. Block verification decides row k from the rows after it (h_i for i > k), so the two
 * rules cannot be applied to the same row; `--draft-verify block` replaces the coupling's verify
 * and requires the coupled drafting (it switches it on), and it is for a LINEAR chain: a draft tree's
 * rows are verified by recursive rejection sampling (abi/rad_mtp_rrs.h) and Traversal Verification
 * (2505.12398) would be its block-level analogue, which this does not do.
 */
#ifndef RAD_MTP_BLOCK_H
#define RAD_MTP_BLOCK_H

#include "rad_spechub.h"

#ifdef __cplusplus

#ifdef __HIPCC__
#define RAD_BLK_HD __host__ __device__
#else
#define RAD_BLK_HD
#endif

#define RAD_MTP_BLOCK_MAX      RAD_MTP_COUPLE_MAX_ROUNDS   /* g: the chain's longest block */
#define RAD_MTP_BLOCK_KEY_STOP 0x424c4b53544f5000ull       /* the h_i uniforms, per block */

/* One verify row's candidate list: the sampler's plane, either dense (lj == nullptr, n slots, holes
 * id < 0 and -inf scores skipped) or compacted to the live slots in ascending order (lj[j] indexes
 * ids/val, n of them), with `valid` and `top` the scan of the whole plane the device already made
 * (rad_mtp_block_scan makes the same on the host). Summation order is ascending slot order in both,
 * so the two give bit-identical FP64. */
struct RadBlockRow {
    const int32_t* ids;
    const float*   val;
    const short*   lj;
    int64_t        n;
    int            valid;
    float          top;
};

/* The dense row's validity and maximum, as rad_mtp_couple_decide scans them. */
static inline RAD_BLK_HD void rad_mtp_block_scan(RadBlockRow* r, int64_t vocab) {
    float top = -INFINITY;
    int valid = 1;
    for (int64_t j = 0; j < r->n; ++j) {
        if (r->ids[j] < 0) continue;
        if (r->ids[j] >= vocab || r->val[j] != r->val[j] || r->val[j] == INFINITY) valid = 0;
        else if (r->val[j] > top) top = r->val[j];
    }
    r->valid = valid;
    r->top = top;
}

/* Visit order helper: slot j's index into the plane, or -1 for a dead slot. */
static inline RAD_BLK_HD int64_t rad_mtp_block_slot(const RadBlockRow& r, int64_t j) {
    const int64_t i = r.lj ? (int64_t)r.lj[j] : j;
    return (r.ids[i] < 0 || r.val[i] == -INFINITY) ? -1 : i;
}

/* P_i over Q_{i+1}'s support (pq, normalised, duplicate candidates aggregated), the mass off that
 * support (pout), the normaliser `sum` and the index `ix` of the draft token in the state's ids. */
struct RadBlockStat {
    double pq[8];
    double pout, sum;
    int    ix;
};

static inline RAD_BLK_HD bool rad_mtp_block_stat(const RadBlockRow& r, const RadMtpCoupleState& s,
                                                 RadBlockStat* o) {
    if (!r.valid || r.top == -INFINITY) return false;
    double sum = 0, wout = 0;
    for (int k = 0; k < 8; ++k) o->pq[k] = 0;
    for (int64_t j = 0; j < r.n; ++j) {
        const int64_t i = rad_mtp_block_slot(r, j);
        if (i < 0) continue;
        const double w = ::exp((double)r.val[i] - r.top);
        sum += w;
        int k = 0;
        while (k < s.count && r.ids[i] != s.ids[k]) ++k;
        if (k < s.count) o->pq[k] += w; else wout += w;
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    o->ix = -1;
    for (int k = 0; k < s.count; ++k) {
        o->pq[k] /= sum;
        if (s.ids[k] == s.x) o->ix = k;
    }
    o->pout = wout / sum;
    o->sum = sum;
    return o->ix >= 0;
}

/* The block's weights. p[i] (i = 0 .. g), h[i] (i = 0 .. g). */
struct RadBlockPlan {
    int    g;
    double p[RAD_MTP_BLOCK_MAX + 1];
    double h[RAD_MTP_BLOCK_MAX + 1];
    double s[RAD_MTP_BLOCK_MAX + 1];   /* S_i, the residual's mass (s[g] unused) */
};

/* The residual mass of one token of row i at weight p: max(p P(x) - Q(x), 0) on Q's support,
 * p P(x) off it. */
static inline RAD_BLK_HD double rad_mtp_block_resid_sum(const RadMtpCoupleState& s,
                                                        const RadBlockStat& t, double p) {
    double r = p * t.pout;
    for (int k = 0; k < s.count; ++k) r += rad_spechub_positive(p * t.pq[k] - s.q[k]);
    return r;
}

/* st[i] / stat[i] for i = 0 .. g-1. */
static inline RAD_BLK_HD void rad_mtp_block_plan(const RadMtpCoupleState* st,
                                                 const RadBlockStat* stat, int g,
                                                 RadBlockPlan* o) {
    o->g = g;
    o->p[0] = 1.0;
    for (int i = 0; i < g; ++i) {
        const double ratio = stat[i].pq[stat[i].ix] / st[i].q[stat[i].ix];
        const double v = o->p[i] * ratio;
        o->p[i + 1] = v < 1.0 ? v : 1.0;
    }
    for (int i = 0; i < g; ++i) {
        const double S = rad_mtp_block_resid_sum(st[i], stat[i], o->p[i]);
        o->s[i] = S;
        const double den = S + (1.0 - o->p[i]);
        o->h[i] = (den > 0 && rad_spechub_finite(den)) ? S / den : 1.0;
    }
    o->s[g] = 0;
    o->h[g] = o->p[g];
}

/* tau: the first i, counting down from g, whose uniform falls below h_i. The uniforms are keyed by
 * the block's first output position, so no row's own draws are reused. */
static inline RAD_BLK_HD int rad_mtp_block_tau(const RadBlockPlan& pl, uint64_t seed, uint64_t pos0) {
    for (int i = pl.g; i > 0; --i)
        if (rad_spechub_uniform(seed, pos0, RAD_MTP_BLOCK_KEY_STOP + (uint64_t)i) < pl.h[i]) return i;
    return 0;
}

/* Y ~ max(p P - Q, 0) normalised (state s, row r, stat t) with the uniform u in [0, 1). When the
 * residual's total underflows to zero (FP64 cancellation with p = 1: P == Q) Y is drawn from P
 * itself, a branch that depends on (P, Q) only. False: no live candidate. */
static inline RAD_BLK_HD bool rad_mtp_block_resid_draw(const RadBlockRow& r,
                                                       const RadMtpCoupleState& s,
                                                       const RadBlockStat& t, double p, double u,
                                                       int32_t* out) {
    double total = 0;
    for (int pass = 0; pass < 2; ++pass) {
        unsigned seen = 0;
        double c = u * total;
        int32_t last = -1;
        for (int64_t j = 0; j < r.n; ++j) {
            const int64_t i = rad_mtp_block_slot(r, j);
            if (i < 0) continue;
            int k = 0;
            while (k < s.count && r.ids[i] != s.ids[k]) ++k;
            double m;
            if (k < s.count) {
                if (seen & (1u << k)) continue;
                seen |= 1u << k;
                m = rad_spechub_positive(p * t.pq[k] - s.q[k]);
            } else {
                m = p * (::exp((double)r.val[i] - r.top) / t.sum);
            }
            if (pass == 0) { total += m; continue; }
            if (m > 0) { last = r.ids[i]; c -= m; if (c < 0) { *out = last; return true; } }
        }
        if (pass == 1) { *out = last; return last >= 0; }
        if (!(total > 0 && rad_spechub_finite(total))) break;
    }
    /* direct draw from P */
    double c = u * t.sum;
    int32_t last = -1;
    for (int64_t j = 0; j < r.n; ++j) {
        const int64_t i = rad_mtp_block_slot(r, j);
        if (i < 0) continue;
        const double w = ::exp((double)r.val[i] - r.top);
        if (w > 0) { last = r.ids[i]; c -= w; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return last >= 0;
}

/* The whole decision on the host: rows[0 .. g-1] / st[0 .. g-1] are the block's verify rows and
 * their coupled states (round i + 1 at row i), pos0 the first row's output position. Writes tau and
 * *y; *y = -1 when tau == g (the g-th row's ordinary draw stands). False for an invalid target. */
static inline RAD_BLK_HD bool rad_mtp_block_decide(const RadMtpCoupleState* st,
                                                   const RadBlockRow* rows, int g, uint64_t seed,
                                                   uint64_t pos0, int* tau, int32_t* y) {
    RadBlockStat stat[RAD_MTP_BLOCK_MAX];
    RadBlockPlan pl;
    if (g < 1 || g > RAD_MTP_BLOCK_MAX) return false;
    for (int i = 0; i < g; ++i)
        if (!rad_mtp_block_stat(rows[i], st[i], &stat[i])) return false;
    rad_mtp_block_plan(st, stat, g, &pl);
    const int t = rad_mtp_block_tau(pl, seed, pos0);
    *tau = t;
    *y = -1;
    if (t >= g) return true;
    return rad_mtp_block_resid_draw(rows[t], st[t], stat[t], pl.p[t],
                                    rad_spechub_uniform(seed, pos0 + (uint64_t)t, RAD_SAMPLE_KEY_RESID), y);
}

#endif /* __cplusplus */
#endif /* RAD_MTP_BLOCK_H */

/* rad_mtp_adapt.h -- cost-aware, calibrated node selection for a serial draft head's tree
 * (--draft-adaptive on). One header, shared by the grower (abi/rad_mtp_tree.h, host and device),
 * the stage-1 leaf rules and the tests; the design and the losslessness argument are in
 * docs/DESIGN-adaptive.md.
 *
 * The rule (EVICT-style, argmax of tokens / time). A tree of k nodes beside the root verifies k+1
 * rows and is expected to commit
 *
 *     E(k) = 1 + sum over its nodes of exp(cumulative CALIBRATED log probability)
 *
 * tokens in T(k) = C(k) + D milliseconds (C the measured verify-step cost of k+1 rows, D the
 * serial head's rounds). The best-first grower offers its next candidate (cumulative probability
 * p); it is created only while it raises the ratio:
 *
 *     (E + p) / T(k+1) > E / T(k)    <=>    p * T(k) > margin * E * (C(k+1) - C(k))
 *
 * The first `floor` nodes are always created. A candidate the rule refuses stays in the pool, so
 * a later pass whose candidates beat it creates it.
 *
 * LOSSLESSNESS. Every quantity the rule reads is a function of the nodes' (and the candidate's)
 * cumulative scores. Under --draft-coupling those scores are built from SLOT priorities, the log
 * of the j-th largest q of each node's own law plus the parent's score (abi/rad_mtp_tree_rrs.h),
 * never from a drawn token, and the pool always creates its maximum first, so a node's children
 * are a prefix of its draft order: the conditions recursive rejection sampling needs
 * (DESIGN-coupling.md 3.5). The calibration is a monotone map of a node's own slot values (a
 * per-depth temperature), so it keeps both properties.
 *
 * CALIBRATION. The head's top-8 log-softmax is re-expressed as softmax(v / T_d) for the depth d of
 * the children (T_d > 1 flattens an overconfident head). Fixed per depth (--draft-calib); the
 * host-side RadMtpCalib below fits the T_d from (predicted, observed) acceptance history.
 */
#ifndef RAD_MTP_ADAPT_H
#define RAD_MTP_ADAPT_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef RAD_MTP_HD
#ifdef __HIPCC__
#define RAD_MTP_HD __host__ __device__
#else
#define RAD_MTP_HD
#endif
#endif

#define RAD_MTP_ADAPT_MAXC    24   /* cost table entries: k = 0 .. 23 nodes */
#define RAD_MTP_ADAPT_DEPTHS  8    /* calibration depths 1 .. 8, the last repeats */

typedef struct RadMtpAdapt {
    int   on;          /* 0: every rule off, the grower is bit for bit what it was */
    int   floor;       /* nodes always created */
    int   ncost;       /* table entries given (0: the linear default) */
    int   ntemp;       /* temperatures given (0: no calibration) */
    float margin;      /* > 1 stricter, < 1 looser */
    float draft_ms;    /* D: the head's rounds a step, ms */
    float chain_cont;  /* stage 1's estimate of one chain level's continuation probability */
    float base_ms;     /* the linear default: C(k) = base_ms + row_ms * (k - base_k) */
    float row_ms;
    int   base_k;
    float cost[RAD_MTP_ADAPT_MAXC];
    float temp[RAD_MTP_ADAPT_DEPTHS];
    int   nlift;       /* lifts given (0: none) */
    float lift[RAD_MTP_ADAPT_DEPTHS];   /* p' = lift + (1 - lift) p of the children at depth d */
} RadMtpAdapt;

/* The profile default (Qwen3.8-Flash-Next TP2, batch 1): ~12.3 ms a 4-row step, +1.0 ms a row, a
 * 0.5 ms head round. */
static inline void rad_mtp_adapt_defaults(RadMtpAdapt* a) {
    memset(a, 0, sizeof *a);
    a->floor = 1;
    a->margin = 1.0f;
    a->draft_ms = 1.5f;
    a->chain_cont = 0.7f;
    a->base_ms = 12.3f;
    a->row_ms = 1.0f;
    a->base_k = 3;
}

/* C(k): the verify step's ms with k nodes (k + 1 rows). A table entry below its predecessor is
 * raised to it (the cost never falls with rows); past the table the last slope goes on. */
RAD_MTP_HD static inline float rad_mtp_adapt_cost(const RadMtpAdapt* a, int k) {
    if (k < 0) k = 0;
    if (a->ncost <= 0) return a->base_ms + a->row_ms * (float)(k - a->base_k);
    const int n = a->ncost < RAD_MTP_ADAPT_MAXC ? a->ncost : RAD_MTP_ADAPT_MAXC;
    float prev = a->cost[0], slope = a->row_ms;
    for (int i = 1; i < n && i <= k; ++i) {
        float v = a->cost[i];
        if (v < prev) v = prev;
        slope = v - prev;
        prev = v;
    }
    if (k < n) return prev;
    if (n < 2) slope = a->row_ms;
    if (slope < 0.05f) slope = 0.05f;
    return prev + slope * (float)(k - (n - 1));
}

/* Whether a node of cumulative (calibrated) probability p is worth a verify row when the tree has
 * k nodes and expects E tokens. */
RAD_MTP_HD static inline int rad_mtp_adapt_admit(const RadMtpAdapt* a, int k, float E, float p) {
    if (!a->on || k < a->floor) return 1;
    float dc = rad_mtp_adapt_cost(a, k + 1) - rad_mtp_adapt_cost(a, k);
    if (dc < 0.01f) dc = 0.01f;
    const float t = rad_mtp_adapt_cost(a, k) + a->draft_ms;
    return p * t > a->margin * E * dc;
}

/* The calibration temperature of the children at depth d >= 1 (1 when none). */
RAD_MTP_HD static inline float rad_mtp_adapt_temp(const RadMtpAdapt* a, int depth) {
    if (!a->on || a->ntemp <= 0) return 1.0f;
    int i = depth - 1;
    if (i < 0) i = 0;
    if (i >= a->ntemp) i = a->ntemp - 1;
    if (i >= RAD_MTP_ADAPT_DEPTHS) i = RAD_MTP_ADAPT_DEPTHS - 1;
    const float t = a->temp[i];
    return t > 0.05f ? t : 1.0f;
}

/* The probability floor of the children at depth d >= 1 (0 when none). */
RAD_MTP_HD static inline float rad_mtp_adapt_lift(const RadMtpAdapt* a, int depth) {
    if (!a->on || a->nlift <= 0) return 0.0f;
    int i = depth - 1;
    if (i < 0) i = 0;
    if (i >= a->nlift) i = a->nlift - 1;
    if (i >= RAD_MTP_ADAPT_DEPTHS) i = RAD_MTP_ADAPT_DEPTHS - 1;
    const float f = a->lift[i];
    return f > 0.0f && f < 1.0f ? f : 0.0f;
}

/* log_softmax(lp / T) over the valid (non-NaN) entries of a log-softmaxed row, in place. */
RAD_MTP_HD static inline void rad_mtp_adapt_recal(float* lp, int n, float T) {
    if (T == 1.0f) return;
    float mx = -INFINITY;
    for (int c = 0; c < n; ++c) if (lp[c] == lp[c] && lp[c] > mx) mx = lp[c];
    if (!(mx > -INFINITY)) return;
    float sum = 0.0f;
    for (int c = 0; c < n; ++c)
        if (lp[c] == lp[c] && lp[c] > -INFINITY) sum += expf((lp[c] - mx) / T);
    const float lse = logf(sum);
    for (int c = 0; c < n; ++c)
        if (lp[c] == lp[c] && lp[c] > -INFINITY) lp[c] = (lp[c] - mx) / T - lse;
}

/* lp <- log(lift + (1 - lift) exp(lp)) over the valid entries: a monotone floor on the probabilities. */
RAD_MTP_HD static inline void rad_mtp_adapt_floor(float* lp, int n, float lift) {
    if (!(lift > 0.0f)) return;
    for (int c = 0; c < n; ++c)
        if (lp[c] == lp[c] && lp[c] > -INFINITY) lp[c] = logf(lift + (1.0f - lift) * expf(lp[c]));
}

/* E: 1 + the sum of the created nodes' probabilities. `cum` the nodes' cumulative log
 * probabilities (-inf: an inert copy, no mass). */
RAD_MTP_HD static inline float rad_mtp_adapt_expect(const float* cum, int n) {
    float e = 1.0f;
    for (int i = 0; i < n; ++i) if (cum[i] > -INFINITY) e += expf(cum[i]);
    return e;
}

/* Stage 1 (a chain of `chain_nodes` and depth-1 leaves): how many of the first `want` runner-up
 * leaves of a round-1 row to keep. `lp` the row's calibrated log-softmax in DESCENDING order
 * (column 0 the chain's first node), 8 entries. The chain beyond its first node is not visible
 * to the row, so it is estimated: each level continues with probability chain_cont. The kept
 * leaves are a prefix of the descending order. */
RAD_MTP_HD static inline int rad_mtp_adapt_leaf_count(const RadMtpAdapt* a, const float* lp,
                                                      int chain_nodes, int want) {
    if (!a->on || want < 1) return want;
    if (want > 7) want = 7;
    float E = 1.0f, lvl = expf(lp[0]);
    for (int i = 0; i < chain_nodes; ++i) { E += lvl; lvl *= a->chain_cont; }
    int k = chain_nodes, m = 0;
    while (m < want) {
        const float pj = lp[m + 1] == lp[m + 1] ? expf(lp[m + 1]) : 0.0f;
        if (!(pj > 0.0f) || !rad_mtp_adapt_admit(a, k, E, pj)) break;
        E += pj; ++k; ++m;
    }
    return m;
}

#ifdef __cplusplus
#include <stdlib.h>

/* "key=value;key=a,b,c;..." (host only). Keys: on, floor, margin, draft (ms), cont, base (ms),
 * row (ms a row), basek, cost (ms for 0,1,2.. nodes), temp (depth 1,2,..). Unknown keys are
 * ignored. False on a malformed number. */
static inline bool rad_mtp_adapt_parse(const char* s, RadMtpAdapt* a) {
    rad_mtp_adapt_defaults(a);
    if (!s || !*s) return true;
    const char* p = s;
    while (*p) {
        const char* e = p;
        while (*e && *e != ';') ++e;
        const char* eq = p;
        while (eq < e && *eq != '=') ++eq;
        if (eq < e) {
            char key[16] = {};
            const size_t kl = (size_t)(eq - p);
            if (kl >= sizeof key) return false;
            memcpy(key, p, kl);
            char buf[256] = {};
            const size_t vl = (size_t)(e - eq - 1);
            if (vl >= sizeof buf) return false;
            memcpy(buf, eq + 1, vl);
            float vals[RAD_MTP_ADAPT_MAXC];
            int nv = 0;
            const char* q = buf;
            while (*q && nv < RAD_MTP_ADAPT_MAXC) {
                char* end = nullptr;
                const double d = strtod(q, &end);
                if (end == q || !(d == d) || d > 1e9 || d < -1e9) return false;   /* nan / inf / absurd */
                vals[nv++] = (float)d;
                q = end;
                if (*q == ',') ++q; else if (*q) return false;
            }
            if (nv < 1 || *q) return false;   /* empty, or a list longer than the tables */
            if (!strcmp(key, "on")) a->on = vals[0] != 0.0f;
            else if (!strcmp(key, "floor")) a->floor = (int)vals[0];
            else if (!strcmp(key, "margin")) a->margin = vals[0];
            else if (!strcmp(key, "draft")) a->draft_ms = vals[0];
            else if (!strcmp(key, "cont")) a->chain_cont = vals[0];
            else if (!strcmp(key, "base")) a->base_ms = vals[0];
            else if (!strcmp(key, "row")) a->row_ms = vals[0];
            else if (!strcmp(key, "basek")) a->base_k = (int)vals[0];
            else if (!strcmp(key, "cost")) {
                a->ncost = nv;
                for (int i = 0; i < nv; ++i) a->cost[i] = vals[i] > 0.0f ? vals[i] : 0.0f;
            } else if (!strcmp(key, "lift")) {
                a->nlift = nv < RAD_MTP_ADAPT_DEPTHS ? nv : RAD_MTP_ADAPT_DEPTHS;
                for (int i = 0; i < a->nlift; ++i) a->lift[i] = vals[i];
            } else if (!strcmp(key, "temp")) {
                a->ntemp = nv < RAD_MTP_ADAPT_DEPTHS ? nv : RAD_MTP_ADAPT_DEPTHS;
                for (int i = 0; i < a->ntemp; ++i) a->temp[i] = vals[i];
            }
        }
        p = *e ? e + 1 : e;
    }
    if (a->floor < 1) a->floor = 1;
    if (!(a->margin > 0.0f)) a->margin = 1.0f;
    return true;
}

/* The same, memoised per thread on the text: the launch path parses a few hundred bytes a pass. */
static inline bool rad_mtp_adapt_parse_cached(const char* s, RadMtpAdapt* a) {
    static thread_local char last[256];
    static thread_local RadMtpAdapt val;
    static thread_local bool have = false;
    if (!s || !*s) return rad_mtp_adapt_parse(s, a);
    if (have && strlen(s) < sizeof last && !strcmp(s, last)) { *a = val; return true; }
    if (!rad_mtp_adapt_parse(s, a)) return false;
    if (strlen(s) < sizeof last) { strcpy(last, s); val = *a; have = true; }
    return true;
}

/* Online calibration (host). Per depth, the head's predicted probability of the token it drafted
 * (the calibrated softmax's) against whether the verify accepted it; the temperature follows by a
 * damped multiplicative step on the gap, so an overconfident depth (predicted > observed) rises.
 * Observations are conditional: a node whose parent was rejected is no observation. */
struct RadMtpCalib {
    double pred[RAD_MTP_ADAPT_DEPTHS] = {}, obs[RAD_MTP_ADAPT_DEPTHS] = {}, n[RAD_MTP_ADAPT_DEPTHS] = {};
    float  temp[RAD_MTP_ADAPT_DEPTHS] = {1, 1, 1, 1, 1, 1, 1, 1};
    double decay = 0.995, eta = 1.5, t_lo = 0.5, t_hi = 3.0;
    void observe(int depth, double predicted_p, bool accepted) {
        int i = depth - 1;
        if (i < 0) i = 0;
        if (i >= RAD_MTP_ADAPT_DEPTHS) i = RAD_MTP_ADAPT_DEPTHS - 1;
        pred[i] = pred[i] * decay + predicted_p;
        obs[i]  = obs[i] * decay + (accepted ? 1.0 : 0.0);
        n[i]    = n[i] * decay + 1.0;
    }
    /* One refit step; depths with fewer than min_n effective observations keep their temperature. */
    void refit(double min_n = 50.0) {
        for (int i = 0; i < RAD_MTP_ADAPT_DEPTHS; ++i) {
            if (n[i] < min_n) continue;
            const double gap = (pred[i] - obs[i]) / n[i];   /* > 0: overconfident */
            double t = (double)temp[i] * ::exp(eta * gap);
            t = t < t_lo ? t_lo : t > t_hi ? t_hi : t;
            temp[i] = (float)t;
        }
    }
};
#endif /* __cplusplus */

#endif /* RAD_MTP_ADAPT_H */

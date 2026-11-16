/* rad_mtp_tree.h -- the best-first tree grower of a serial draft head (op `mtp_tree_grow`), as one
 * sequence's worth of plain code that libref runs on the host and libr4d on the device, so the two
 * cannot drift. docs/OPS.md ("Serial draft tree growth") is the contract.
 *
 * A head of R rounds grows a tree of N nodes, W frontier rows a pass. Node 0 is the root (the token
 * the verify starts from); nodes are numbered from 1 in creation order, so a parent always
 * precedes its child. After every head pass the grower reads the top-K candidates (K = 8, ids and
 * logits) of each frontier row it embedded, scores each candidate by its parent's cumulative log
 * probability plus the candidate's log-softmax over the row's own K logits (the head's coarse
 * shortlist is all it keeps of the distribution), pools them, and creates the best of the pool as
 * the next frontier (round < R: up to W, the budget N permitting) or, after the last pass, every
 * node the budget has left (they are never embedded). Ties go to the candidate pooled first
 * (lower frontier row, then lower candidate column).
 *
 * Confidence pruning (optional, off by default: then every statement above holds bit for bit). A
 * candidate becomes a pool entry only if its cumulative probability exp(cum) >= prune_abs AND its
 * probability >= prune_rel * the probability of the best candidate of its own parent (the first
 * column of the row's largest logit); a threshold of 0 is off. The best candidate of every frontier
 * row, the trunk, is never pruned, so with both thresholds high the tree is the plain chain of
 * best children. Pruned growth never invents a node: a pool that runs dry ends the creation (the
 * pass's frontier is shorter, a token of -1 ends its list) and the final tree's remaining entries
 * are INERT, token -1 and parent 0, which the core makes a duplicate of node 1 (the trunk's first
 * node) that no acceptance walk takes. The thresholds are compared in the log domain against
 * values the walk computes anyway: log(prune_abs) against the candidate's cumulative log
 * probability, log(prune_rel) against logit - max logit, both floats the host rounds once
 * (rad_mtp_tree_set_prune), so libref and libr4d decide identically.
 *
 * The state is plain 32-bit words per sequence, persistent across the passes of a step:
 *
 *     [0] head        the head's index of node 0 (its K and V sit at slot head, node k's at head + k)
 *     [1] count       nodes created (not counting the root)
 *     [2] nfront      the frontier rows the last pass embedded
 *     [3] pooled      candidates in the pool
 *     [4 .. 4+W)      the frontier's node ids, row order
 *     then N+1 nodes of 6 words: token, parent, depth, cumulative log probability (f32 bits),
 *     ancestors-or-self as a 64-bit mask over node ids >= 1 (low word, high word)
 *     then K*W*R pool entries of 4 words: parent, token, cumulative log probability (f32), used.
 *
 * The outputs after a pass that is not the last: `tok[w]` / `par[w]` the w-th new node's token and
 * parent id (a token of -1 ends the list), and for each of the next pass's W rows the tree
 * attention's operands (the head's attention over the committed pages plus the row's ancestors,
 * docs/OPS.md `attn_paged_tree_sel`): `sel` the committed pages below the row's region (the last
 * `sel_pitch` when there are more -- the head then reads a recency window, not QSA's choice),
 * `nsel` their count, `vmask` the region keys [rb, rb + 64) the row reads, `rb`
 * = block * floor(head / block). `src` / `dst` (W entries each, optional) route the head's hidden
 * state: row `seq * Q + id` of a per-node store (Q = 1 + (R - 1) * W) holds node `id`'s head
 * output; `src[w]` is the store row of new row w's parent (the row to continue from), `dst[w]` the
 * store row new row w's own output goes to (-1: none). After the last pass: the final tree, `tok[j-1]` / `par[j-1]` for
 * j = 1..N.
 */
#ifndef RAD_MTP_TREE_H
#define RAD_MTP_TREE_H

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
#include "rad_mtp_adapt.h"

#define RAD_MTP_TREE_K 8          /* candidates a row reports (the head's reranked top 8) */
#define RAD_MTP_TREE_MAX_NODES 60 /* the masks are 64 bits wide */

typedef struct RadMtpTreeCfg {
    int round;       /* 1 .. rounds: the pass whose candidates are read */
    int rounds;      /* R */
    int nodes;       /* N */
    int width;       /* W */
    int block;       /* the head's KV page, in tokens */
    int sel_pitch;   /* the page list's width; 0 when the attention operands are not written */
    int pitch;       /* tok/par row pitch, at least N */
    /* Confidence pruning (rad_mtp_tree_set_prune); a zeroed cfg prunes nothing. */
    int   prune;     /* nonzero when either threshold is on */
    float log_abs;   /* log(prune_abs), -inf when off: keep a candidate whose cum log prob >= it */
    float log_rel;   /* log(prune_rel), -inf when off: keep one whose logit - max logit >= it */
    /* Cost-aware node selection and calibration (abi/rad_mtp_adapt.h); a zeroed cfg is off. */
    RadMtpAdapt ad;
    /* Nonzero: the grower leaves the W page lists (sel) to the caller (the device grower writes them
     * with a whole workgroup, rad_mtp_tree_sel_span names what to write) and still writes nsel,
     * vmask and rb. Zero (every host caller): it writes them itself. */
    int skip_sel_lists;
} RadMtpTreeCfg;

static inline int rad_mtp_tree_state_words(int nodes, int width, int rounds) {
    return 4 + width + 6 * (nodes + 1) + 4 * RAD_MTP_TREE_K * width * rounds;
}

/* The thresholds (probabilities, 0 = off) into the cfg's log domain. */
RAD_MTP_HD static inline void rad_mtp_tree_set_prune(RadMtpTreeCfg* c, double prune_abs,
                                                     double prune_rel) {
    const int a = prune_abs > 0.0, r = prune_rel > 0.0;
    c->prune   = a || r;
    c->log_abs = a ? logf((float)prune_abs) : -INFINITY;
    c->log_rel = r ? logf((float)prune_rel) : -INFINITY;
}

/* One candidate row's normalised scores, the arithmetic every consumer shares (the grower's pool,
 * the core's stage-1 leaf capture): out[c] = (v_c - mx) - lse for a valid candidate (id >= 0, logit
 * a number above -inf), NaN for an invalid one; *mx_out the largest valid logit, *lse_out the log
 * of the sum of exp(v - mx) in column order. Returns the first column of the largest logit, or -1
 * when no candidate is valid (then nothing is written but out). */
RAD_MTP_HD static inline int rad_mtp_tree_row_logp(const int* idx, const float* val, float* out,
                                                    float* mx_out, float* lse_out) {
    float mx = -INFINITY;
    int best = -1;
    for (int c = 0; c < RAD_MTP_TREE_K; ++c) {
        out[c] = NAN;
        const float v = val[c];
        if (idx[c] >= 0 && v == v && v > mx) { mx = v; best = c; }
    }
    if (best < 0) return -1;
    float sum = 0.0f;
    for (int c = 0; c < RAD_MTP_TREE_K; ++c) {
        const float v = val[c];
        if (idx[c] >= 0 && v == v && v > -INFINITY) sum += expf(v - mx);
    }
    const float lse = logf(sum);
    for (int c = 0; c < RAD_MTP_TREE_K; ++c) {
        const float v = val[c];
        if (idx[c] >= 0 && v == v && v > -INFINITY) out[c] = (v - mx) - lse;
    }
    *mx_out = mx;
    *lse_out = lse;
    return best;
}

/* Whether candidate c of a row scored by rad_mtp_tree_row_logp is kept by the cfg's pruning, given
 * its parent's cumulative log probability `base`: the best column always, any column when pruning
 * is off, else both thresholds. `v` its logit. */
RAD_MTP_HD static inline int rad_mtp_tree_keep(const RadMtpTreeCfg* cfg, int c, int best,
                                               float base, float lp, float v, float mx) {
    if (!cfg->prune || c == best) return 1;
    return (base + lp) >= cfg->log_abs && (v - mx) >= cfg->log_rel;
}

/* The page list a pass's attention operands carry: the head's pages table[p0 .. p0 + n), then -1 up to
 * sel_pitch, the same for every frontier row of the sequence (it depends on the head's position and
 * the table alone, not on the node). Returns n, the nsel value. */
RAD_MTP_HD static inline int rad_mtp_tree_sel_span(const RadMtpTreeCfg* cfg, int head, int* p0_out) {
    const int rb = cfg->block * (head / cfg->block);
    const int nb = rb / cfg->block;
    const int p0 = nb > cfg->sel_pitch ? nb - cfg->sel_pitch : 0;
    *p0_out = p0;
    return nb - p0;
}

RAD_MTP_HD static inline float rad_mtp_tree_bits_f(int v) { float f; memcpy(&f, &v, 4); return f; }
RAD_MTP_HD static inline int   rad_mtp_tree_f_bits(float f) { int v; memcpy(&v, &f, 4); return v; }

/* One sequence. `st` its state words; `didx` / `dval` the candidate rows of this pass (`row0` the
 * sequence's first row, `row_stride` elements between rows, `stride` between rows of one sequence
 * in a pass of `width` rows a sequence, so row w is row0 + w * stride); `head_in` the head index of
 * node 0 for round 1 (negative: keep the state's). `tok` / `par` the sequence's output rows;
 * `sel` / `nsel` / `vmask` the W rows of the next pass's operands (null: none), `table` the
 * sequence's block-table row (null: none), `rb_out` its region base, `seq` its index and `src_out` / `dst_out`
 * the hidden-state routing (null: none). Returns 0, or a negative
 * code: -1 a bad configuration. */
RAD_MTP_HD static inline int rad_mtp_tree_grow_seq(
    const RadMtpTreeCfg* cfg, int* st, const int* didx, const float* dval, long row0,
    long row_stride, long stride, int head_in, const int* table, int* tok, int* par, int* sel,
    int* nsel, long long* vmask, int* rb_out, long seq, int* src_out, int* dst_out) {
    const int K = RAD_MTP_TREE_K, N = cfg->nodes, W = cfg->width, R = cfg->rounds;
    if (N < 1 || N > RAD_MTP_TREE_MAX_NODES || W < 1 || R < 1 || cfg->round < 1 ||
        cfg->round > R || cfg->pitch < N)
        return -1;
    int* front = st + 4;
    int* nd    = st + 4 + W;                       /* node k: nd[6k .. 6k+6) */
    int* pool  = st + 4 + W + 6 * (N + 1);         /* entry e: pool[4e .. 4e+4) */
    const int pcap = K * W * R;

    if (cfg->round == 1) {
        st[1] = 0; st[2] = 1; st[3] = 0;
        st[0] = head_in >= 0 ? head_in : 0;
        front[0] = 0;
        nd[0] = -1; nd[1] = -1; nd[2] = 0; nd[3] = rad_mtp_tree_f_bits(0.0f); nd[4] = 0; nd[5] = 0;
    }
    const int nfront = st[2];

    /* pool this pass's candidates */
    for (int w = 0; w < nfront && w < W; ++w) {
        const int f = front[w];
        const long r = row0 + (long)w * stride;
        float lp[RAD_MTP_TREE_K], mx = 0.0f, lse = 0.0f;
        const int bestc = rad_mtp_tree_row_logp(didx + r * row_stride, dval + r * row_stride, lp,
                                                &mx, &lse);
        if (bestc < 0) continue;
        /* the calibration of this row's law (a monotone map of its own values: slot priorities
         * stay slot priorities); the relative-confidence test reads the same scaled gap */
        const float tcal = rad_mtp_adapt_temp(&cfg->ad, nd[6 * f + 2] + 1);
        rad_mtp_adapt_recal(lp, K, tcal);
        rad_mtp_adapt_floor(lp, K, rad_mtp_adapt_lift(&cfg->ad, nd[6 * f + 2] + 1));
        const float base = rad_mtp_tree_bits_f(nd[6 * f + 3]);
        for (int c = 0; c < K; ++c) {
            if (!(lp[c] == lp[c])) continue;     /* NaN: an invalid candidate */
            if (st[3] >= pcap) break;
            if (!rad_mtp_tree_keep(cfg, c, bestc, base, lp[c],
                                   tcal == 1.0f ? dval[r * row_stride + c]
                                                : mx + (dval[r * row_stride + c] - mx) / tcal,
                                   mx))
                continue;
            int* e = pool + 4 * st[3]++;
            e[0] = f; e[1] = didx[r * row_stride + c]; e[2] = rad_mtp_tree_f_bits(base + lp[c]);
            e[3] = 0;
        }
    }

    /* create. Adaptive selection acts like pruning: a pool that runs dry or a refused candidate
     * ends the creation and the tree's remaining entries are inert. */
    const int shrink = cfg->prune || cfg->ad.on;
    int n_new = cfg->round < R ? W : N;
    if (n_new > N - st[1]) n_new = N - st[1];
    int made[RAD_MTP_TREE_MAX_NODES + 1];
    int n_made = 0;
    for (int j = 0; j < n_new; ++j) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < st[3]; ++e) {
            if (pool[4 * e + 3]) continue;
            const float v = rad_mtp_tree_bits_f(pool[4 * e + 2]);
            if (best < 0 || v > bv) { best = e; bv = v; }
        }
        /* A pruned pool that runs dry ends the creation: no placeholder node is made (but node 1,
         * the trunk's first, always exists), the rest of the tree stays inert. */
        if (best < 0 && shrink && st[1] > 0) break;
        if (best >= 0 && cfg->ad.on && st[1] > 0) {
            float cum[RAD_MTP_TREE_MAX_NODES];
            for (int q = 1; q <= st[1]; ++q) cum[q - 1] = rad_mtp_tree_bits_f(nd[6 * q + 3]);
            if (!rad_mtp_adapt_admit(&cfg->ad, st[1], rad_mtp_adapt_expect(cum, st[1]), expf(bv)))
                break;
        }
        const int id = ++st[1];
        int* n = nd + 6 * id;
        if (best < 0) {
            /* a dry pool: a copy of the first node's token under the root, which the verify's walk
             * never takes (the first copy is the one that is) */
            n[0] = id > 1 ? nd[6 * 1 + 0] : 0; n[1] = 0; n[2] = 1;
            n[3] = rad_mtp_tree_f_bits(-INFINITY);
            n[4] = id < 32 ? (1 << id) : 0; n[5] = id >= 32 ? (1 << (id - 32)) : 0;
        } else {
            int* e = pool + 4 * best;
            e[3] = 1;
            const int p = e[0];
            n[0] = e[1]; n[1] = p; n[2] = nd[6 * p + 2] + 1; n[3] = e[2];
            n[4] = nd[6 * p + 4] | (id < 32 ? (1 << id) : 0);
            n[5] = nd[6 * p + 5] | (id >= 32 ? (1 << (id - 32)) : 0);
        }
        made[n_made++] = id;
    }

    if (cfg->round >= R) {
        for (int j = 1; j <= N; ++j) {
            /* pruned: the entries past the created nodes are inert (-1), else a copy of node 1's */
            tok[j - 1] = j <= st[1] ? nd[6 * j + 0]
                                    : shrink ? -1 : (st[1] >= 1 ? nd[6 * 1 + 0] : 0);
            par[j - 1] = j <= st[1] ? nd[6 * j + 1] : 0;
        }
        st[2] = 0;
        return 0;
    }

    /* The hidden-state routing of the next pass. A head pass's hidden rows are kept per node in a
     * store of Q = 1 + (R - 1) * W rows a sequence (row `seq * Q + id`): the pass scatters each
     * frontier row to its node's store row (`dst`, -1 for a row past the new nodes), and the next
     * pass gathers each of its rows from its parent's (`src`). A pooled candidate may hang off a
     * node of any earlier pass, not only the last, so the store keeps them all; a row past the new
     * nodes reads the root's, which is finite and whose output is discarded. */
    if (src_out || dst_out) {
        const long q = 1 + (long)(R - 1) * W;
        for (int w = 0; w < W; ++w) {
            if (src_out) src_out[w] = (int)(seq * q + (w < n_made ? nd[6 * made[w] + 1] : 0));
            if (dst_out) dst_out[w] = w < n_made ? (int)(seq * q + made[w]) : -1;
        }
    }

    /* the next frontier */
    for (int w = 0; w < W; ++w) {
        if (w < n_made) {
            front[w] = made[w];
            tok[w] = nd[6 * made[w] + 0];
            par[w] = nd[6 * made[w] + 1];
        } else {
            tok[w] = -1;
            par[w] = 0;
        }
    }
    st[2] = n_made;
    if (cfg->sel_pitch > 0 && sel && nsel && vmask && table && rb_out && cfg->block > 0) {
        const int head = st[0];
        const int B = cfg->block;
        const int rb = B * (head / B);
        const int nb = rb / B;
        const int p0 = nb > cfg->sel_pitch ? nb - cfg->sel_pitch : 0;
        *rb_out = rb;
        const long long own = ((long long)1 << (head - rb + 1)) - 1;   /* [rb, head], the root */
        for (int w = 0; w < W; ++w) {
            if (!cfg->skip_sel_lists)
                for (int k = 0; k < cfg->sel_pitch; ++k)
                    sel[(long)w * cfg->sel_pitch + k] = (k < nb - p0) ? table[p0 + k] : -1;
            nsel[w] = nb - p0;
            if (w < n_made) {
                const int id = made[w];
                const unsigned lo = (unsigned)nd[6 * id + 4], hi = (unsigned)nd[6 * id + 5];
                const unsigned long long anc = ((unsigned long long)hi << 32) | lo;
                vmask[w] = own | (long long)(anc << (head - rb));
            } else {
                vmask[w] = 1;   /* an inert row still reads one key and its output is discarded */
            }
        }
    }
    return 0;
}

#endif /* RAD_MTP_TREE_H */

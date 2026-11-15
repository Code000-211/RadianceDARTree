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
 * = block * floor(head / block). After the last pass: the final tree, `tok[j-1]` / `par[j-1]` for
 * j = 1..N.
 */
#ifndef RAD_MTP_TREE_H
#define RAD_MTP_TREE_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __HIPCC__
#define RAD_MTP_HD __host__ __device__
#else
#define RAD_MTP_HD
#endif

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
} RadMtpTreeCfg;

static inline int rad_mtp_tree_state_words(int nodes, int width, int rounds) {
    return 4 + width + 6 * (nodes + 1) + 4 * RAD_MTP_TREE_K * width * rounds;
}

RAD_MTP_HD static inline float rad_mtp_tree_bits_f(int v) { float f; memcpy(&f, &v, 4); return f; }
RAD_MTP_HD static inline int   rad_mtp_tree_f_bits(float f) { int v; memcpy(&v, &f, 4); return v; }

/* One sequence. `st` its state words; `didx` / `dval` the candidate rows of this pass (`row0` the
 * sequence's first row, `row_stride` elements between rows, `stride` between rows of one sequence
 * in a pass of `width` rows a sequence, so row w is row0 + w * stride); `head_in` the head index of
 * node 0 for round 1 (negative: keep the state's). `tok` / `par` the sequence's output rows;
 * `sel` / `nsel` / `vmask` the W rows of the next pass's operands (null: none), `table` the
 * sequence's block-table row (null: none) and `rb_out` its region base. Returns 0, or a negative
 * code: -1 a bad configuration. */
RAD_MTP_HD static inline int rad_mtp_tree_grow_seq(
    const RadMtpTreeCfg* cfg, int* st, const int* didx, const float* dval, long row0,
    long row_stride, long stride, int head_in, const int* table, int* tok, int* par, int* sel,
    int* nsel, long long* vmask, int* rb_out) {
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
        float mx = -INFINITY;
        for (int c = 0; c < K; ++c) {
            const float v = dval[r * row_stride + c];
            if (didx[r * row_stride + c] >= 0 && v == v && v > mx) mx = v;
        }
        if (!(mx > -INFINITY)) continue;
        float sum = 0.0f;
        for (int c = 0; c < K; ++c) {
            const float v = dval[r * row_stride + c];
            if (didx[r * row_stride + c] >= 0 && v == v && v > -INFINITY) sum += expf(v - mx);
        }
        const float lse = logf(sum);
        const float base = rad_mtp_tree_bits_f(nd[6 * f + 3]);
        for (int c = 0; c < K; ++c) {
            const float v = dval[r * row_stride + c];
            const int   t = didx[r * row_stride + c];
            if (t < 0 || !(v == v) || !(v > -INFINITY)) continue;
            if (st[3] >= pcap) break;
            int* e = pool + 4 * st[3]++;
            e[0] = f; e[1] = t; e[2] = rad_mtp_tree_f_bits(base + ((v - mx) - lse)); e[3] = 0;
        }
    }

    /* create */
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
            tok[j - 1] = j <= st[1] ? nd[6 * j + 0] : (st[1] >= 1 ? nd[6 * 1 + 0] : 0);
            par[j - 1] = j <= st[1] ? nd[6 * j + 1] : 0;
        }
        st[2] = 0;
        return 0;
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

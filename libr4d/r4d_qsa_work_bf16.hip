// The QSA indexer's work list: which blocks complete this step, where their raw keys are, and
// where their block key goes.
//
// qsa_block_key reads each block's `ratio` raw keys from consecutive rows of one plane, but a
// block completing at decode has its keys partly in the tail and partly in this step, and blocks
// belong to different sequences. This op finds the completed blocks, gathers each one's raw keys
// into consecutive rows of a staging plane, and writes its destination page and first-token
// position: exactly qsa_block_key's three operands (docs/QSA.md). The copy is small (1 KiB per
// completing block) and keeps qsa_block_key single-source.
//
// The tail is a per-layer linear KV state ([slots, 1, ring, n]; `sidx` names the sequence's slot,
// not the step-local batch row); see r4d_qsa_tail_store_bf16.hip, including why ring is
// ratio + n_spec. Keys are read at row q % ring to match.
//
// The launch is bounded on the host (ceil(T / ratio) + n_seq: one block per `ratio` tokens plus
// one straddling block per sequence). Work items past the real count get page = -1, which
// qsa_block_key skips.
//
// TREE VERIFY AND HEAL (docs/QSA.md, "Tree verify"; libref/ref_qsa.cpp states the contract):
//   * `num_accepted` (heal): a sequence that committed na >= 2 tokens last step also owes the blocks
//     whose last token is one of the na - 1 non-anchor committed positions, which a tree verify
//     never writes. They are ordinary work items ahead of the sequence's own, every key from the
//     (already compacted) tail.
//   * `spec_parent`: a sequence whose first row has parent -1 is a tree verify and completes only
//     its anchor's block (its other rows' blocks may never be committed); nc is p0 / ratio. Every
//     row completing a block additionally gets a SCRATCH item, launched as `T` more workgroups after
//     the `max_work` ones, whose keys are gathered along the row's ancestors, and every row gets its
//     region tables (rtab, rnc) and rcu[w] = w for the second, per-row qsa_score.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
/* The sequence loop's bound, checked by the launcher against the real count from cu_seqlens. Not a
 * row constraint on the declared seqs (rad-convert declares 256, which must still resolve). */
constexpr int kMaxSeq  = 256;
/* The widest block (ratio) the scratch gather keeps its source rows for. */
constexpr int kMaxRatio = 32;

#define QSA_WORK_PARAMS                                                                         \
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu, const uint16_t* __restrict__ state,                         \
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ bt,                           \
    uint16_t* __restrict__ stage, int32_t* __restrict__ page, int32_t* __restrict__ bpos,       \
    int32_t* __restrict__ nc, int n, int ratio, int ring, int seqs, int max_work, int k_ld,     \
    int s_slot, int s_row, int sidx_ld, int bt_ld, int g_ld,                                    \
    const int32_t* __restrict__ rope, int rplane, int bstep,                                    \
    const int32_t* __restrict__ sp, const int32_t* __restrict__ na,                             \
    uint16_t* __restrict__ sstage, int32_t* __restrict__ spage, int32_t* __restrict__ sbpos,    \
    int32_t* __restrict__ rtab, int32_t* __restrict__ rnc, int32_t* __restrict__ rcu,           \
    int T, int rt_ld, int rt_cols, int sg_ld, int sbstep

/* Tree sequence: its first row has parent -1 (spec_parent absent: never). */
__device__ __forceinline__ bool qw_tree(const int32_t* sp, const int32_t* cu, int i) {
    return sp && cu[i + 1] > cu[i] && sp[cu[i]] == -1;
}
/* The ancestor-or-self of row x (of the sequence starting at t0) at position q. */
__device__ __forceinline__ int qw_anc(const int32_t* pos, const int32_t* sp, int t0, int x, int q) {
    int y = x;
    for (int g = 0; g < 64 && pos[y] > q; ++g) {
        const int pj = sp[y];
        if (pj < 0) break;
        y = t0 + pj;
    }
    return y;
}

/* The scratch item of row x (workgroup max_work + x): its region tables, and, when its position
 * completes a block, the block's keys along its ancestors. */
__device__ __forceinline__ void qsa_work_scratch(QSA_WORK_PARAMS, int x) {
    const int t = (int)threadIdx.x;
    __shared__ int sh_src[kMaxRatio];
    if (x == 0 && t == 0) rcu[T] = T;
    if (t == 0) {
        rcu[x] = x; spage[x] = -1; rnc[x] = 0;
        for (int c = 0; c < (rope ? 3 : 1); ++c) sbpos[(size_t)x * sbstep + c] = 0;
    }
    for (int i = t; i < rt_cols; i += kThreads) rtab[(size_t)x * rt_ld + i] = -1;
    int s = -1;
    for (int i = 0; i < seqs && i < kMaxSeq; ++i)
        if (x >= cu[i] && x < cu[i + 1]) { s = i; break; }
    if (s < 0 || !qw_tree(sp, cu, s)) return;
    const int t0 = cu[s], p0 = pos[t0], px = pos[x], b0 = p0 / ratio;
    const int si = sidx[(size_t)s * sidx_ld];
    if (si < 0) return;                      /* no slot: no keys, rnc 0, rtab -1 */
    int nr = (px + 1) / ratio - b0;
    if (nr > rt_cols) nr = rt_cols;
    if (nr < 0) nr = 0;
    /* The table's -1 fill above and its entries below are different threads' writes to one row. */
    __syncthreads();
    for (int i = t; i < nr; i += kThreads)
        rtab[(size_t)x * rt_ld + i] = qw_anc(pos, sp, t0, x, (b0 + i) * ratio + ratio - 1);
    if (t == 0) rnc[x] = nr;
    if ((px + 1) % ratio != 0 || si < 0) return;
    if (t == 0) spage[x] = x;
    if (t < ratio) {
        const int q = px - ratio + 1 + t;
        sh_src[t] = q >= p0 ? qw_anc(pos, sp, t0, x, q) : -1;
    }
    __syncthreads();
    if (t == 0) {
        /* The block's first token: its ancestor row's own components when in this step, else text
         * at its index plus the shift of the sequence's first row (as the committed items). */
        const int f0 = px - ratio + 1;
        if (!rope) {
            sbpos[(size_t)x * sbstep] = f0;
        } else {
            for (int c = 0; c < 3; ++c)
                sbpos[(size_t)x * sbstep + c] =
                    sh_src[0] >= 0 ? rope[(size_t)c * rplane + sh_src[0]] : f0 + (rope[t0] - p0);
        }
    }
    for (int i = 0; i < ratio; ++i) {
        const int q = px - ratio + 1 + i;
        const uint16_t* src = sh_src[i] >= 0 ? k + (size_t)sh_src[i] * k_ld
                                             : state + (size_t)si * s_slot + (size_t)(q % ring) * s_row;
        uint16_t* dst = sstage + (size_t)(x * ratio + i) * sg_ld;
        for (int d = t; d < n; d += kThreads) dst[d] = src[d];
    }
}

__device__ __forceinline__ void qsa_work_body(QSA_WORK_PARAMS) {

    const int w = (int)blockIdx.x;
    /* Complete blocks per sequence (needed by the scorer and the selection), written by the first
     * `seqs` workgroups instead of a separate launch. A tree sequence's are the committed blocks
     * below the one holding its anchor. */
    if (w < seqs && threadIdx.x == 0) {
        const int t1 = cu[w + 1];
        nc[w] = t1 <= cu[w] ? 0 : qw_tree(sp, cu, w) ? pos[cu[w]] / ratio : (pos[t1 - 1] + 1) / ratio;
    }
    if (w >= max_work) {
        if (sp && w - max_work < T)
            qsa_work_scratch(k, pos, cu, state, sidx, bt, stage, page, bpos, nc, n, ratio, ring,
                             seqs, max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, rope,
                             rplane, bstep, sp, na, sstage, spage, sbpos, rtab, rnc, rcu, T, rt_ld,
                             rt_cols, sg_ld, sbstep, w - max_work);
        return;
    }

    /* The sequence owning work item w, by a running count recomputed in every workgroup (seqs is
     * small; a prefix would need a second launch). */
    int s = -1, j = 0;
    int run = 0;
    for (int i = 0; i < seqs && i < kMaxSeq; ++i) {
        const int t0 = cu[i], t1 = cu[i + 1];
        if (t1 <= t0) continue;
        const int p0 = pos[t0];
        /* A tree sequence completes only its anchor's block (see the header). */
        const int p1 = qw_tree(sp, cu, i) ? p0 : pos[t1 - 1];
        /* A block completes this step when its last token, b * ratio + ratio - 1, is in [p0, p1].
         * Both bounds use non-negative numerators: C division truncates toward zero, so
         * (p1 - (ratio - 1)) / ratio would be 0, not -1, for p1 < ratio - 1. */
        int lo = p0 / ratio;                                        /* ceil((p0-ratio+1)/ratio) */
        /* Heal: the blocks whose last token is one of the na - 1 non-anchor positions the last
         * step committed, [p0 - na + 1, p0 - 1], ahead of the sequence's own. */
        if (na && na[i] >= 2 && p0 - na[i] >= 0) {   /* a stale na on a short sequence heals nothing */
            const int x = p0 - na[i] - ratio + 2;
            const int lh = x <= 0 ? 0 : (x + ratio - 1) / ratio;
            if (lh < lo) lo = lh;
        }
        const int hi = (p1 + 1) / ratio - 1;                        /* floor((p1-ratio+1)/ratio) */
        const int cnt = hi >= lo ? hi - lo + 1 : 0;
        if (w < run + cnt) { s = i; j = w - run + lo; break; }
        run += cnt;
    }
    if (s < 0) {
        /* Past the real count: page = -1 makes the host's bound safe. */
        if (threadIdx.x == 0) {
            page[w] = -1;
            for (int c = 0; c < (rope ? 3 : 1); ++c) bpos[(size_t)w * bstep + c] = 0;
        }
        return;
    }

    const int b  = j;                       /* the block index within the sequence */
    const int t0 = cu[s];
    const int p0 = pos[t0];
    /* Column 0 is the committed slot; -1 marks a sequence with no state, whose blocks cannot be
     * built (page -1, no gather). */
    const int si = sidx[(size_t)s * sidx_ld];

    if (threadIdx.x == 0) {
        page[w] = si < 0 ? -1 : bt[(size_t)s * bt_ld + b];
        if (!rope) {
            bpos[(size_t)w * bstep] = b * ratio;
        } else {
            /* Rotary position of the block's first token, three components side by side, items
             * bstep apart. A first token in this step reads its own components. One from an
             * earlier step is text (only a decode row or a prompt's last chunk leaves a block open,
             * and a prompt ends in text), so it rotates at its index plus the shift carried by
             * this step's first row of the sequence. */
            const int f = b * ratio;
            for (int c = 0; c < 3; ++c)
                bpos[(size_t)w * bstep + c] =
                    f >= p0 ? rope[(size_t)c * rplane + t0 + (f - p0)] : f + (rope[t0] - p0);
        }
    }
    if (si < 0) return;

    /* The block's raw keys, each from this step or from the tail. */
    for (int i = 0; i < ratio; ++i) {
        const int q = b * ratio + i;                 /* this key's position */
        const uint16_t* src;
        if (q >= p0) src = k + (size_t)(t0 + (q - p0)) * k_ld;
        else         src = state + (size_t)si * s_slot + (size_t)(q % ring) * s_row;
        uint16_t* dst = stage + (size_t)(w * ratio + i) * g_ld;
        for (int d = (int)threadIdx.x; d < n; d += kThreads) dst[d] = src[d];
    }
}

/* The kernel around the body, whose blocks and threads return where their work ends: CH `r4dchain`
 * is the chained form (r4d_chain.h). */
template <class CH>
__global__ __launch_bounds__(kThreads)
void qsa_work_kernel(QSA_WORK_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_work_body(k, pos, cu, state, sidx, bt, stage, page, bpos, nc, n, ratio, ring, seqs,
                  max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, rope, rplane, bstep, sp, na,
                  sstage, spage, sbpos, rtab, rnc, rcu, T, rt_ld, rt_cols, sg_ld, sbstep);
    if (CH::on) r4d_chain_done();
}
#undef QSA_WORK_PARAMS

}  // namespace

//   k      [T][n]                  bf16, this step's raw index keys
//   pos    [T]                     i32
//   cu     [seqs + 1]              i32
//   state  [slots][ring][n]        bf16, the tail group's cache for this layer
//   sidx   [seqs][sidx_ld]         i32, the tail group's state row; column 0 is the slot
//   bt     [seqs][bt_ld]           i32, the block-key group's block table
//   stage  [max_work * ratio][n]   bf16
//   page   [max_work]              i32, -1 past the real count
//   bpos   [max_work][bstep]       i32, a work item's first-token position at [w * bstep]
//   rope   [3][rplane]             i32, optional: this step's rotary components (RadBatch::
//                                  rope_pos); present, a work item's three components are
//                                  bpos[w * bstep + 0..2]
//   sp     [T]                     i32, optional: spec_parent (tree verify)
//   na     [seqs]                  i32, optional: num_accepted (heal)
//   sstage [T * ratio][n]          bf16, with sp: row x's scratch keys
//   spage, rnc [T]                 i32, with sp
//   sbpos  [T][sbstep]             i32, with sp: row x's scratch block's first-token position; with
//                                  rope its three components, as bpos
//   rtab   [T][rt_cols]            i32, with sp: row x's region blocks' scratch rows, -1 past rnc[x]
//   rcu    [T + 1]                 i32, with sp: rcu[w] = w
static int r4d_qsa_work_bf16_emit(long k, long pos, long cu, long state, long sidx, long bt,
                                 long stage, long page, long bpos, long nc, int n, int ratio,
                                 int ring, int seqs, int max_work, int k_ld, int s_slot,
                                 int s_row, int sidx_ld, int bt_ld, int g_ld, long rope,
                                 int rplane, int bstep, long sp, long na, long sstage, long spage,
                                 long sbpos, long rtab, long rnc, long rcu, int T, int rt_ld,
                                 int rt_cols, int sg_ld, int sbstep, RadStream stream, bool chain) {
    if (!k || !pos || !cu || !state || !sidx || !bt || !stage || !page || !bpos || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (n <= 0 || ratio <= 0 || ring < ratio || seqs <= 0 || max_work <= 0 || seqs > kMaxSeq)
        return R4D_NO(RAD_E_SHAPE);
    if (!k_ld) k_ld = n;
    if (!s_row) s_row = n;
    if (!g_ld) g_ld = n;
    if (!sidx_ld) sidx_ld = 1;
    const bool scr = sp != 0;
    if (scr && (!sstage || !spage || !sbpos || !rtab || !rnc || !rcu || ratio > kMaxRatio || T <= 0))
        return R4D_NO(RAD_E_INVAL);
    if (!scr && (sstage || spage || sbpos || rtab || rnc || rcu)) return R4D_NO(RAD_E_INVAL);
    if (!sg_ld) sg_ld = n;
    if (!sbstep) sbstep = 1;
    /* The scratch items are `T` more workgroups after the `max_work` ones. */
    const unsigned grid = (unsigned)(max_work + (scr ? T : 0));
    /* Each form named at its own launch: a test reads the kernel a launch site names. */
#define R4D_QWORK(CH_)                                                                        \
    R4D_EMIT(stream, qsa_work_kernel<CH_>, dim3(grid), dim3(kThreads), 0,                     \
             (const uint16_t*)k, (const int32_t*)pos, (const int32_t*)cu,                     \
             (const uint16_t*)state, (const int32_t*)sidx, (const int32_t*)bt,                \
             (uint16_t*)stage, (int32_t*)page, (int32_t*)bpos, (int32_t*)nc, n, ratio, ring,  \
             seqs, max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, (const int32_t*)rope, \
             rplane, bstep, (const int32_t*)sp, (const int32_t*)na, (uint16_t*)sstage,        \
             (int32_t*)spage, (int32_t*)sbpos, (int32_t*)rtab, (int32_t*)rnc, (int32_t*)rcu,  \
             T, rt_ld, rt_cols, sg_ld, sbstep)
    const int rc = chain ? R4D_QWORK(r4dchain) : R4D_QWORK(R4dUnchained);
#undef R4D_QWORK
    return rc;
}

// ============================================================== the plugin ABI entry point

enum { QW_K = 0, QW_POS, QW_CU, QW_STATE, QW_SIDX, QW_BT, QW_ROPE, QW_STAGE, QW_PAGE, QW_BPOS,
       QW_NC, QW_SP, QW_NA, QW_SSTAGE, QW_SPAGE, QW_SBPOS, QW_RTAB, QW_RNC, QW_RCU };

extern "C" int r4d_rad_qsa_work(const RadArgs* a, RadStream stream) {
    const RadTensor* k  = r4d_opt(a, QW_K);
    const RadTensor* po = r4d_opt(a, QW_POS);
    const RadTensor* cu = r4d_opt(a, QW_CU);
    const RadTensor* tl = r4d_opt(a, QW_STATE);
    const RadTensor* sx = r4d_opt(a, QW_SIDX);
    const RadTensor* bt = r4d_opt(a, QW_BT);
    const RadTensor* rp = r4d_opt(a, QW_ROPE);       /* optional: multi-component positions */
    const RadTensor* st = r4d_opt(a, QW_STAGE);
    const RadTensor* pg = r4d_opt(a, QW_PAGE);
    const RadTensor* bp = r4d_opt(a, QW_BPOS);
    const RadTensor* nc = r4d_opt(a, QW_NC);
    const RadTensor* sp = r4d_opt(a, QW_SP);          /* optional: tree verify */
    const RadTensor* na = r4d_opt(a, QW_NA);          /* optional: heal */
    const RadTensor* ss = r4d_opt(a, QW_SSTAGE);
    const RadTensor* sg = r4d_opt(a, QW_SPAGE);
    const RadTensor* sb = r4d_opt(a, QW_SBPOS);
    const RadTensor* rt = r4d_opt(a, QW_RTAB);
    const RadTensor* rn = r4d_opt(a, QW_RNC);
    const RadTensor* rc = r4d_opt(a, QW_RCU);
    if (!k || !po || !cu || !tl || !sx || !bt || !st || !pg || !bp || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (k->rank != 2 || bt->rank != 2 || st->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* The tail state arrives as its group declared it: [slots, heads, ring, n], heads 1. See
     * r4d_qsa_tail_store_bf16.hip. */
    if (tl->rank != 4 || tl->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (tl->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    /* bt is the block-key group's table, not the attention group's; both are paged at `ratio`,
     * so they share block indices (docs/QSA.md). */
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long ring  = r4d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R4D_NO(RAD_E_INVAL);
    const long long seqs = r4d_numel(cu) - 1;
    const long long T    = k->shape[0];
    if (seqs < 1 || T < 1 || r4d_numel(po) < T) return R4D_NO(RAD_E_SHAPE);
    if (k->shape[1] != n || st->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (tl->shape[2] < ring || tl->shape[3] != n) return R4D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* The work count comes from `page`, and `stage` must hold ratio rows per item. */
    const long long mw = r4d_numel(pg);
    if (mw < 1 || st->shape[0] < mw * ratio) return R4D_NO(RAD_E_SHAPE);
    /* bpos is one position per work item, or a rank-2 [>= work, 1 or 3] column slice of a
     * [max_work, 3] buffer (components side by side, items a row pitch apart); three columns are
     * required with `rope`. rope is [3, rplane] component-major, rplane >= the step's tokens. */
    long long bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return R4D_NO(RAD_E_SHAPE);
        bstep = bp->stride[0];
    } else if (rp || r4d_numel(bp) < mw) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T || !r4d_contig(rp) ||
               rp->dtype != RAD_I32))
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(nc) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* nc is written by the first `seqs` workgroups, so the launch must be at least that wide. */
    if (mw < seqs) return R4D_NO(RAD_E_SHAPE);
    /* The tree operands go together: spec_parent with its six outputs. */
    const bool scr = sp != nullptr;
    long long sbstep = 1;
    if (scr != (ss && sg && sb && rt && rn && rc) || (!scr && (ss || sg || sb || rt || rn || rc)))
        return R4D_NO(RAD_E_INVAL);
    if (scr) {
        /* sbpos: [T] or a [>= T, 1 or 3] column slice (three columns with `rope`), as bpos. */
        if (sb->rank == 2) {
            if (sb->shape[0] < T || sb->stride[1] != 1 || sb->stride[0] < sb->shape[1] ||
                (rp && sb->shape[1] < 3))
                return R4D_NO(RAD_E_SHAPE);
            sbstep = sb->stride[0];
        } else if (rp || !r4d_contig(sb)) {
            return R4D_NO(rp ? RAD_E_SHAPE : RAD_E_STRIDE);
        }
        if (ratio > kMaxRatio) return R4D_NO(RAD_E_SHAPE);
        if (r4d_numel(sp) < T || ss->rank != 2 || ss->shape[0] < T * ratio || ss->shape[1] != n ||
            r4d_numel(sg) < T || r4d_numel(sb) < T || rt->rank != 2 || rt->shape[0] < T ||
            r4d_numel(rn) < T || r4d_numel(rc) < T + 1)
            return R4D_NO(RAD_E_SHAPE);
        if (sp->dtype != RAD_I32 || sg->dtype != RAD_I32 || sb->dtype != RAD_I32 ||
            rt->dtype != RAD_I32 || rn->dtype != RAD_I32 || rc->dtype != RAD_I32 ||
            !r4d_contig(sp) || !r4d_contig(sg) || !r4d_contig(rn) ||
            !r4d_contig(rc) || rt->stride[1] != 1 || ss->stride[1] != 1 || ss->dtype != RAD_BF16)
            return R4D_NO(RAD_E_STRIDE);
    }
    if (na && (r4d_numel(na) < seqs || na->dtype != RAD_I32 || !r4d_contig(na)))
        return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k);
    const long long bp2 = r4d_row_pitch(bt), sp2 = r4d_row_pitch(st);
    if (kp < 0 || bp2 < 0 || sp2 < 0) return R4D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || tl->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    return r4d_qsa_work_bf16_emit(r4d_addr(k), r4d_addr(po), r4d_addr(cu), r4d_addr(tl),
                                  r4d_addr(sx), r4d_addr(bt), r4d_addr(st), r4d_addr(pg),
                                  r4d_addr(bp), r4d_addr(nc), (int)n, (int)ratio, (int)ring,
                                  (int)seqs, (int)mw, (int)kp, (int)tl->stride[0],
                                  (int)tl->stride[2], (int)sxp, (int)bp2, (int)sp2,
                                  r4d_addr(rp), rp ? (int)rp->shape[1] : 0, (int)bstep,
                                  r4d_addr(sp), r4d_addr(na), r4d_addr(ss), r4d_addr(sg),
                                  r4d_addr(sb), r4d_addr(rt), r4d_addr(rn), r4d_addr(rc), (int)T,
                                  rt ? (int)rt->stride[0] : 0, rt ? (int)rt->shape[1] : 0,
                                  ss ? (int)r4d_row_pitch(ss) : 0, (int)sbstep, stream,
                                  r4d_geti_or(a, "chain", 0) != 0);
}


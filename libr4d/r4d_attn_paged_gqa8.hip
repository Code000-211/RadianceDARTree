// R4D paged attention dispatch at eight queries per KV head, head_dim 256 and 128.
//
// Instantiations of the h256/gqa6 kernel templates (templated on head_dim and gqa), for the gqa-8
// models; the constraint tables gate on the exact ratio:
//
//   Qwen3.6-35B-A3B   16 query heads over 2 KV heads, head_dim 256
//   MiniCPM5-2B       16 query heads over 2 KV heads, head_dim 128
//
// Gqa changes only two bounds. A decode workgroup holds 64 rows of q_len * gqa, so the verify
// width is 8 here (10 at gqa 6); the selector bands on that value, stated in the row. A prefill
// block carries NWARPS * 16 / gqa query positions (48 here, 64 at gqa 6).
//
// Head_dim changes how K is fetched. O_KWIDE's thread-to-(key, channel) map covers exactly 16 x 16
// channels (head_dim 256); at 128 it would stage past the tile, so the h128 prefill runs without it.
// Its fragments take half the registers, so it keeps K prefetch 4 where h256 uses 2. All other
// option bits are as defined in r4d_attn_prefill_h256_gqa6.hip.
//
// No r4d_attn_dims here: that entry point (and check_dims_once) belongs to the h256/gqa6 unit,
// and a second definition would not link. Geometry is gated by the row constraints and the
// launchers below.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel uses helpers defined in the prefill unit, so prefill is included first.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"
#include "r4d_attn_prefill_split.h"

#define CHK8() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* Queries per KV head, and the symbol names built from it.
 *
 * Another ratio reuses this file: it defines these macros and includes this one (e.g.
 * r4d_attn_paged_h256_gqa12.hip), and the entry points paste the ratio into their names. Only
 * the two bounds change: decode verify width 64/gqa (8 here, 5 at gqa 12) and prefill block
 * 384/gqa query positions (48 here, 32 at gqa 12). Both must divide, and both are stated in the
 * row so the selector bands on them. */
#ifndef R4D_ATTN_GQA
#define R4D_ATTN_GQA  8
#define R4D_ATTN_H128 1      /* this ratio also serves head_dim 128 -- MiniCPM5-2B is 16 over 2 */
#endif
/* Head_dim 64 prefill leg, off unless the including unit enables it (LFM2: 32 query heads over 8,
 * r4d_attn_paged_h256_gqa4.hip). Uses the h128 option set (no O_KWIDE). No decode leg: the
 * split-KV merge holds a lane's columns as one or two dwords of codes (head_dim 128 or 256), so
 * the query-tiled kernel serves every query length at this width. */
#ifndef R4D_ATTN_H64
#define R4D_ATTN_H64 0
#endif
/* Head_dim 256 leg, on unless the including unit serves only head_dim 128 (as R4D_ATTN_H128 0
 * leaves the 128 leg out). */
#ifndef R4D_ATTN_H256
#define R4D_ATTN_H256 1
#endif
#define GQ_JOIN_(a, b, c) a##b##c
#define GQ_JOIN(a, b, c)  GQ_JOIN_(a, b, c)
#define GQ_SYM(pre, suf)  GQ_JOIN(pre, R4D_ATTN_GQA, suf)

#define G8_GQA         R4D_ATTN_GQA
/* KV page size in tokens; the kernel body is generic in it, so an including unit may override.
 * Qwen4-Exp's QSA uses 4: its indexer selects 4-token blocks, so the selection is just a block
 * table (docs/QSA.md). */
#ifndef G8_BLOCK_SIZE
#define G8_BLOCK_SIZE 16
#endif
/* q_len * gqa a single decode workgroup can hold: q_len <= 8 here. An including unit may raise it
 * to 128 (eight waves): gqa 16 then verifies 7 drafts in the decode kernel, as gqa 8 does. A unit
 * may also lower it (see the gqa7 unit: 63 rows and 21 warps at gqa 7), so each bound below is a
 * guard the including unit can win; the defaults are the numbers the row table states for gqa 8. */
#ifndef G8_MAX_DECODE_ROWS
#define G8_MAX_DECODE_ROWS 64
#endif

#ifndef G8_P_NWARPS
#define G8_P_NWARPS 24
#endif
#ifndef G8_P_TILE
#define G8_P_TILE   48
#endif
// K prefetch 2 at head_dim 256, as in the h256/gqa6 prefill (P_OPT there): 4 spills the m-tile
// loop. The head_dim-128 set keeps prefetch 4, which fits.
#define G8_P_OPT_256 3954187
#define G8_P_OPT_128 3430923        // without O_KWIDE, prefetch 4
#define G8_P_OPT_64  G8_P_OPT_128
#define G8_D_TILE   16
#define G8_D_OPT    3429947
#define G8_D_JPB     2
#define G8_D_VECMIN 256

// ---- prefill -----------------------------------------------------------------------------------
// Query positions a workgroup: 384 / gqa.
#define G8_P_BQ ((G8_P_NWARPS * 16) / G8_GQA)

// a->splits > 1 with a->scratch set is the KV split (r4d_attn_prefill_split.h): segments in the
// scratch, then the merge. Only the head_dim 256 entries set it (their rows declare the scratch);
// the 128 and 64 legs have no split instance and always run the plain form, unchanged.
template <int HEAD_DIM, int OPT, int KVP>
static int g8_prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    constexpr int BQ = G8_P_BQ;
    constexpr bool CAN_SPLIT = HEAD_DIM == 256;
    const int splits = (CAN_SPLIT && a->scratch && a->splits > 1) ? a->splits : 1;
    g_prefill_last_splits = splits;
    dim3 grid((a->q_len + BQ - 1) / BQ * splits, a->kv_heads, a->num_seqs);
    if constexpr (CAN_SPLIT) {
        if (splits > 1) {
            hipLaunchKernelGGL((r4d_attn_prefill_kernel<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA,
                                                        G8_BLOCK_SIZE, KVP, OPT | O_PSPLIT>),
                               grid, dim3(G8_P_NWARPS * 32), 0, s, *a);
            CHK8();
            return r4d_prefill_combine<HEAD_DIM, G8_D_JPB, G8_D_VECMIN>(a, splits, G8_P_TILE, s);
        }
    }
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA,
                                                G8_BLOCK_SIZE, KVP, OPT>),
                       grid, dim3(G8_P_NWARPS * 32), 0, s, *a);
    CHK8();
    return 0;
}

// ---- decode ------------------------------------------------------------------------------------
// The h256/gqa6 split law (r4d_decode_splits).
extern "C" long GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits(a, G8_D_TILE), G8_D_OPT);
}

template <int HEAD_DIM, int KVP, int XOPT = 0>
/* Two dispatches: the attention kernel and the split-KV combine. R4DArgs is passed by value (184
 * bytes) like any other kernel argument. Non-null `gq` adds the output gate and quantiser to the
 * combine (attn_paged_gate_quant; R4DGateQuant in r4d_attn_decode_h256_gqa6.hip). `XOPT` adds
 * option bits to the attention kernel's (O_KVI8: the int8 cache). */
static int g8_decode_launch(const R4DArgs* a, RadStream stream,
                            const R4DGateQuant* gq = nullptr) {
    int emit_rc = RAD_OK;
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    const int rows = a->q_len * G8_GQA;
    if (rows > G8_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = r4d_decode_splits(a, G8_D_TILE);

    /* `a->chain`: the chained forms (r4d_chain.h) of the attention kernel and of the merge
     * without a gate; the merge with a gate and quantiser has none. */
#define G8_LAUNCH_K(K_, NW, OPT_)                                                              \
    emit_rc = R4D_EMIT(stream, (K_<NW, G8_D_TILE, HEAD_DIM, G8_GQA, G8_BLOCK_SIZE, KVP,        \
                                   G8_D_OPT | OPT_ | XOPT>),                                   \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, *a, splits)
#define G8_LAUNCH(NW, OPT_)                                                                    \
    do { if (a->chain) G8_LAUNCH_K(r4dchain_attn_decode, NW, OPT_);                            \
         else          G8_LAUNCH_K(r4d_attn_decode_kernel, NW, OPT_); } while (0)
    /* One-wave workgroups stage V as codes (O_V8): same conversion count, half the LDS. The int8
     * cache stages V widened (the kernel drops O_V8 under it). */
    if      (nw == 1) G8_LAUNCH(1, O_V8);
    else if (nw == 2) G8_LAUNCH(2, 0);
    else if (nw == 3) G8_LAUNCH(3, 0);
#if G8_MAX_DECODE_ROWS > 64
    /* Five to eight waves take eight; a wave past the last query row computes a copy of it and
     * stores nothing. */
    else if (nw > 4)  G8_LAUNCH(8, 0);
#endif
    else              G8_LAUNCH(4, 0);
#undef G8_LAUNCH
#undef G8_LAUNCH_K
    if (emit_rc != RAD_OK) return -5;
    CHK8();

    // Merge form, as in the other dispatches: one wave per row when there are enough rows and the
    // segment count fits a wave, else the general LDS-tree form. Both give identical bytes.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (gq) {
        /* GQ 3: rotated int8 codes, GQ 2: int8 codes, GQ 1: E4M3; see R4DGateQuant. */
#define G8_GQ_VEC(PF, GQV)                                                                     \
        R4D_EMIT(stream,                                                                       \
                 (r4d_attn_splitkv_combine_vec_gq_kernel<HEAD_DIM, 4, PF, G8_D_JPB, 0, GQV>),   \
                 dim3((jobs3 + G8_D_JPB - 1) / G8_D_JPB), dim3(32 * G8_D_JPB), 0, *a, splits,   \
                 G8_D_TILE, *gq)
#define G8_GQ_GEN(PF, GQV)                                                                     \
        R4D_EMIT(stream, (r4d_attn_splitkv_combine_gq_kernel<HEAD_DIM, 4, PF, 0, GQV>),         \
                 dim3(jobs3), dim3(HEAD_DIM), splits * sizeof(float), *a, splits, G8_D_TILE,   \
                 *gq)
        const bool i8 = gq->i8 != 0, rot = gq->rot != 0;
        if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
            if (G8_D_OPT & 32)
                emit_rc = rot ? G8_GQ_VEC(1, 3) : i8 ? G8_GQ_VEC(1, 2) : G8_GQ_VEC(1, 1);
            else
                emit_rc = rot ? G8_GQ_VEC(0, 3) : i8 ? G8_GQ_VEC(0, 2) : G8_GQ_VEC(0, 1);
        } else if (G8_D_OPT & 32) {
            emit_rc = rot ? G8_GQ_GEN(1, 3) : i8 ? G8_GQ_GEN(1, 2) : G8_GQ_GEN(1, 1);
        } else {
            emit_rc = rot ? G8_GQ_GEN(0, 3) : i8 ? G8_GQ_GEN(0, 2) : G8_GQ_GEN(0, 1);
        }
#undef G8_GQ_VEC
#undef G8_GQ_GEN
    } else if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
        const int blocks = (jobs3 + G8_D_JPB - 1) / G8_D_JPB;
#define G8_VEC(K_, PF)                                                                         \
        emit_rc = R4D_EMIT(stream, (K_<HEAD_DIM, 4, PF, G8_D_JPB>), dim3(blocks),              \
                           dim3(32 * G8_D_JPB), 0, *a, splits, G8_D_TILE)
        if (G8_D_OPT & 32) {
            if (a->chain) G8_VEC(r4dchain_attn_splitkv_combine_vec, 1);
            else          G8_VEC(r4d_attn_splitkv_combine_vec_kernel, 1);
        } else {
            if (a->chain) G8_VEC(r4dchain_attn_splitkv_combine_vec, 0);
            else          G8_VEC(r4d_attn_splitkv_combine_vec_kernel, 0);
        }
#undef G8_VEC
    } else {
        /* Block of HEAD_DIM threads, not 256: the general combine is __launch_bounds__(HEAD_DIM),
         * one thread per channel, and the runtime refuses a larger block. */
#define G8_GEN(K_, PF)                                                                         \
        emit_rc = R4D_EMIT(stream, (K_<HEAD_DIM, 4, PF>), dim3(jobs3), dim3(HEAD_DIM),         \
                           splits * sizeof(float), *a, splits, G8_D_TILE)
        if (G8_D_OPT & 32) {
            if (a->chain) G8_GEN(r4dchain_attn_splitkv_combine, 1);
            else          G8_GEN(r4d_attn_splitkv_combine_kernel, 1);
        } else {
            if (a->chain) G8_GEN(r4dchain_attn_splitkv_combine, 0);
            else          G8_GEN(r4d_attn_splitkv_combine_kernel, 0);
        }
#undef G8_GEN
    }
    if (emit_rc != RAD_OK) return -5;
    CHK8();
    return 0;
}

// ============================================================== the plugin ABI entry points

#if R4D_ATTN_H256
/* The head_dim 256 prefill's KV split at this ratio's workgroup height (r4d_attn_prefill_split.h).
 * Test hooks (r4d_selftest), not plugin entries, named by ratio and page since gqa 6 exists at 16-
 * and 4-token pages in different units: the law at this card's WGP count, and the split count this
 * unit's last prefill launch used. */
#define G8_HOOK_(g, b, suf) r4d_attn_prefill_h256_gqa##g##_bs##b##suf
#define G8_HOOK(g, b, suf)  G8_HOOK_(g, b, suf)
extern "C" int G8_HOOK(R4D_ATTN_GQA, G8_BLOCK_SIZE, _splits)(int q_len, int total_q, int num_seqs,
                                                             int kv_heads, int max_ctx) {
    return r4d_prefill_splits_law(G8_P_BQ, G8_P_TILE, q_len, total_q, num_seqs, kv_heads, max_ctx,
                                  r4d_prefill_wgps());
}
extern "C" int G8_HOOK(R4D_ATTN_GQA, G8_BLOCK_SIZE, _last_splits)(void) {
    return g_prefill_last_splits;
}
#undef G8_HOOK
#undef G8_HOOK_

/* The split's f32 partials at declare, by the cache's token row (r4d_prefill_scratch). */
static int64_t g8_prefill_scratch(const RadArgs* a, long long row) {
    return r4d_prefill_scratch(a, 256, row, G8_P_BQ, G8_P_TILE);
}
extern "C" int64_t GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_prefill_scratch(a, 512);
}

extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    r.splits = r4d_prefill_splits(a, &r, G8_P_BQ, G8_P_TILE);
    return g8_prefill_launch<256, G8_P_OPT_256, 1>(&r, (hipStream_t)s);
}
#endif
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128, 1>(&r, (hipStream_t)s);
}
#endif
#if R4D_ATTN_H64
/* The int8 cache at this width: a token's row is 2 * 64 + 4 = 132 bytes. */
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h64_gqa, _i8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged_row(a, &r, 64, 2 * 64 + 4);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<64, G8_P_OPT_64 | O_KVI8, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h64_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 64);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<64, G8_P_OPT_64, 1>(&r, (hipStream_t)s);
}
#endif

/* Split-KV partial buffer size at declare time, from the band's upper bound. Declare passes
 * parameters and no tensors, so parameters come first and tensors are the fallback (see the
 * h256/gqa6 dispatch). The size is the maximum over every sequence count up to max_seqs and every
 * query length up to the band's: S * floor(128 / (kv_heads * S)) is not monotone in S, and the
 * split law's targets (r4d_decode_splits: 256 workgroups at q_len * gqa <= 32; above, 128 to 16384
 * tokens deep and more past it) can give a shorter query more splits: at gqa 8 a 4-row step needs
 * 1.6x the partials of the band bound 5. */
/* `row` is a token's bytes in a (block, head) page, 0 for the bf16 cache's 2 * head_dim elements
 * (fill_paged_row). */
#if R4D_ATTN_H256
/* The tree-sel split count. A tree launch's num_seqs is its ROW count, but the sequential decode of
 * the same rows runs with the SEQUENCE count, and r4d_decode_splits depends on num_seqs (units =
 * kv_heads * num_seqs). So tsel evaluates the law at `n_seq` sequences (q_len 1, the tsel row
 * constraint), which is what the chain / sequential launch of those sequences evaluates: a tree row
 * and its path's sequential decode then split, accumulate and merge identically (same max_ctx, same
 * `splits` tunable). The count is independent of how many tree rows a sequence carries. */
static int g8_tsel_splits(const R4DArgs& r, int sp, int n_seq) {
    R4DArgs law = r;
    law.splits   = sp;
    law.num_seqs = n_seq;
    law.q_len    = 1;
    return r4d_decode_splits(&law, G8_D_TILE);
}
#endif

static int64_t g8_decode_scratch(const RadArgs* a, int head_dim, long long row = 0,
                                 bool tsel = false) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim   = head_dim;
        r.q_heads    = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads   = (int)(gqa > 0 ? n_head / gqa : n_head);
        r.num_seqs   = (int)max_seqs;
        const long long q_hi = std::max(1LL, r4d_geti_or(a, "q_len", 1));
        r.max_ctx    = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits     = sp;
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
#if R4D_ATTN_H256
        if (tsel) {
            /* rows = max_seqs, splits at the law's maximum (one sequence); nonincreasing in n_seq. */
            r.q_len    = 1;
            r.num_seqs = (int)max_seqs;
            r.splits   = g8_tsel_splits(r, sp, 1);
            return GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
        }
#endif
        for (long long q = 1; q <= q_hi; ++q)
            for (long long n = 1; n <= max_seqs; ++n) {
                r.q_len    = (int)q;
                r.num_seqs = (int)n;
                const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
                if (need > worst) worst = need;
            }
        return worst;
    }
    const int rc = fill_paged_row(a, &r, head_dim, row > 0 ? row : 2LL * head_dim);
    if (rc != RAD_OK) return rc;
    r.splits = sp;
#if R4D_ATTN_H256
    if (tsel) r.splits = g8_tsel_splits(r, sp, 1);
#endif
    return GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
}

#if R4D_ATTN_H256
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 256);
}
#endif
#if R4D_ATTN_H128
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h128_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128);
}
#endif

static int g8_decode_common(const RadArgs* a, R4DArgs* r, int head_dim, long long row = 0) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    const int rc = fill_paged_row(a, r, head_dim, row > 0 ? row : 2LL * head_dim);
    if (rc != RAD_OK) return rc;
    r->splits = sp;
    const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(r);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

#if R4D_ATTN_H256
/* ---- attn_paged_gate_quant: attn_paged's eight operands, then the gate, codes and scales.
 * The gate is a [T, n_head, head_dim] column view of the [q|gate] projection; its (token, head)
 * rows are one pitch apart only if the view narrowed the last axis alone. */
enum { AP_GQ_G = AP_OUT + 1, AP_GQ_Q, AP_GQ_S };

static long long gq_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* `rot`: attn_paged_gate_had_quant, whose codes are in the rotated domain. */
static int g8_gate_quant_args(const RadArgs* a, const R4DArgs& r, R4DGateQuant* gq,
                              bool rot = false) {
    const RadTensor* g = r4d_opt(a, AP_GQ_G);
    const RadTensor* q = r4d_opt(a, AP_GQ_Q);
    const RadTensor* s = r4d_opt(a, AP_GQ_S);
    if (!g || !q || !s) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype picks the quantiser: E4M3, or int8 for an int8 out projection. */
    if (g->dtype != RAD_BF16 || (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) ||
        s->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (rot && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "sigmoid")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(a, "group", kGqGroup) != kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long n = r.head_dim;
    if (n % kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long nkb  = n / kGqGroup;
    const long long rows = (long long)r4d_q_total(r) * r.q_heads;
    if (r4d_numel(g) < rows * n || r4d_numel(q) < rows * n || r4d_numel(s) < rows * nkb)
        return R4D_NO(RAD_E_SHAPE);
    const long long gp = gq_row_pitch(g, n), qp = gq_row_pitch(q, n), sp = gq_row_pitch(s, nkb);
    if (gp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The vector merge reads a lane's gate in one 16-byte access and stores its codes in one
     * 8-byte access. */
    if ((r4d_addr(g) & 15) || (gp % 8) || (r4d_addr(q) & 7) || (qp % 8))
        return R4D_NO(RAD_E_STRIDE);
    gq->g    = (const uint16_t*)r4d_addr(g);
    gq->q    = (uint8_t*)r4d_addr(q);
    gq->s    = (float*)r4d_addr(s);
    gq->g_ld = (unsigned)gp;
    gq->q_ld = (unsigned)qp;
    gq->s_ld = (unsigned)sp;
    gq->i8   = q->dtype == RAD_I8 ? 1 : 0;
    gq->rot  = rot ? 1 : 0;
    return RAD_OK;
}

/* bf16-cache decode, plain and with the gate and quantiser in the merge (`_gq` E4M3 or int8 codes,
 * `_ghq` rotated int8 codes for an out projection stored rotated). */
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 1>(&r, stream);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, stream, &gq);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_ghq_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq, true);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, stream, &gq);
}
#endif
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 1>(&r, s);
}
#endif

#if R4D_ATTN_H128
/* ---- the int8 cache at head_dim 128: a token's row is 2 * 128 + 4 = 260 bytes. Prefill (K staged
 * eight dims a thread, having no wide fetch at this width), and the decode that also serves a
 * drafter's non-causal block. */
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _i8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged_row(a, &r, 128, 2 * 128 + 4);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128 | O_KVI8, 0>(&r, (hipStream_t)s);
}
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h128_gqa, _i8_scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128, 2 * 128 + 4);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _i8)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128, 2 * 128 + 4);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 0, O_KVI8>(&r, stream);
}
#endif

#if R4D_ATTN_H256
/* ---- the int8 cache at head_dim 256 (a token's row 2 * 256 + 4 = 516 bytes): prefill, decode
 * and decode with the gate and quantiser in the merge. `_ghq` is `_gq` with the rotated
 * quantiser. */
#define G8_I8_ENTRIES(TAG, ROWB, XOPT)                                                        \
extern "C" int64_t GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _##TAG##_scratch)(const RadArgs* a) { \
    return g8_prefill_scratch(a, ROWB);                                                        \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) { \
    R4DArgs r;                                                                                 \
    const int rc = fill_paged_row(a, &r, 256, ROWB);                                           \
    if (rc != RAD_OK) return rc;                                                               \
    r.splits = r4d_prefill_splits(a, &r, G8_P_BQ, G8_P_TILE);                                  \
    return g8_prefill_launch<256, G8_P_OPT_256 | (XOPT), 0>(&r, (hipStream_t)s);               \
}                                                                                              \
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h256_gqa, _##TAG##_scratch)(const RadArgs* a) {  \
    return g8_decode_scratch(a, 256, ROWB);                                                    \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _##TAG)(const RadArgs* a,                  \
                                                            RadStream stream) {                \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream);                                       \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _##TAG)(const RadArgs* a,               \
                                                               RadStream stream) {             \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    R4DGateQuant gq{};                                                                         \
    const int gc = g8_gate_quant_args(a, r, &gq);                                              \
    if (gc != RAD_OK) return gc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream, &gq);                                  \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_ghq_h256_gqa, _##TAG)(const RadArgs* a,              \
                                                                RadStream stream) {            \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    R4DGateQuant gq{};                                                                         \
    const int gc = g8_gate_quant_args(a, r, &gq, true);                                        \
    if (gc != RAD_OK) return gc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream, &gq);                                  \
}
G8_I8_ENTRIES(i8, 516, O_KVI8)
#undef G8_I8_ENTRIES
#endif

#if R4D_ATTN_H256 && defined(G8_TSEL)
// ============================================================== attn_paged_tree_sel
// QSA's attention at a tree verify (docs/OPS.md, attn_paged_tree_sel): per query row, the row's
// selected pages and then the keys of a 64-key region that its mask names, in one softmax. It is
// the split-KV decode kernel run on a key list instead of a block table (O_TSEL): the list is the
// row's `nsel` pages, then its visible region keys packed in ascending order, each given by its
// flat cache slot, built per row by r4d_tsel_prep:
//
//     nkeys[row]   = nsel * BS               keys read through the page list (`sel`, as is)
//     slots[row]   = flat slot of each visible region key, ascending, 64 entries a row
//     seqused[row] = nsel * BS + popcount(vmask)
//
// Nothing is masked: the kernel sees exactly the keys of the row, at consecutive indices. A chain
// row (a prefix of the region) therefore runs the plain sparse row's kernel over the plain
// sparse row's keys, tile for tile, and a tree row is the sequential decode of its root-to-node
// path laid out as that path's keys are -- the same bytes, however the keys sit in the pages. The
// split law, the partial buffer and the merge are the sparse row's (the q_len 1 launch, one row a
// "sequence"). Scratch: the decode partials, then nkeys and seqused [rows] and slots [rows, 64].
enum { kTselKeys = 64 };

static __global__ void r4d_tsel_prep_kernel(const int* __restrict__ nsel, int sel_pitch,
                                     const unsigned long long* __restrict__ vmask,
                                     const int* __restrict__ bt, int bt_pitch,
                                     const int* __restrict__ rb, const int* __restrict__ cu,
                                     int n_seq, int rows, int* __restrict__ nkeys,
                                     int* __restrict__ ctxv, int* __restrict__ slots)
{
    const int r = blockIdx.x;
    if (r >= rows) return;
    int s = 0;
    if (cu) { while (s + 1 < n_seq && cu[s + 1] <= r) ++s; }
    else    { s = r / (rows / n_seq); }
    int ns = nsel[r];
    ns = ns < 0 ? 0 : (ns > sel_pitch ? sel_pitch : ns);
    const unsigned long long m = vmask[r];
    const int k = threadIdx.x;                         // 64 threads, one a region key
    if ((m >> k) & 1ull) {
        const int rank = (int)__builtin_popcountll(m & ((1ull << k) - 1ull));
        const int pg   = rb[s] / G8_BLOCK_SIZE + k / G8_BLOCK_SIZE;
        int blk = pg < bt_pitch ? bt[(size_t)s * bt_pitch + pg] : 0;
        if (blk < 0) blk = 0;
        slots[(size_t)r * kTselKeys + rank] = blk * G8_BLOCK_SIZE + k % G8_BLOCK_SIZE;
    }
    if (k == 0) {
        nkeys[r] = ns * G8_BLOCK_SIZE;
        ctxv[r]  = ns * G8_BLOCK_SIZE + (int)__builtin_popcountll(m);
    }
}

/* The launch-time view of a tree-sel call: the base op's operands through fill_paged_row (sel as
 * its block table, nsel as seqused), then the tree's. `tb` is the first tree operand: 8 for the
 * plain op, 11 behind the gated op's gate, codes and scales. */
struct G8Tsel {
    const unsigned long long* vmask;
    const int *bt, *rb, *cu;
    int bt_pitch, n_seq, sel_pitch;
};

static int g8_tsel_operands(const RadArgs* a, const R4DArgs& r, int tb, G8Tsel* t) {
    if (r4d_opt(a, AP_KDS) || r4d_opt(a, AP_VDS) || r4d_opt(a, AP_CU))
        return R4D_NO(RAD_E_UNSUPPORTED);
    if (r.q_len != 1) return R4D_NO(RAD_E_SHAPE);
    if (r.causal != 1 || r.window != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor* vm = r4d_opt(a, tb);
    const RadTensor* bt = r4d_opt(a, tb + 1);
    const RadTensor* rb = r4d_opt(a, tb + 2);
    const RadTensor* cu = r4d_opt(a, tb + 3);
    const RadTensor* sel = r4d_opt(a, AP_BT);
    const RadTensor* ns  = r4d_opt(a, AP_SEQ);
    if (!vm || !bt || !rb) return R4D_NO(RAD_E_INVAL);
    if (sel->dtype != RAD_I32 || ns->dtype != RAD_I32 || bt->dtype != RAD_I32 ||
        rb->dtype != RAD_I32 || (cu && cu->dtype != RAD_I32) || vm->dtype != RAD_I64)
        return R4D_NO(RAD_E_DTYPE);
    if (bt->rank != 2 || !r4d_contig(bt) || !r4d_contig(rb) || !r4d_contig(vm) ||
        (cu && !r4d_contig(cu)))
        return R4D_NO(RAD_E_STRIDE);
    const long long rows = r.num_seqs, n_seq = bt->shape[0];
    if (r4d_numel(vm) != rows || n_seq <= 0 || r4d_numel(rb) != n_seq) return R4D_NO(RAD_E_SHAPE);
    if (cu ? r4d_numel(cu) != n_seq + 1 : rows % n_seq != 0) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(vm->data, 8)) return R4D_NO(RAD_E_ALIGN);
    t->vmask     = (const unsigned long long*)vm->data;
    t->bt        = (const int*)bt->data;
    t->rb        = (const int*)rb->data;
    t->cu        = cu ? (const int*)cu->data : nullptr;
    t->bt_pitch  = (int)bt->shape[1];
    t->n_seq     = (int)n_seq;
    t->sel_pitch = (int)sel->shape[1];
    return RAD_OK;
}

/* nkeys, seqused and slots for `rows` rows. */
static long g8_tsel_prep_bytes(long long rows) { return (long)(rows * (kTselKeys + 2) * 4); }
static long g8_tsel_align(long v) { return (v + 255) & ~255L; }

/* Declare-time size: the decode partials of the sparse row at the declared bound, plus the prep
 * arrays at `max_seqs` rows. */
static int64_t g8_tsel_scratch(const RadArgs* a, long long row) {
    const int64_t dec = g8_decode_scratch(a, 256, row, true);
    if (dec < 0) return dec;
    long long rows = r4d_geti_or(a, "max_seqs", 0);
    if (rows <= 0) {
        const RadTensor* q = r4d_opt(a, AP_Q);
        if (!q) return R4D_NO(RAD_E_INVAL);
        rows = q->shape[0];
    }
    return g8_tsel_align(dec) + g8_tsel_prep_bytes(rows);
}

template <int KVP, int XOPT>
static int g8_tsel_launch(const RadArgs* a, RadStream stream, long long row, int tb,
                          bool gated, bool rot) {
    R4DArgs r;
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    int rc = fill_paged_row(a, &r, 256, row);
    if (rc != RAD_OK) return rc;
    G8Tsel t;
    rc = g8_tsel_operands(a, r, tb, &t);
    if (rc != RAD_OK) return rc;
    r.splits = g8_tsel_splits(r, sp, t.n_seq);
    const long dec = g8_tsel_align(GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r));
    const long need = dec + g8_tsel_prep_bytes(r.num_seqs);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    int* nkeys = (int*)((char*)a->scratch + dec);
    int* ctxv  = nkeys + r.num_seqs;
    int* slots = ctxv + r.num_seqs;

    const int* nsel = r.seqused_k;
    R4DGateQuant gq{};
    if (gated) {
        rc = g8_gate_quant_args(a, r, &gq, rot);
        if (rc != RAD_OK) return rc;
    }
    if (R4D_EMIT(stream, r4d_tsel_prep_kernel, dim3(r.num_seqs), dim3(kTselKeys), 0, nsel,
                 t.sel_pitch, t.vmask, t.bt, t.bt_pitch, t.rb, t.cu, t.n_seq, r.num_seqs, nkeys,
                 ctxv, slots) != RAD_OK)
        return -5;
    CHK8();
    r.seqused_k  = ctxv;
    r.tsel_nkeys = nkeys;
    r.tsel_slots = slots;
    return g8_decode_launch<256, KVP, XOPT | O_TSEL>(&r, stream, gated ? &gq : nullptr);
}

/* attn_paged_tree_sel_gate_quant's gate / codes / scales sit at operands 8-10 (AP_GQ_*), the tree
 * operands after them; the plain op has the tree operands at 8. */
#define G8_TSEL_ENTRIES(TAG, KVP, ROWB, XOPT)                                                  \
extern "C" int64_t GQ_SYM(r4d_rad_attn_tsel_h256_gqa, _##TAG##_scratch)(const RadArgs* a) {    \
    return g8_tsel_scratch(a, ROWB);                                                           \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_tsel_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) {     \
    return g8_tsel_launch<KVP, (XOPT)>(a, s, ROWB, 8, false, false);                           \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_tsel_gq_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) {  \
    return g8_tsel_launch<KVP, (XOPT)>(a, s, ROWB, 11, true, false);                           \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_tsel_ghq_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) { \
    return g8_tsel_launch<KVP, (XOPT)>(a, s, ROWB, 11, true, true);                            \
}
G8_TSEL_ENTRIES(bf16, 1, 512, 0)
G8_TSEL_ENTRIES(i8, 0, 516, O_KVI8)
#undef G8_TSEL_ENTRIES
#endif

// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip -- the gated-delta-net recurrent state
// update, run by decode (prefill runs the chunked scan). One kernel for the whole step:
//
//     g = -exp(A_log) . softplus(a + dt_bias)      beta = sigmoid(b)      q, k l2-normalised
//     S  *= e^g
//     u   = beta . (v - S k)                       S += u k^T              o = S q
//
// per token, against a paged state cache: the initial state is read from the slot the last
// accepted token wrote, and every candidate token writes its own slot, since which survives
// verification is unknown until after this layer. Replaces FLA's
// fused_sigmoid_gating_delta_rule_update; the non-speculative case is width 1.
//
// State traffic is the whole cost: one fp32 [V,K] state is 64 KB, written per candidate token per
// head (113 MB per layer per rank at width 9, at the memory bandwidth limit). So the design keeps
// the state in registers: one thread owns one v row and half its k range, so
//
//   * S k and S q are in-lane dot products closed by a single lane<->lane^1 exchange, not a
//     tree reduction;
//   * the outer-product update is 64 independent FMAs with no cross-lane traffic at all;
//   * a thread's 64 state elements are contiguous in k, so the load and every store is four
//     16-byte accesses over one 256-byte run.
//
// The state cache is fp32 by default (the model's HF config sets mamba_ssm_dtype=float32, which
// vLLM honours); the conv state is bf16. An fp16 state cache is also supported (SDT).
//
// Slot and head strides are arguments, not H*V*K: vLLM's hybrid cache pads a mamba page to the
// attention page size (e.g. 421888 elements per slot vs 393216 from the shape), so a shape-derived
// stride would read every slot but the first at the wrong offset there.
#include <hip/hip_runtime.h>
#include "r4d_keyed.h"
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include "r4d_gdn_conv_tap.h"
#include "r4d_q8.h"         // the 8-bit codes the folded quantiser writes: E4M3 or int8
#include "r4d_fwht.h"       // NORM 3: had_quant_act_i8g's rotation before the int8 codes
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

#define RU_K   128              // head_k
#define RU_V   128              // head_v
#define RU_HK   64              // k elements per thread (RU_K / 2)
#define RU_THR (RU_V * 2)       // two threads per v row

// The state written per step (~12 MB a layer, ~660 MB a step) is read again only once, a step
// later, and cannot stay in the 64 MB last-level cache; its write-back slows the next GEMM (a
// (5120, 3072) GEMM goes from 14.9 us to 20.4 us behind a 12 MB write flood, 18.4 us when the
// flood is non-temporal). A non-temporal hint on the state load and store changes no result, and
// whether it pays depends on the flood size, so it is chosen per launch (RU_NT_MB below).
typedef float ru_f4 __attribute__((ext_vector_type(4)));
typedef _Float16 ru_h8 __attribute__((ext_vector_type(8)));

// SDT is the state cache's element type (0 = fp32, 1 = fp16); all arithmetic is fp32. At eight
// sequences x eight candidates a call is ~68 us, of which ~45 us is 24 MB of state traffic at 88%
// of achievable bandwidth; halving the state width saves about a third.
//
// Cost model (eight sequences, 16k context, anchor state): ~3.2 us per token walked plus ~57 us
// fixed per layer. State traffic largely overlaps the token loop (fp16 state saves only about a
// fifth of what its byte count predicts), and the loop is on the critical path.
//
// Shipped instantiation: 207 VGPRs, 7 waves/SIMD, no spill, 384 B LDS. At seven waves the serial
// `dot +=` and `on +=` chains are hidden; splitting them would only change the summation order.
//
// fp16 rather than bf16: state elements are O(0.1) (sums of beta (v - Sk) k^T with l2-normalised
// k), so range is no issue and fp16's mantissa gives ~5e-4 relative round-trip error. vLLM's
// --mamba-ssm-cache-dtype already accepts float16.
template <int SDT> struct ru_state { using T = float; };
template <> struct ru_state<1>     { using T = _Float16; };

// Eight elements per call: two 16-byte accesses in fp32, one in fp16.
template <int NT, int SDT>
__device__ __forceinline__ void ru_load8(float* d, const typename ru_state<SDT>::T* p) {
  if constexpr (SDT) {
    const ru_h8 t = NT ? __builtin_nontemporal_load((const ru_h8*)p) : *(const ru_h8*)p;
#pragma unroll
    for (int e = 0; e < 8; ++e) d[e] = (float)t[e];
  } else {
    const ru_f4* q = (const ru_f4*)p;
    *(ru_f4*)d       = NT ? __builtin_nontemporal_load(q)     : *q;
    *(ru_f4*)(d + 4) = NT ? __builtin_nontemporal_load(q + 1) : *(q + 1);
  }
}

template <int NT, int SDT>
__device__ __forceinline__ void ru_store8(typename ru_state<SDT>::T* p, const float* d) {
  if constexpr (SDT) {
    ru_h8 t;
#pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = (_Float16)d[e];
    if (NT) __builtin_nontemporal_store(t, (ru_h8*)p);
    else    *(ru_h8*)p = t;
  } else {
    ru_f4* q = (ru_f4*)p;
    if (NT) {
      __builtin_nontemporal_store(*(const ru_f4*)d,       q);
      __builtin_nontemporal_store(*(const ru_f4*)(d + 4), q + 1);
    } else {
      *q       = *(const ru_f4*)d;
      *(q + 1) = *(const ru_f4*)(d + 4);
    }
  }
}
// VS: workgroups sharing a head's 128 v rows. At one sequence the (sequences, heads) grid is 24
// workgroups on 64 CUs, and the working set fits the last-level cache. v rows are independent
// (dot products contract over k; the update is per (v, k)); only the fused RMS norm spans the
// row, so VS > 1 is compiled without it.
#define WGTARGET 64             // workgroups the v-row split aims for

__device__ __forceinline__ float ru_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// a and b are a chunk() of the ba projection (see r4d_gdn_conv_w4_h128_bf16.hip).
template <int ABF16>
__device__ __forceinline__ float ru_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// The layer's gated RMS norm, folded into the epilogue: with NORM a workgroup owns one (sequence,
// head) and so the whole 128-channel output row, and the norm is one workgroup reduction per
// token. Prefill owns half a row and uses r4d_gdn_gated_rmsnorm_h128_bf16 instead.
__device__ __forceinline__ float ru_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;
}

// One token's replay factors (anchor state), inside the head's block of a factor slot:
//
//     [0    .. 4*K) normalised k, FOUR copies -- one per fixed 32-row group
//     [4*K  .. 4*K+V) the v row, one float per v row
//     [4*K+V .. +8)   decay and beta, one pair per row group
//
// Four k copies avoid a cross-workgroup barrier: a head's v rows may be split over up to four
// workgroups, and fixed 32-row groups make every byte belong to exactly one workgroup at any split.
// Read-before-write within a workgroup is ordered by a barrier.
#define RU_FSTR (4 * RU_K + RU_V + 8)

// RU_TMAX bounds the candidate block STG can stage (LDS for four partials and two scales per
// token). The launcher checks the caller's width against it.
#define RU_TMAX 16
#ifndef RU_TREE_DEFER
#define RU_TREE_DEFER 1
#endif
#ifndef RU_SISEL
#define RU_SISEL 1
#endif
#ifndef RU_PRE
#define RU_PRE 1
#endif
#ifndef RU_FUSE
#define RU_FUSE 1
#endif
#ifndef RU_PD
#define RU_PD 3
#endif

// RU_ANC_DUP=0 drops the two neighbouring-page copies of the replay factors (and, without the
// marker, of the anchor). Incorrect without RU_ANC_MARK (a moved page index reads an unwritten
// slot). With the marker it writes the factor block once; `-DRU_ANC_MARK=1 -DRU_ANC_DUP=0` is the
// leanest build, since the factor block (RU_TMAX * RU_FSTR) outweighs the state (RU_V * RU_K).
#ifndef RU_ANC_DUP
#define RU_ANC_DUP 1
#endif

// Shift marker: instead of writing the anchor three times (one per possible page shift), each of
// the three factor-block copies records the column it was written to. The reader finds `m` at its
// fixed factor column, so the page index moved by `m - 1`, and reads the anchor at the matching
// column; the anchor is written once (6.3 MB a layer instead of 18.9 at eight sequences, roughly a
// fifth of the layer's time).
//
// The marker lies past all factor data (RU_TMAX tokens of RU_FSTR) within one head's slice, so no
// workgroup reads a marker another is writing. Stored values are 1, 2, 3 (one-based; see the read),
// exact in fp16 and fp32.
//
// Invariant: the anchor is written at column 2 and read at 3 - m, never column 0. vLLM's
// precopy_mamba_align_fused_kernel copies state[bt[src_col + num_accepted - 1]] into this window's
// column 0 before the forward on a page crossing, so a read of column 0 (anchor at 1, read at
// 2 - m, shift +1) would see migrated data and degenerate intermittently.
#ifndef RU_ANC_MARK
#define RU_ANC_MARK 1
#endif
#define RU_MARK_OFF (RU_TMAX * RU_FSTR)

// RU_ANC_FRD=1 (diagnostic): use the displaced factor read (column 5 - m) while still writing all
// three factor copies, isolating the marker's read-column logic from the write side.
#ifndef RU_ANC_FRD
#define RU_ANC_FRD 0
#endif

// Decode convolution folded in (CV). This kernel is gdn_conv_update's only decode reader, so the
// fold saves a launch and a dependent-dispatch gap. Each workgroup computes the convolution its
// token loop reads (its key head's 128 q and 128 k channels and its value head's 128 v channels)
// from the projection rows into LDS; q, k and v never reach memory. The arithmetic is
// gdn_conv_update's (same taps, order, cp_silu and bf16 rounding), so results are byte-identical.
//
// The conv state is rewritten in place, and a key head's channels are read by all H / Hg value-head
// workgroups of its group, so none may rewrite them until all have read the history. Each
// workgroup arrives on a counter after reading; the last arriver rewrites the key head's q and k
// channels and resets the counter (no waiting). A value head's v channels have one reader and are
// rewritten by it. Two launches sharing a counter block must not overlap; blocks are keyed on
// (stream, output), as split_ctr in r4d_gemm_fp8a8.hip.
//
// CV is also the weight's type: 1 bf16, 2 f32, read as gdn_conv_update reads it.
struct RuConv {
  const unsigned short* x; long xpitch;   // the projection's [q | k | v] columns, row pitch
  const void* w;                          // [conv_dim, R4D_GDN_CONV_W], bf16 or f32 (CV)
  const unsigned short* b;                // [conv_dim] or null
  unsigned short* cs;                     // the conv state cache
  long cs_seq, cs_dim, cs_tok;
  const int* cidx; long ci_stride;        // conv state slot per sequence; negative is padding
  unsigned int* ctr;                      // one arrival counter per (sequence, key head)
};
// NORM 4's arrival counters, one per (sequence, pair of value heads), are the same block's from
// here on: the key heads' take the entries below it.
constexpr unsigned kRuPairCtr = 2048;
// Max tokens per sequence for the folded convolution (preloaded into registers, static indices).
#define RU_CVT 8

// A channel's conv state rewritten as gdn_conv_update does: shifted by one, the old history's
// tail then this step's tokens. The launcher requires the window split to be R4D_GDN_CONV_W - 2,
// so these are the same stores.
__device__ __forceinline__ void ru_conv_rewrite(unsigned short* sp, const unsigned* hist,
                                                const unsigned* xs, int T, long cs_tok) {
  constexpr int NH = R4D_GDN_CONV_W - 2;
#pragma unroll
  for (int i = 0; i < NH; ++i) sp[(size_t)i * cs_tok] = hist[i + 1];
#pragma unroll
  for (int t = 0; t < RU_CVT; ++t)
    if (t < T) sp[(size_t)(t + NH) * cs_tok] = xs[t];
}

// ---- Tree verification (RadBatch::spec_parent / spec_path; docs/OPS.md, "Tree verification of the
// delta net"). Anchor state, f32 state only, one workgroup a head (VS 1).
//
// spec_parent [T]: a row's parent within its sequence's rows; an entry that does not name an
//   earlier row means the row before it (nothing for row 0), so -2 everywhere is the chain.
// spec_path [N, pitch]: the rows of the PREVIOUS block that were committed, in order. The step's
//   replay applies the records of THOSE rows (a chain replays rows 0..nacc-1), and the folded
//   convolution reads its history through them.
//
// Every row starts from its parent's state and writes its own record at its own row index, so the
// next step can replay any ordered subset of them. The workgroup holds ONE state in registers, so
// the rows are visited in a depth-first order planned up front (ru_tree_plan):
//   * a row with no children (a leaf) is computed from the live state WITHOUT advancing it: the
//     same multiply, dot, rank-1 update and q-dot as an advancing row, on a temporary (the
//     register array is not written), so its output is bit-identical to the advancing form;
//   * a node's non-leaf children after the first start from the node's state again, re-derived
//     from the published anchor by replaying the node's ancestors' records (kept in LDS, exact
//     copies of the values the replay would read) with the replay's own arithmetic. In f32 that
//     equals the in-register chain bit for bit, which is what the step-to-step replay already
//     relies on; the fp16 state cache rounds records and is therefore not served.
// Row order in memory is untouched: outputs and records are indexed by row, not visit order.
struct RuSpec {
  const int* parent;     // [T] or null (a chain)
  const int* path;       // [N, pitch] or null (the identity)
  int pitch;
};
#define RU_SCHED_MAX 256
// The schedule entry's kind (bits 8..): 0 advance the live state by the row, 1 the row as a leaf of
// the live state (read only), 2 reload the published anchor, 3 replay the row's record onto it.
// Run by the first wave (lanes 0..31), every lane the same instruction stream. The tree's tables
// live one entry a lane (row r in lane r, T <= RU_TMAX <= 16) and as 32-bit masks in scalar
// registers, so the walk below is scalar code with lane reads, not a thread chasing indices
// through LDS (that form took 20k cycles at seven rows: a third of the kernel). The schedule it
// writes is the same one the sequential walk wrote. The plan is shared through LDS, read after the
// barrier that follows the anchor's publication.
__device__ __forceinline__ void ru_tree_plan(const int* __restrict__ spar, int bos, int T,
                                             int* par, int* anc, int* sched, int* tmp) {
  const int lane = (int)(threadIdx.x & 31);
  const bool valid = lane < T;
  int pr = -1;
  if (valid) {
    const int pv = spar ? spar[bos + lane] : -2;
    pr = (pv >= 0 && pv < lane) ? pv : (lane > 0 ? lane - 1 : -1);
  }
  if (valid) par[lane] = pr;
  // The ancestor chain of every row as the convolution's taps want it: entry d is the d-th
  // ancestor, a row index, or past the sequence's first row -1 - e for the window's e-th newest.
  if (valid) {
    int cur = lane;
    anc[lane * 4] = lane;
#pragma unroll
    for (int d = 1; d < 4; ++d) {
      const int up = __builtin_amdgcn_ds_bpermute((cur >= 0 ? cur : 0) << 2, pr);
      cur = cur >= 0 ? up : cur - 1;
      anc[lane * 4 + d] = cur;
    }
  }
  // kid: the mask of a row's children, in the row's lane. root: the rows whose parent is -1.
  unsigned kid = 0u;
#pragma unroll
  for (int r = 0; r < RU_TMAX; ++r) {
    const unsigned b = __builtin_amdgcn_ballot_w32(valid && pr == r);
    if (lane == r) kid = b;
  }
  const unsigned root = __builtin_amdgcn_ballot_w32(valid && pr == -1);
  const unsigned nonleaf = __builtin_amdgcn_ballot_w32(valid && kid != 0u);
  const unsigned allm = T >= 32 ? 0xFFFFFFFFu : ((1u << T) - 1u);
  const unsigned leafm = allm & ~nonleaf;
  // ancm: a row's ancestors and itself, one bit a row (ancestors have smaller indices).
  unsigned ancm = valid ? (1u << lane) : 0u;
#pragma unroll
  for (int r = 0; r < RU_TMAX; ++r) {
    const unsigned a = (unsigned)__builtin_amdgcn_readlane((int)ancm, r);
    if (valid && pr == r) ancm |= a;
  }
  // rem: the non-leaf children a node has yet to start, in the node's lane.
  unsigned rem = kid & ~leafm;
  unsigned started = 0u;      // bit p: a non-leaf child of p has been started
  int n = 0;
  int p = -1;
  bool rootdone = false;
#define RU_EMIT(v) do { if (lane == 0 && n < RU_SCHED_MAX) sched[n] = (v); ++n; } while (0)
  while (true) {
    unsigned r;
    if (p < 0) r = rootdone ? 0u : root;
    else r = (unsigned)__builtin_amdgcn_readlane((int)rem, p);
    if (r == 0u) {
      if (p < 0) break;
      p = __builtin_amdgcn_readlane(pr, p);
      continue;
    }
    const int c = __builtin_ctz(r);
    if (p < 0) rootdone = true;
    else if (lane == p) rem &= ~(1u << c);
    if (p >= 0 && ((started >> p) & 1u)) {
      // A later non-leaf child: the first consumed the node's state; re-derive it from the anchor.
      RU_EMIT(2 << 8);
      unsigned am = (unsigned)__builtin_amdgcn_readlane((int)ancm, p);
      while (am) { RU_EMIT((3 << 8) | __builtin_ctz(am)); am &= am - 1u; }
    } else if (p >= 0) {
      started |= 1u << p;
    }
    RU_EMIT(c);
    // The leaves of c come first, while the live state is c's.
    unsigned lm = (unsigned)__builtin_amdgcn_readlane((int)kid, c) & leafm;
    while (lm) { RU_EMIT((1 << 8) | __builtin_ctz(lm)); lm &= lm - 1u; }
    p = c;
  }
#undef RU_EMIT
  if (lane == 0) tmp[127] = n;
}

// The conv state entry that holds the e-th entry of the committed sequence: the chain reads entry e,
// a tree's previous block committed rows path[0..), whose inputs sit at entry (width - 2) + row.
template <int TREE>
__device__ __forceinline__ int ru_hist_entry(int e, const RuSpec& spec, int n) {
  if constexpr (TREE) {
    constexpr int NH = R4D_GDN_CONV_W - 2;
    if (spec.path && e >= NH) {
      int pi = e - NH;
      pi = pi < spec.pitch ? pi : spec.pitch - 1;
      return NH + spec.path[(size_t)n * spec.pitch + pi];
    }
  }
  return e;
}

// The scaled q and k rows in LDS are read by two half-row groups of lanes at once (the two threads of
// a v row); a plain row puts the halves 64 words apart, the same banks, so every wave read is a
// two-way conflict. Each row is laid out with a 4-word gap after its first half, which moves the
// second half four banks over: element e of a row is at e + 4 * (e >= RU_HK), a row is RU_LP long.
#define RU_HP (RU_HK + 4)
#define RU_LP (RU_K + 8)
#define RU_LDSK(c4) (4 * (c4) + ((c4) >= RU_HK / 4 ? 4 : 0))
// A workgroup barrier that orders LDS only. __syncthreads also waits for every global store the
// thread has issued to be acknowledged, a round trip to memory behind each barrier that follows the
// anchor, factor or output stores. Nothing here reads another thread's global store before the
// kernel (or the chain's completion, which keeps __syncthreads) ends; the loads that precede a
// barrier were consumed before it, which is all a later overwrite needs.
__device__ __forceinline__ void ru_bar() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
#define RU_PARAMS                                                                              \
  const unsigned short* __restrict__ q, const unsigned short* __restrict__ k, const           \
  unsigned short* __restrict__ v, const void* __restrict__ av, const void* __restrict__       \
  bv, long ab_stride, const float* __restrict__ A_log, const float* __restrict__ dt_bias,     \
  typename ru_state<SDT>::T* __restrict__ state, long st_slot, long st_head, unsigned         \
  short* __restrict__ o, unsigned char* __restrict__ oq, float* __restrict__ os, long         \
  oq_pitch, long os_ld, const int* __restrict__ cu, const int* __restrict__ sidx, long        \
  sidx_stride, const int* __restrict__ naccept, const unsigned short* __restrict__ zg,        \
  long zg_pitch, const float* __restrict__ nw, float neps, int nact, int H, int Hg, float     \
  scale, float sp_thr, const RuConv cv, const RuSpec spec
template <int ABF16, int NORM, int VS, int NT, int ANC, int STG = 0, int SDT = 0,
          int CV = 0, int TREE = 0>
__device__ __forceinline__ void ru_update_body(RU_PARAMS)
{
  using ST = typename ru_state<SDT>::T;
  constexpr int NR = RU_V / VS;         // v rows this workgroup owns
  constexpr int THR = NR * 2;
  static_assert(!(NORM && VS > 1), "the fused row norm needs the whole 128-wide row");
  static_assert(NORM != 4 || CV, "the pair's arrival counter is the folded convolution's block");
  static_assert(!TREE || (ANC && SDT == 0 && VS == 1), "a tree verify is the f32 anchor form");
  // The tree's plan and the replay records of this step's rows (exact copies, see RuSpec).
  __shared__ int s_par[TREE ? RU_TMAX : 1];
  __shared__ int s_anc[TREE ? RU_TMAX * 4 : 1];
  __shared__ int s_sched[TREE ? RU_SCHED_MAX : 1];
  __shared__ int s_tmp[TREE ? 128 : 1];
  __shared__ int s_rtok[TREE ? THR / 32 : 1];
  __shared__ int s_ord[TREE ? RU_CVT : 1];            // the row each emission slot holds
  __shared__ __attribute__((aligned(16))) float s_rk[TREE ? RU_TMAX * RU_K : 1];
  __shared__ __attribute__((aligned(16))) float s_rv[TREE ? RU_TMAX * RU_V : 1];
  __shared__ float s_rgb[TREE ? RU_TMAX * 2 : 1];
  __shared__ float wsum[THR / 32];
  __shared__ float wsum2[RU_CVT][THR / 32];   // the row norm's partials, one slot a row (the fused loop has no barrier)
  // NORM 3: a block of tokens' bf16 output rows, for the rotation (one token a wave afterwards)
  constexpr int RB = THR / 32;
  __shared__ __attribute__((aligned(8))) unsigned short srot[NORM == 3 ? RB * RU_V : 1];
  // RU_PRE: with the folded convolution and staged scales, every token's l2-scaled q and k as f32 in
  // LDS (the scaled values the token loop would otherwise rebuild per row: widen, scale), written
  // once by all threads between two barriers instead of inside the serial chain.
  constexpr bool PRE = RU_PRE && CV && STG && !TREE;
  constexpr bool FUSE = RU_FUSE && PRE && ANC && SDT == 0;
  // DEFER: the rows' epilogues (norm, stores, quantiser) run after the token chain. The fused loop
  // is one; a folded tree verify is the other, its rows in the planned order (and at most RU_CVT).
  constexpr bool DEFER = FUSE || (TREE && CV && RU_TREE_DEFER);
  // One array: the replay's staging (below) spreads the four group copies of up to RU_CVT k rows over
  // both halves of it.
  __shared__ __attribute__((aligned(16))) float spqk[PRE ? 2 * RU_CVT * RU_LP : 1];
  float* const spq = spqk;
  float* const spk = spqk + (PRE ? RU_CVT * RU_LP : 0);
  __shared__ float sgb[FUSE ? RU_CVT * 8 : 1];      // the staged rows' (g, beta) pairs, one per group
  __shared__ float son[DEFER ? RU_CVT * THR : 1];     // each row's output dot, finished after the loop
  __shared__ float sact[FUSE ? RU_CVT * RU_V : 1];   // the output gate's activation per (token, row)
  __shared__ float ssc[FUSE ? RU_CVT * 4 : 1];     // per token: e^g, beta, g
  __shared__ float spart[STG ? RU_TMAX * 4 : 1];   // per (token, half): sum q^2, sum k^2
  __shared__ float sscl [STG ? RU_TMAX * 2 : 1];   // per token: the two closed scales
  const int n = blockIdx.x, hv = blockIdx.y;
  const int bos = cu[n], T = cu[n + 1] - bos;
  // With ANC, T == 0 is valid: it folds the previous step's accepted prefix into the anchor before
  // something else reads the state cache. Without ANC there is nothing to do.
  const bool empty = (T <= 0);
  if (empty && !ANC) return;

  const int tid = threadIdx.x;
  // The plan is read after the barrier that follows the anchor's publication, which every
  // anchor-form workgroup reaches.
  const int row = (tid >> 1) + (VS > 1 ? blockIdx.z * NR : 0), half = tid & 1, k0 = half * RU_HK;
  const int hk = hv / (H / Hg);
  /* Read early so their latency overlaps the prologue's loads. */
  const float a_log_h = A_log[hv], dtb = dt_bias[hv];

  // ---- CV: issue the convolution's inputs first so they are in flight during the anchor replay.
  // Thread tid owns one q|k channel of the key head (q below RU_K, k above) and, below RU_V, one v
  // channel of the value head.
  static_assert(!CV || (NORM && VS == 1), "the folded convolution is the decode shape");
  static_assert(RU_K == RU_V, "one LDS row pitch serves the q, k and v taps");
  constexpr int CW = R4D_GDN_CONV_W, CS = R4D_GDN_CONV_W - 1;
  __shared__ __attribute__((aligned(16))) unsigned short cvq[CV ? RU_CVT * RU_K : 1],
      cvk[CV ? RU_CVT * RU_K : 1];
  __shared__ unsigned short cvv[CV ? RU_CVT * RU_V : 1];
  __shared__ int cvlast;
  /* The key head's history for the final rewrite (two newest history entries and this step's
   * inputs, one channel per thread), parked in LDS: in registers across the token loop they spill. */
  __shared__ unsigned short cvh[CV ? (CS - 1 + RU_CVT) * THR : 1];
  /* The token loop's per-token global reads (a, b and the output gate row) are also staged: at
   * decode there is too little occupancy to hide a global load per token. They are issued here,
   * parked in LDS at the convolution's barrier and read from there: `ca_b` is one token's a
   * (tid < T) or b (T <= tid < 2T), `czg` this thread's strided share of the gate rows. */
  constexpr int ZPT = (RU_CVT * RU_V + THR - 1) / THR;  // gate values a thread stages
  __shared__ float sab[CV ? 2 * RU_CVT : 1];
  __shared__ unsigned short szg[CV ? RU_CVT * RU_V : 1];
  float ca_b = 0.0f;
  unsigned czg[ZPT];
  const int G = H / Hg;                                 // workgroups that read one key head
  float cwa[CW], cwb[CW], cba = 0.0f, cbb = 0.0f;
  // 32-bit holders: packed 16-bit ones make the compiler merge loads with d16_hi, each waiting for
  // the one before it (a serial chain of memory latencies in the prologue).
  unsigned cha[CS], chb[CS], cxa[RU_CVT], cxb[RU_CVT];
  unsigned short* spa = nullptr;
  unsigned short* spb = nullptr;
  unsigned int arrived = 0;
  if constexpr (CV) {
    if (!empty) {
      const int cslot = cv.cidx[(size_t)n * cv.ci_stride];
      // A negative slot is padding: every workgroup of the sequence leaves here, so none arrives
      // on the counter.
      if (cslot < 0) return;
      const int off = naccept ? naccept[n] - 1 : 0;
      const int ca = tid < RU_K ? hk * RU_K + tid : (Hg + hk) * RU_K + (tid - RU_K);
      const int cb = (2 * Hg + hv) * RU_V + tid;
      using CWT = std::conditional_t<CV == 2, float, unsigned short>;
      const CWT* const cw = (const CWT*)cv.w;
      const unsigned short* xr = cv.x + (size_t)bos * cv.xpitch;
      // Every load below is unconditional (indices clamped to the valid range, the extra ones
      // unused): a load inside a branch makes the compiler wait for it at the join, which turns the
      // prologue into a chain of memory latencies.
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t)
        cxa[t] = xr[(size_t)(t < T ? t : T - 1) * cv.xpitch + ca];
      {
        const cp_taps<CWT> w4 = *(const cp_taps<CWT>*)(cw + (size_t)ca * CW);
        cwa[0] = cp_tap(w4.x); cwa[1] = cp_tap(w4.y); cwa[2] = cp_tap(w4.z); cwa[3] = cp_tap(w4.w);
        const unsigned short bra = *(cv.b ? cv.b + ca : cv.x);
        cba = cv.b ? bf2f(bra) : 0.0f;
      }
      spa = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)ca * cv.cs_dim;
#pragma unroll
      for (int i = 0; i < CS; ++i)
        cha[i] = spa[(size_t)ru_hist_entry<TREE>(off + i, spec, n) * cv.cs_tok];
      if (tid < RU_V) {
#pragma unroll
        for (int t = 0; t < RU_CVT; ++t)
          cxb[t] = xr[(size_t)(t < T ? t : T - 1) * cv.xpitch + cb];
        const cp_taps<CWT> w4 = *(const cp_taps<CWT>*)(cw + (size_t)cb * CW);
        cwb[0] = cp_tap(w4.x); cwb[1] = cp_tap(w4.y); cwb[2] = cp_tap(w4.z); cwb[3] = cp_tap(w4.w);
        const unsigned short brb = *(cv.b ? cv.b + cb : cv.x);
        cbb = cv.b ? bf2f(brb) : 0.0f;
        spb = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)cb * cv.cs_dim;
#pragma unroll
        for (int i = 0; i < CS; ++i)
          chb[i] = spb[(size_t)ru_hist_entry<TREE>(off + i, spec, n) * cv.cs_tok];
      }
      {
        const bool isa = tid < T;
        int tt = isa ? tid : tid - T;
        tt = tt < 0 ? 0 : (tt >= T ? T - 1 : tt);
        ca_b = ru_ab<ABF16>(isa ? av : bv, (size_t)(bos + tt) * ab_stride + hv);
      }
#pragma unroll
      for (int i = 0; i < ZPT; ++i) {
        const int zi = tid + i * THR;
        const int zc = zi < T * RU_V ? zi : 0;
        czg[i] = zg[(size_t)(bos + zc / RU_V) * zg_pitch + (size_t)hv * RU_V + (zc % RU_V)];
      }
    }
  }

  // Without ANC the initial state is the one the last accepted token wrote; with ANC there is one
  // state per sequence and the accepted prefix is replayed onto it below.
  const int nacc = naccept ? naccept[n] : 1;
  // Paged anchor. vLLM addresses mamba state by page, re-derived each step as
  // (seq_lens - 1) / block_size, so the column holding a sequence's state can move. seq_lens
  // includes this step's query (the draft depth, which varies), so
  //     start' - start = ((ctx + acc + q') - 1)/bs - ((ctx + q) - 1)/bs
  // is +1 when the committed context crosses a page, usually 0, and -1 when a step drafts fewer
  // tokens than the previous one.
  //
  // With the shift in {-1, 0, +1}, a fixed read column works only if all three pages it can land
  // on were written. So the anchor is written to columns 0, 1, 2 and read from column 1, and the
  // replay factors are written to columns 3, 4, 5 and read from column 4:
  //     shift -1 -> next column 1 is this column 0     (and next column 4 is this column 3)
  //     shift  0 -> next column 1 is this column 1     (and next column 4 is this column 4)
  //     shift +1 -> next column 1 is this column 2     (and next column 4 is this column 5)
  // The six columns are distinct writable slots: vLLM's per-candidate path writes to the same
  // indices, and indices_stride is num_spec + 1 = 8 whatever the draft depth.
  //
  // vLLM also migrates state itself: precopy_mamba_align_fused_kernel copies
  // state[bt[src_col + num_accepted - 1]] into state[bt[dst_col]] on a page crossing, assuming the
  // per-candidate layout. The anchor and factor read columns resolve to pages it never writes.
  // Any change to these columns must be re-derived against that kernel too.
  //
  // Nothing else may write this cache: advancing one copy alone leaves the others stale. Every step
  // shape, including non-speculative decode (`kind == "decode1"` in radiance_gdn.py), goes through
  // the fused path, and any other step first folds the accepted prefix into a single state.
  const int sa0 = ANC ? sidx[(size_t)n * sidx_stride + 0] : 0;   // anchor, page before
  const int sa1 = ANC ? sidx[(size_t)n * sidx_stride + 1] : 0;   // anchor, written here
  const int sa2 = ANC ? sidx[(size_t)n * sidx_stride + 2] : 0;   // anchor, page after
  const int sf0 = ANC ? sidx[(size_t)n * sidx_stride + 3] : 0;   // factors, marker 0
  const int sf  = ANC ? sidx[(size_t)n * sidx_stride + 4] : 0;   // factors, marker 1, read here
  const int sf2 = ANC ? sidx[(size_t)n * sidx_stride + 5] : 0;   // factors, marker 2
  // A negative slot is padding; 0 is a real slot (as for the per-candidate read below).
  if (ANC && (sa0 < 0 || sa1 < 0 || sa2 < 0 || sf0 < 0 || sf < 0 || sf2 < 0)) return;
  ST* const fb  = ANC ? (state + (size_t)sf  * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb0 = ANC ? (state + (size_t)sf0 * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb2 = ANC ? (state + (size_t)sf2 * st_slot + (size_t)hv * st_head) : nullptr;
  // Anchor column: without markers, column 1 (all three copies written); with them, 3 - m, never 0
  // (column 0 is the migration's target, derived from the committed context; see RU_ANC_MARK).
  // A caller whose state does not page (this engine's linear pool: one slot per sequence for its
  // life) names the same slot in every column; then duplicates are skipped, saving two thirds of
  // the factor traffic.
  const bool dup_a = ANC && RU_ANC_DUP && (sa0 != sa2 || sa1 != sa2);
  const bool dup_f = ANC && RU_ANC_DUP && (sf0 != sf || sf2 != sf);
  int acol = RU_ANC_MARK ? 2 : 1;
  int mrk = 1;
  if (ANC && RU_ANC_MARK) {
    // The marker is one-based so a zeroed (freshly allocated) factor block reads as absent and
    // falls back to the default anchor column. Zero-based, it would steer the first read after a
    // prefill to column 3, a factor column of zeros, discarding the prefill's state.
    const int m = (int)((float)fb[RU_MARK_OFF] + 0.5f) - 1;
    mrk = (m >= 0 && m <= 2) ? m : 1;
    acol = 3 - mrk;
  }
  // With RU_ANC_DUP=0 (or RU_ANC_FRD) the factors are read at 5 - m (3, 4 or 5) and written only
  // at column 4. The marker itself stays at fixed column 4 and is stamped into all three copies.
  const ST* const fr = (ANC && RU_ANC_MARK && (RU_ANC_FRD || !RU_ANC_DUP))
      ? (const ST*)(state + (size_t)(mrk == 0 ? sf2 : (mrk == 1 ? sf : sf0)) * st_slot
                    + (size_t)hv * st_head)
      : (const ST*)fb;
  // The anchor columns 1, 2 and 3 are the entries loaded above (sa1, sa2, sf0): select rather than
  // issue a second dependent global load on the marker's critical path.
  const int si = (RU_SISEL && ANC && RU_ANC_MARK) ? (acol == 1 ? sa1 : (acol == 2 ? sa2 : sf0))
                                      : sidx[(size_t)n * sidx_stride + (ANC ? acol : (nacc - 1))];
  // A negative slot is padding; 0 is a real slot.
  if (si < 0) return;
  /* Fixed 32-row group (see RU_FSTR), uniform per wave for any VS (a wave is 16 consecutive rows
   * from a multiple of 16). readfirstlane makes the factor addresses scalar instead of 64-bit
   * VGPR offsets live across the token loop. */
  const int grp = __builtin_amdgcn_readfirstlane(row >> 5);

  float h[RU_HK];
  {
    const ST* p = state + (size_t)si * st_slot + (size_t)hv * st_head
                + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) ru_load8<NT, SDT>(&h[j], p + j);
  }
  // FUSE: the replay's factor rows (up to RU_CVT of them) are fetched together with the state, by all
  // threads, and parked in LDS (the arrays the token loop fills later) behind one barrier, so the
  // replay is not a chain of global round trips. Every copy is staged, not group 0's alone: the
  // four copies are written equal, but a replay must read what its row group's copy holds. A row is
  // 160 float4: four k copies of 32, then the v row of 32.
  constexpr int SU = 5;                     // RU_CVT rows * 160 float4 / RU_V * 2 threads
  ru_f4 stg[FUSE ? SU : 1];
  float stgb = 0.0f;
  const bool viaLds = FUSE && nacc <= RU_CVT;
  if constexpr (FUSE) {
#pragma unroll
    for (int i = 0; i < SU; ++i) {
      const int u = tid + i * THR, r = u / 160, w = u - r * 160;
      const int rr = r < nacc ? r : 0;
      stg[i] = *(const ru_f4*)(fr + (size_t)rr * RU_FSTR + (w < 128 ? 4 * w : 4 * RU_K + 4 * (w - 128)));
    }
    {
      const int gr = (tid >> 3) < nacc ? (tid >> 3) : 0;
      stgb = fr[(size_t)gr * RU_FSTR + 4 * RU_K + RU_V + (tid & 7)];
    }
  }
  // The convolution and the q/k scales need no state: when the replay's factor rows are staged
  // (FUSE with up to RU_CVT rows) they run while the state is still in flight, and only the
  // global stores of this step's factors wait for the replay's reads to be done.
  auto cv_stage = [&]() __attribute__((always_inline)) {
  if constexpr (CV) {
    // gdn_conv_update's token body, one channel per thread, into LDS instead of q, k and v.
    auto conv = [&](const float* wt, float bs, const unsigned* hist,
                    const unsigned* xs, unsigned short* dst) {
      float win[CW];
#pragma unroll
      for (int i = 0; i < CS; ++i) win[i] = bf2f((unsigned short)hist[i]);
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t) {
        if (t < T) {
          win[CW - 1] = bf2f((unsigned short)xs[t]);
          float acc = bs;
#pragma unroll
          for (int i = 0; i < CW; ++i) acc += wt[i] * win[i];
          dst[t * RU_K] = f2bf(cp_silu(acc));
#pragma unroll
          for (int i = 0; i < CW - 1; ++i) win[i] = win[i + 1];
        }
      }
    };
    // A tree row's window is its ancestor chain (s_anc): the same taps and arithmetic, the window
    // gathered per row instead of slid. Row inputs and history are registers selected statically.
    auto conv_tree = [&](const float* wt, float bs, const unsigned* hist,
                         const unsigned* xs, unsigned short* dst) {
#pragma unroll
      for (int t = 0; t < RU_CVT; ++t) {
        if (t < T) {
          float win[CW];
          win[CW - 1] = bf2f((unsigned short)xs[t]);
#pragma unroll
          for (int m = 1; m < CW; ++m) {
            const int an = s_anc[t * 4 + m];
            unsigned r = hist[0];
#pragma unroll
            for (int j = 1; j < CS; ++j)
              if (CS + an == j) r = hist[j];
#pragma unroll
            for (int j = 0; j < RU_CVT; ++j)
              if (an == j) r = xs[j];
            win[CW - 1 - m] = bf2f((unsigned short)r);
          }
          float acc = bs;
#pragma unroll
          for (int i = 0; i < CW; ++i) acc += wt[i] * win[i];
          dst[t * RU_K] = f2bf(cp_silu(acc));
        }
      }
    };
    if constexpr (TREE) conv_tree(cwa, cba, cha, cxa, (tid < RU_K ? cvq : cvk) + (tid & (RU_K - 1)));
    else conv(cwa, cba, cha, cxa, (tid < RU_K ? cvq : cvk) + (tid & (RU_K - 1)));
#pragma unroll
    for (int i = 1; i < CS; ++i) cvh[(i - 1) * THR + tid] = cha[i];
#pragma unroll
    for (int t = 0; t < RU_CVT; ++t)
      if (t < T) cvh[(CS - 1 + t) * THR + tid] = cxa[t];
    if (tid < RU_V) {
      if constexpr (TREE) conv_tree(cwb, cbb, chb, cxb, cvv + tid);
      else conv(cwb, cbb, chb, cxb, cvv + tid);
      ru_conv_rewrite(spb, chb, cxb, T, cv.cs_tok);
    }
    if (tid < 2 * T) sab[tid] = ca_b;
#pragma unroll
    for (int i = 0; i < ZPT; ++i) {
      const int zi = tid + i * THR;
      if (zi < T * RU_V) szg[zi] = czg[i];
    }
    // All history is consumed (taps are in LDS), so after the barrier this workgroup no longer
    // reads the key head's conv state and may arrive on the counter.
    ru_bar();
    if (G > 1 && tid == 0)
      arrived = __hip_atomic_fetch_add(cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  }

  };
  auto stg_scale = [&](const bool gst) __attribute__((always_inline)) {
  const float alog = __expf(a_log_h);
  if (STG) {
    // One lane per (token, half) computes the same sums each v row computes on the STG=0 path.
    if (tid < 2 * T) {
      const int ut = tid >> 1, uk0 = (tid & 1) * RU_HK;
      const unsigned short* qp = CV ? cvq + ut * RU_K + uk0
                                    : q + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      const unsigned short* kp = CV ? cvk + ut * RU_K + uk0
                                    : k + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      float ssq = 0.0f, ssk = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) {
        const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
        const unsigned short* aq = (const unsigned short*)&wq;
        const unsigned short* ak = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float fq = bf2f(aq[e]), fk = bf2f(ak[e]);
          ssq += fq * fq; ssk += fk * fk;
        }
      }
      spart[tid * 2] = ssq;
      spart[tid * 2 + 1] = ssk;
    }
    ru_bar();
    if (tid < T) {
      // Same order as the STG=0 exchange: own half plus the other half.
      const float ssq = spart[tid * 4]     + spart[tid * 4 + 2];
      const float ssk = spart[tid * 4 + 1] + spart[tid * 4 + 3];
      sscl[tid * 2]     = __frsqrt_rn(ssq + 1e-6f) * scale;
      sscl[tid * 2 + 1] = __frsqrt_rn(ssk + 1e-6f);
      if constexpr (FUSE) {
        // The token's decay and beta, as the token loop computes them, and the replay factors'
        // pair; the loop reads them from LDS instead of running the transcendentals in the chain.
        const float ra = sab[tid], rb = sab[T + tid];
        const float g_ = -alog * ru_softplus(ra + dtb, sp_thr);
        const float bt_ = 1.0f / (1.0f + __expf(-rb));
        ssc[tid * 4]     = __expf(g_);
        ssc[tid * 4 + 1] = bt_;
        ssc[tid * 4 + 2] = g_;
        if (gst) {
        ST* const f = fb + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
        ST* const f0 = fb0 + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
        ST* const f2 = fb2 + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
#pragma unroll
        for (int gp = 0; gp < 4; ++gp) {
          f[2 * gp] = (ST)g_; f[2 * gp + 1] = (ST)bt_;
          if (dup_f) { f0[2 * gp] = (ST)g_; f0[2 * gp + 1] = (ST)bt_; f2[2 * gp] = (ST)g_; f2[2 * gp + 1] = (ST)bt_; }
        }
        }
      }
    }
    ru_bar();
  }
  };
  const bool early = FUSE && viaLds;
  if constexpr (FUSE) {
    if (early && !empty) { cv_stage(); stg_scale(false); }
  }
  if constexpr (FUSE) {
    if (viaLds) {
#pragma unroll
      for (int i = 0; i < SU; ++i) {
        const int u = tid + i * THR, r = u / 160, w = u - r * 160;
        if (r < nacc)
          *(ru_f4*)(w < 128 ? spqk + (r * 4 + (w >> 5)) * RU_LP + RU_LDSK(w & 31)
                            : sact + r * RU_V + 4 * (w - 128)) = stg[i];
      }
      if (tid < 8 * nacc) sgb[tid] = stgb;
      ru_bar();
    }
  }
  // The plan (first wave) runs while the state's loads are in flight; it is read after the barrier
  // that follows the anchor's publication, which every anchor-form workgroup reaches.
  if constexpr (TREE) {
    if (!empty && tid < 32) ru_tree_plan(spec.parent, bos, T < RU_TMAX ? T : RU_TMAX, s_par, s_anc,
                                         s_sched, s_tmp);
  }

  if (ANC) {
    for (int t = 0; t < nacc; ++t) {
      // The record of the t-th committed row: row t of a chain, spec_path[t] after a tree.
      int rec = t;
      if constexpr (TREE) {
        if (spec.path) {
          const int pi = t < spec.pitch ? t : spec.pitch - 1;
          rec = spec.path[(size_t)n * spec.pitch + pi];
          rec = rec < 0 ? 0 : (rec >= RU_TMAX ? RU_TMAX - 1 : rec);
        }
      }
      const ST* f = fr + (size_t)rec * RU_FSTR;
      float kk[RU_HK];
      float vv, eg, bt;
      if (FUSE && viaLds) {
#pragma unroll
        for (int j = 0; j < RU_HK; j += 4)
          *(ru_f4*)&kk[j] = *(const ru_f4*)(spqk + (t * 4 + grp) * RU_LP + half * RU_HP + j);
        vv = sact[t * RU_V + row];
        eg = __expf(sgb[t * 8 + 2 * grp]);
        bt = sgb[t * 8 + 2 * grp + 1];
      } else {
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) ru_load8<0, SDT>(&kk[j], f + grp * RU_K + k0 + j);
        vv = (float)f[4 * RU_K + row];
        eg = __expf((float)f[4 * RU_K + RU_V + 2 * grp]);
        bt = (float)f[4 * RU_K + RU_V + 2 * grp + 1];
      }
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
      dot += r4d_xor_lane<1>(dot);          // DPP, no LDS round trip
      const float u = (vv - dot) * bt;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) h[j] += u * kk[j];
    }
    // Publish the advanced anchor, then barrier: every replay read above must complete before the
    // token loop overwrites the factor block with this step's factors.
    const size_t rowoff = (size_t)hv * st_head + (size_t)row * RU_K + k0;
    ST* p  = state + (size_t)(RU_ANC_MARK ? sa2 : sa1) * st_slot + rowoff;
    ST* p0 = state + (size_t)sa0 * st_slot + rowoff;
    ST* p2 = state + (size_t)sa2 * st_slot + rowoff;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
    if (dup_a && !RU_ANC_MARK) {
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p0 + j, &h[j]);
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p2 + j, &h[j]);
    }
    ru_bar();
    // Stamp after the barrier: every thread read fb's marker at the top, and stamping earlier
    // could steer a late wave to a different column from the rest of the workgroup.
    if (RU_ANC_MARK && tid == 0) {                      // the three copies say where they are
      fb0[RU_MARK_OFF] = (ST)1.0f;
      fb[RU_MARK_OFF]  = (ST)2.0f;
      fb2[RU_MARK_OFF] = (ST)3.0f;
    }
  }
  if (empty) return;

  if constexpr (FUSE) {
    if (early) {
      // This step's decay and beta into the factor block (the replay has read the old ones).
      if (tid < T) {
        const float g_ = ssc[tid * 4 + 2], bt_ = ssc[tid * 4 + 1];
        ST* const f = fb + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
        ST* const f0 = fb0 + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
        ST* const f2 = fb2 + (size_t)tid * RU_FSTR + 4 * RU_K + RU_V;
#pragma unroll
        for (int gp = 0; gp < 4; ++gp) {
          f[2 * gp] = (ST)g_; f[2 * gp + 1] = (ST)bt_;
          if (dup_f) { f0[2 * gp] = (ST)g_; f0[2 * gp + 1] = (ST)bt_; f2[2 * gp] = (ST)g_; f2[2 * gp + 1] = (ST)bt_; }
        }
      }
    } else {
      cv_stage();
      stg_scale(true);
    }
  } else {
    cv_stage();
    stg_scale(true);
  }
  const float alog = __expf(a_log_h);
  if (STG) {
    if constexpr (PRE) {
#pragma unroll
      for (int i = 0; i < (RU_CVT * 64 + THR - 1) / THR; ++i) {
        const int u = tid + i * THR;
        if (u < T * 64) {
          const int ut = u >> 6, isk = (u >> 5) & 1, c = u & 31;
          const unsigned short* src = (isk ? cvk : cvq) + ut * RU_K + 4 * c;
          const float sc = sscl[ut * 2 + isk];
          const uint2 w = *(const uint2*)src;
          ru_f4 r;
          r[0] = bf2f((unsigned short)(w.x & 0xFFFFu)) * sc;
          r[1] = bf2f((unsigned short)(w.x >> 16)) * sc;
          r[2] = bf2f((unsigned short)(w.y & 0xFFFFu)) * sc;
          r[3] = bf2f((unsigned short)(w.y >> 16)) * sc;
          *(ru_f4*)((isk ? spk : spq) + ut * RU_LP + RU_LDSK(c)) = r;
          if constexpr (FUSE) {
            // The replay factors' four k copies (one per 32-row group), written here once by all
            // threads instead of by the group leaders inside the token loop.
            if (isk) {
              ST* const f = fb + (size_t)ut * RU_FSTR + 4 * c;
              ST* const f0 = fb0 + (size_t)ut * RU_FSTR + 4 * c;
              ST* const f2 = fb2 + (size_t)ut * RU_FSTR + 4 * c;
#pragma unroll
              for (int gp = 0; gp < 4; ++gp) {
                *(ru_f4*)(f + gp * RU_K) = r;
                if (dup_f) { *(ru_f4*)(f0 + gp * RU_K) = r; *(ru_f4*)(f2 + gp * RU_K) = r; }
              }
            }
          }
        }
      }
      if constexpr (FUSE) {
        for (int u = tid; u < T * RU_V; u += THR) {
          const int ut = u / RU_V, vr = u % RU_V;
          const ST vv = (ST)bf2f(cvv[u]);
          sact[u] = ru_act(bf2f(szg[u]), nact);
          fb[(size_t)ut * RU_FSTR + 4 * RU_K + vr] = vv;
          if (dup_f) { fb0[(size_t)ut * RU_FSTR + 4 * RU_K + vr] = vv; fb2[(size_t)ut * RU_FSTR + 4 * RU_K + vr] = vv; }
        }
      }
      ru_bar();
    }
  }

  // A tree is walked in its planned order (ru_tree_plan); `oi` counts the rows emitted so far.
  const int nsteps = TREE ? __builtin_amdgcn_readfirstlane(s_tmp[127]) : T;
  int oi = 0;
  // The row's epilogue: gated norm, bf16 store, the folded quantiser, a per-candidate state store.
  // The norm weight is the same every row: one load, not one behind each row's barrier.
  const float nwr = NORM ? nw[row] : 0.0f;
  // The norm's two halves: this thread's partial of the row's sum of squares into LDS, and (after
  // the barrier, or in the pipelined form after the next row's pass) the total and the rest.
  auto post_a = [&](const int slot, const float on) {
    if (NORM) {
      // sum of squares over the row: one channel per thread pair, 16 per wave
      float sq = (half == 0) ? on * on : 0.0f;
      // compile-time masks: 16 uses v_permlanex16, the rest DPP row_xmask
      sq += r4d_xor_lane<16>(sq); sq += r4d_xor_lane<8>(sq); sq += r4d_xor_lane<4>(sq);
      sq += r4d_xor_lane<2>(sq);  sq += r4d_xor_lane<1>(sq);
      float* const wn = DEFER ? wsum2[slot] : wsum;
      if ((threadIdx.x & 31) == 0) wn[threadIdx.x >> 5] = sq;
    }
  };
  auto post_b = [&](const int slot, const int t, const int tok, float on) {
    if (NORM) {
      float* const wn = DEFER ? wsum2[slot] : wsum;
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) tot += wn[i];
      const float sc = __frsqrt_rn(tot * (1.0f / RU_V) + neps);
      float actv;
      if constexpr (FUSE) {
        actv = sact[t * RU_V + row];
      } else {
        const unsigned short zr = CV ? szg[t * RU_V + row]
                                     : zg[(size_t)tok * zg_pitch + (size_t)hv * RU_V + row];
        actv = ru_act(bf2f(zr), nact);
      }
      on *= sc * nwr * actv;
      // The pipelined form double buffers the partials (the next row's lands in the other buffer,
      // and the one after it only after the next row's barrier, which every thread reaches after
      // reading this one); the plain form reuses one and fences it.
      if (!DEFER) ru_bar();
    }

    const unsigned short ob = f2bf(on);
    if (half == 0) o[((size_t)tok * H + hv) * RU_V + row] = ob;

    // The out projection's quantiser, folded in (saves 48 launches per decode step, each with less
    // work than its launch gap). The quant group (128) is the row and NORM gives one workgroup per
    // (token, head), so the amax is a workgroup reduction and the scale index is (tok, hv).
    // Bit-identical to the standalone quantiser: it quantises the stored bf16 `ob`, not the f32
    // `on`, with r4d_fused_quant_fp8.hip's quant_group arithmetic (including the clamp that keeps
    // the group maximum from becoming NaN). NORM == 2 writes int8 codes instead (quant_act_i8g:
    // scale amax / 127, round to nearest-even).
    // NORM == 3 writes had_quant_act_i8g's codes: the row's 128 stored bf16 values through
    // r4d_fwht_wave<128> in one wave (a lane four consecutive channels, that function's mapping),
    // then its amax / 127 scale and round-to-nearest-even, so the bytes are that quantiser's over
    // `o`. Nothing in the token loop reads the codes, so the rows wait in LDS and every RB tokens
    // (and after the last) each wave takes one token's: the rotation leaves the serial chain of
    // the state update, which otherwise pays a barrier pair and one wave's work a token.
    if (NORM == 3 && oq) {
      // Rows are batched in the order they are emitted; a tree's are not consecutive tokens.
      const int pos = TREE ? oi : t;
      const int tb = pos % RB;
      if (half == 0) srot[tb * RU_V + row] = ob;
      if constexpr (TREE) { if (tid == 0) s_rtok[tb] = tok; }
      if (tb == RB - 1 || pos == T - 1) {
        ru_bar();
        const int rw = tid >> 5, rl = tid & 31;
        if (rw <= tb) {
          const int rtok = TREE ? s_rtok[rw] : tok - tb + rw;
          const uint2 rr = *(const uint2*)(srot + rw * RU_V + 4 * rl);
          float rv[4] = { bf2f((unsigned short)(rr.x & 0xFFFFu)), bf2f((unsigned short)(rr.x >> 16)),
                          bf2f((unsigned short)(rr.y & 0xFFFFu)), bf2f((unsigned short)(rr.y >> 16)) };
          r4d_fwht_wave<128>(rv, rl);
          float m = fmaxf(fmaxf(fabsf(rv[0]), fabsf(rv[1])), fmaxf(fabsf(rv[2]), fabsf(rv[3])));
          m = r4d_wave_max(m);
          const float qs = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
          if (rl == 0) os[(size_t)rtok * os_ld + hv] = qs;
          const float inv = 1.0f / qs;
          uint32_t pk = 0;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            pk |= ((uint32_t)(int)rintf(fminf(fmaxf(rv[j] * inv, -127.0f), 127.0f)) & 0xFFu) << (8 * j);
          *(uint32_t*)(oq + (size_t)rtok * oq_pitch + (size_t)hv * RU_V + 4 * rl) = pk;
        }
        ru_bar();             // srot is reused by the next block of tokens
      }
    }
    if ((NORM == 1 || NORM == 2) && oq) {
      const float ov = bf2f(ob);
      float mv = (half == 0) ? fabsf(ov) : 0.0f;
      mv = fmaxf(mv, r4d_xor_lane<16>(mv)); mv = fmaxf(mv, r4d_xor_lane<8>(mv));
      mv = fmaxf(mv, r4d_xor_lane<4>(mv));  mv = fmaxf(mv, r4d_xor_lane<2>(mv));
      mv = fmaxf(mv, r4d_xor_lane<1>(mv));
      if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = mv;
      ru_bar();
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) amax = fmaxf(amax, wsum[i]);
      const float qs = amax > 0.0f ? amax * (1.0f / q8_top<NORM == 2>()) : 1.0f;
      if (tid == 0) os[(size_t)tok * os_ld + hv] = qs;
      if (half == 0) {
        unsigned char code;
        if constexpr (NORM == 2) {
          code = (unsigned char)(int)rintf(fminf(fmaxf(ov * (1.0f / qs), -127.0f), 127.0f));
        } else {
          const float cv = fminf(fmaxf(ov * (1.0f / qs), -448.0f), 448.0f);
          code = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(cv, cv, 0, false) & 0xff);
        }
        oq[(size_t)tok * oq_pitch + (size_t)hv * RU_V + row] = code;
      }
      ru_bar();               // wsum is reused by the next token
    }

    if (!ANC) {
      const int so = sidx[(size_t)n * sidx_stride + t];
      // `>= 0`, not `> 0`: a negative slot is padding and 0 is a real slot (a single-sequence
      // decode's only slot), as for the read above.
      if (so >= 0) {
        ST* p = state + (size_t)so * st_slot + (size_t)hv * st_head
              + (size_t)row * RU_K + k0;
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
      }
    }
  };
  auto post_row = [&](const int t, const int tok, float on) {
    post_a(t, on);
    if (NORM) ru_bar();
    post_b(t, t, tok, on);
  };
  if constexpr (FUSE) {
    // The delta rule's two passes over the state fuse across rows: row t's update and output dot
    // (h += u k; on += h q) and row t+1's decay and key dot (h *= e^g; dot += h k) touch each
    // element once, in the order each chain had, so every sum and every rounding is the sequential
    // form's. The two chains are independent, which doubles the adds in flight. The scaled q and k
    // rows (and the per-token decay and beta) come from LDS, staged by all threads before the
    // loop together with the replay factors' stores, so the chain holds nothing else. The LDS
    // reads run RU_PD chunks of four ahead of the arithmetic: the chain is latency bound, and a
    // read issued next to its use would put an LDS round trip behind every four elements.
    constexpr int NC = RU_HK / 4;
    constexpr int PD = RU_PD;
    float kc[RU_HK];                       // the row's k: the previous pass's k' for this row
    float dot = 0.0f;
    {
      const float eg0 = ssc[0];
      ru_f4 ka[PD];
#pragma unroll
      for (int c = 0; c < PD; ++c) ka[c] = *(const ru_f4*)(spk + half * RU_HP + 4 * c);
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        const ru_f4 kv = ka[c % PD];
        if (c + PD < NC) ka[c % PD] = *(const ru_f4*)(spk + half * RU_HP + 4 * (c + PD));
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          kc[4 * c + e] = kv[e];
          h[4 * c + e] *= eg0; dot += h[4 * c + e] * kv[e];
        }
      }
    }
    dot += r4d_xor_lane<1>(dot);
    for (int t = 0; t < T; ++t) {
      const float vraw = bf2f(cvv[t * RU_V + row]);
      const float u = (vraw - dot) * ssc[t * 4 + 1];
      const bool last = (t + 1 >= T);
      const int tn = last ? t : t + 1;
      const float egn = last ? 1.0f : ssc[tn * 4];
      const float* const qrow = spq + t * RU_LP + half * RU_HP;
      const float* const nrow = spk + tn * RU_LP + half * RU_HP;
      ru_f4 qa[PD], na[PD];
#pragma unroll
      for (int c = 0; c < PD; ++c) { qa[c] = *(const ru_f4*)(qrow + 4 * c); na[c] = *(const ru_f4*)(nrow + 4 * c); }
      float on = 0.0f, dotn = 0.0f;
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        const ru_f4 qt = qa[c % PD], kn = na[c % PD];
        if (c + PD < NC) {
          qa[c % PD] = *(const ru_f4*)(qrow + 4 * (c + PD));
          na[c % PD] = *(const ru_f4*)(nrow + 4 * (c + PD));
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float x = h[4 * c + e] + u * kc[4 * c + e];
          on += x * qt[e];
          x = x * egn;
          dotn += x * kn[e];
          h[4 * c + e] = x;
          kc[4 * c + e] = kn[e];
        }
      }
      on += r4d_xor_lane<1>(on);
      dotn += r4d_xor_lane<1>(dotn);
      dot = dotn;
      son[t * THR + tid] = on;
    }
    // The rows' epilogues run after the chain: nothing in it reads them, so the loop carries no
    // barrier, no reduction and no output store, and the T norms' reductions overlap.
    for (int t = 0; t < T; ++t) post_a(t, son[t * THR + tid]);
    ru_bar();
    for (int t = 0; t < T; ++t) post_b(t, t, bos + t, son[t * THR + tid]);
  }
  if constexpr (!FUSE) {
  for (int si = 0; si < nsteps; ++si) {
    int t = si;
    bool leaf = false;
    if constexpr (TREE) {
      const int se = __builtin_amdgcn_readfirstlane(s_sched[si]);   // uniform: scalar registers
      const int kind = se >> 8;
      t = se & 255;
#ifndef RU_X1
      if (kind == 2) {
        // Back to the anchor: the barrier makes every earlier row's LDS record (and the published
        // anchor, which the same thread wrote) visible and ordered before the reload.
        __syncthreads();
        const ST* pa = state + (size_t)(RU_ANC_MARK ? sa2 : sa1) * st_slot + (size_t)hv * st_head
                     + (size_t)row * RU_K + k0;
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) ru_load8<0, SDT>(&h[j], pa + j);
        continue;
      }
      if (kind == 3) {
        // The replay's arithmetic, over the record this workgroup kept for row t.
        const float* const kr = s_rk + t * RU_K + k0;   // read from LDS where used
        const float vr = s_rv[t * RU_V + row];
        const float egr = __expf(s_rgb[2 * t]);
        const float btr = s_rgb[2 * t + 1];
        float dotr = 0.0f;
#pragma unroll
        for (int j = 0; j < RU_HK; ++j) { h[j] *= egr; dotr += h[j] * kr[j]; }
        dotr += r4d_xor_lane<1>(dotr);
        const float ur = (vr - dotr) * btr;
#pragma unroll
        for (int j = 0; j < RU_HK; ++j) h[j] += ur * kr[j];
        continue;
      }
#endif
      leaf = kind == 1;
    }
    const int tok = bos + t;
    const unsigned short* qp = CV ? cvq + t * RU_K + k0 : q + ((size_t)tok * Hg + hk) * RU_K + k0;
    const unsigned short* kp = CV ? cvk + t * RU_K + k0 : k + ((size_t)tok * Hg + hk) * RU_K + k0;

    // A tree row keeps q as the packed bf16 words it loaded and widens (and scales) it where the
    // output dot uses it: the same values, 32 registers instead of 64 across the state pass.
    float qq[TREE ? 1 : RU_HK], kk[RU_HK];
    uint4 qw[TREE ? RU_HK / 8 : 1];
    if constexpr (PRE) {
#pragma unroll
      for (int j = 0; j < RU_HK; j += 4) {
        *(ru_f4*)&qq[j] = *(const ru_f4*)(spq + t * RU_LP + half * RU_HP + j);
        *(ru_f4*)&kk[j] = *(const ru_f4*)(spk + t * RU_LP + half * RU_HP + j);
      }
    } else {
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) {
      const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
      if constexpr (TREE) qw[j / 8] = wq;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if constexpr (!TREE) qq[j + e] = bf2f(sq[e]);
        kk[j + e] = bf2f(sk[e]);
      }
    }
    }
    auto qraw = [&](int j) {
      if constexpr (TREE) {
        const unsigned w = ((const unsigned*)&qw[j >> 3])[(j & 7) >> 1];
        return __uint_as_float((j & 1) ? (w & 0xFFFF0000u) : (w << 16));
      } else {
        return qq[j];
      }
    };

    // l2 norms: 64 in-lane terms, then one exchange with the lane holding the other half
    float qs = 0.0f, ks = 0.0f;
    if (PRE) {
    } else if (STG) {
      qs = sscl[t * 2]; ks = sscl[t * 2 + 1];
    } else {
      float sq = 0.0f, sk = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { const float qj = qraw(j); sq += qj * qj; sk += kk[j] * kk[j]; }
      sq += r4d_xor_lane<1>(sq);
      sk += r4d_xor_lane<1>(sk);
      qs = __frsqrt_rn(sq + 1e-6f) * scale; ks = __frsqrt_rn(sk + 1e-6f);
    }
    if constexpr (!PRE) {
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { if constexpr (!TREE) qq[j] *= qs; kk[j] *= ks; }
    }

    const float ra  = CV ? sab[t]     : ru_ab<ABF16>(av, (size_t)tok * ab_stride + hv);
    const float rb  = CV ? sab[T + t] : ru_ab<ABF16>(bv, (size_t)tok * ab_stride + hv);
    const float g   = -alog * ru_softplus(ra + dtb, sp_thr);
    const float bt  = 1.0f / (1.0f + __expf(-rb));
    const float eg  = __expf(g);

    float dot = 0.0f;
    if constexpr (TREE) {
      // A leaf reads the live state without advancing it: the same products on a temporary, which
      // the second pass recomputes (h * eg is the same multiply) instead of storing.
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { const float hj = h[j] * eg; dot += hj * kk[j]; }
    } else {
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    }
    dot += r4d_xor_lane<1>(dot);

    const float vraw = bf2f(CV ? cvv[t * RU_V + row] : v[((size_t)tok * H + hv) * RU_V + row]);
    const float u = (vraw - dot) * bt;

    if (ANC) {
      // This step's factors for the next step's replay: one thread pair per row group writes the
      // group's k copy and decay/beta; every row writes its own v.
      ST* f  = fb  + (size_t)t * RU_FSTR;
      ST* f0 = fb0 + (size_t)t * RU_FSTR;
      ST* f2 = fb2 + (size_t)t * RU_FSTR;
      if ((row & 31) == 0) {
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) {
          float kn[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) kn[e] = kk[j + e];
          ru_store8<0, SDT>(f + grp * RU_K + k0 + j, kn);
          if (dup_f) {
            ru_store8<0, SDT>(f0 + grp * RU_K + k0 + j, kn);
            ru_store8<0, SDT>(f2 + grp * RU_K + k0 + j, kn);
          }
        }
        if (half == 0) {
          f[4 * RU_K + RU_V + 2 * grp]      = (ST)g;
          f[4 * RU_K + RU_V + 2 * grp + 1]  = (ST)bt;
          if (dup_f) {
            f0[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f0[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
            f2[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f2[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
          }
        }
      }
      if (half == 0) {
        f[4 * RU_K + row] = (ST)vraw;
        if (dup_f) { f0[4 * RU_K + row] = (ST)vraw; f2[4 * RU_K + row] = (ST)vraw; }
      }
    }

#ifndef RU_X3
    if constexpr (TREE) {
      // This row's record for a later restore: exact copies of what the factor block receives.
      if (row == 0) {
#pragma unroll
        for (int j = 0; j < RU_HK; ++j) s_rk[t * RU_K + k0 + j] = kk[j];
      }
      if (half == 0) s_rv[t * RU_V + row] = vraw;
      if (tid == 0) { s_rgb[2 * t] = g; s_rgb[2 * t + 1] = bt; }
    }
#endif

    float on = 0.0f;
    if constexpr (TREE) {
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) {
        const float hn = h[j] * eg + u * kk[j];
        on += hn * (qraw(j) * qs);
        h[j] = leaf ? h[j] : hn;
      }
    } else {
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
    }
    on += r4d_xor_lane<1>(on);

    if constexpr (TREE && CV && RU_TREE_DEFER) {
      son[oi * THR + tid] = on;
      if (tid == 0) s_ord[oi] = t;
    } else {
      post_row(t, tok, on);
    }
    ++oi;
  }
  if constexpr (TREE && CV && RU_TREE_DEFER) {
    // The same epilogues, after the chain, in emission order (the quantiser's batching is by it).
    const int ne = oi;
    ru_bar();                    // s_ord
    for (int e = 0; e < ne; ++e) post_a(e, son[e * THR + tid]);
    ru_bar();
    for (int e = 0; e < ne; ++e) {
      oi = e;
      const int te = s_ord[e];
      post_b(e, te, bos + te, son[e * THR + tid]);
    }
  }
  }

  // NORM 4: had_quant_act_i8g's codes at IQ8R's group, one scale per 256 columns of `o`: two value
  // heads' rows. A workgroup owns one head, so the second of a pair to arrive quantises both heads'
  // rows for every token of the sequence, from the stored bf16 `o`: each head's 128 values through
  // r4d_fwht_wave<128> in one wave (wave 0 the even head, wave 1 the odd), the amax over the pair,
  // then that quantiser's amax / 127 scale and round-to-nearest-even, so the bytes are its over
  // `o`. The arrival is router_gemm_topk_scatter_kernel's idiom: every store made visible
  // device-wide before the arrival is published, and the workgroup completing the count resets
  // it for the next launch and acquires its partner's rows. Every workgroup of a sequence takes
  // the same early returns above, so a pair's two arrive or neither does.
  if constexpr (NORM == 4) {
    if (oq) {
      __shared__ unsigned int s_pair;
      __threadfence();
      __syncthreads();
      unsigned int* const pc = cv.ctr + kRuPairCtr + (size_t)n * (H / 2) + (hv >> 1);
      if (tid == 0)
        s_pair = __hip_atomic_fetch_add(pc, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      __syncthreads();
      if (s_pair == 1u) {
        if (tid == 0) __hip_atomic_store(pc, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
        const int hp = hv & ~1, pw = tid >> 5, pl = tid & 31;
        for (int t = 0; t < T; ++t) {
          const int tok = bos + t;
          float rv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
          if (tid < 64) {
            const unsigned short* src = o + ((size_t)tok * H + hp + pw) * RU_V + 4 * pl;
#pragma unroll
            for (int j = 0; j < 4; ++j) rv[j] = bf2f(src[j]);
            r4d_fwht_wave<128>(rv, pl);
            float m = fmaxf(fmaxf(fabsf(rv[0]), fabsf(rv[1])), fmaxf(fabsf(rv[2]), fabsf(rv[3])));
            m = r4d_wave_max(m);
            if (pl == 0) wsum[pw] = m;
          }
          ru_bar();
          if (tid < 64) {
            const float m = fmaxf(wsum[0], wsum[1]);
            const float qs = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
            if (tid == 0) os[(size_t)tok * os_ld + (hp >> 1)] = qs;
            const float inv = 1.0f / qs;
            uint32_t pk = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
              pk |= ((uint32_t)(int)rintf(fminf(fmaxf(rv[j] * inv, -127.0f), 127.0f)) & 0xFFu) << (8 * j);
            *(uint32_t*)(oq + (size_t)tok * oq_pitch + (size_t)(hp + pw) * RU_V + 4 * pl) = pk;
          }
          ru_bar();             // wsum is reused by the next token
        }
      }
    }
  }

  if constexpr (CV) {
    // The last arriving reader rewrites the key head's q and k channels. The count was taken
    // before the token loop, hiding its latency.
    if (G > 1) {
      if (tid == 0) {
        cvlast = arrived == (unsigned)(G - 1);
        if (cvlast)
          __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
      }
      ru_bar();
      if (!cvlast) return;
    }
    unsigned rh[CS], rx[RU_CVT];
    rh[0] = 0;                                  /* never read: the oldest entry is shifted out */
#pragma unroll
    for (int i = 1; i < CS; ++i) rh[i] = cvh[(i - 1) * THR + tid];
#pragma unroll
    for (int t = 0; t < RU_CVT; ++t)
      rx[t] = t < T ? cvh[(CS - 1 + t) * THR + tid] : 0u;
    ru_conv_rewrite(spa, rh, rx, T, cv.cs_tok);
  }
}

// The kernel around the body, whose workgroups return where their work ends: CH `r4dchain` is the
// chained form (r4d_chain.h), every workgroup waiting before its first read and adding once after
// its last write.
// The chain's wait and completion (r4d_chain.h), with the barrier and fence unconditional: the
// header's forms return early where the engine's record is absent, and that branch around a
// barrier lets the compiler clone the whole body per side (two token loops, and the clone spills a
// tree form's state). The record is the same one; only a launch from outside the engine (no
// record) pays a barrier it did not need.
#if defined(__HIP__)
__device__ __forceinline__ void ru_chain_wait() {
  const R4dChainRec* r = r4d_chain_rec();
  if (r && threadIdx.x == 0) {
    const unsigned long long target = r->target;
    unsigned long long* ctr = r->ctr;
    int n = 0;
    while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
      if (n < R4D_CHAIN_FAST) { __builtin_amdgcn_s_sleep(1); ++n; }
      else __builtin_amdgcn_s_sleep(R4D_CHAIN_SLOW);
    }
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
__device__ __forceinline__ void ru_chain_done() {
  const R4dChainRec* r = r4d_chain_rec();
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (r && threadIdx.x == 0) __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#else
__device__ __forceinline__ void ru_chain_wait() { r4d_chain_wait(); }
__device__ __forceinline__ void ru_chain_done() { r4d_chain_done(); }
#endif

template <int ABF16, int NORM, int VS, int NT, int ANC, int STG, int SDT, int CV, int TREE,
          class CH>
__global__ __launch_bounds__(RU_V / VS * 2)
void r4d_gdn_recurrent_update_kernel(RU_PARAMS)
{
  if (CH::on) { if (TREE) ru_chain_wait(); else r4d_chain_wait(); }
  ru_update_body<ABF16, NORM, VS, NT, ANC, STG, SDT, CV, TREE>(
      q, k, v, av, bv, ab_stride, A_log, dt_bias, state, st_slot, st_head, o, oq, os, oq_pitch,
      os_ld, cu, sidx, sidx_stride, naccept, zg, zg_pitch, nw, neps, nact, H, Hg, scale, sp_thr,
      cv, spec);
  if (CH::on) { if (TREE) ru_chain_done(); else r4d_chain_done(); }
}
#undef RU_PARAMS

// Non-temporal hint on the state load and store (TH_LOAD_NT / TH_STORE_NT): no result changes;
// chosen per launch because its cost and saving land in different kernels. In isolation it makes
// this kernel 43.8% slower (DRAM-write bound), but no state survives the 64 MB last-level cache
// within a step, so the hint only decides who pays for the write-back, and it saves the following
// GEMMs more (about -140 us on the tiled GEMM vs +25 us here under anchor state).
//
// Anchor state: one state per sequence with the accepted prefix replayed onto it. The caller's
// prefill path must zero state slot sidx[n][1] for every sequence it writes. Off (per_candidate),
// every candidate writes its own slot, matching vLLM's Triton path. A radiance caller declares the
// form as the op's `state_form` (ru_state_form); the raw C entry, which vLLM's radiance_gdn.py
// calls, has no op parameters and reads RADIANCE_GDN_ANCHOR, as that caller does.
static bool ru_anchor() {
  static const bool v = [] {
    const char* e = getenv("RADIANCE_GDN_ANCHOR");
    return e && e[0] == '1';
  }();
  return v;
}

// STG: the q/k l2 scales are computed once per head and shared by the 128 v rows (bit-identical to
// per row). Folding the scales into the reductions instead is slower and moves about one output in
// 600 by a bf16 ULP.
//
// Flood size (MB of state a launch writes) up to which the non-temporal hint is used. It wins when
// the state would otherwise stay in the last-level cache and become the next GEMM's write-back
// (6.3 MB at one sequence: ~3% of a step) and loses when nothing survives anyway (50 MB at eight:
// ~2% slower). The threshold leaves cache room for the streaming weights. Under anchor state the
// hint nets about -100 us a step; under per-candidate state the flood is ~8x larger.
#define RU_NT_MB 16

// `oq_i8` picks the folded quantiser's codes: E4M3 (0) or int8 (1); see the kernel's epilogue.
static int ru_update_launch(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
    long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
    long oq_pitch, long os_ld, const void* cu,
    const void* ssm_state_indices, long indices_stride, const void* num_accepted,
    const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
    int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
    void* stream, int oq_i8, bool anc, RuSpec spec = RuSpec{}, int qlen = 0)
{
  if (K != RU_K || V != RU_V) return -1;
  if (H % Hg) return -2;
  const int nm0 = (z_gate && norm_weight) ? 1 : 0;
  // Fill the GPU: split a head's v rows until there are at least WGTARGET workgroups, at most
  // four ways (64 threads, two waves).
  int vs = 1;
  if (!nm0) { const int want = (WGTARGET + N * H - 1) / (N * H);
              vs = want >= 4 ? 4 : (want >= 2 ? 2 : 1); }
  // The shift marker forbids splitting a head: with vs > 1 one workgroup could stamp this step's
  // marker before another read the previous one, and they would disagree on the page shift. The
  // wave count is the same either way.
  if (anc && RU_ANC_MARK) vs = 1;
  dim3 grid(N, H, vs), blk(RU_V / vs * 2);
  // Every switch is a template parameter; the nested pick macros enumerate the instantiations.
#define RU_LAUNCH(AB, NM, VSC, NTC, ANCC, STGC, SDTC)                                         \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<AB, NM, VSC, NTC, ANCC, STGC, SDTC, 0,  \
                                                      0, R4dUnchained>),                      \
                     grid, blk, 0, (hipStream_t)stream,                                       \
                     (const unsigned short*)q, (const unsigned short*)k,                      \
                     (const unsigned short*)v, a, b, ab_stride,                               \
                     (const float*)A_log, (const float*)dt_bias,                              \
                     (ru_state<SDTC>::T*)state, state_slot_stride, state_head_stride,         \
                     (unsigned short*)o, (unsigned char*)oq, (float*)os, oq_pitch, os_ld,    \
                     (const int*)cu, (const int*)ssm_state_indices,                          \
                     indices_stride, (const int*)num_accepted,                                \
                     (const unsigned short*)z_gate, z_pitch, (const float*)norm_weight, norm_eps,      \
                     norm_act, H, Hg, scale, softplus_thr, RuConv{}, RuSpec{})
// The tree form: bf16 a/b, f32 anchor state, one workgroup a head, no non-temporal hint (the
// instantiations are few on purpose; the hint is a bandwidth choice, not a result).
#define RU_LAUNCH_T(NM, STGC)                                                                 \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<1, NM, 1, 0, 1, STGC, 0, 0, 1,          \
                                                      R4dUnchained>),                         \
                     grid, blk, 0, (hipStream_t)stream,                                       \
                     (const unsigned short*)q, (const unsigned short*)k,                      \
                     (const unsigned short*)v, a, b, ab_stride,                               \
                     (const float*)A_log, (const float*)dt_bias,                              \
                     (ru_state<0>::T*)state, state_slot_stride, state_head_stride,            \
                     (unsigned short*)o, (unsigned char*)oq, (float*)os, oq_pitch, os_ld,    \
                     (const int*)cu, (const int*)ssm_state_indices,                          \
                     indices_stride, (const int*)num_accepted,                                \
                     (const unsigned short*)z_gate, z_pitch, (const float*)norm_weight, norm_eps,      \
                     norm_act, H, Hg, scale, softplus_thr, RuConv{}, spec)
#define RU_PICKV(AB, NTC, ANCC, SDTC)                                                         \
  do { if (vs == 4)      RU_LAUNCH(AB, 0, 4, NTC, ANCC, 0, SDTC);                             \
       else if (vs == 2) RU_LAUNCH(AB, 0, 2, NTC, ANCC, 0, SDTC);                             \
       else              RU_LAUNCH(AB, 0, 1, NTC, ANCC, 0, SDTC); } while (0)
#define RU_PICKNM(AB, NTC, ANCC, SDTC)                                                        \
  do { if (oq_i8) RU_LAUNCH(AB, 2, 1, NTC, ANCC, 0, SDTC);                                    \
       else       RU_LAUNCH(AB, 1, 1, NTC, ANCC, 0, SDTC); } while (0)
#define RU_PICKAB(NTC, ANCC, SDTC)                                                            \
  do { if (ab_is_bf16) { if (nm0) RU_PICKNM(1, NTC, ANCC, SDTC);                              \
                         else     RU_PICKV(1, NTC, ANCC, SDTC); }                             \
       else            { if (nm0) RU_PICKNM(0, NTC, ANCC, SDTC);                              \
                         else     RU_PICKV(0, NTC, ANCC, SDTC); } } while (0)
#define RU_PICKNT(ANCC, SDTC)                                                                 \
  do { if (nt) RU_PICKAB(1, ANCC, SDTC); else RU_PICKAB(0, ANCC, SDTC); } while (0)
#define RU_PICKANC(SDTC)                                                                      \
  do { if (anc) RU_PICKNT(1, SDTC); else RU_PICKNT(0, SDTC); } while (0)

  // The staged path (STG) is instantiated only for the decode shape: bf16 a/b, no fused row norm,
  // VS 1. Everything else uses the per-row path (STG 0).
  const int sdt = state_fp16 ? 1 : 0;
  // Bytes of state this launch writes: one slot per sequence under the anchor, one per candidate
  // without it.
  const long slots_written = anc ? 1 : (long)indices_stride;
  const long flood = (long)N * slots_written * state_slot_stride * (sdt ? 2 : 4);
  const bool nt = flood <= (long)RU_NT_MB * 1024 * 1024;
  // Anchor bounds: six columns are addressed (three anchor, three factor), and the factor block
  // (indices_stride tokens of RU_FSTR) must fit in one head's slice of a slot. Keep these before
  // the staged path, which returns early and is the shipped anchor path.
  if (anc && indices_stride < 6) return -4;
  if (anc && (long)indices_stride * RU_FSTR > state_head_stride) return -3;
  if (spec.parent || spec.path) {
    // Refused with a code the entry point maps to RAD_E_UNSUPPORTED before it gets here; these
    // are the backstop for the raw C entry.
    if (!anc || sdt || !ab_is_bf16) return -5;
    if (qlen > RU_TMAX) return -6;
    // The staged scales (STG) hold the l2 norms outside the state pass, which a tree row's
    // register budget needs (the unstaged form spills) whenever the rows fit its LDS.
    if (indices_stride <= RU_TMAX) {
      if (nm0) { if (oq_i8) RU_LAUNCH_T(2, 1); else RU_LAUNCH_T(1, 1); }
      else RU_LAUNCH_T(0, 1);
    } else {
      if (nm0) { if (oq_i8) RU_LAUNCH_T(2, 0); else RU_LAUNCH_T(1, 0); }
      else RU_LAUNCH_T(0, 0);
    }
    return (int)hipGetLastError();
  }
  if (ab_is_bf16 && !nm0 && vs == 1 && indices_stride <= RU_TMAX) {
#define RU_PICKSTG2(ANCC, SDTC)                                                               \
  do { if (nt) RU_LAUNCH(1, 0, 1, 1, ANCC, 1, SDTC);                                          \
       else    RU_LAUNCH(1, 0, 1, 0, ANCC, 1, SDTC); } while (0)
#define RU_PICKSTG(SDTC) do { if (anc) RU_PICKSTG2(1, SDTC); else RU_PICKSTG2(0, SDTC); } while (0)
    if (sdt) RU_PICKSTG(1); else RU_PICKSTG(0);
#undef RU_PICKSTG
#undef RU_PICKSTG2
    return (int)hipGetLastError();
  }
  if (sdt) RU_PICKANC(1); else RU_PICKANC(0);
#undef RU_PICKANC
#undef RU_PICKNT
#undef RU_PICKAB
#undef RU_PICKNM
#undef RU_PICKV
#undef RU_LAUNCH_T
#undef RU_LAUNCH
  return (int)hipGetLastError();
}

extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
    long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
    long oq_pitch, long os_ld, const void* cu,
    const void* ssm_state_indices, long indices_stride, const void* num_accepted,
    const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
    int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
    void* stream)
{
  return ru_update_launch(q, k, v, a, b, ab_stride, ab_is_bf16, A_log, dt_bias, state,
                          state_slot_stride, state_head_stride, o, oq, os, oq_pitch, os_ld, cu,
                          ssm_state_indices, indices_stride, num_accepted, z_gate, z_pitch,
                          norm_weight, norm_eps, norm_act, N, H, Hg, K, V, scale, softplus_thr,
                          state_fp16, stream, 0, ru_anchor());
}

// ============================================================== the folded convolution

// Arrival counters for the folded convolution: one block per (stream, output), zeroed once and
// reset by the last arriver of each group. As with split_ctr in r4d_gemm_fp8a8.hip, launches
// sharing a block must not overlap: different streams get different blocks, and launches on one
// stream sharing an output are ordered by their dependency.
constexpr unsigned kMaxConvCtr = 4096;
static_assert(kRuPairCtr * 2 == kMaxConvCtr, "the pairs' counters are the block's upper half");

static unsigned int* ru_conv_ctr(RadStream s, const void* out) {
  return r4d::keyed_block(r4d::Keyed::GdnConv, s, out,
                          (size_t)kMaxConvCtr * sizeof(unsigned int));
}

// The recurrent update with gdn_conv_update folded in, decode shape only (bf16 a/b, fused row
// norm, one workgroup per (sequence, value head)). State contract, non-temporal choice and
// factor-block bounds are as in ru_update_launch. `w_f32` is the conv weight's type. `oq_i8`:
// E4M3 codes (0), int8 (1), had_quant_act_i8g's rotated int8 (2), or those at IQ8R's one scale
// per 256 columns (3). `chain`: the chained form (r4d_chain.h); the launch is the op's only one.
static int ru_conv_recurrent_launch(const RuConv& cv, const void* a, const void* b, long ab_stride,
                                    const void* A_log, const void* dt_bias, void* state,
                                    long st_slot, long st_head, void* o, void* oq, void* os,
                                    long oq_pitch, long os_ld, const void* cu, const void* sidx,
                                    long sidx_stride, const void* nacc, const void* z,
                                    long z_pitch, const void* nw, float neps, int nact, int N,
                                    int H, int Hg, float scale, float thr, int state_fp16,
                                    RadStream stream, int oq_i8, bool anc, bool w_f32,
                                    bool chain, RuSpec spec = RuSpec{}, bool fuse = false) {
  if (anc && sidx_stride < 6) return RAD_E_SHAPE;
  if (anc && sidx_stride * RU_FSTR > st_head) return RAD_E_SHAPE;
  const int sdt = state_fp16 ? 1 : 0;
  const long slots_written = anc ? 1 : sidx_stride;
  const long flood = (long)N * slots_written * st_slot * (sdt ? 2 : 4);
  const bool nt = flood <= (long)RU_NT_MB * 1024 * 1024;
  const dim3 grid(N, H, 1), blk(RU_V * 2);
#define RC_LAUNCH(NTC, ANCC, SDTC, CVC, STGC)                                                 \
  do { if (oq_i8 == 3) RC_LAUNCH_NM(4, NTC, ANCC, SDTC, CVC, STGC);                            \
       else if (oq_i8 == 2) RC_LAUNCH_NM(3, NTC, ANCC, SDTC, CVC, STGC);                       \
       else if (oq_i8) RC_LAUNCH_NM(2, NTC, ANCC, SDTC, CVC, STGC);                            \
       else            RC_LAUNCH_NM(1, NTC, ANCC, SDTC, CVC, STGC); } while (0)
#define RC_LAUNCH_NM(NM, NTC, ANCC, SDTC, CVC, STGC)                                          \
  do { if (chain) RC_LAUNCH_CH(NM, NTC, ANCC, SDTC, CVC, STGC, r4dchain);                      \
       else       RC_LAUNCH_CH(NM, NTC, ANCC, SDTC, CVC, STGC, R4dUnchained); } while (0)
#define RC_LAUNCH_CH(NM, NTC, ANCC, SDTC, CVC, STGC, CH_)                                     \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<1, NM, 1, NTC, ANCC, STGC, SDTC, CVC, 0, \
                                                      CH_>),                                   \
                     grid, blk, 0, (hipStream_t)stream, nullptr, nullptr, nullptr, a, b,       \
                     ab_stride, (const float*)A_log, (const float*)dt_bias,                    \
                     (ru_state<SDTC>::T*)state, st_slot, st_head, (unsigned short*)o,          \
                     (unsigned char*)oq, (float*)os, oq_pitch, os_ld, (const int*)cu,          \
                     (const int*)sidx, sidx_stride, (const int*)nacc,                          \
                     (const unsigned short*)z, z_pitch, (const float*)nw, neps, nact, H, Hg,   \
                     scale, thr, cv, RuSpec{})
// The tree form: f32 anchor state, no non-temporal hint; chained like the chain form.
#define RC_TREE_NM(NM, CVC)                                                                   \
  do { if (chain) RC_TREE_CH(NM, CVC, r4dchain); else RC_TREE_CH(NM, CVC, R4dUnchained); } while (0)
#define RC_TREE_CH(NM, CVC, CH_)                                                              \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<1, NM, 1, 0, 1, 1, 0, CVC, 1,           \
                                                      CH_>),                                  \
                     grid, blk, 0, (hipStream_t)stream, nullptr, nullptr, nullptr, a, b,       \
                     ab_stride, (const float*)A_log, (const float*)dt_bias,                    \
                     (ru_state<0>::T*)state, st_slot, st_head, (unsigned short*)o,             \
                     (unsigned char*)oq, (float*)os, oq_pitch, os_ld, (const int*)cu,          \
                     (const int*)sidx, sidx_stride, (const int*)nacc,                          \
                     (const unsigned short*)z, z_pitch, (const float*)nw, neps, nact, H, Hg,   \
                     scale, thr, cv, spec)
#define RC_TREE(CVC)                                                                          \
  do { if (oq_i8 == 3) RC_TREE_NM(4, CVC);                                                    \
       else if (oq_i8 == 2) RC_TREE_NM(3, CVC);                                               \
       else if (oq_i8) RC_TREE_NM(2, CVC);                                                    \
       else            RC_TREE_NM(1, CVC); } while (0)
// STGC 1 is the fused multi-row form (RU_FUSE): the anchor form over an f32 state only, picked by
// `fuse`. The entry points always pick it, one row a sequence included: the unfused anchor form
// spills 20 registers and is no faster at one row.
#define RC_PICKNT(ANCC, SDTC, CVC, STGC) \
  do { if (nt) RC_LAUNCH(1, ANCC, SDTC, CVC, STGC); else RC_LAUNCH(0, ANCC, SDTC, CVC, STGC); } while (0)
#define RC_PICKANC(SDTC, CVC) \
  do { if (anc) RC_PICKNT(1, SDTC, CVC, 0); else RC_PICKNT(0, SDTC, CVC, 0); } while (0)
#define RC_PICKSDT(CVC) \
  do { if (sdt) RC_PICKANC(1, CVC); \
       else if (anc) { if (fuse) RC_PICKNT(1, 0, CVC, 1); else RC_PICKNT(1, 0, CVC, 0); } \
       else RC_PICKNT(0, 0, CVC, 0); } while (0)
  if (spec.parent || spec.path) {
    if (!anc || sdt) return RAD_E_UNSUPPORTED;
    if (w_f32) RC_TREE(2); else RC_TREE(1);
  } else if (w_f32) RC_PICKSDT(2); else RC_PICKSDT(1);
#undef RC_TREE
#undef RC_TREE_CH
#undef RC_TREE_NM
#undef RC_PICKSDT
#undef RC_PICKANC
#undef RC_PICKNT
#undef RC_LAUNCH
#undef RC_LAUNCH_NM
#undef RC_LAUNCH_CH
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h. Operand enums use an OPD_ prefix because RU_K and RU_V
// are kernel constants.
/* ------------------------------------------------------------------ recurrent update */

enum { OPD_RU_Q = 0, OPD_RU_K, OPD_RU_V, OPD_RU_A, OPD_RU_B, OPD_RU_ALOG, OPD_RU_DTB, OPD_RU_STATE, OPD_RU_SIDX, OPD_RU_NACC, OPD_RU_CU,
       OPD_RU_Z, OPD_RU_NORMW, OPD_RU_O, OPD_RU_OQ, OPD_RU_OS, OPD_RU_SPAR, OPD_RU_SPATH };

/* The tree verify's two planes (RuSpec), checked here as far as the host can see: dtype and extents,
 * and `pitch`. `rows` is the step's token count, `n_seq` the sequences. A refusal is returned as
 * the status; RAD_OK with both planes absent leaves `spec` empty (the chain). */
static int ru_spec_planes(const RadTensor* spar, const RadTensor* spath, long long rows, int n_seq,
                          RuSpec* spec) {
    *spec = RuSpec{};
    if (spar) {
        R4D_DT(spar, RAD_I32);
        if (spar->rank != 1 || spar->shape[0] < rows) return R4D_NO(RAD_E_SHAPE);
        if (!r4d_contig(spar)) return R4D_NO(RAD_E_STRIDE);
        spec->parent = (const int*)spar->data;
    }
    if (spath) {
        R4D_DT(spath, RAD_I32);
        if (spath->rank != 2 || spath->shape[0] < n_seq || spath->shape[1] < 1)
            return R4D_NO(RAD_E_SHAPE);
        if (spath->stride[0] < spath->shape[1]) return R4D_NO(RAD_E_STRIDE);
        if (spath->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
        spec->path = (const int*)spath->data;
        spec->pitch = (int)spath->stride[0];
    }
    return RAD_OK;
}

/* The op's `state_form`: 1 for anchor, 0 for per_candidate (also the default), -1 for a form this
 * kernel does not implement. The pool behind `state` was sized for the declared form. */
static int ru_state_form(const RadArgs* a) {
    const char* sf = r4d_gets(a, "state_form");
    if (!sf || std::strcmp(sf, "per_candidate") == 0) return 0;
    return std::strcmp(sf, "anchor") == 0 ? 1 : -1;
}

/* The tree form's pair path instantiates bf16 a/b only. */
static inline bool ta_bf16_ok(const RadTensor* ta) { return ta && ta->dtype == RAD_BF16; }

extern "C" int r4d_rad_gdn_recurrent_update(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* q     = r4d_opt(a, OPD_RU_Q);
    const RadTensor* k     = r4d_opt(a, OPD_RU_K);
    const RadTensor* v     = r4d_opt(a, OPD_RU_V);
    const RadTensor* ta    = r4d_opt(a, OPD_RU_A);
    const RadTensor* tb    = r4d_opt(a, OPD_RU_B);
    const RadTensor* alog  = r4d_opt(a, OPD_RU_ALOG);
    const RadTensor* dtb   = r4d_opt(a, OPD_RU_DTB);
    const RadTensor* state = r4d_opt(a, OPD_RU_STATE);
    const RadTensor* sidx  = r4d_opt(a, OPD_RU_SIDX);
    const RadTensor* nacc  = r4d_opt(a, OPD_RU_NACC);
    const RadTensor* cu    = r4d_opt(a, OPD_RU_CU);
    const RadTensor* z     = r4d_opt(a, OPD_RU_Z);
    const RadTensor* nw    = r4d_opt(a, OPD_RU_NORMW);
    const RadTensor* o     = r4d_opt(a, OPD_RU_O);
    const RadTensor* oq    = r4d_opt(a, OPD_RU_OQ);
    const RadTensor* os    = r4d_opt(a, OPD_RU_OS);
    const RadTensor* spar  = r4d_opt(a, OPD_RU_SPAR);
    const RadTensor* spath = r4d_opt(a, OPD_RU_SPATH);
    if (!q || !k || !v || !ta || !tb || !alog || !dtb || !state || !sidx || !cu || !o)
        return R4D_NO(RAD_E_INVAL);
    if (q->rank != 3 || k->rank != 3 || v->rank != 3 || o->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (state->rank < 2 || sidx->rank < 1 || ta->rank != 2 || tb->rank != 2) return R4D_NO(RAD_E_SHAPE);

    /* N comes from cu (the kernel reads cu[n + 1] for every n in the grid); state_idx is checked
     * against it. */
    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;
    if (sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);

    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0 || r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(o))
        return R4D_NO(RAD_E_STRIDE);

    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    const int anc = ru_state_form(a);
    if (anc < 0) return R4D_NO(RAD_E_UNSUPPORTED);

    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const float thr   = (float)r4d_getf(a, "softplus_thr", 20.0);
    /* Folded gated norm parameters. */
    const float neps  = (float)r4d_getf(a, "eps", 0.0);
    const int   nact  = r4d_act_code(a, "act", "norm_act", 0);
    if (nact < 0) return R4D_NO(RAD_E_INVAL);
    /* The l2 norm epsilon is compiled in, as in r4d_rad_gdn_conv_prep. */
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    /* State width from the cache's dtype, as in the chunk scan. */
    const int   sfp16 = state->dtype == RAD_F16;
    const int   ab_is_bf16 = (ta->dtype == RAD_BF16) ? 1 : 0;
    if (tb->dtype != ta->dtype) return R4D_NO(RAD_E_DTYPE);
    /* Optional n_head_v must agree with the value tensor. */
    const long long nhv = r4d_geti_or(a, "n_head_v", 0);
    if (nhv > 0 && nhv != H) return R4D_NO(RAD_E_SHAPE);

    /* bf16 q/k/v/out/z; f32 or f16 state; fp32 A_log, dt_bias and norm weight. */
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16); R4D_DT(o, RAD_BF16);
    R4D_DT(z, RAD_BF16);
    if (state->dtype != RAD_F32 && state->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(nw, RAD_F32);

    /* The folded quantiser needs oq and os together, and the fused norm: without it a head's rows
     * may be split across workgroups (VS > 1), and a group amax cannot span workgroups. */
    if ((oq != nullptr) != (os != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (oq) {
        if (!z || !nw) return R4D_NO(RAD_E_UNSUPPORTED);
        if (V != 128) return R4D_NO(RAD_E_SHAPE);
        if (oq->dtype != RAD_F8E4M3 && oq->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
        R4D_DT(os, RAD_F32);
        if (!r4d_contig(oq)) return R4D_NO(RAD_E_STRIDE);
    }

    /* z may be a column view of the fused in-projection (rad_block_gdn_fp8.h): tokens are z_row
     * elements apart, not H*V. */
    const long long z_row = z ? gn_row_pitch(z, H * V) : 0;
    if (z && z_row < 0) return R4D_NO(RAD_E_STRIDE);

    /* Tree verification: the anchor form with an f32 state and bf16 a/b, at most RU_TMAX rows a
     * sequence (the records' bound; the device cannot refuse a longer one, so q_len, which the
     * block fixes, is the check). */
    RuSpec spec{};
    /* `q_len` is the DECLARED band (up to the step's max_tok, 2048), which prefill never issues
     * here (chunk ops serve it), so it is not the check. What this step issues is q->shape[0]
     * rows over N sequences; a sequence longer than RU_TMAX would overrun the tree's LDS records. */
    const long long qlen = q->shape[0];
    if (spar || spath) {
        rc = ru_spec_planes(spar, spath, q->shape[0], N, &spec);
        if (rc != RAD_OK) return rc;
        if (!anc || state->dtype != RAD_F32 || !ta_bf16_ok(ta)) return R4D_NO(RAD_E_UNSUPPORTED);
        if (qlen > (long long)RU_TMAX * N) {
            std::fprintf(stderr, "[r4d] gdn_recurrent_update tree verify: %lld rows over %d "
                         "sequence(s) exceeds the %d rows a sequence may verify (a draft tree "
                         "wider than %d nodes cannot run on the delta net)\n",
                         qlen, N, (int)RU_TMAX, (int)RU_TMAX);
            return R4D_NO(RAD_E_SHAPE);
        }
    }

    /* Slot and head strides come from the tensor, so a cache with padded slots is served too
     * (see the header); the engine's own state cache is dense ([n_states, H, V, K] per layer). */
    return R4D_RAW(ru_update_launch(
        q->data, k->data, v->data, ta->data, tb->data, (long)ab_stride, ab_is_bf16,
        alog->data, dtb->data, state->data,
        (long)state->stride[0], (long)state->stride[1],
        o->data, oq ? oq->data : nullptr, os ? os->data : nullptr,
        oq ? (long)r4d_row_pitch(oq) : 0, os ? (long)r4d_row_pitch(os) : 0,
        cu->data, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, z ? z->data : nullptr, (long)z_row,
        nw ? nw->data : nullptr,
        neps, nact, N, (int)H, (int)Hg, (int)K, (int)V, scale, thr, sfp16, s,
        oq && oq->dtype == RAD_I8 ? 1 : 0, anc != 0, spec, (int)((qlen + N - 1) / (N > 0 ? N : 1))));
}


/* ------------------------------------------------------------------ conv + recurrent update */

/* gdn_conv_update's inputs, then gdn_recurrent_update's minus q, k and v (never materialised).
 * Inputs first, outputs last (docs/OPS.md). */
enum { OPD_CR_X = 0, OPD_CR_CW, OPD_CR_CB, OPD_CR_CSTATE, OPD_CR_CIDX, OPD_CR_A, OPD_CR_B,
       OPD_CR_ALOG, OPD_CR_DTB, OPD_CR_STATE, OPD_CR_SIDX, OPD_CR_NACC, OPD_CR_CU, OPD_CR_Z,
       OPD_CR_NORMW, OPD_CR_O, OPD_CR_OQ, OPD_CR_OS, OPD_CR_SPAR, OPD_CR_SPATH };

/* `had`: gdn_conv_recurrent_update_had, whose codes are had_quant_act_i8g's over `o` (required). */
static int gdn_conv_recurrent_entry(const RadArgs* a, RadStream s, bool had) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x     = r4d_opt(a, OPD_CR_X);
    const RadTensor* cw    = r4d_opt(a, OPD_CR_CW);
    const RadTensor* cb    = r4d_opt(a, OPD_CR_CB);
    const RadTensor* cst   = r4d_opt(a, OPD_CR_CSTATE);
    const RadTensor* cidx  = r4d_opt(a, OPD_CR_CIDX);
    const RadTensor* ta    = r4d_opt(a, OPD_CR_A);
    const RadTensor* tb    = r4d_opt(a, OPD_CR_B);
    const RadTensor* alog  = r4d_opt(a, OPD_CR_ALOG);
    const RadTensor* dtb   = r4d_opt(a, OPD_CR_DTB);
    const RadTensor* state = r4d_opt(a, OPD_CR_STATE);
    const RadTensor* sidx  = r4d_opt(a, OPD_CR_SIDX);
    const RadTensor* nacc  = r4d_opt(a, OPD_CR_NACC);
    const RadTensor* cu    = r4d_opt(a, OPD_CR_CU);
    const RadTensor* z     = r4d_opt(a, OPD_CR_Z);
    const RadTensor* nw    = r4d_opt(a, OPD_CR_NORMW);
    const RadTensor* o     = r4d_opt(a, OPD_CR_O);
    const RadTensor* oq    = r4d_opt(a, OPD_CR_OQ);
    const RadTensor* os    = r4d_opt(a, OPD_CR_OS);
    const RadTensor* spar  = r4d_opt(a, OPD_CR_SPAR);
    const RadTensor* spath = r4d_opt(a, OPD_CR_SPATH);
    /* The row norm is required: the fold is the decode shape, one workgroup per (sequence, value
     * head). */
    if (!x || !cw || !cst || !cidx || !ta || !tb || !alog || !dtb || !state || !sidx || !cu ||
        !z || !nw || !o)
        return R4D_NO(RAD_E_INVAL);
    if ((oq != nullptr) != (os != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (had && (!oq || oq->dtype != RAD_I8)) return R4D_NO(RAD_E_INVAL);
    if (x->rank != 2 || cst->rank != 3 || o->rank != 3 || ta->rank != 2 || tb->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    if (state->rank < 2 || sidx->rank < 1 || cidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;
    if (sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);

    /* Head geometry from o ([tokens, H, V]) and x ([q | k | v] columns, 2 Hg K + H V wide). */
    const long long T  = o->shape[0];
    const long long H  = o->shape[1];
    const long long V  = o->shape[2];
    const long long K  = r4d_geti_or(a, "head_k", 0);
    if (K != RU_K || V != RU_V || r4d_geti_or(a, "head_v", 0) != RU_V) return R4D_NO(RAD_E_SHAPE);
    if (x->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    const long long conv_dim = x->shape[1];
    if (H <= 0 || conv_dim <= H * V || (conv_dim - H * V) % (2 * K)) return R4D_NO(RAD_E_SHAPE);
    const long long Hg = (conv_dim - H * V) / (2 * K);
    if (H % Hg) return R4D_NO(RAD_E_SHAPE);
    const long long nhv = r4d_geti_or(a, "n_head_v", 0);
    if (nhv > 0 && nhv != H) return R4D_NO(RAD_E_SHAPE);
    if ((long long)N * Hg > (long long)kMaxConvCtr) return R4D_NO(RAD_E_SHAPE);
    /* `group` 256 (the `_had` op alone): one scale per two value heads, whose arrival counters
     * share the block with the key heads'. */
    const long long qgrp = r4d_geti_or(a, "group", RU_V);
    const bool wide = had && qgrp == 2 * RU_V;
    if (qgrp != RU_V && !wide) return R4D_NO(RAD_E_UNSUPPORTED);
    if (wide && (H % 2 || (long long)N * Hg > (long long)kRuPairCtr ||
                 (long long)N * (H / 2) > (long long)kRuPairCtr))
        return R4D_NO(RAD_E_SHAPE);

    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(cw) || !r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);
    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0 || r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);

    R4D_DT(x, RAD_BF16); R4D_DT(cb, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(ta, RAD_BF16); R4D_DT(tb, RAD_BF16); R4D_DT(o, RAD_BF16); R4D_DT(z, RAD_BF16);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(nw, RAD_F32);
    if (cw->dtype != RAD_BF16 && cw->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (state->dtype != RAD_F32 && state->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (oq) {
        if (oq->dtype != RAD_F8E4M3 && oq->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
        R4D_DT(os, RAD_F32);
        if (!r4d_contig(oq)) return R4D_NO(RAD_E_STRIDE);
    }

    /* The window as in gdn_conv_update: max_query_len = state_len_max - (width - 2). What the fold
     * holds in registers is THIS step's tokens of a sequence, at most RU_CVT: the issue's `q_len`
     * (the band the core selected it for, which bounds the rows a sequence has this step) must fit
     * it. The window's own length does not matter to the kernel: the history it reads is
     * conv_width - 1 entries at an offset (ru_hist_entry, up to the previous step's accepted count
     * past the window's start) and the rewrite stores conv_width - 2 + T entries, all inside any
     * window of at least that many. A window longer than RU_CVT is a draft tree's or a long
     * chain's (conv_width - 1 + max_spec entries), and the fold serves its short steps. */
    if (r4d_geti_or(a, "conv_width", 0) != R4D_GDN_CONV_W) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long mql = cst->shape[2] - (R4D_GDN_CONV_W - 2);
    if (mql < 1) return R4D_NO(RAD_E_SHAPE);
    const long long qband = r4d_geti_or(a, "q_len", 0);
    if (qband < 1 || qband > RU_CVT) return R4D_NO(RAD_E_UNSUPPORTED);
    if (T > (long long)RU_CVT * N) return R4D_NO(RAD_E_UNSUPPORTED);

    /* The recurrent half's checks, as in r4d_rad_gdn_recurrent_update. */
    const int anc = ru_state_form(a);
    if (anc < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const float thr   = (float)r4d_getf(a, "softplus_thr", 20.0);
    const float neps  = (float)r4d_getf(a, "eps", 0.0);
    const int   nact  = r4d_act_code(a, "act", "norm_act", 0);
    if (nact < 0) return R4D_NO(RAD_E_INVAL);
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long z_row = gn_row_pitch(z, H * V);
    if (z_row < 0) return R4D_NO(RAD_E_STRIDE);

    /* Tree verification (see RuSpec): the anchor form over an f32 state, at most RU_CVT rows a
     * sequence (the window check above), not chained. */
    RuSpec spec{};
    if (spar || spath) {
        rc = ru_spec_planes(spar, spath, T, N, &spec);
        if (rc != RAD_OK) return rc;
        if (!anc || state->dtype != RAD_F32) return R4D_NO(RAD_E_UNSUPPORTED);
    }

    RuConv cv{};
    cv.x = (const unsigned short*)x->data;  cv.xpitch = (long)xpitch;
    cv.w = cw->data; cv.b = cb ? (const unsigned short*)cb->data : nullptr;
    cv.cs = (unsigned short*)cst->data;
    cv.cs_seq = (long)cst->stride[0]; cv.cs_dim = (long)cst->stride[1];
    cv.cs_tok = (long)cst->stride[2];
    cv.cidx = (const int*)cidx->data; cv.ci_stride = (long)cidx->stride[0];
    cv.ctr = ru_conv_ctr(s, o->data);
    if (!cv.ctr) return R4D_NO(RAD_E_NOMEM);

    return ru_conv_recurrent_launch(
        cv, ta->data, tb->data, (long)ab_stride, alog->data, dtb->data, state->data,
        (long)state->stride[0], (long)state->stride[1], o->data, oq ? oq->data : nullptr,
        os ? os->data : nullptr, oq ? (long)r4d_row_pitch(oq) : 0,
        os ? (long)r4d_row_pitch(os) : 0, cu->data, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, z->data, (long)z_row, nw->data, neps, nact, N, (int)H,
        (int)Hg, scale, thr, state->dtype == RAD_F16 ? 1 : 0, s,
        wide ? 3 : had ? 2 : oq && oq->dtype == RAD_I8 ? 1 : 0, anc != 0,
        cw->dtype == RAD_F32,
        r4d_geti_or(a, "chain", 0) != 0, spec, /*fuse*/ true);
}

extern "C" int r4d_rad_gdn_conv_recurrent_update(const RadArgs* a, RadStream s) {
    return gdn_conv_recurrent_entry(a, s, false);
}

extern "C" int r4d_rad_gdn_conv_recurrent_update_had(const RadArgs* a, RadStream s) {
    return gdn_conv_recurrent_entry(a, s, true);
}

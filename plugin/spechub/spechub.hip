/* plugin/spechub/spechub.hip -- the device launches for the coupling ops.
 *
 * The propose kernel is one thread a row over abi/rad_spechub.h's helpers. The verify is one
 * workgroup a row: the masses are computed in parallel, and the sums and walks run on one thread
 * in ascending candidate order, the order rad_mtp_couple_decide uses, so the token matches the
 * host reference bit for bit.
 *
 * Strict FP64: fast-math or FMA contraction would reorder those sums and change the answer, hence
 * the #error below and -ffp-contract=off in CMakeLists.txt.
 */
#ifdef __FAST_MATH__
#error "the MTP coupling requires strict FP64 arithmetic: build without -ffast-math"
#endif

#include <hip/hip_runtime.h>

#include "spechub.h"
#include "../../libr4d/r4d_emit.h"

namespace {

/* One thread a row: both ops are row-local and the M is the number of sampled positions. */
__global__ void propose(RadMtpCoupleProposeCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_propose_row(c, row);
}

/* One workgroup a row, 128 threads. The weight and residual planes are in LDS: 16 KiB of FP64 at
 * n_cand 1024. */
constexpr int kThreads = 128;
constexpr int64_t kMaxCand = 1024;

/* The row's live candidates (id >= 0, value != -inf), compacted in ascending index order into `lj`, with
 * the validity flag (no id >= vocab, NaN or +inf among the ids >= 0) and the maximum value over the
 * ids >= 0: what the serial scans of the host reference read, in its order. The sampler's plane is
 * kSampleCandidates wide and mostly holes (top_k 20 leaves about 20 live), so the serial part afterwards
 * walks that many entries, not n_cand. Every thread of the workgroup calls it. */
__device__ void couple_live(const int32_t* ids, const float* val, int64_t n, int64_t vocab,
                            short* lj, int* n_live, int* valid, float* top) {
    __shared__ int wcnt[kThreads / 32], wbad[kThreads / 32], sbase;
    __shared__ float wmax[kThreads / 32];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) sbase = 0;
    int bad = 0;
    float mx = -INFINITY;
    for (int64_t c0 = 0; c0 < n; c0 += kThreads) {
        const int64_t j = c0 + tid;
        bool live = false;
        if (j < n) {
            const int32_t id = ids[j];
            const float v = val[j];
            if (id >= 0) {
                if (id >= vocab || v != v || v == INFINITY) bad = 1;
                else if (v > mx) mx = v;
                live = v != -INFINITY;
            }
        }
        const uint64_t ball = __ballot(live);
        const int below = __popcll(ball & ((1ull << lane) - 1ull));
        if (lane == 0) wcnt[wave] = __popcll(ball);
        __syncthreads();
        int off = sbase, tot = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wave) off += wcnt[w];
            tot += wcnt[w];
        }
        if (live) lj[off + below] = (short)j;
        __syncthreads();
        if (tid == 0) sbase += tot;
    }
    for (int d = 16; d > 0; d >>= 1) {
        bad |= __shfl_xor(bad, d);
        mx = fmaxf(mx, __shfl_xor(mx, d));
    }
    if (lane == 0) { wbad[wave] = bad; wmax[wave] = mx; }
    __syncthreads();
    if (tid == 0) {
        int b = 0;
        float m = -INFINITY;
        for (int w = 0; w < kThreads / 32; ++w) { b |= wbad[w]; m = fmaxf(m, wmax[w]); }
        *valid = !b;
        *top = m;
        *n_live = sbase;
    }
    __syncthreads();
}

/* The live candidates of a row as the shared decisions read them (rad_mtp_rrs_decide_over,
 * rad_mtp_trav_*): the sampler's ids, the compaction (`lj`, ascending candidate order), each live
 * candidate's FP64 weight exp(val - top) and its slot in the node's support (count when outside),
 * all computed by the workgroup and stored in LDS. */
struct LiveCands {
    const int32_t* ids; const short* lj; const double* w; const unsigned char* k; int L;
    template <class F> __device__ void each(F&& f) const {
        for (int i = 0; i < L; ++i)
            if (!f(ids[lj[i]], w[i], (int)k[i])) return;
    }
};

/* --draft-verify block (abi/rad_mtp_block.h). The workgroup of a block's round-1 row decides the
 * whole block and writes the tokens of rows 0 .. tau; the block's other rows return. Per row the
 * live candidates are compacted by the workgroup (couple_live) and thread 0 runs the shared helpers
 * over them in ascending slot order, so the result is the host reference's bit for bit. Called by
 * every thread of the workgroup (couple_live synchronises); `lj` is the kMaxCand scratch. */
__device__ void block_verify(const RadMtpCoupleVerifyCall& c, int64_t row, short* lj) {
    __shared__ RadMtpCoupleState sst[RAD_MTP_BLOCK_MAX];
    __shared__ RadBlockStat      sstat[RAD_MTP_BLOCK_MAX];
    __shared__ RadBlockPlan      plan;
    __shared__ int   g_s, ok_s, tau_s, nl_s, valid_s;
    __shared__ float top_s;
    __shared__ int32_t y_s;
    const int tid = threadIdx.x;
    if (tid == 0) {
        const auto& s0 = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
        g_s = s0.blk;
        ok_s = s0.round == 1 && g_s >= 1 && g_s <= RAD_MTP_BLOCK_MAX && row + g_s <= c.M;
        if (ok_s) {
            const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
            for (int i = 0; i < g_s && ok_s; ++i) {
                sst[i] = *reinterpret_cast<const RadMtpCoupleState*>(c.state + (row + i) * c.ld[3]);
                const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + (row + i) * c.ld[2]);
                ok_s = rad_mtp_block_row_ok(sst[0], sst[i], p0, p, i, g_s, c.vocab);
            }
        }
    }
    __syncthreads();
    if (!ok_s) return;
    const int g = g_s;
    for (int i = 0; i < g; ++i) {
        couple_live(c.idx + (row + i) * c.ld[0], c.val + (row + i) * c.ld[1], c.n_cand, c.vocab, lj,
                    &nl_s, &valid_s, &top_s);
        if (tid == 0) {
            const RadBlockRow r{ c.idx + (row + i) * c.ld[0], c.val + (row + i) * c.ld[1], lj,
                                 (int64_t)nl_s, valid_s, top_s };
            if (!rad_mtp_block_stat(r, sst[i], &sstat[i])) ok_s = 0;
        }
        __syncthreads();
    }
    if (!ok_s) return;
    const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    if (tid == 0) {
        rad_mtp_block_plan(sst, sstat, g, &plan);
        tau_s = rad_mtp_block_tau(plan, p0.seed, p0.pos);
        y_s = -1;
    }
    __syncthreads();
    const int tau = tau_s;
    if (tau < g) {
        couple_live(c.idx + (row + tau) * c.ld[0], c.val + (row + tau) * c.ld[1], c.n_cand, c.vocab,
                    lj, &nl_s, &valid_s, &top_s);
        if (tid == 0) {
            const RadBlockRow r{ c.idx + (row + tau) * c.ld[0], c.val + (row + tau) * c.ld[1], lj,
                                 (int64_t)nl_s, valid_s, top_s };
            if (!rad_mtp_block_resid_draw(r, sst[tau], sstat[tau], plan.p[tau],
                                          rad_spechub_uniform(p0.seed, p0.pos + (uint64_t)tau,
                                                              RAD_SAMPLE_KEY_RESID), &y_s))
                ok_s = 0;
        }
        __syncthreads();
        if (!ok_s) return;
    }
    if (tid == 0) rad_mtp_block_write(c, row, sst, tau, g, y_s);
}

__global__ void couple_verify(const int32_t* idx, const float* values, const uint32_t* params,
                              const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                              int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                              int64_t ld4) {
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    /* Above kMaxCand the LDS planes would overflow, so one thread runs the serial helper. */
    if (n_cand > kMaxCand) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row(c, row);
        return;
    }
    /* `weight` is P's unnormalised FP64 mass per live candidate and `mass` its residual (P-Q)+,
     * both indexed by the candidate's place in `lj`. */
    __shared__ double weight[kMaxCand], mass[kMaxCand], pq[8], sum;
    __shared__ short  lj[kMaxCand];
    __shared__ float  top;
    __shared__ int    eligible, nl, valid, first[8], ix, dominated;
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float*   val = c.val + row * c.ld[1];
    /* The same eligibility gate as rad_mtp_couple_verify_row: an unbound row keeps the ordinary
     * token. */
    if (threadIdx.x == 0)
        eligible = s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
                   s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab);
    __syncthreads();
    if (!eligible) return;
    /* A chain row of a block-verified draft (--draft-verify block). */
    if (s.blk >= 1) {
        block_verify(c, row, lj);
        return;
    }
    /* A RheoSampling node (abi/rad_mtp_rrs.h): one thread runs the shared serial helper, which is
     * the host reference itself. Only rows of a --draft-siblings rheo tree take it. */
    if (s.scheme != 0) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row(c, row);
        return;
    }
    couple_live(ids, val, c.n_cand, c.vocab, lj, &nl, &valid, &top);
    if (!valid || top == -INFINITY) return;
    const int L = nl;
    for (int i = threadIdx.x; i < L; i += blockDim.x) weight[i] = ::exp((double)val[lj[i]] - top);
    /* A tree node with several drafted children (abi/rad_mtp_rrs.h, recursive rejection sampling):
     * the workgroup has compacted the live candidates and computed every weight and every
     * candidate's slot in the node's support; one lane then runs the host reference's own decision
     * (rad_mtp_rrs_decide_over) over those stored values, in the same order, so the token is the
     * host's bit for bit. */
    if (s.nkids >= 2) {
        __shared__ unsigned char kslot[kMaxCand];
        __shared__ double        pq_x[8], sum_x;
        for (int i = threadIdx.x; i < L; i += blockDim.x) {
            const int32_t id = ids[lj[i]];
            int k = 0;
            while (k < s.count && id != s.ids[k]) ++k;
            kslot[i] = (unsigned char)k;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t out = -1;
            if (rad_mtp_rrs_decide_over(s, LiveCands{ ids, lj, weight, kslot, L }, p.seed, p.pos, &out,
                                        pq_x, &sum_x)) {
                c.token[row * c.ld[4]] = out;
                /* the export (rad_mtp_couple_export_p): P on the node's support, from the same sums */
                if (s.draw.dpad_ & RAD_MTP_DRAW_EXPORT_BIT) {
                    RadMtpCoupleState* sm = const_cast<RadMtpCoupleState*>(&s);
                    for (int k = 0; k < s.count; ++k) sm->q[k] = pq_x[k] / sum_x;
                    sm->status = RAD_MTP_COUPLE_STATUS_EXPORTED;
                }
            }
        }
        return;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
        for (int i = 0; i < L; ++i) {
            const int32_t id = ids[lj[i]];
            const double w = weight[i];
            sum += w;
            for (int k = 0; k < s.count; ++k)
                if (id == s.ids[k]) {
                    if (first[k] < 0) first[k] = i;
                    pq[k] += w;
                    break;
                }
        }
        if (!(sum > 0) || !rad_spechub_finite(sum)) eligible = 0;
        if (eligible) {
            ix = -1;
            dominated = true;
            for (int k = 0; k < s.count; ++k) {
                pq[k] /= sum;
                dominated = dominated && pq[k] >= s.q[k];
                if (s.ids[k] == s.x) ix = k;
            }
            if (ix < 0) eligible = 0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* Residual per live slot: a Q token contributes max(0, P-Q) once, at its first occurrence;
     * every other candidate contributes its own P. */
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        const int32_t id = ids[lj[i]];
        double m = 0;
        bool in_q = false;
        for (int k = 0; k < s.count; ++k)
            if (s.ids[k] == id) {
                in_q = true;
                if (first[k] == i) m = rad_spechub_positive(pq[k] - s.q[k]);
                break;
            }
        if (!in_q) m = weight[i] / sum;
        mass[i] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    /* One thread, ascending candidate order: parallelising these sums would reorder them and
     * break bit-exactness with the host reference. */
    double resid = 0;
    for (int i = 0; i < L; ++i) resid += mass[i];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int i = 0; i < L; ++i) {
            const double w = weight[i];
            if (w > 0) { out = ids[lj[i]]; cdf -= w; if (cdf < 0) break; }
        }
    } else if (pq[ix] >= s.q[ix] ||
               rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT) * s.q[ix] < pq[ix]) {
        out = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int i = 0; i < L; ++i) {
            const int32_t id = ids[lj[i]];
            if (id == s.x) continue;
            const double m = mass[i];
            if (m > 0) { out = id; cdf -= m; if (cdf < 0) break; }
        }
    }
    c.token[row * c.ld[4]] = out;
    /* The export (rad_mtp_couple_export_p): P on Q's support, from the shared pq, after the decision
     * and after every other thread's last read of the state (they passed the barrier above). */
    if (s.draw.dpad_ & RAD_MTP_DRAW_EXPORT_BIT) {
        RadMtpCoupleState* sm = const_cast<RadMtpCoupleState*>(&s);
        for (int k = 0; k < s.count; ++k) sm->q[k] = pq[k];
        sm->status = RAD_MTP_COUPLE_STATUS_EXPORTED;
    }
}

/* DFlash2's coupled walk: one workgroup a sequence. Per step the workgroup forms the
 * candidate-independent half `sa` of the edge, then thread (c, j) computes part j of candidate c's
 * edge (rad_dflash_edge_part, the host's serial fmaf chain over the same slice), and thread 0 adds
 * the parts in order, takes the step (rad_dflash_couple_step) and publishes the token as the next
 * predecessor. The arithmetic and its order are the host reference's, so tokens and states match
 * rad_ref_dflash_couple bit for bit. */
constexpr int kDfThreads = RAD_DFLASH_COUPLE_MAX_K * RAD_DFLASH_EDGE_PARTS;   /* 256 */

__global__ __launch_bounds__(kDfThreads) void dflash_couple(
    const int32_t* cand, const float* unary, const uint16_t* hp, const int32_t* anchor,
    const uint16_t* pred, const uint16_t* succ, const int32_t* card, const uint32_t* draw,
    int32_t* tokens, uint32_t* state, int64_t steps, int64_t K, int64_t R, int64_t vocab,
    int64_t astr, int64_t ld_cand, int64_t ld_unary, int64_t ld_hp, int64_t ld_draw,
    int64_t ld_tok, int64_t st_round, int64_t st_row, float q_temp) {
    __shared__ float   sa[RAD_DFLASH_COUPLE_MAX_R];
    __shared__ float   part[kDfThreads];
    __shared__ float   sc[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t sid[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t spid;
    /* The step's Q in the making: its live (id, score / q_temp) in selection order, their count
     * (-1: no Q), the top score, the FP64 weights, their sum and the normalised q. */
    __shared__ int32_t lid[8], slot_id[8];
    __shared__ float   lv[8], slot_v[8];
    __shared__ int     sn;
    __shared__ float   stop;
    __shared__ double  w[8], wq[8], ssum;
    __shared__ RadMtpCoupleState sst;
    const int64_t s   = blockIdx.x;
    const int     tid = threadIdx.x;
    const RadMtpCoupleDraw d = *reinterpret_cast<const RadMtpCoupleDraw*>(draw + s * ld_draw);
    int32_t* tok = tokens + s * ld_tok;
    uint32_t* st0 = state + s * st_row;
    if (d.mode != 1) {
        /* A greedy request: dflash_select's walk stands, and each round records it (status 0). */
        if (tid == 0)
            for (int64_t l = 0; l < steps; ++l)
                (void)rad_dflash_couple_step(sc, sid, (int)K, q_temp, d, false, (int)l + 1, vocab,
                                             tok[l],
                                             reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round));
        return;
    }
    const bool card_ok = d.card == card[s];
    if (tid == 0) spid = anchor[s * astr];
    __syncthreads();
    const int c = tid / RAD_DFLASH_EDGE_PARTS, j = tid % RAD_DFLASH_EDGE_PARTS;
    for (int64_t l = 0; l < steps; ++l) {
        const int64_t row = s * steps + l;
        /* What does not depend on the predecessor is loaded first, so its latency overlaps the
         * predecessor's codebook row: the candidate, its successor slice and its unary score. */
        const int32_t id = c < K ? cand[row * ld_cand + c] : -1;
        const bool    id_ok = id >= 0 && id < vocab;
        const bool    wide = R == 32 * RAD_DFLASH_EDGE_PARTS;
        uint4 v[4] = {};
        if (id_ok && wide) {
            /* DFlash2's rank 256: the slice's 32 values in four 16-byte loads issued together. A
             * load inside the fmaf chain would wait out one memory latency per element. */
            const uint4* src = reinterpret_cast<const uint4*>(succ + (int64_t)id * R + j * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = src[i];
        }
        const float un = tid < K ? unary[row * ld_unary + tid] : 0.0f;
        const int32_t pid = spid;
        const bool ok = pid >= 0 && pid < vocab;
        for (int64_t r = tid; r < R; r += kDfThreads)
            sa[r] = ok ? rad_dflash_bf2f(pred[(int64_t)pid * R + r]) *
                             (hp ? rad_dflash_bf2f(hp[row * ld_hp + r]) : 1.0f)
                       : 0.0f;
        __syncthreads();
        if (c < K) {
            float pv = 0.0f;
            if (id_ok) {
                if (wide) {
                    /* rad_dflash_edge_part's fmaf chain over the slice, in its order. */
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(v);
                    const float* a = sa + j * 32;
#pragma unroll
                    for (int r = 0; r < 32; ++r) pv = fmaf(a[r], rad_dflash_bf2f(h[r]), pv);
                } else {
                    pv = rad_dflash_edge_part(sa, succ + (int64_t)id * R, R, j);
                }
            }
            part[tid] = pv;
            if (j == 0) sid[c] = id;
        }
        __syncthreads();
        /* Each candidate's score from its parts, in the host's order. */
        if (tid < K) {
            float acc = 0.0f;
            if (sid[tid] >= 0 && sid[tid] < vocab)
                for (int q = 0; q < RAD_DFLASH_EDGE_PARTS; ++q)
                    acc += part[tid * RAD_DFLASH_EDGE_PARTS + q];
            sc[tid] = un + acc;
        }
        if (tid < 8) slot_id[tid] = -2;
        __syncthreads();
        RadMtpCoupleState* so = reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round);
        /* rad_dflash_couple_step, spread over the workgroup: one lane walking the selection and
         * build_q's FP64 work serially waits out an LDS or FP64 latency per element. Each operation
         * and its order are the helper's, so the bits are too.
         *
         * The eight: candidate k's place is the number of eligible candidates ahead of it (a higher
         * score, or an equal one at a lower index), which is exactly the order the helper's repeated
         * first-maximum scan takes them in. */
        if (tid < K) {
            const int32_t id = sid[tid];
            const float   v  = sc[tid];
            if (id >= 0 && id < vocab && v == v) {
                int rank = 0;
                for (int k = 0; k < K; ++k) {
                    const int32_t ik = sid[k];
                    const float   vk = sc[k];
                    rank += (ik >= 0 && ik < vocab && vk == vk && (vk > v || (vk == v && k < tid)));
                }
                if (rank < 8) { slot_id[rank] = id; slot_v[rank] = v / q_temp; }
            }
        }
        __syncthreads();
        if (tid == 0) {
            sn = -1;
            if (card_ok) {
                int live = 0;
                bool bad = false;
                for (int n = 0; n < 8 && slot_id[n] != -2; ++n) {
                    const float v = slot_v[n];
                    /* build_q: +inf refuses the whole Q, -inf carries no mass, a repeat refuses. */
                    if (v == INFINITY) { bad = true; break; }
                    if (v == -INFINITY) continue;
                    for (int i = 0; i < live; ++i) bad = bad || lid[i] == slot_id[n];
                    if (bad) break;
                    lid[live] = slot_id[n];
                    lv[live++] = v;
                }
                if (!bad && live > 0) {
                    float top = lv[0];
                    for (int i = 1; i < live; ++i)
                        if (lv[i] > top) top = lv[i];
                    stop = top;
                    sn = live;
                }
            }
        }
        __syncthreads();
        if (tid < 8 && tid < sn) w[tid] = ::exp((double)lv[tid] - stop);
        __syncthreads();
        if (tid == 0 && sn > 0) {
            double sum = 0;
            for (int i = 0; i < sn; ++i) sum += w[i];
            ssum = sum;
        }
        __syncthreads();
        if (tid < 8 && tid < sn) wq[tid] = w[tid] / ssum;
        __syncthreads();
        /* The state is built in LDS: a private struct indexed by a runtime count lives in scratch,
         * and every access to it is a memory round trip. */
        if (tid == 0) {
            RadMtpCoupleState& s = sst;
            s.draw = d;
            s.pos = 0;
            s.round = (int)l + 1;
            s.status = -1;
            s.count = 0;
            s.nkids = 0;   /* the walk's position is one draw: the chain's wire */
            s.scheme = 0;
            for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
            for (int i = 0; i < sn; ++i)
                if (wq[i] > 0) { s.ids[s.count] = lid[i]; s.q[s.count++] = wq[i]; }
            if (s.count > 0) {
                const double u = rad_spechub_uniform(d.seed, d.base,
                                                     RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)(l + 1));
                s.x = s.ids[rad_mtp_couple_pick(s, u)];
                s.status = 1;
            } else {
                s.count = 0;
                for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
                int   best = 0;
                float bv = -INFINITY;
                for (int k = 0; k < K; ++k)
                    if (sc[k] > bv) { bv = sc[k]; best = k; }
                s.x = sid[best];
            }
            tok[l] = s.x;
            spid = s.x;
        }
        __syncthreads();
        /* One word a lane out to the state plane. */
        if (tid < (int)RAD_MTP_COUPLE_STATE_WORDS)
            reinterpret_cast<uint32_t*>(so)[tid] = reinterpret_cast<const uint32_t*>(&sst)[tid];
        /* The next step reads spid and overwrites sa, part and the Q scratch (the copy above reads
         * sst, which the next step writes only after its own barriers). */
    }
}

}  /* namespace */

int rad_mine_dflash_couple_launch(const RadArgs* a, RadStream stream) {
    RadDflashCoupleCall c{};
    const int rc = rad_dflash_couple_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, dflash_couple, dim3((unsigned)c.M), dim3(kDfThreads), 0,
                    c.cand, c.unary, c.hp, c.anchor, c.pred, c.succ, c.card, c.draw, c.tokens,
                    c.state, c.steps, c.K, c.R, c.vocab, c.anchor_stride, c.ld_cand, c.ld_unary,
                    c.ld_hp, c.ld_draw, c.ld_tok, c.st_round, c.st_row, c.q_temp);
}

int rad_mine_mtp_couple_propose_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, propose, dim3((unsigned)((c.M + 63) / 64)), dim3(64), 0, c);
}

/* The traversal walk (abi/rad_mtp_traversal.h): one workgroup a row, only a request's root row (topo
 * parent -1) does work. The walk itself is the host reference's, run by every lane of the workgroup
 * redundantly (it is serial, short -- a tree of at most 60 rows -- and its control flow reads only
 * data every lane reads alike); what costs is the candidate scans, one per visited node over the
 * sampler's whole plane (1024 slots, FP64 exp each, at top_k 0), and those are the workgroup's: live()
 * compacts the node's live candidates and computes every weight and support slot in parallel, after
 * which the ordered sums run over the stored values on each lane, in the host's order. Tokens and
 * accepted paths are the host's bit for bit. Above kMaxCand candidates one lane walks as before. */
struct DevTravRows : RadMtpTravRows {
    short* lj; double* w; unsigned char* kk; int* nl; int* valid; float* top;
    __device__ void live(int v, const int32_t* sid, int cnt, RadTravLaw* out) const {
        const int32_t* ids; const float* val; int64_t nc;
        cand(v, &ids, &val, &nc);
        __syncthreads();   /* every lane is done with the previous gather */
        couple_live(ids, val, nc, vocab(), lj, nl, valid, top);
        const int L = *nl;
        out->valid = *valid && *top != -INFINITY;
        out->top = *top; out->L = L; out->ids = ids; out->lj = lj; out->w = w; out->k = kk;
        if (out->valid) {
            const float t = *top;
            for (int i = threadIdx.x; i < L; i += blockDim.x) {
                const int32_t id = ids[lj[i]];
                w[i] = ::exp((double)val[lj[i]] - t);
                int k = 0;
                while (k < cnt && id != sid[k]) ++k;
                kk[i] = (unsigned char)k;
            }
        }
        __syncthreads();
    }
};

__global__ void trav_verify(const int32_t* idx, const float* values, const uint32_t* params,
                            const uint32_t* state, int32_t* token, const int32_t* topo, int64_t M,
                            int64_t n_cand, int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2,
                            int64_t ld3, int64_t ld4, int64_t ld_topo) {
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    RadMtpTravCall c{ { idx, values, params, state, token, M, n_cand, vocab, { ld0, ld1, ld2, ld3, ld4 } },
                      topo, ld_topo };
    if (c.topo[row * c.ld_topo] >= 0) return;   /* not a request's root */
    if (n_cand > kMaxCand) {
        if (threadIdx.x == 0) rad_mtp_trav_verify_root(c, row);
        return;
    }
    __shared__ short         lj[kMaxCand];
    __shared__ double        wt[kMaxCand];
    __shared__ unsigned char kk[kMaxCand];
    __shared__ int           nl, valid;
    __shared__ float         top;
    int64_t n = 1;
    while (row + n < c.v.M && c.topo[(row + n) * c.ld_topo] >= 0) ++n;
    const DevTravRows T{ RadMtpTravRows{ c, row, n }, lj, wt, kk, &nl, &valid, &top };
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.v.params + row * c.v.ld[2]);
    RadMtpTravOut o;
    if (!rad_mtp_trav_walk(T, 0, p.seed, p.pos, &o)) return;
    if (threadIdx.x != 0) return;
    for (int i = 0; i + 1 < o.n; ++i)
        c.v.token[(row + o.node[i]) * c.v.ld[4]] = c.topo[(row + o.node[i + 1]) * c.ld_topo + 1];
    if (o.has_y) c.v.token[(row + o.node[o.n - 1]) * c.v.ld[4]] = o.y;
}

int rad_mine_mtp_trav_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpTravCall c{};
    const int rc = rad_mtp_trav_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, trav_verify, dim3((unsigned)c.v.M), dim3(kThreads), 0,
                    c.v.idx, c.v.val, c.v.params, c.v.state, c.v.token, c.topo, c.v.M, c.v.n_cand,
                    c.v.vocab, c.v.ld[0], c.v.ld[1], c.v.ld[2], c.v.ld[3], c.v.ld[4], c.ld_topo);
}

int rad_mine_mtp_couple_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    /* Separate pointer arguments let graph replay rebind each operand address independently: a
     * struct on the host would freeze the addresses it was built with. */
    return R4D_EMIT(stream, couple_verify, dim3((unsigned)c.M), dim3(kThreads), 0,
                    c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                    c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
}

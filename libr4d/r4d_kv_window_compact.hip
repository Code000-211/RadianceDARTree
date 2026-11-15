// r4d_kv_window_compact.hip -- closes a rolling window over the rows a tree verify committed.
//
// The window is a RAD_KV_CONV-layout cache [slots, channels, entries]; a verify block's row r wrote
// its entry at `off + r`, and the committed rows are spec_path[n][0..num_accepted), so entry
// off + t takes the entry of row path[t] for t = 1..num_accepted-1, ascending, for every channel of
// the sequence's slot (state_idx[n][0]). Ascending in place is hazard free because a path's rows
// increase (path[t] >= t): the source of step t is never one an earlier step wrote. A chain
// (path[t] = t) moves nothing.
//
// One thread a (sequence, channel), walking t serially: channels are independent and a window is a
// few entries, so there is nothing to split and the launch is one wave per few sequences. The
// elements are moved whole, so the element width (1, 2, 4 or 8 bytes) is the only dtype fact used.
// The device cannot refuse what it reads: a step whose path does not ascend or names an entry past
// the window is skipped there (as kda_update skips a sequence whose slots are out of bounds); the
// host checks the planes' dtype and extents.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_gdn_args.h"   // R4D_DT

#include <cstdint>

namespace {

enum { kWcThreads = 128 };

template <typename E>
__global__ __launch_bounds__(kWcThreads)
void kv_window_compact_kernel(E* __restrict__ cache, long long cs_seq, long long cs_ch,
                              long long cs_ent, int slots, int channels, int entries, int off,
                              const int* __restrict__ sidx, long long sidx_stride,
                              const int* __restrict__ nacc, const int* __restrict__ path,
                              long long pitch)
{
    const int n = blockIdx.x;
    const int c = blockIdx.y * kWcThreads + threadIdx.x;
    if (c >= channels) return;
    const int slot = sidx[(long long)n * sidx_stride];
    if (slot < 0 || slot >= slots) return;
    int na = nacc[n];
    if (na > pitch) na = (int)pitch;
    E* const base = cache + (long long)slot * cs_seq + (long long)c * cs_ch;
    for (int t = 1; t < na; ++t) {
        const int src = path[(long long)n * pitch + t];
        if (src < t || off + src >= entries) continue;
        if (src == t) continue;
        base[(long long)(off + t) * cs_ent] = base[(long long)(off + src) * cs_ent];
    }
}

template <typename E>
int launch(const RadTensor* cache, const RadTensor* sidx, const RadTensor* nacc,
           const RadTensor* path, int off, RadStream s)
{
    const int n_seq = (int)nacc->shape[0];
    const int channels = (int)cache->shape[1];
    dim3 grid(n_seq, (channels + kWcThreads - 1) / kWcThreads), blk(kWcThreads);
    hipLaunchKernelGGL((kv_window_compact_kernel<E>), grid, blk, 0, (hipStream_t)s,
                       (E*)cache->data, (long long)cache->stride[0], (long long)cache->stride[1],
                       (long long)cache->stride[2], (int)cache->shape[0], channels,
                       (int)cache->shape[2], off, (const int*)sidx->data,
                       (long long)sidx->stride[0], (const int*)nacc->data,
                       (const int*)path->data, (long long)path->stride[0]);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

enum { KC_CACHE = 0, KC_SIDX, KC_NACC, KC_PATH };

}  // namespace

extern "C" int r4d_rad_kv_window_compact(const RadArgs* a, RadStream s) {
    const RadTensor* cache = r4d_opt(a, KC_CACHE);
    const RadTensor* sidx  = r4d_opt(a, KC_SIDX);
    const RadTensor* nacc  = r4d_opt(a, KC_NACC);
    const RadTensor* path  = r4d_opt(a, KC_PATH);
    if (!cache || !sidx || !nacc || !path) return R4D_NO(RAD_E_INVAL);
    if (cache->rank != 3 || nacc->rank != 1 || sidx->rank < 1 || path->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    const long long off = r4d_geti_or(a, "off", -1);
    if (off < 0 || off > 1000000) return R4D_NO(RAD_E_INVAL);
    if (nacc->shape[0] < 1 || sidx->shape[0] < nacc->shape[0] || path->shape[0] < nacc->shape[0] ||
        path->shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (cache->shape[0] < 1 || cache->shape[1] < 1 || cache->shape[2] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (cache->shape[0] > 0x7fffffffLL || cache->shape[1] > 0x7fffffffLL ||
        cache->shape[2] > 0x7fffffffLL)
        return R4D_NO(RAD_E_SHAPE);
    R4D_DT(sidx, RAD_I32); R4D_DT(nacc, RAD_I32); R4D_DT(path, RAD_I32);
    if (!r4d_contig(nacc)) return R4D_NO(RAD_E_STRIDE);
    if (path->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    /* A cache dtype is its element width here: the entries are moved, not interpreted. */
    const long long eb = rad_dtype_bytes(cache->dtype, 1);
    switch (eb) {
        case 1: return launch<uint8_t>(cache, sidx, nacc, path, (int)off, s);
        case 2: return launch<uint16_t>(cache, sidx, nacc, path, (int)off, s);
        case 4: return launch<uint32_t>(cache, sidx, nacc, path, (int)off, s);
        case 8: return launch<uint64_t>(cache, sidx, nacc, path, (int)off, s);
        default: return R4D_NO(RAD_E_DTYPE);
    }
}

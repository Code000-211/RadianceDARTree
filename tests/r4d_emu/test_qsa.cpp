/* tests/r4d_emu: the QSA block selection (r4d_qsa_select_bf16.hip) on the host SIMT emulator, each
 * of its four kernels against a port of libref's ref_qsa_select, byte for byte: the page-id table
 * (sentinel-filled before every launch, the row pitch's padding and a row past M included), nsel
 * and seqused.
 *
 * Every fixture runs three ways: pinned to the single-workgroup kernel and pinned to the wide one,
 * both without scratch, and the launcher's own choice with the split path's scratch given (the
 * local + merge pair for a decode-shaped launch over several chunks, the wide kernel above 64 rows,
 * else the single-workgroup one). The kernel that ran is checked by name.
 *
 * The fixtures follow r4d_selftest's qsa_select cases at sizes the emulator runs in seconds, plus a
 * seeded sweep of small odd shapes (NaN, +-inf, -0.0, denormal and -1e30 scores, pitches, a table
 * narrower than the scores, rows no sequence holds, topk 1, ratio 1). Which branch a row takes
 * inside the kernels is predicted from their constants, restated in `Path`; the run fails when the
 * fixtures stop reaching one of them. `sweep N` runs N seeded fixtures at serving sizes instead.
 *
 * Fixtures marked signed run with the op's `signed` parameter: the kernels' signed key order,
 * against the port's signed ranking, over negative, mixed and the sweep's odd scores. The unsigned
 * fixtures are the same as without the parameter. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_qsa_select_pin_form(int form);
extern int emu_grids;
extern const char* emu_last_kernel;

namespace {

constexpr int32_t kSentinel = 0x55555555;

struct Fix {
    std::string name;
    int S = 1, M = 1, nb_s = 1, nb_t = 1;   /* sequences, query rows, score and table widths */
    int s_ld = 0, t_ld = 0, o_ld = 0;       /* row pitches; 0 = tight */
    int topk = 1, ratio = 1;
    bool sgn = false;                       /* the op's `signed` parameter */
    std::vector<float> sc;                  /* [M][s_ld] */
    std::vector<int32_t> bt;                /* [S][t_ld] */
    std::vector<int32_t> nc, pos, cu;       /* [S], [M], [S + 1] */
    void shape(int s, int m, int nbs, int nbt, int tk, int r, int spad = 0, int tpad = 0, int opad = 0) {
        S = s; M = m; nb_s = nbs; nb_t = nbt; topk = tk; ratio = r;
        s_ld = nbs + spad; t_ld = nbt + tpad; o_ld = tk + 1 + opad;
        sc.assign((size_t)M * s_ld, -INFINITY);
        bt.assign((size_t)S * t_ld, -1);
        nc.assign((size_t)S, 0); pos.assign((size_t)M, 0); cu.assign((size_t)S + 1, 0);
    }
    float& score(int m, int b) { return sc[(size_t)m * s_ld + b]; }
    int32_t& page(int s, int b) { return bt[(size_t)s * t_ld + b]; }
    int nb() const { return std::min(nb_s, nb_t); }
    int seq_of(int m) const {
        for (int s = 0; s < S; ++s) if (m >= cu[s] && m < cu[s + 1]) return s;
        return 0;
    }
    int n_of(int m) const {
        const int s = seq_of(m);
        return (int)std::min<int64_t>(std::min<int64_t>(nc[s], nb()), ((int64_t)pos[m] + 1) / ratio);
    }
};

/* ---- libref's ref_qsa_select (libref/ref_qsa.cpp), restated over the fixture's arrays: the same
 * steps in the same order, both rankings. */
void ref_select(const Fix& f, std::vector<int32_t>& se, std::vector<int32_t>& ns, std::vector<int32_t>& su) {
    const int64_t topk = f.topk, ratio = f.ratio;
    se.assign((size_t)f.M * (topk + 1), 0); ns.assign((size_t)f.M, 0); su.assign((size_t)f.M, 0);
    for (int64_t m = 0; m < f.M; ++m) {
        const int64_t s = f.seq_of((int)m), p = f.pos[m];
        const int64_t n = std::min(std::min<int64_t>(f.nc[s], f.nb()), (p + 1) / ratio);
        std::vector<float> val((size_t)n);
        std::vector<int64_t> live;
        for (int64_t b = 0; b < n; ++b) {
            const float v = f.sc[(size_t)(m * f.s_ld + b)];
            if (f.bt[(size_t)(s * f.t_ld + b)] < 0) continue;
            if (f.sgn ? !std::isfinite(v) : !(v > -1e30f)) continue;
            val[(size_t)b] = f.sgn ? v : v > 0.0f ? v : 0.0f;
            live.push_back(b);
        }
        std::vector<int64_t> take;
        if ((int64_t)live.size() <= topk) {
            take = live;
        } else {
            std::vector<float> ord;
            for (int64_t b : live) ord.push_back(val[(size_t)b]);
            std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(), std::greater<float>());
            const float thr = ord[(size_t)(topk - 1)];
            for (int64_t b : live)
                if (val[(size_t)b] > thr) take.push_back(b);
            for (int64_t b : live)
                if ((int64_t)take.size() < topk && val[(size_t)b] == thr) take.push_back(b);
        }
        int32_t* row = &se[(size_t)(m * (topk + 1))];
        int64_t out = 0;
        for (int64_t b : take) row[out++] = f.bt[(size_t)(s * f.t_ld + b)];
        if (n < f.nb() && (p + 1) % ratio != 0) {
            const int32_t part = f.bt[(size_t)(s * f.t_ld + n)];
            if (part >= 0) row[out++] = part;
        }
        for (int64_t i = out; i <= topk; ++i) row[i] = -1;
        ns[(size_t)m] = (int32_t)out;
        su[(size_t)m] = (int32_t)(out == 0 ? 0 : (out - 1) * ratio + p % ratio + 1);
    }
}

/* ---- Which branch a row takes inside the kernels: the launcher's and kernels' constants
 * (r4d_qsa_select_bf16.hip) on a 64 KiB-LDS device, and their branch conditions. Coverage only: the
 * tables are checked against the port above whatever the prediction says. */
namespace Path {
constexpr int kCacheCap = 8192, kCacheMaxM = 64;                    /* the key cache */
constexpr int kSamp = 2048, kSampRun = 32, kSampRuns = 64, kSampMinN = 3 * kSamp, kSampCap = 1920;
constexpr int kWideShift = 19, kWideCand = 3840, kMergeAbove = 512, kMergeCand = 1664;
constexpr int kWideShiftSigned = 20;                                /* the signed key's 32 bits */
int wide_shift(bool sgn) { return sgn ? kWideShiftSigned : kWideShift; }
constexpr int kParM = 32, kParWMax = 16, kParCap = 8192;            /* the split path */
int par_chunks(int nb) { return std::max(1, std::min(kParWMax, (nb + 4095) / 4096)); }
}

uint32_t keyof(float v, int32_t page) {
    if (page < 0 || !(v > -1e30f)) return 0u;
    const float c = v > 0.0f ? v : 0.0f;
    uint32_t u;
    std::memcpy(&u, &c, 4);
    return u + 1u;
}

/* The signed order: the float's total order over 32 bits, -0 keyed as +0, nonfinite dead. */
uint32_t keyof_signed(float v, int32_t page) {
    if (page < 0 || !std::isfinite(v)) return 0u;
    uint32_t u = 0;
    if (v != 0.0f) std::memcpy(&u, &v, 4);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

enum Form { kSingle, kWide, kAuto };
const char* const kFormName[] = { "single", "wide", "auto" };

bool split_applies(const Fix& f) {
    const int nch = Path::par_chunks(f.nb()), chunk = (f.nb() + nch - 1) / nch;
    return f.M <= Path::kParM && nch > 1 && chunk <= Path::kParCap && nch * f.topk <= Path::kParCap;
}

std::map<std::string, int> g_paths;

/* The branches row m reaches under `form`. */
void note_paths(const Fix& f, int m, Form form, bool split) {
    const int s = f.seq_of(m), n = f.n_of(m), topk = f.topk;
    std::vector<uint32_t> k((size_t)std::max(n, 0));
    for (int b = 0; b < n; ++b) {
        const float v = f.sc[(size_t)m * f.s_ld + b];
        const int32_t pg = f.bt[(size_t)s * f.t_ld + b];
        k[(size_t)b] = f.sgn ? keyof_signed(v, pg) : keyof(v, pg);
    }
    std::vector<uint32_t> live;
    for (uint32_t x : k) if (x) live.push_back(x);
    std::sort(live.begin(), live.end(), std::greater<uint32_t>());
    const int L = (int)live.size();
    auto hit = [](const char* p) { ++g_paths[p]; };
    if (n <= topk) hit("rows of at most topk blocks");
    else if (L == topk) hit("exactly topk live over more blocks");
    else if (L < topk) hit("fewer than topk live over more blocks");
    else {
        hit("a threshold");
        const uint32_t T = live[(size_t)topk - 1];
        int above = 0, eq = 0;
        for (uint32_t x : live) { above += x > T; eq += x == T; }
        if (eq > topk - above) hit("a threshold whose ties are cut short");
        if (f.sgn && T < 0x80000000u) hit("signed: a negative threshold");
    }
    if (split) {
        const int nch = Path::par_chunks(f.nb()), chunk = (f.nb() + nch - 1) / nch;
        int tot = 0;
        bool over = false;
        for (int c = 0; c < nch && c * chunk < n; ++c) {
            const int lo = c * chunk, hi = std::min(lo + chunk, n);
            int lc = 0;
            for (int b = lo; b < hi; ++b) lc += k[(size_t)b] != 0;
            over |= hi - lo > topk && lc > topk;
            tot += std::min(lc, topk);
        }
        if (n <= topk || (!over && tot <= topk)) hit("split: topk or fewer live, every one taken");
        else if (!over) hit("split: no chunk over topk, more than topk live");
        else if (tot == topk) hit("split: exactly topk candidates, from a chunk over topk");
        else hit("split: a chunk over topk");
        return;
    }
    if (form == kSingle || (form == kAuto && f.M <= Path::kCacheMaxM)) {
        const int cap = f.M <= Path::kCacheMaxM ? std::min(Path::kCacheCap, f.nb()) : 0;
        hit(n <= cap ? "single: keys cached in LDS" : "single: keys read from global memory");
        return;
    }
    if (n <= topk) return;
    if (n >= Path::kSampMinN && topk <= Path::kMergeAbove) {
        std::vector<uint32_t> smp;
        for (int i = 0; i < Path::kSamp; ++i) {
            const long b = (long)(i / Path::kSampRun) * (n - Path::kSampRun) / (Path::kSampRuns - 1) + i % Path::kSampRun;
            if (k[(size_t)b]) smp.push_back(k[(size_t)b]);
        }
        std::sort(smp.begin(), smp.end(), std::greater<uint32_t>());
        const uint32_t ms = std::min<uint32_t>(
            (uint32_t)(((uint64_t)topk * 2u * Path::kSamp + (uint64_t)n - 1u) / (uint64_t)n) + 8u, Path::kSamp);
        const uint32_t lo = smp.size() > ms ? smp[ms - 1] : 0u;
        int cnt = 0;
        for (uint32_t x : live) cnt += x >= lo;
        const bool vec = (((uintptr_t)&f.sc[(size_t)m * f.s_ld] | (uintptr_t)&f.bt[(size_t)s * f.t_ld]) & 15u) == 0;
        hit(vec ? "wide: the sampled pass, rows 16-byte aligned" : "wide: the sampled pass, rows not 16-byte aligned");
        if (cnt > topk && cnt <= Path::kSampCap) {
            hit(lo ? "wide: the sampled list" : "wide: the sampled list, holding every live key");
            return;
        }
        hit("wide: a sample that fell back");
    }
    if (L <= topk) return;
    const int shift = Path::wide_shift(f.sgn);
    const uint32_t B = live[(size_t)topk - 1] >> shift;
    int in = 0;
    for (uint32_t x : live) in += (x >> shift) == B;
    hit(in <= Path::kMergeCand && topk <= Path::kMergeAbove ? "wide: merged lists"
        : in <= Path::kWideCand ? "wide: the threshold's bin in LDS" : "wide: the threshold's bin over global memory");
}

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape, int64_t ld = 0) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    if (ld && t.rank == 2) t.stride[0] = ld;
    return t;
}

/* One launch of `form` against the port. Returns the failures (0 or 1). */
int run(Fix& f, Form form) {
    std::vector<int32_t> want, want_n, want_su;
    ref_select(f, want, want_n, want_su);
    const int M = f.M, W = f.topk + 1;
    std::vector<int32_t> se((size_t)(M + 1) * f.o_ld, kSentinel), ns((size_t)M + 1, kSentinel), su((size_t)M + 1, kSentinel);
    RadTensor t[8] = {
        tens(f.sc.data(), RAD_F32, {M, f.nb_s}, f.s_ld), tens(f.bt.data(), RAD_I32, {f.S, f.nb_t}, f.t_ld),
        tens(f.nc.data(), RAD_I32, {f.S}),             tens(f.pos.data(), RAD_I32, {M}),
        tens(f.cu.data(), RAD_I32, {f.S + 1}),         tens(se.data(), RAD_I32, {M, W}, f.o_ld),
        tens(ns.data(), RAD_I32, {M}),                 tens(su.data(), RAD_I32, {M}),
    };
    RadParam p[6]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb()}; p[2] = {"topk", RAD_P_INT, f.topk};
    p[3] = {"ratio", RAD_P_INT, f.ratio}; p[4] = {"seqs", RAD_P_INT, f.S}; p[5] = {"signed", RAD_P_INT, 1};
    RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = f.sgn ? 6 : 5;
    const int64_t need = r4d_rad_qsa_select_scratch(&a);
    std::vector<uint32_t> scratch((size_t)(need / 4 + 1), 0xA5A5A5A5u);
    if (form == kAuto) { a.scratch = scratch.data(); a.scratch_bytes = need; }
    r4d_qsa_select_pin_form(form == kSingle ? 1 : form == kWide ? 2 : 0);
    const int g0 = emu_grids;
    const int rc = r4d_rad_qsa_select(&a, nullptr);
    r4d_qsa_select_pin_form(0);
    const int grids = emu_grids - g0;
    const bool split = form == kAuto && split_applies(f);
    const char* expect = split ? "qsa_merge_kernel"
                       : form == kWide || (form == kAuto && f.M > Path::kCacheMaxM) ? "qsa_select_wide_kernel"
                       : "qsa_select_kernel";
    const char* fn = kFormName[form];
    if (rc != RAD_OK) { std::printf("FAIL %-48s %-6s launch returned %d\n", f.name.c_str(), fn, rc); return 1; }
    /* The launch site names the template in parentheses, `(kernel<Signed, form>)`: the name is
     * what stands between the parenthesis and the arguments. */
    const char* const kname = emu_last_kernel + (emu_last_kernel[0] == '(' ? 1 : 0);
    const std::string ran(kname, std::strcspn(kname, "<"));
    if (ran != expect || grids != (split ? 2 : 1)) {
        std::printf("FAIL %-48s %-6s ran %s in %d grid(s), expected %s in %d\n", f.name.c_str(), fn,
                    emu_last_kernel, grids, expect, split ? 2 : 1);
        return 1;
    }
    int bad_rows = 0;
    for (int m = 0; m <= M; ++m) {
        if (m < M) note_paths(f, m, form, split);
        int diff = 0, first = -1;
        for (int i = 0; i < f.o_ld; ++i) {
            const int32_t w = m < M && i < W ? want[(size_t)m * W + i] : kSentinel;
            if (se[(size_t)m * f.o_ld + i] != w) { ++diff; if (first < 0) first = i; }
        }
        const int32_t wn = m < M ? want_n[(size_t)m] : kSentinel, wsu = m < M ? want_su[(size_t)m] : kSentinel;
        if (!diff && ns[(size_t)m] == wn && su[(size_t)m] == wsu) continue;
        if (++bad_rows > 4) continue;
        std::printf("FAIL %-48s %-6s row %d (n %d): nsel %d want %d, seqused %d want %d", f.name.c_str(), fn, m,
                    m < M ? f.n_of(m) : -1, ns[(size_t)m], wn, su[(size_t)m], wsu);
        if (diff) {
            const int32_t w = m < M && first < W ? want[(size_t)m * W + first] : kSentinel;
            std::printf(", %d slot(s) differ, first slot %d: %d want %d", diff, first, se[(size_t)m * f.o_ld + first], w);
        }
        std::printf("\n");
    }
    return bad_rows ? 1 : 0;
}

uint64_t g_x = 0;
uint64_t rnd() { g_x ^= g_x << 13; g_x ^= g_x >> 7; g_x ^= g_x << 17; return g_x; }

/* r4d_selftest's qsa_select_case: decode rows, ties on a coarse grid, a dead page with a winning
 * score, a tail on alternate rows; rows 0 and 1 have at most topk live. */
Fix fix_decode() {
    Fix f; f.name = "decode: ties, a tail and a dead page";
    const int M = 6, nb = 900, topk = 64, pages = 4096, ratio = 4;
    f.shape(M, M, nb, nb, topk, ratio);
    std::fill(f.sc.begin(), f.sc.end(), 0.0f);
    for (int m = 0; m < M; ++m) {
        const int live = m == 0 ? 9 : m == 1 ? topk : 100 + m * 137;
        f.nc[m] = live; f.cu[m + 1] = m + 1;
        for (int b = 0; b < nb; ++b) {
            if (b < live) {
                f.score(m, b) = (float)((b * 17 + m * 7) % 40) * 0.25f;
                f.page(m, b) = (b * 31 + m * 11) % pages;
                if (b == live / 3) { f.score(m, b) = 1e6f; f.page(m, b) = -1; }
            } else if (b == live) {
                f.page(m, b) = m % 2 == 0 ? (m * 97 + 5) % pages : -1;
            }
        }
        f.pos[m] = f.page(m, live) >= 0 ? live * ratio + m % 3 : live * ratio - 1;
    }
    return f;
}

/* r4d_selftest's qsa_select_prefill_case: two chunks over two histories, each query bounded by
 * its own position, the partial page present exactly when (pos + 1) % ratio != 0. */
Fix fix_prefill() {
    Fix f; f.name = "prefill: per-query bound and the partial page";
    const int S = 2, nb = 96, topk = 5, pages = 512, ratio = 4, q0 = 9, q1 = 14, M = q0 + q1;
    const int base[2] = { 137, 4 };
    f.shape(S, M, nb, nb, topk, ratio);
    f.cu = { 0, q0, M };
    for (int i = 0; i < q0; ++i) f.pos[i] = base[0] + i;
    for (int i = 0; i < q1; ++i) f.pos[q0 + i] = base[1] + i;
    f.nc = { (base[0] + q0) / ratio, (base[1] + q1) / ratio };
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < nb; ++b)
            f.page(s, b) = b <= (base[s] + (s == 0 ? q0 : q1) - 1) / ratio ? (b * 29 + s * 7) % pages : -1;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 23 + m * 5) % 37) * 0.5f;
    return f;
}

/* r4d_selftest's qsa_select_wide_case at emulator size: the first sequence's rows see about 3000,
 * 5000, 7000 and 8800 blocks (the two-pass form below the sample's 6144, the sampled form above,
 * the key cache's 8192 crossed), the second's fewer than topk; dead pages and -inf scattered; five
 * score distributions. Over 4096 blocks at six rows, the automatic choice is the split path. */
Fix fix_wide(int dist) {
    static const char* const kDist[] = { "continuous", "coarse, heavy ties", "an eighth in one first-digit bin",
                                         "all in one first-digit bin", "fine ties at the threshold",
                                         "signed, heavy ties", "signed, all negative, heavy ties",
                                         "signed, continuous across zero" };
    Fix f; f.name = std::string("wide: ") + kDist[dist];
    f.sgn = dist >= 5;
    const int S = 2, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
    f.shape(S, 6, nb, nb, topk, ratio);
    f.cu = { 0, 4, 6 };
    f.pos = { 4 * 3000 + 1, 4 * 5000 + 2, 4 * 7000 + 3, 4 * 8800 + 1, 4 * 300, 4 * 337 + 2 };
    f.nc = { 8900, 340 };
    g_x = 0x243F6A8885A308D3ull + (uint64_t)dist;
    for (auto& v : f.bt) v = rnd() % 97 == 0 ? -1 : (int32_t)(rnd() % pages);
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        if (u % 211 == 0) { v = -INFINITY; continue; }
        switch (dist) {
            case 0: v = (float)(u % 1000003) / 1000.0f; break;
            case 1: v = (float)(u % 23) * 0.25f; break;
            case 2: v = u % 8 == 0 ? 1.0f + (float)(u % 60000) * 1e-6f : (float)(u % 1000) / 2000.0f; break;
            case 3: v = 1.0f + (float)(u % 60000) * 1e-6f; break;
            case 5: v = (float)(u % 23) * 0.25f - 3.0f; break;
            case 6: v = -(float)(u % 23) * 0.25f - 1.0f; break;
            case 7: v = (float)(u % 1000003) / 1000.0f - 500.0f; break;
            default: v = (float)(u % 4999) * 0.01f; break;
        }
    }
    return f;
}

/* r4d_selftest's qsa_select_split_one at emulator size: three chunks, 96 distinct scores so the
 * tie rule decides, one page in 64 dead, `per` draft rows a sequence. */
Fix fix_split(int S, int per) {
    Fix f; f.name = "split: " + std::to_string(S) + " seq x " + std::to_string(per) + " rows, 96 scores";
    const int M = S * per, nb = 12001, topk = 512, ratio = 4, pages = 8192, n = 11000;
    f.shape(S, M, nb, nb, topk, ratio);
    for (int s = 0; s < S; ++s) {
        const int ns = n - s * 1300;
        f.nc[s] = ns; f.cu[s] = s * per;
        for (int b = 0; b < nb; ++b)
            f.page(s, b) = b >= ns ? -1 : b % 64 == 37 ? -1 : (b * 4099 + 11 + s * 71) % pages;
        for (int j = 0; j < per; ++j) f.pos[s * per + j] = ns * ratio + j - 2;
    }
    f.cu[S] = M;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 31 + m * 17) % 96) * 0.125f;
    return f;
}

/* Three chunks, none with more than topk live: the merge's candidates are every live key, more
 * than topk in the first row (its threshold the list's own, coarse scores tied at it), topk or
 * fewer in the second. The first row's sample holds fewer live keys than its bound, so the wide
 * kernel's list is every live key. */
Fix fix_split_sparse() {
    Fix f; f.name = "split: no chunk over topk";
    const int nb = 9000, topk = 512, ratio = 4;
    f.shape(2, 2, nb, nb, topk, ratio);
    f.cu = { 0, 1, 2 }; f.nc = { 8990, 8990 }; f.pos = { 4 * 8990 + 1, 4 * 8990 + 3 };
    for (int s = 0; s < 2; ++s)
        for (int b = 0; b < nb; ++b) f.page(s, b) = b % (s ? 20 : 9) == 3 ? (b * 131 + s) % 65536 : -1;
    for (int m = 0; m < 2; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 7 + m) % 13) * 0.5f;
    return f;
}

/* r4d_selftest's qsa_select_few_live_case: over more than topk blocks with topk or fewer live.
 *   0  few live pages, continuous scores           1  few live, most of them scoring zero or less
 *   2  live pages, most scores -inf                3  exactly topk live, a third tied at zero
 *   4  all live keys in the first chunk, more than topk, ties at the threshold
 *   5  nothing live                                6  at most topk blocks (control)
 *   7  everything live (control) */
Fix fix_few_live() {
    Fix f; f.name = "few live: topk or fewer live, in every way";
    const int S = 8, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
    const int nrow[S] = { 8000, 8000, 8000, 8000, 2000, 8000, 300, 8000 };
    f.shape(S, S, nb, nb, topk, ratio);
    g_x = 0x9E3779B97F4A7C15ull;
    for (int s = 0; s < S; ++s) {
        const int n = nrow[s];
        f.nc[s] = n; f.cu[s] = s; f.pos[s] = n * ratio + s % 3;
        for (int b = 0; b < n; ++b) {
            float& v = f.score(s, b);
            int32_t& pg = f.page(s, b);
            const uint64_t u = rnd();
            const float cont = (float)(u % 1000003) / 1000.0f + 0.001f;
            pg = (int32_t)(u % pages);
            switch (s) {
                case 0: if (b % 80 != 3) pg = -1; v = cont; break;
                case 1: if (b % 20 != 7) pg = -1;
                        v = (b / 20) % 4 == 0 ? cont : (b / 20) % 3 == 0 ? -2.5f : (b / 20) % 3 == 1 ? -0.0f : 0.0f;
                        break;
                case 2: v = b % 40 == 1 ? cont : -INFINITY; break;
                case 3: if (b % 15 != 2 || b >= 15 * 512) pg = -1;
                        v = b % 45 == 2 ? 0.0f : cont; break;
                case 4: v = b % 7 == 0 ? cont : (b % 2 ? 0.0f : -0.0f); break;
                case 5: pg = -1; v = cont; break;
                case 6: if (b % 9 == 4) pg = -1; v = (float)(u % 13) * 0.5f; break;
                default: v = cont; break;
            }
        }
        f.page(s, n) = s % 2 == 0 ? (int32_t)(rnd() % pages) : -1;
    }
    f.cu[S] = S;
    return f;
}

/* A 70-row prefill chunk: wider than the key cache's gate, so the automatic choice is the wide
 * kernel and the pinned single-workgroup kernel reads its keys from global memory. */
Fix fix_wide_launch() {
    Fix f; f.name = "a 70-row chunk: the wide kernel by default";
    const int M = 70, nb = 160, topk = 16, ratio = 4;
    f.shape(1, M, nb, nb, topk, ratio);
    f.cu = { 0, M }; f.nc = { 110 };
    for (int m = 0; m < M; ++m) f.pos[m] = 4 * 90 + m;
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 11 == 5 ? -1 : (b * 37 + 3) % 4096;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 13 + m * 3) % 9) * 0.5f;
    return f;
}

/* topk above the merged form's 512: the wide kernel's threshold bin goes to LDS alone, and the
 * split path still applies (two chunks of 2500, two lists of up to 1000 keys). */
Fix fix_big_topk() {
    Fix f; f.name = "topk 1000: above the merged form";
    const int nb = 5000, topk = 1000, ratio = 4;
    f.shape(1, 3, nb, nb, topk, ratio);
    f.cu = { 0, 3 }; f.nc = { 4990 }; f.pos = { 4 * 4990 + 2, 4 * 3000 - 1, 4 * 900 + 1 };
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 13 == 6 ? -1 : (b * 211 + 7) % 100000;
    for (int m = 0; m < 3; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 11 + m * 5) % 50) * 0.125f;
    return f;
}

/* Where the launcher leaves the split path: 33 query rows (one past its 32), and topk 3000, whose
 * three chunk lists overflow its 8192 keys. Both take the single-workgroup kernel. */
Fix fix_no_split(bool rows) {
    Fix f; f.name = rows ? "33 rows: past the split path's 32" : "topk 3000: past the split path's lists";
    const int M = rows ? 33 : 2, nb = rows ? 4500 : 9000, topk = rows ? 64 : 3000, ratio = 4;
    f.shape(1, M, nb, nb, topk, ratio);
    f.cu = { 0, M }; f.nc = { nb - 10 };
    for (int m = 0; m < M; ++m) f.pos[m] = 4 * (nb - 10 - 97 * m) + m % 4;
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 17 == 1 ? -1 : (b * 389 + 1) % 200000;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 29 + m * 3) % 211) * 0.0625f;
    return f;
}

/* A seeded small fixture: odd values, pitches, widths and row-to-sequence maps. */
Fix fix_sweep(int it) {
    g_x = 0xD1B54A32D192ED03ull ^ ((uint64_t)(it + 1) * 0x9E3779B97F4A7C15ull);
    for (int i = 0; i < 4; ++i) rnd();
    static const int kTopk[] = { 1, 2, 3, 5, 8, 17, 33, 64 }, kRatio[] = { 1, 2, 3, 4, 7 };
    const int topk = kTopk[rnd() % 8], ratio = kRatio[rnd() % 5], S = 1 + (int)(rnd() % 3);
    const int nb_t = 1 + (int)(rnd() % 300), nb_s = rnd() % 3 ? nb_t : 1 + (int)(rnd() % 300);
    std::vector<int> rows((size_t)S);
    int M = 0;
    for (auto& r : rows) { r = (int)(rnd() % 5); M += r; }
    const int lead = rnd() % 4 == 0 ? 1 : 0, trail = rnd() % 4 == 0 ? 1 : 0;   /* rows no sequence holds */
    M += lead + trail;
    if (M == 0) { rows[0] = 1; M = 1; }
    Fix f; f.name = "sweep " + std::to_string(it);
    f.shape(S, M, nb_s, nb_t, topk, ratio, (int)(rnd() % 4), (int)(rnd() % 4), (int)(rnd() % 3));
    f.cu[0] = lead;
    for (int s = 0; s < S; ++s) f.cu[s + 1] = f.cu[s] + rows[(size_t)s];
    const int nb = f.nb();
    for (int s = 0; s < S; ++s) f.nc[s] = (int)(rnd() % (uint64_t)(nb + 6));
    for (int m = 0; m < M; ++m) f.pos[m] = (int)(rnd() % (uint64_t)(ratio * (nb + 3)));
    const unsigned dead = (unsigned)(rnd() % 4);      /* none, a tenth, half, nearly all */
    for (auto& v : f.bt) {
        const uint64_t u = rnd();
        v = (dead == 1 && u % 10 == 0) || (dead == 2 && u % 2) || (dead == 3 && u % 20) ? -1 : (int32_t)(u % 100000);
    }
    const unsigned grid = 1 + (unsigned)(rnd() % 12);  /* distinct values: ties when small */
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        switch (u % 23) {
            case 0: v = NAN; break;
            case 1: v = INFINITY; break;
            case 2: v = -INFINITY; break;
            case 3: v = -0.0f; break;
            case 4: v = 0.0f; break;
            case 5: v = 1e-40f * (float)(1 + (u >> 8) % 4); break;   /* denormal */
            case 6: v = -1e30f; break;                                /* not live: the bound is strict */
            case 7: v = -9e29f; break;                                /* live, ranked as zero */
            case 8: v = -(float)((u >> 8) % 100); break;
            default: v = (float)((u >> 8) % grid) * 0.75f; break;
        }
    }
    return f;
}

/* A seeded fixture at serving sizes for `sweep`: 500 to 14500 blocks, topk on both sides of the
 * merged form's 512 and of the split path's list budget, one to three sequences of a few draft
 * rows or of a prefill chunk past 64 rows, and dead pages, -inf, NaN and ties in varying amounts. */
Fix fix_random(int it) {
    g_x = 0xA0761D6478BD642Full ^ ((uint64_t)(it + 7) * 0xE7037ED1A0B428DBull);
    for (int i = 0; i < 4; ++i) rnd();
    static const int kTopk[] = { 1, 16, 64, 200, 511, 512, 513, 700, 1500, 2600 }, kRatio[] = { 1, 4, 4, 8 };
    const int topk = kTopk[rnd() % 10], ratio = kRatio[rnd() % 4], S = 1 + (int)(rnd() % 3);
    const int nb = 500 + (int)(rnd() % 14000);
    const bool chunk = rnd() % 6 == 0;
    std::vector<int> rows((size_t)S);
    int M = 0;
    for (auto& r : rows) { r = chunk ? 20 + (int)(rnd() % 30) : 1 + (int)(rnd() % 5); M += r; }
    Fix f; f.name = "sweep " + std::to_string(it) + ": " + std::to_string(M) + " rows, " + std::to_string(nb) +
                    " blocks, topk " + std::to_string(topk);
    f.shape(S, M, nb, nb + (int)(rnd() % 3), topk, ratio, (int)(rnd() % 5), (int)(rnd() % 5), (int)(rnd() % 3));
    for (int s = 0; s < S; ++s) f.cu[s + 1] = f.cu[s] + rows[(size_t)s];
    for (int s = 0; s < S; ++s) f.nc[s] = nb - (int)(rnd() % (uint64_t)(nb / 2 + 1)) + (int)(rnd() % 3);
    for (int m = 0; m < M; ++m)
        f.pos[m] = std::max(0, ratio * f.nc[f.seq_of(m)] - (int)(rnd() % (uint64_t)(ratio * nb / 3 + 1)) + (int)(rnd() % 7));
    const unsigned dead = (unsigned)(rnd() % 6), dist = (unsigned)(rnd() % 6), grid = 1 + (unsigned)(rnd() % 400);
    for (auto& v : f.bt) {
        const uint64_t u = rnd();
        const bool d = dead == 1 ? u % 50 == 0 : dead == 2 ? u % 2 == 0 : dead == 3 ? u % 16 != 0
                     : dead == 4 ? u % 100 != 0 : false;
        v = d ? -1 : (int32_t)(u % 1000000);
    }
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        if (u % 97 == 0) { v = -INFINITY; continue; }
        if (u % 389 == 0) { v = NAN; continue; }
        switch (dist) {
            case 0: v = (float)(u % 1000003) / 1000.0f; break;
            case 1: v = (float)((u >> 8) % grid) * 0.25f; break;
            case 2: v = u % 8 == 0 ? 1.0f + (float)(u % 60000) * 1e-6f : (float)(u % 1000) / 2000.0f; break;
            case 3: v = 1.0f + (float)((u >> 8) % 60000) * 1e-6f; break;
            case 4: v = (u >> 8) % 3 == 0 ? 0.0f : (u >> 8) % 3 == 1 ? -0.0f : -1.0f; break;
            default: v = (float)((u >> 8) % 4999) * 0.01f; break;
        }
    }
    return f;
}

/* A fixture under the signed order, its finite scores negated on odd rows when `flip` and moved
 * by `shift`, so live scores straddle zero or fall below it. */
Fix as_signed(Fix f, float shift, bool flip = false) {
    f.sgn = true;
    f.name = "signed " + f.name;
    if (shift == 0.0f && !flip) return f;          /* as built: keeps -0 apart from +0 */
    for (int m = 0; m < f.M; ++m)
        for (int b = 0; b < f.nb_s; ++b) {
            float& v = f.score(m, b);
            if (!std::isfinite(v)) continue;
            if (flip && m % 2) v = -v;
            v += shift;
        }
    return f;
}

/* ---- tree verify (docs/QSA.md, "Tree verify"): the tree launch of r4d_rad_qsa_select against a port of
 * libref's tree rows. One tree sequence (a random tree of `R` rows over a history of `b0` committed
 * blocks) beside an ordinary decode sequence and a prefill-shaped one with parent -2 rows, so a
 * launch mixes the three; scores on a coarse grid so ties decide, a dead committed page, a region
 * that the committed blocks outrank or that outranks them. Every output (sel, nsel, seqused, vmask)
 * is compared byte for byte, the padding and a row past M included. */
struct TreeFix {
    std::string name;
    int topk = 3, ratio = 4, nb = 0, RW = 8, M = 0;
    std::vector<float> sc, rs;             /* [M][nb], [M][RW] */
    std::vector<int32_t> bt, nc, pos, cu, sp;
};

void ref_tree_select(const TreeFix& f, std::vector<int32_t>& se, std::vector<int32_t>& ns,
                     std::vector<int32_t>& su, std::vector<int64_t>& vm) {
    const int64_t topk = f.topk, ratio = f.ratio, nb = f.nb, W = topk + 1;
    const int S = (int)f.cu.size() - 1;
    se.assign((size_t)f.M * W, 0); ns.assign((size_t)f.M, 0); su.assign((size_t)f.M, 0); vm.assign((size_t)f.M, 0);
    for (int64_t m = 0; m < f.M; ++m) {
        int s = 0;
        for (int i = 0; i < S; ++i) if (m >= f.cu[i] && m < f.cu[i + 1]) s = i;
        const int64_t p = f.pos[m], t0 = f.cu[s];
        const bool tree = f.sp[t0] == -1;
        const int64_t base = f.pos[t0], b0 = base / ratio, ntot = (p + 1) / ratio;
        const int64_t nC = tree ? b0 : std::min<int64_t>(std::min<int64_t>(f.nc[s], nb), ntot);
        const int64_t n = tree ? ntot : nC;
        std::vector<float> val((size_t)n);
        std::vector<int64_t> live;
        for (int64_t u = 0; u < n; ++u) {
            float v;
            if (u < nC) {
                v = f.sc[(size_t)(m * nb + u)];
                if (f.bt[(size_t)(s * nb + u)] < 0) continue;
            } else {
                v = f.rs[(size_t)(m * f.RW + (u - nC))];
            }
            if (!(v > -1e30f)) continue;
            val[(size_t)u] = v > 0.0f ? v : 0.0f;
            live.push_back(u);
        }
        std::vector<int64_t> take;
        if ((int64_t)live.size() <= topk) {
            take = live;
        } else {
            std::vector<float> ord;
            for (int64_t u : live) ord.push_back(val[(size_t)u]);
            std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(), std::greater<float>());
            const float thr = ord[(size_t)(topk - 1)];
            for (int64_t u : live) if (val[(size_t)u] > thr) take.push_back(u);
            for (int64_t u : live) if ((int64_t)take.size() < topk && val[(size_t)u] == thr) take.push_back(u);
        }
        int32_t* row = &se[(size_t)(m * W)];
        std::vector<char> taken((size_t)n, 0);
        int64_t out = 0;
        for (int64_t u : take) { taken[(size_t)u] = 1; if (u < nC) row[out++] = f.bt[(size_t)(s * nb + u)]; }
        if (!tree && n < nb && (p + 1) % ratio != 0) {
            const int32_t part = f.bt[(size_t)(s * nb + n)];
            if (part >= 0) row[out++] = part;
        }
        for (int64_t i = out; i <= topk; ++i) row[i] = -1;
        ns[(size_t)m] = (int32_t)out;
        if (!tree) { su[(size_t)m] = (int32_t)(out == 0 ? 0 : (out - 1) * ratio + p % ratio + 1); continue; }
        su[(size_t)m] = (int32_t)(out * ratio);
        const int64_t rb = b0 * ratio;
        auto vis = [&](int64_t c) { return c >= ntot || taken[(size_t)c] != 0; };
        uint64_t bits = 0;
        for (int64_t q = rb; q < base; ++q) if (vis(q / ratio)) bits |= 1ull << (q - rb);
        for (int64_t y = m; y >= 0; y = f.sp[y] < 0 ? -1 : t0 + f.sp[y]) {
            if (vis(f.pos[y] / ratio)) bits |= 1ull << (base - rb + (y - t0));
        }
        vm[(size_t)m] = (int64_t)bits;
    }
}

TreeFix fix_tree(int it, bool coarse) {
    TreeFix f;
    f.name = "tree " + std::to_string(it);
    f.ratio = 4; f.topk = 2 + it % 5; f.nb = 40 + it % 3 * 150;
    const int R = 3 + (int)(rnd() % 11), base = 30 + (int)(rnd() % 40);
    /* sequence 0: the tree; 1: a one-row decode (parent -2); 2: three prefill-shaped rows (-2). */
    f.M = R + 1 + 3;
    f.cu = { 0, R, R + 1, R + 4 };
    f.nc.assign(3, 0); f.pos.assign((size_t)f.M, 0); f.sp.assign((size_t)f.M, -2);
    std::vector<int> depth((size_t)R, 0);
    f.sp[0] = -1; f.pos[0] = base;
    for (int j = 1; j < R; ++j) {
        int par;
        do par = (int)(rnd() % (uint64_t)j); while (depth[(size_t)par] >= 6);
        f.sp[(size_t)j] = par; depth[(size_t)j] = depth[(size_t)par] + 1; f.pos[(size_t)j] = base + depth[(size_t)j];
    }
    f.nc[0] = base / 4;
    const int d1 = 60 + (int)(rnd() % 30), p2 = 70 + (int)(rnd() % 20);
    f.pos[(size_t)R] = d1; f.nc[1] = (d1 + 1) / 4;
    for (int i = 0; i < 3; ++i) f.pos[(size_t)(R + 1 + i)] = p2 + i;
    f.nc[2] = (p2 + 3) / 4;
    f.bt.assign((size_t)3 * f.nb, -1);
    for (int s = 0; s < 3; ++s)
        for (int b = 0; b < f.nb; ++b) f.bt[(size_t)(s * f.nb + b)] = (b * 7 + s * 3) % 5000 + 1;
    f.bt[(size_t)(0 * f.nb + 2)] = -1;                  /* a dead committed page */
    f.sc.assign((size_t)f.M * f.nb, -INFINITY);
    f.rs.assign((size_t)f.M * f.RW, -INFINITY);
    for (int m = 0; m < f.M; ++m) {
        for (int b = 0; b < f.nb; ++b)
            f.sc[(size_t)(m * f.nb + b)] = coarse ? (float)(rnd() % 6) * 0.5f : (float)(rnd() % 100000) * 1e-3f;
        for (int i = 0; i < f.RW; ++i)
            f.rs[(size_t)(m * f.RW + i)] = coarse ? (float)(rnd() % 6) * 0.5f : (float)(rnd() % 100000) * 1e-3f;
    }
    return f;
}

int run_tree(const TreeFix& f) {
    std::vector<int32_t> want, want_n, want_su;
    std::vector<int64_t> want_vm;
    ref_tree_select(f, want, want_n, want_su, want_vm);
    const int M = f.M, W = f.topk + 1, S = (int)f.cu.size() - 1;
    std::vector<int32_t> se((size_t)(M + 1) * W, kSentinel), ns((size_t)M + 1, kSentinel), su((size_t)M + 1, kSentinel);
    std::vector<int64_t> vm((size_t)M + 1, (int64_t)0x5555555555555555ll);
    std::vector<float> sc = f.sc, rs = f.rs;
    std::vector<int32_t> bt = f.bt, nc = f.nc, pos = f.pos, cu = f.cu, sp = f.sp;
    RadTensor t[11] = {
        tens(sc.data(), RAD_F32, {M, f.nb}),  tens(bt.data(), RAD_I32, {S, f.nb}),
        tens(nc.data(), RAD_I32, {S}),        tens(pos.data(), RAD_I32, {M}),
        tens(cu.data(), RAD_I32, {S + 1}),    tens(se.data(), RAD_I32, {M, W}),
        tens(ns.data(), RAD_I32, {M}),        tens(su.data(), RAD_I32, {M}),
        tens(sp.data(), RAD_I32, {M}),        tens(rs.data(), RAD_F32, {M, f.RW}),
        tens(vm.data(), RAD_I64, {M}),
    };
    RadParam p[5]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb}; p[2] = {"topk", RAD_P_INT, f.topk};
    p[3] = {"ratio", RAD_P_INT, f.ratio}; p[4] = {"seqs", RAD_P_INT, S};
    RadArgs a{}; a.t = t; a.n_t = 11; a.p = p; a.n_p = 5;
    const int rc = r4d_rad_qsa_select(&a, nullptr);
    if (rc != RAD_OK) { std::printf("FAIL %-48s tree launch returned %d\n", f.name.c_str(), rc); return 1; }
    const char* const kname = emu_last_kernel + (emu_last_kernel[0] == '(' ? 1 : 0);
    if (std::string(kname, std::strcspn(kname, "<")) != "qsa_select_tree_kernel") {
        std::printf("FAIL %-48s ran %s\n", f.name.c_str(), emu_last_kernel);
        return 1;
    }
    int bad = 0;
    for (int m = 0; m < M; ++m) {
        bool ok = ns[(size_t)m] == want_n[(size_t)m] && su[(size_t)m] == want_su[(size_t)m] && vm[(size_t)m] == want_vm[(size_t)m];
        for (int i = 0; i < W; ++i) ok = ok && se[(size_t)m * W + i] == want[(size_t)m * W + i];
        if (ok) continue;
        if (++bad <= 3)
            std::printf("FAIL %-48s row %d: nsel %d want %d, seqused %d want %d, vmask %llx want %llx\n", f.name.c_str(), m,
                        ns[(size_t)m], want_n[(size_t)m], su[(size_t)m], want_su[(size_t)m],
                        (unsigned long long)vm[(size_t)m], (unsigned long long)want_vm[(size_t)m]);
    }
    /* Nothing past the rows is written. */
    if (ns[(size_t)M] != kSentinel || su[(size_t)M] != kSentinel || vm[(size_t)M] != 0x5555555555555555ll) {
        std::printf("FAIL %-48s wrote past M\n", f.name.c_str());
        ++bad;
    }
    return bad ? 1 : 0;
}

int tree_cases(int* launches) {
    int fails = 0;
    for (int it = 0; it < 40; ++it) {
        g_x = 0x9E3779B97F4A7C15ull * (uint64_t)(it + 1) + 1;
        fails += run_tree(fix_tree(it, it % 2 == 0));
        ++*launches;
    }
    return fails;
}

}  // namespace

/* No argument: the fixtures above and the branch check. `sweep N`: N fix_random fixtures, about
 * 0.15 s each. */
int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool sweep = argc > 1 && std::string(argv[1]) == "sweep";
    std::vector<std::function<Fix()>> fixtures;
    if (sweep) {
        const int n = argc > 2 ? std::atoi(argv[2]) : 100;
        for (int i = 0; i < n; ++i) fixtures.push_back([i] { return fix_random(i); });
        for (int i = 0; i < n; ++i) fixtures.push_back([i] { return as_signed(fix_random(i), -1.0f, true); });
    } else {
        fixtures = { fix_decode, fix_prefill, fix_few_live, fix_wide_launch,
                     [] { return fix_split(1, 4); }, [] { return fix_split(2, 4); }, fix_split_sparse, fix_big_topk,
                     [] { return fix_no_split(true); }, [] { return fix_no_split(false); } };
        for (int d = 0; d < 5; ++d) fixtures.push_back([d] { return fix_wide(d); });
        for (int i = 0; i < 150; ++i) fixtures.push_back([i] { return fix_sweep(i); });
        fixtures.push_back([] { return as_signed(fix_decode(), -5.0f); });
        fixtures.push_back([] { return as_signed(fix_decode(), -20.0f); });
        fixtures.push_back([] { return as_signed(fix_few_live(), 0.0f); });
        fixtures.push_back([] { return as_signed(fix_split(2, 4), -6.0f); });
        fixtures.push_back([] { return as_signed(fix_wide_launch(), -1.0f, true); });
        for (int d = 5; d < 8; ++d) fixtures.push_back([d] { return fix_wide(d); });
        for (int i = 0; i < 150; ++i) fixtures.push_back([i] { return as_signed(fix_sweep(i), 0.0f); });
    }
    int fails = 0, launches = 0;
    fails += tree_cases(&launches);
    for (auto& make : fixtures) {
        Fix f = make();
        for (Form form : { kSingle, kWide, kAuto }) { fails += run(f, form); ++launches; }
    }
    if (sweep) {
        for (const auto& [p, c] : g_paths) std::printf("  %5d row(s): %s\n", c, p.c_str());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("qsa_select sweep: %zu fixtures, %d launches, %d failure(s), %.1f s\n", fixtures.size(),
                    launches, fails, sec);
        return fails ? 1 : 0;
    }
    /* Every branch the fixtures are built to reach. */
    static const char* const kRequired[] = {
        "rows of at most topk blocks", "exactly topk live over more blocks", "fewer than topk live over more blocks",
        "a threshold", "a threshold whose ties are cut short",
        "single: keys cached in LDS", "single: keys read from global memory",
        "wide: the sampled pass, rows 16-byte aligned", "wide: the sampled pass, rows not 16-byte aligned",
        "wide: the sampled list", "wide: the sampled list, holding every live key",
        "wide: a sample that fell back", "wide: merged lists",
        "wide: the threshold's bin in LDS", "wide: the threshold's bin over global memory",
        "split: topk or fewer live, every one taken", "split: no chunk over topk, more than topk live",
        "split: a chunk over topk", "split: exactly topk candidates, from a chunk over topk",
        "signed: a negative threshold",
    };
    int missing = 0;
    for (const char* p : kRequired)
        if (!g_paths.count(p)) { std::printf("FAIL no row reaches: %s\n", p); ++missing; }
    for (const auto& [p, c] : g_paths) std::printf("  %5d row(s): %s\n", c, p.c_str());
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("qsa_select: %zu fixtures, %d launches, %d failure(s), %d branch(es) unreached, %.1f s\n",
                fixtures.size(), launches, fails, missing, sec);
    return fails || missing ? 1 : 0;
}

/* ref_test.cpp -- libref, op by op, on shapes small enough to check by hand.
 *
 * libref is the oracle rad-kbench holds every resolved kernel to (spec §17); if it is wrong,
 * wrong kernels are certified and the engine's output is fluent, wrong and unbisectable. So every
 * expected value is computed on paper and written as a literal, never produced by a second call
 * into the same code. Where a literal is impossible (the seeded sampler's draw) the test checks
 * properties: determinism, range, a non-constant generator. Two cross-checks compare different
 * routes instead: the gated-delta-net chunked scan against the recurrent update, and paged
 * attention at q_len 2 against a hand-computed softmax.
 *
 * The plugin is reached through RadKernelInfo::launch with a hand-built RadArgs, as the engine
 * reaches it, so the operand order and parameter names in libref/ref_registry.cpp are tested too.
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "../libref/ref_common.h"
#include "../core/sample/host_ref.h"   /* the host chain, for the whitelist's op-for-op form */
#include "rad_sample.h"
#include "rad_spechub.h"   /* the coupling state, for mtp_tree_grow's coupled form */
#include "rad_philox.h"
#include "rad_runtime.h"   /* RAD_VGEOM_* */
#include "rad_planes.h"
#include "rad_mtp_tree.h"   /* the grower's own walk, for the pruned fuzz */

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>

using namespace ref;

/* ================================================================== reaching the plugin
 * Weak references first: linked directly against the plugin sources, the symbols are present.
 * Built by CMake it links rad_core only, the weak references stay null, and the plugin is
 * dlopen'd from $RADIANCE_HOME as the engine loads it. Both paths exercise the same table. */
extern "C" {
__attribute__((weak)) int                  rad_kernel_count(void);
__attribute__((weak)) const RadKernelInfo* rad_kernel_at(int);
__attribute__((weak)) int                  rad_kernel_schema_count(void);
__attribute__((weak)) const RadOpSchema*   rad_kernel_schema_at(int);
__attribute__((weak)) uint32_t             rad_plugin_abi_version(void);
__attribute__((weak)) const RadPluginInfo* rad_plugin_info(void);
}

namespace {

struct Api {
    void*                handle                       = nullptr;   /* null when linked in */
    int                  (*kernel_count)(void)        = nullptr;
    const RadKernelInfo* (*kernel_at)(int)            = nullptr;
    int                  (*schema_count)(void)        = nullptr;
    const RadOpSchema*   (*schema_at)(int)            = nullptr;
    uint32_t             (*abi_version)(void)         = nullptr;
    const RadPluginInfo* (*info)(void)                = nullptr;
};

Api load_api() {
    Api a;
    if (&rad_kernel_count) {
        a.kernel_count = rad_kernel_count;
        a.kernel_at    = rad_kernel_at;
        a.schema_count = rad_kernel_schema_count;
        a.schema_at    = rad_kernel_schema_at;
        a.abi_version  = rad_plugin_abi_version;
        a.info         = rad_plugin_info;
        return a;
    }
    std::vector<std::string> paths;
    if (const char* home = std::getenv("RADIANCE_HOME"))
        paths.push_back(std::string(home) + "/kernels/libref.so");
    paths.push_back("radiance_home/kernels/libref.so");
    paths.push_back("build/radiance_home/kernels/libref.so");
    paths.push_back("./libref.so");
    for (auto& p : paths) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        a.handle       = h;
        a.kernel_count = (int (*)(void))dlsym(h, "rad_kernel_count");
        a.kernel_at    = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        a.schema_count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
        a.schema_at    = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        a.abi_version  = (uint32_t (*)(void))dlsym(h, "rad_plugin_abi_version");
        a.info         = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
        if (a.kernel_count) return a;
    }
    fprintf(stderr, "ref_test: libref not found. Set RADIANCE_HOME or link the plugin.\n");
    std::exit(2);
}

const Api& api() { static Api a = load_api(); return a; }

/* A test hook the plugin exports beside the ABI entry points (ref_collective_timeout_ms). */
void* plugin_sym(const char* name) {
    return dlsym(api().handle ? api().handle : RTLD_DEFAULT, name);
}

const RadKernelInfo* kern(const char* op) {
    for (int i = 0; i < api().kernel_count(); ++i) {
        const RadKernelInfo* k = api().kernel_at(i);
        if (k && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

const RadOpSchema* schema(const char* op) {
    for (int i = 0; i < api().schema_count(); ++i) {
        const RadOpSchema* s = api().schema_at(i);
        if (s && std::strcmp(s->op, op) == 0) return s;
    }
    return nullptr;
}

/* ================================================================== tiny tensors */
int64_t dt_bytes(uint32_t dt, int64_t n) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32:            return n * 4;
        case RAD_I64:                                        return n * 8;
        case RAD_F16: case RAD_BF16: case RAD_I16:           return n * 2;
        case RAD_I8: case RAD_U8: case RAD_BOOL:
        case RAD_F8E4M3: case RAD_F8E5M2:                    return n;
        case RAD_I4: case RAD_FP4E2M1:                         return (n + 1) / 2;
        case RAD_I2:                                         return (n + 3) / 4;
        default:                                             return n * 4;
    }
}

struct T {
    std::vector<uint8_t> mem;
    RadTensor t{};

    T(uint32_t dt, std::vector<int64_t> shape) {
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; n *= shape[i]; }
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
        mem.assign((size_t)dt_bytes(dt, n) + 8, 0);
        t.data = mem.data();
    }
    T& f(std::vector<float> v) {
        for (size_t i = 0; i < v.size(); ++i) st_dt(t.dtype, t.data, (int64_t)i, v[i]);
        return *this;
    }
    T& raw(std::vector<uint8_t> v) {
        for (size_t i = 0; i < v.size(); ++i) mem[i] = v[i];
        return *this;
    }
    T& n(std::vector<int64_t> v) {
        for (size_t i = 0; i < v.size(); ++i) st_int(t.dtype, t.data, (int64_t)i, v[i]);
        return *this;
    }
    float    at(int64_t i) const  { return ld_dt(t.dtype, t.data, i); }
    int64_t  iat(int64_t i) const { return ld_int(t.dtype, t.data, i); }
};

/* ================================================================== the sampler's params row */
/* Every sampler op takes its scalars as an operand of RadSampleParams rows, not as parameters: a
 * parameter is geometry and freezes at declare, so top_p as a parameter would make two requests
 * with different top_p two resolutions of one op (abi/rad_sample.h). This builds that operand:
 * identity defaults, then the case sets the one field it is about. */
struct SP {
    std::vector<RadSampleParams> rows;
    T                            t;

    explicit SP(int64_t n)
        : rows((size_t)n), t(RAD_U32, {n, (int64_t)(sizeof(RadSampleParams) / 4)}) {
        for (auto& r : rows) {
            std::memset(&r, 0, sizeof r);
            r.temp = 1.0f;
            r.top_p = 1.0f;
            r.min_p = 0.0f;
            r.typical_p = 1.0f;
            r.rep_penalty = 1.0f;
            r.mask_row = -1;
            /* The whole history; 0 would turn the history stages off (llama.cpp's meaning). */
            r.penalty_last_n = -1;
            r.dry_penalty_last_n = -1;
        }
    }
    RadSampleParams& operator[](size_t i) { return rows[i]; }
    /* Copied on demand, so a case can set fields after construction and before the call. */
    const T& operand() {
        std::memcpy(t.mem.data(), rows.data(), rows.size() * sizeof(RadSampleParams));
        return t;
    }
};

/* ================================================================== argument builder */
struct A {
    std::vector<RadTensor>  ts;
    std::vector<RadParam>   ps;
    std::deque<std::string> strs;      /* stable addresses, unlike a vector's */
    std::vector<float>      scratch;
    RadArgs                 a{};

    A& t(const T& x)   { ts.push_back(x.t); return *this; }
    A& tnull()         { ts.push_back(RadTensor{}); return *this; }
    A& i(const char* k, long long v) { ps.push_back(RAD_INT(k, v)); return *this; }
    /* A float parameter is RAD_P_F64 (docs/OPS.md). This test calls the launch directly, with no
     * declare to catch a kind mismatch, so a case spelling it otherwise would prove nothing about
     * the schema it claims to test. */
    A& d(const char* k, double v) { ps.push_back(RAD_F64(k, v)); return *this; }
    A& s(const char* k, const char* v) {
        strs.emplace_back(v);
        ps.push_back(RAD_STR(k, strs.back().c_str()));
        return *this;
    }
    A& scr(int64_t n) { scratch.assign((size_t)n, 0.0f); return *this; }

    const RadArgs* get(int rank = 0, int world = 1) {
        a.t = ts.data();
        a.n_t = (int)ts.size();
        a.p = ps.data();
        a.n_p = (int)ps.size();
        a.scratch = scratch.empty() ? nullptr : scratch.data();
        a.scratch_bytes = (int64_t)scratch.size() * 4;
        a.instance = nullptr;
        a.rank = rank;
        a.world_size = world;
        return &a;
    }
};

int run(const char* op, A& args, int rank = 0, int world = 1) {
    const RadKernelInfo* k = kern(op);
    if (!k || !k->launch) return RAD_E_NOKERNEL;
    return k->launch(args.get(rank, world), nullptr);
}

}  /* namespace */

TEST(manifold_read_uses_whole_stream_rms_and_unaveraged_pre_mix) {
    T h(RAD_F32, {1, 2}); h.f({1, 3});
    T fn(RAD_F32, {8, 2});
    fn.f({3.0f * std::log(3.0f), 0});
    T scale(RAD_F32, {3}); scale.f({1, 1, 1});
    T base(RAD_F32, {8}); base.f({0, 0, std::log(3.0f), 0, 0, 0, 0, 0});
    T x(RAD_F32, {1, 1}), post(RAD_F32, {1, 2}), comb(RAD_F32, {1, 4});
    A a; a.t(h).t(fn).t(scale).t(base).t(x).t(post).t(comb)
        .i("M", 1).i("n", 1).i("hc", 2).i("iterations", 20)
        .d("eps", 4).d("hc_eps", 0).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_pre", a), RAD_OK);
    /* RMS inverse = 1/3, pre = [3/4,1/2], post = [3/2,1]. */
    CHECK_NEAR(x.at(0), 2.25f, 1e-6);
    CHECK_NEAR(post.at(0), 1.5f, 1e-6);
    CHECK_EQ(post.at(1), 1.0f);
    for (int i = 0; i < 4; ++i) CHECK_EQ(comb.at(i), 0.5f);
    a.ps[3].ival = 0;
    CHECK_EQ(run("mhc_pre", a), RAD_E_SHAPE);
}

/* Two ranks each project half of the chunks; their partials, joined row by row as the
 * row-interleaved all_gather lays them out, finish to mhc_pre. Quarter-step inputs keep every
 * product and sum exact, so the split is compared bit for bit. */
TEST(manifold_read_split_by_chunks_finishes_to_the_whole_read) {
    const int64_t M = 3, n = 8, hc = 4, W = hc * n, mix = hc * (hc + 2), S = 28;
    T h(RAD_F32, {M, W}), fn(RAD_F32, {mix, W}), scale(RAD_F32, {3}), base(RAD_F32, {mix});
    std::vector<float> hv((size_t)(M * W)), fv((size_t)(mix * W)), bv((size_t)mix);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = (float)((int)((i * 7) % 9) - 4) * 0.25f;
    for (size_t i = 0; i < fv.size(); ++i) fv[i] = (float)((int)((i * 5) % 7) - 3) * 0.125f;
    for (size_t i = 0; i < bv.size(); ++i) bv[i] = (float)((int)(i % 5) - 2) * 0.5f;
    h.f(hv); fn.f(fv); scale.f({0.5f, 0.25f, 1.0f}); base.f(bv);
    T x(RAD_F32, {M, n}), post(RAD_F32, {M, hc}), comb(RAD_F32, {M, hc * hc});
    A whole; whole.t(h).t(fn).t(scale).t(base).t(x).t(post).t(comb)
        .i("M", M).i("n", n).i("hc", hc).i("iterations", 20)
        .d("eps", 1e-5).d("hc_eps", 1e-6).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_pre", whole), RAD_OK);
    T gathered(RAD_F32, {M, 4 * S});
    for (int rank = 0; rank < 2; ++rank) {
        T part(RAD_F32, {M, 2 * S});
        part.f(std::vector<float>((size_t)(M * 2 * S), 9.0f));
        A pa; pa.t(h).t(fn).t(part).i("M", M).i("n", n).i("hc", hc).i("chunk", 8)
            .i("first", 2 * rank).i("chunks", 2).s("dtype", "f32");
        REQUIRE_EQ(run("mhc_pre_partial", pa), RAD_OK);
        for (int64_t r = 0; r < M; ++r) for (int64_t c = 0; c < 2; ++c)
            for (int64_t j = mix + 1; j < S; ++j) CHECK_EQ(part.at(r * 2 * S + c * S + j), 0.0f);
        for (int64_t r = 0; r < M; ++r) for (int64_t i = 0; i < 2 * S; ++i)
            st_dt(RAD_F32, gathered.t.data, r * 4 * S + rank * 2 * S + i, part.at(r * 2 * S + i));
    }
    T x2(RAD_F32, {M, n}), post2(RAD_F32, {M, hc}), comb2(RAD_F32, {M, hc * hc});
    A fa; fa.t(gathered).t(h).t(scale).t(base).t(x2).t(post2).t(comb2)
        .i("M", M).i("n", n).i("hc", hc).i("iterations", 20)
        .d("eps", 1e-5).d("hc_eps", 1e-6).i("chunk", 8).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_pre_finish", fa), RAD_OK);
    for (int64_t i = 0; i < M * n; ++i) CHECK_EQ(x2.at(i), x.at(i));
    for (int64_t i = 0; i < M * hc; ++i) CHECK_EQ(post2.at(i), post.at(i));
    for (int64_t i = 0; i < M * hc * hc; ++i) CHECK_EQ(comb2.at(i), comb.at(i));
    /* A chunk that does not divide the residual, or a range past its end, is refused. */
    T part(RAD_F32, {M, 2 * S});
    A bad; bad.t(h).t(fn).t(part).i("M", M).i("n", n).i("hc", hc).i("chunk", 8)
        .i("first", 3).i("chunks", 2).s("dtype", "f32");
    CHECK_EQ(run("mhc_pre_partial", bad), RAD_E_SHAPE);
    fa.ps[6].ival = 7;
    CHECK_EQ(run("mhc_pre_finish", fa), RAD_E_SHAPE);
}

TEST(manifold_post_mixes_sources_into_destinations_and_contracts_by_mean) {
    T h(RAD_F32, {1, 6}); h.f({10, 20, 30, 40, 50, 60});
    T x(RAD_F32, {1, 2}); x.f({2, 4});
    T post(RAD_F32, {1, 3}); post.f({1, 2, 3});
    T comb(RAD_F32, {1, 9}); comb.f({0, 1, 0, 0, 0, 1, 1, 0, 0});
    T out(RAD_F32, {1, 6});
    A a; a.t(x).t(h).t(post).t(comb).t(out).i("M", 1).i("n", 2).i("hc", 3).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_post", a), RAD_OK);
    const float expected[] = {52, 64, 14, 28, 36, 52};
    for (int i = 0; i < 6; ++i) CHECK_EQ(out.at(i), expected[i]);
    /* The input remains the unnormalized residual. */
    CHECK_EQ(h.at(0), 10.0f);
    a.ts[4] = h.t;
    CHECK_EQ(run("mhc_post", a), RAD_E_SHAPE);
    T contracted(RAD_F32, {1, 2});
    A c; c.t(h).t(contracted).i("M", 1).i("n", 2).i("hc", 3).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_contract", c), RAD_OK);
    CHECK_EQ(contracted.at(0), 30.0f);
    CHECK_EQ(contracted.at(1), 40.0f);
}

TEST(manifold_sinkhorn_first_pass_and_epsilon_are_distinct) {
    T h(RAD_F32, {1, 2}); h.f({1, 3});
    T fn(RAD_F32, {8, 2}), scale(RAD_F32, {3}); scale.f({1, 1, 1});
    T base(RAD_F32, {8}); base.f({0, 0, 0, 0, std::log(3.0f), 0, 0, 0});
    T x(RAD_F32, {1, 1}), post(RAD_F32, {1, 2}), comb(RAD_F32, {1, 4});
    A a; a.t(h).t(fn).t(scale).t(base).t(x).t(post).t(comb)
        .i("M", 1).i("n", 1).i("hc", 2).i("iterations", 1)
        .d("eps", 1e-5).d("hc_eps", 0).s("dtype", "f32");
    REQUIRE_EQ(run("mhc_pre", a), RAD_OK);
    const float first[] = {0.6f, 1.0f / 3, 0.4f, 2.0f / 3};
    for (int i = 0; i < 4; ++i) CHECK_NEAR(comb.at(i), first[i], 1e-6);
    a.ps[3].ival = 20;
    REQUIRE_EQ(run("mhc_pre", a), RAD_OK);
    CHECK_NEAR(comb.at(0), 0.633974596f, 1e-6);
    CHECK_NEAR(comb.at(1), 0.366025404f, 1e-6);
    /* Uniform logits: first pass is (.5 + .25) / (1.5 + .25) = 3/7. */
    base.f({0, 0, 0, 0, 0, 0, 0, 0});
    a.ps[3].ival = 1;
    a.ps[5].dval = 0.25;
    REQUIRE_EQ(run("mhc_pre", a), RAD_OK);
    CHECK_EQ(x.at(0), 3.0f);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(comb.at(i), 3.0f / 7, 1e-6);
}

TEST(kda_candidate_states_restore_the_accepted_prefix) {
    T q(RAD_F32, {2, 1, 2}); q.f({1, 0, 1, 0});
    T k(RAD_F32, {2, 1, 2}); k.f({1, 0, 1, 0});
    T v(RAD_F32, {2, 1, 1}); v.f({2, 2});
    T g(RAD_F32, {2, 1, 2}); g.f({0, std::log(3.0f), 0, std::log(3.0f)});
    T b(RAD_F32, {2, 1}), alog(RAD_F32, {1}), bias(RAD_F32, {2});
    T state(RAD_F32, {3, 1, 1, 2}); state.f({4, 8, 0, 0, 123, 456});
    T idx(RAD_I32, {1, 2}); idx.n({0, 1});
    T cu(RAD_I32, {2}); cu.n({0, 2});
    T accepted(RAD_I32, {1}); accepted.n({1});
    T out(RAD_F32, {2, 1, 1});
    A a; a.t(q).t(k).t(v).t(g).t(b).t(alog).t(bias).t(state).t(idx).t(cu).t(accepted).tnull().t(out)
        .i("M", 2).i("heads", 1).i("head_k", 2).i("head_v", 1)
        .d("lower_bound", -std::log(16.0)).d("scale", 1).d("l2_eps", 1e-6)
        .s("mode", "verify").s("dtype", "f32");
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    /* Decays [1/4,1/8] distinguish the per-channel gate from a scalar gate. */
    CHECK_NEAR(out.at(0), 1.5f, 3e-6);
    CHECK_NEAR(out.at(1), 1.1875f, 3e-6);
    CHECK_NEAR(state.at(0), 1.5f, 3e-6);
    CHECK_NEAR(state.at(1), 1.0f, 3e-6);
    CHECK_NEAR(state.at(2), 1.1875f, 3e-6);
    CHECK_NEAR(state.at(3), 0.125f, 3e-6);
    CHECK_EQ(state.at(4), 123.0f);
    CHECK_EQ(state.at(5), 456.0f);
    /* The next step rejects candidate 1 and reads candidate 0, then writes candidate 0's slot. */
    cu.n({0, 1});
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 1.1875f, 3e-6);
    /* Accepting both candidates instead starts from slot 1. */
    accepted.n({2});
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 1.1484375f, 3e-6);
    accepted.n({3});
    CHECK_EQ(run("kda_update", a), RAD_E_SHAPE);
}

TEST(kda_prefill_commits_final_state_and_uses_absolute_cu_offsets) {
    T q(RAD_F32, {3, 1, 2}); q.f({1000, 1000, 1, 0, 1, 0});
    T k(RAD_F32, {3, 1, 2}); k.f({1000, 1000, 1, 0, 1, 0});
    T v(RAD_F32, {3, 1, 1}); v.f({1000, 2, 2});
    T g(RAD_F32, {3, 1, 2}); g.f({0, 0, 0, std::log(3.0f), 0, std::log(3.0f)});
    T b(RAD_F32, {3, 1}), alog(RAD_F32, {1}), bias(RAD_F32, {2});
    T state(RAD_F32, {2, 1, 1, 2}); state.f({4, 8, 123, 456});
    T idx(RAD_I32, {1, 1}); idx.n({0});
    T cu(RAD_I32, {2}); cu.n({1, 3});
    T init(RAD_I32, {1}); init.n({1});
    T out(RAD_F32, {3, 1, 1}); out.f({77, 77, 77});
    A a; a.t(q).t(k).t(v).t(g).t(b).t(alog).t(bias).t(state).t(idx).t(cu).tnull().t(init).t(out)
        .i("M", 3).i("heads", 1).i("head_k", 2).i("head_v", 1)
        .d("lower_bound", -std::log(16.0)).d("scale", 1).d("l2_eps", 1e-6)
        .s("mode", "prefill").s("dtype", "f32");
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    CHECK_EQ(out.at(0), 77.0f);
    CHECK_NEAR(out.at(1), 1.5f, 3e-6);
    CHECK_NEAR(out.at(2), 1.1875f, 3e-6);
    CHECK_NEAR(state.at(0), 1.1875f, 3e-6);
    CHECK_NEAR(state.at(1), 0.125f, 3e-6);
    CHECK_EQ(state.at(2), 123.0f);
    init.n({0});
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    CHECK_NEAR(out.at(1), 1.0f, 3e-6);
    CHECK_NEAR(out.at(2), 1.125f, 3e-6);
    idx.n({-1});
    REQUIRE_EQ(run("kda_update", a), RAD_OK);
    CHECK_EQ(out.at(1), 0.0f);
    CHECK_EQ(out.at(2), 0.0f);
}

/* dt = 0 with no bias gives softplus(0) = ln 2, so A_log 0 and ln 2 decay by exactly 1/2 and 1/4. */
TEST(ssd_prefill_decays_per_head_and_adds_the_skip) {
    const float l2 = std::log(2.0f);
    T x(RAD_F32, {2, 2, 1}); x.f({1, 2, 3, 4});
    T B(RAD_F32, {2, 1, 2}); B.f({1, 0, 0, 1});
    T C(RAD_F32, {2, 1, 2}); C.f({1, 1, 1, 0});
    T dt(RAD_F32, {2, 2}), dtb(RAD_F32, {2});
    T alog(RAD_F32, {2}); alog.f({0, l2});
    T D(RAD_F32, {2}); D.f({1, 0});
    T state(RAD_F32, {1, 2, 1, 2}); state.f({9, 9, 9, 9});
    T idx(RAD_I32, {1, 1}); idx.n({0});
    T cu(RAD_I32, {2}); cu.n({0, 2});
    T init(RAD_I32, {1}); init.n({0});
    T out(RAD_F32, {2, 2, 1});
    A a; a.t(x).t(B).t(C).t(dt).t(alog).t(dtb).t(D).t(state).t(idx).t(cu).tnull().t(init).t(out)
        .i("M", 2).i("heads", 2).i("head_dim", 1).i("d_state", 2).i("groups", 1)
        .s("mode", "prefill").s("dtype", "f32");
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    /* has_init 0 ignores the 9s; head 0 carries D = 1, head 1 none. */
    CHECK_NEAR(out.at(0), l2 + 1, 3e-6);
    CHECK_NEAR(out.at(1), 2 * l2, 3e-6);
    CHECK_NEAR(out.at(2), l2 / 2 + 3, 3e-6);
    CHECK_NEAR(out.at(3), l2 / 2, 3e-6);
    CHECK_NEAR(state.at(0), l2 / 2, 3e-6);
    CHECK_NEAR(state.at(1), 3 * l2, 3e-6);
    CHECK_NEAR(state.at(2), l2 / 2, 3e-6);
    CHECK_NEAR(state.at(3), 4 * l2, 3e-6);
    /* Above softplus's threshold of 20 dt passes through: e^{-25} is zero to f32 at this scale. */
    dt.f({25, 0, 0, 0});
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 25 + 1, 3e-5);
    a.ps[4].ival = 3;
    CHECK_EQ(run("ssd_update", a), RAD_E_SHAPE);
}

TEST(ssd_candidate_states_restore_the_accepted_prefix) {
    const float l2 = std::log(2.0f);
    T x(RAD_F32, {2, 1, 1}); x.f({2, 2});
    T B(RAD_F32, {2, 1, 1}); B.f({1, 1});
    T C(RAD_F32, {2, 1, 1}); C.f({1, 1});
    T dt(RAD_F32, {2, 1}), alog(RAD_F32, {1}), dtb(RAD_F32, {1}), D(RAD_F32, {1});
    T state(RAD_F32, {3, 1, 1, 1}); state.f({4, 0, 123});
    T idx(RAD_I32, {1, 2}); idx.n({0, 1});
    T cu(RAD_I32, {2}); cu.n({0, 2});
    T accepted(RAD_I32, {1}); accepted.n({1});
    T out(RAD_F32, {2, 1, 1});
    A a; a.t(x).t(B).t(C).t(dt).t(alog).t(dtb).t(D).t(state).t(idx).t(cu).t(accepted).tnull().t(out)
        .i("M", 2).i("heads", 1).i("head_dim", 1).i("d_state", 1).i("groups", 1)
        .s("mode", "verify").s("dtype", "f32");
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 2 + 2 * l2, 3e-6);
    CHECK_NEAR(out.at(1), 1 + 3 * l2, 3e-6);
    CHECK_NEAR(state.at(0), 2 + 2 * l2, 3e-6);
    CHECK_NEAR(state.at(1), 1 + 3 * l2, 3e-6);
    CHECK_EQ(state.at(2), 123.0f);
    /* The next step keeps only candidate 0 and reads its slot. */
    cu.n({0, 1});
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 1 + 3 * l2, 3e-6);
    /* Keeping both instead starts from slot 1. */
    state.f({4, 1 + 3 * l2, 123});
    accepted.n({2});
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    CHECK_NEAR(out.at(0), 0.5f * (1 + 3 * l2) + 2 * l2, 3e-6);
    accepted.n({3});
    CHECK_EQ(run("ssd_update", a), RAD_E_SHAPE);
}

/* The anchor form against the per-candidate one over two verify windows: the same outputs, and the
 * committed state after a window is the candidate state its accepted prefix left, bit for bit. */
TEST(ssd_anchor_replays_only_the_accepted_prefix_of_the_previous_window) {
    constexpr int H = 2, P = 8, N = 16, W = 3;
    for (int accepted : { 1, 2, 3 }) {
        T x(RAD_F32, { W, H, P }), B(RAD_F32, { W, 1, N }), C(RAD_F32, { W, 1, N });
        T dt(RAD_F32, { W, H }), al(RAD_F32, { H }), dtb(RAD_F32, { H }), D(RAD_F32, { H });
        T anchor(RAD_F32, { 2, H, P, N }), candidate(RAD_F32, { W, H, P, N });
        T ai(RAD_I32, { 1, 6 }), ci(RAD_I32, { 1, W }), cu(RAD_I32, { 2 }), na(RAD_I32, { 1 }),
            init(RAD_I32, { 1 });
        T ao(RAD_F32, { W, H, P }), co(RAD_F32, { W, H, P });
        std::vector<float> v(W * H * P);
        for (size_t j = 0; j < v.size(); ++j) v[j] = (float)((int)(j % 13) - 6) * 0.125f;
        x.f(v);
        std::vector<float> bc(W * N);
        for (size_t j = 0; j < bc.size(); ++j) bc[j] = (float)((int)(j % 7) - 3) * 0.25f;
        B.f(bc);
        for (size_t j = 0; j < bc.size(); ++j) bc[j] = (float)((int)(j % 5) - 2) * 0.5f;
        C.f(bc);
        dt.f({ 0.5f, -1, 0, 1, -0.5f, 2 });
        al.f({ 0, -0.5f });
        dtb.f({ 0.25f, -0.25f });
        D.f({ 1, 0.5f });
        anchor.f(std::vector<float>(2 * H * P * N, 0));
        candidate.f(std::vector<float>(W * H * P * N, 0));
        ai.n({ 0, 0, 0, 1, 1, 1 });
        ci.n({ 0, 1, 2 });
        cu.n({ 0, W });
        na.n({ 1 });
        init.n({ 1 });
        auto args = [&](T& state, T& idx, T& out, bool anc) {
            A a;
            a.t(x).t(B).t(C).t(dt).t(al).t(dtb).t(D).t(state).t(idx).t(cu).t(na).t(init).t(out)
                .i("M", W).i("heads", H).i("head_dim", P).i("d_state", N).i("groups", 1)
                .s("mode", "verify").s("dtype", "f32");
            if (anc) a.s("state_form", "anchor");
            return a;
        };
        A a = args(anchor, ai, ao, true), c = args(candidate, ci, co, false);
        REQUIRE_EQ(run("ssd_update", a), RAD_OK);
        REQUIRE_EQ(run("ssd_update", c), RAD_OK);
        for (int j = 0; j < W * H * P; ++j) CHECK_EQ(ao.at(j), co.at(j));
        /* Nothing is committed before the next step says how much was accepted. */
        for (int j = 0; j < H * P * N; ++j) CHECK_EQ(anchor.at(j), 0);
        std::vector<float> committed(H * P * N);
        for (int j = 0; j < H * P * N; ++j)
            committed[j] = candidate.at((accepted - 1) * H * P * N + j);
        na.n({ accepted });
        REQUIRE_EQ(run("ssd_update", a), RAD_OK);
        REQUIRE_EQ(run("ssd_update", c), RAD_OK);
        for (int j = 0; j < W * H * P; ++j) CHECK_EQ(ao.at(j), co.at(j));
        for (int j = 0; j < H * P * N; ++j) CHECK_EQ(anchor.at(j), committed[j]);
        const auto saved = anchor.mem;
        na.n({ 4 });
        CHECK_EQ(run("ssd_update", a), RAD_E_SHAPE);
        CHECK(anchor.mem == saved);
    }
}

/* Head h reads group h / (heads / groups): heads 0 and 1 share group 0, not heads 0 and 2. */
TEST(ssd_heads_read_their_group_in_blocks) {
    const float l2 = std::log(2.0f);
    T x(RAD_F32, {1, 4, 1}); x.f({1, 1, 1, 1});
    T B(RAD_F32, {1, 2, 2}); B.f({1, 0, 0, 1});
    T C(RAD_F32, {1, 2, 2}); C.f({1, 0, 0, 2});
    T dt(RAD_F32, {1, 4}), alog(RAD_F32, {4}), dtb(RAD_F32, {4}), D(RAD_F32, {4});
    T state(RAD_F32, {1, 4, 1, 2});
    T idx(RAD_I32, {1, 1}); idx.n({0});
    T cu(RAD_I32, {2}); cu.n({0, 1});
    T out(RAD_F32, {1, 4, 1});
    A a; a.t(x).t(B).t(C).t(dt).t(alog).t(dtb).t(D).t(state).t(idx).t(cu).tnull().tnull().t(out)
        .i("M", 1).i("heads", 4).i("head_dim", 1).i("d_state", 2).i("groups", 2)
        .s("mode", "prefill").s("dtype", "f32");
    REQUIRE_EQ(run("ssd_update", a), RAD_OK);
    const float want[] = {l2, l2, 2 * l2, 2 * l2};
    for (int h = 0; h < 4; ++h) CHECK_NEAR(out.at(h), want[h], 3e-6);
}

TEST(mla_attention_reads_one_latent_and_masks_the_last_selected_page) {
    T q(RAD_F32, { 1, 1, 2 }), kv(RAD_F32, { 2, 1, 2, 2 }), bt(RAD_I32, { 1, 2 });
    T len(RAD_I32, { 1 }), out(RAD_F32, { 1, 1, 2 });
    q.f({ 0, 0 });
    kv.f({ 3, 5, 1000, -1000, 1, 0, 0, 1 });
    bt.n({ 1, 0 });
    len.n({ 3 });
    A a;
    a.t(q)
        .t(kv)
        .t(bt)
        .t(len)
        .t(out)
        .i("M", 7)
        .i("latent_dim", 2)
        .i("heads", 1)
        .i("blocks", 2)
        .i("block_size", 2)
        .d("scale", 1)
        .s("dtype", "f32");
    REQUIRE_EQ(run("mla_attn", a), RAD_OK);
    CHECK_NEAR(out.at(0), 4.0 / 3, 1e-6);
    CHECK_EQ(out.at(1), 2);
    q.f({ std::log(3.0f), 0 });
    kv.f({ 0, 0, 1000, -1000, 1, 0, 0, 0 });
    len.n({ 2 });
    REQUIRE_EQ(run("mla_attn", a), RAD_OK);
    CHECK_NEAR(out.at(0), 0.75, 1e-6);
    CHECK_EQ(out.at(1), 0);
    len.n({ 0 });
    REQUIRE_EQ(run("mla_attn", a), RAD_OK);
    CHECK_EQ(out.at(0), 0);
    CHECK_EQ(out.at(1), 0);
    len.n({ 3 });
    bt.n({ -1, 99 });
    REQUIRE_EQ(run("mla_attn", a), RAD_OK);
    CHECK_EQ(out.at(0), 0);
    CHECK_EQ(out.at(1), 0);
}

/* An E4M3 cache (kv_dtype fp8_e4m3, kv_store_fp8 at descale 1) is read as the codes' values: the
 * first case above with the latent as E4M3 codes and a bf16 query and output. */
TEST(mla_attention_reads_an_e4m3_latent_cache_as_its_values) {
    T q(RAD_BF16, { 1, 1, 2 }), kv(RAD_F8E4M3, { 2, 1, 2, 2 }), bt(RAD_I32, { 1, 2 });
    T len(RAD_I32, { 1 }), out(RAD_BF16, { 1, 1, 2 });
    q.f({ 0, 0 });
    kv.f({ 3, 5, 448, -448, 1, 0, 0, 1 });
    bt.n({ 1, 0 });
    len.n({ 3 });
    A a;
    a.t(q)
        .t(kv)
        .t(bt)
        .t(len)
        .t(out)
        .i("M", 7)
        .i("latent_dim", 2)
        .i("heads", 1)
        .i("blocks", 2)
        .i("block_size", 2)
        .d("scale", 1)
        .s("dtype", "bf16")
        .s("kv_dtype", "fp8_e4m3");
    REQUIRE_EQ(run("mla_attn", a), RAD_OK);
    CHECK_NEAR(out.at(0), 4.0 / 3, 1e-2);
    CHECK_EQ(out.at(1), 2);
}

TEST(dsa_score_relu_is_per_head_before_signed_weights_and_masks_invalid_history) {
    T q(RAD_F8E4M3, { 2, 2, 2 }), qs(RAD_F32, { 2, 2 }), gate(RAD_F32, { 2, 2 });
    T k(RAD_F8E4M3, { 3, 2 }), ks(RAD_F32, { 3 }), bt(RAD_I32, { 2, 3 });
    T cu(RAD_I32, { 3 }), nc(RAD_I32, { 2 }), out(RAD_F32, { 2, 3 });
    q.f({ 1, 0, 0, 1, -1, 0, 0, -1 });
    qs.f({ 2, 0.5, 1, 1 });
    gate.f({ -1, 4, 2, -1 });
    k.f({ 1, -1, 2, 3, -2, -4 });
    ks.f({ 0.25, 0.5, 1 });
    bt.n({ 0, 1, -1, 2, 0, 1 });
    cu.n({ 0, 1, 2 });
    nc.n({ 2, 1 });
    A a;
    a.t(q)
        .t(qs)
        .t(gate)
        .t(k)
        .t(ks)
        .t(bt)
        .t(cu)
        .t(nc)
        .t(out)
        .i("M", 9)
        .i("n", 2)
        .i("heads", 2)
        .i("blocks", 3)
        .i("seqs", 2)
        .s("dtype", "fp8_e4m3");
    REQUIRE_EQ(run("dsa_score", a), RAD_OK);
    CHECK_EQ(out.at(0), -0.25f);
    CHECK_EQ(out.at(1), 0.5f);
    CHECK_EQ(out.at(3), 0);
    for (int i : { 2, 4, 5 }) CHECK_EQ(out.at(i), -INFINITY);
    bt.n({ 99, 1, -1, 2, 0, 1 });
    REQUIRE_EQ(run("dsa_score", a), RAD_OK);
    CHECK_EQ(out.at(0), -INFINITY);
}

TEST(dsa_score_row_part_scores_its_run_of_rows_and_rebases_cu) {
    T q(RAD_F8E4M3, { 2, 2, 2 }), qs(RAD_F32, { 2, 2 }), gate(RAD_F32, { 2, 2 });
    T k(RAD_F8E4M3, { 3, 2 }), ks(RAD_F32, { 3 }), bt(RAD_I32, { 2, 3 });
    T cu(RAD_I32, { 3 }), nc(RAD_I32, { 2 }), out(RAD_F32, { 1, 3 }), cu_part(RAD_I32, { 3 });
    q.f({ 1, 0, 0, 1, -1, 0, 0, -1 });
    qs.f({ 2, 0.5, 1, 1 });
    gate.f({ -1, 4, 2, -1 });
    k.f({ 1, -1, 2, 3, -2, -4 });
    ks.f({ 0.25, 0.5, 1 });
    bt.n({ 0, 1, -1, 2, 0, 1 });
    cu.n({ 0, 1, 2 });
    nc.n({ 2, 1 });
    A a;
    a.t(q).t(qs).t(gate).t(k).t(ks).t(bt).t(cu).t(nc).t(out).t(cu_part);
    a.i("M", 1).i("n", 2).i("heads", 2).i("blocks", 3).i("seqs", 2).s("dtype", "fp8_e4m3");
    a.i("row_part", 1);
    REQUIRE_EQ(run("dsa_score", a), RAD_OK);
    CHECK_EQ(out.at(0), 0);
    CHECK_EQ(out.at(1), -INFINITY);
    CHECK_EQ(out.at(2), -INFINITY);
    CHECK_EQ(cu_part.at(0), -1);
    CHECK_EQ(cu_part.at(1), 0);
    CHECK_EQ(cu_part.at(2), 1);
    /* Part 2 of one-row runs is past the step: no sequence holds its row. */
    T q3(RAD_F8E4M3, { 3, 2, 2 }), qs3(RAD_F32, { 3, 2 }), gate3(RAD_F32, { 3, 2 });
    q3.f({ 1, 0, 0, 1, -1, 0, 0, -1, 1, 1, 1, 1 });
    qs3.f({ 2, 0.5, 1, 1, 1, 1 });
    gate3.f({ -1, 4, 2, -1, 1, 1 });
    A past;
    past.t(q3).t(qs3).t(gate3).t(k).t(ks).t(bt).t(cu).t(nc).t(out).t(cu_part);
    past.i("M", 1).i("n", 2).i("heads", 2).i("blocks", 3).i("seqs", 2).s("dtype", "fp8_e4m3");
    past.i("row_part", 2);
    REQUIRE_EQ(run("dsa_score", past), RAD_OK);
    for (int i : { 0, 1, 2 }) CHECK_EQ(out.at(i), -INFINITY);
    CHECK_EQ(cu_part.at(2), 0);
    /* A part needs query rows for its whole run. */
    A short_q;
    short_q.t(q).t(qs).t(gate).t(k).t(ks).t(bt).t(cu).t(nc).t(out).t(cu_part);
    short_q.i("M", 1).i("n", 2).i("heads", 2).i("blocks", 3).i("seqs", 2).s("dtype", "fp8_e4m3");
    short_q.i("row_part", 2);
    CHECK_EQ(run("dsa_score", short_q), RAD_E_SHAPE);
}

TEST(kda_anchor_replays_only_the_accepted_prefix_of_the_previous_window) {
    constexpr int D = 128;
    for (int accepted : { 1, 2, 3 }) {
        T q(RAD_BF16, { 3, 1, D }), k(RAD_BF16, { 3, 1, D }), v(RAD_BF16, { 3, 1, D }),
            g(RAD_BF16, { 3, 1, D });
        T beta(RAD_BF16, { 3, 1 }), al(RAD_F32, { 1 }), bias(RAD_F32, { D });
        T anchor(RAD_F32, { 2, 1, D, D }), candidate(RAD_F32, { 3, 1, D, D });
        T ai(RAD_I32, { 1, 6 }), ci(RAD_I32, { 1, 3 }), cu(RAD_I32, { 2 }), na(RAD_I32, { 1 }),
            init(RAD_I32, { 1 });
        T ao(RAD_BF16, { 3, 1, D }), co(RAD_BF16, { 3, 1, D });
        std::vector<float> values(3 * D);
        for (int j = 0; j < 3 * D; ++j) values[j] = (float)(j % 17 - 8) * 0.125f;
        q.f(values);
        v.f(values);
        for (int j = 0; j < 3 * D; ++j) values[j] = (float)(j % 11 - 5) * 0.25f;
        k.f(values);
        g.f(std::vector<float>(3 * D, 0));
        beta.f({ 0, 0.5f, -0.5f });
        al.f({ 0 });
        bias.f(std::vector<float>(D, -1));
        anchor.f(std::vector<float>(2 * D * D, 0));
        candidate.f(std::vector<float>(3 * D * D, 0));
        ai.n({ 0, 0, 0, 1, 1, 1 });
        ci.n({ 0, 1, 2 });
        cu.n({ 0, 3 });
        na.n({ 1 });
        init.n({ 1 });
        A a;
        a.t(q)
            .t(k)
            .t(v)
            .t(g)
            .t(beta)
            .t(al)
            .t(bias)
            .t(anchor)
            .t(ai)
            .t(cu)
            .t(na)
            .t(init)
            .t(ao)
            .i("M", 3)
            .i("heads", 1)
            .i("head_k", D)
            .i("head_v", D)
            .d("lower_bound", -5)
            .d("scale", 1.0 / std::sqrt(128.0))
            .d("l2_eps", 1e-6)
            .s("mode", "verify")
            .s("dtype", "bf16")
            .s("state_form", "anchor");
        A c;
        c.t(q)
            .t(k)
            .t(v)
            .t(g)
            .t(beta)
            .t(al)
            .t(bias)
            .t(candidate)
            .t(ci)
            .t(cu)
            .t(na)
            .t(init)
            .t(co)
            .i("M", 3)
            .i("heads", 1)
            .i("head_k", D)
            .i("head_v", D)
            .d("lower_bound", -5)
            .d("scale", 1.0 / std::sqrt(128.0))
            .d("l2_eps", 1e-6)
            .s("mode", "verify")
            .s("dtype", "bf16");
        REQUIRE_EQ(run("kda_update", a), RAD_OK);
        REQUIRE_EQ(run("kda_update", c), RAD_OK);
        for (int j = 0; j < 3 * D; ++j) CHECK_EQ(ao.at(j), co.at(j));
        for (int j = 0; j < D * D; ++j) CHECK_EQ(anchor.at(j), 0);
        std::vector<float> committed(D * D);
        for (int j = 0; j < D * D; ++j) committed[j] = candidate.at((accepted - 1) * D * D + j);
        na.n({ accepted });
        REQUIRE_EQ(run("kda_update", a), RAD_OK);
        REQUIRE_EQ(run("kda_update", c), RAD_OK);
        for (int j = 0; j < 3 * D; ++j) CHECK_EQ(ao.at(j), co.at(j));
        for (int j = 0; j < D * D; ++j) CHECK_EQ(anchor.at(j), committed[j]);
        const auto saved = anchor.mem;
        na.n({ 4 });
        CHECK_EQ(run("kda_update", a), RAD_E_SHAPE);
        CHECK(anchor.mem == saved);
    }
}

TEST(mla_projects_the_key_and_value_halves_with_opposite_orientations) {
    T q(RAD_BF16, { 1, 2, 2 }), w(RAD_BF16, { 8, 3 }), latent(RAD_BF16, { 1, 2, 3 }),
        out(RAD_BF16, { 1, 2, 2 });
    q.f({ 2, -1, 3, 1 });
    w.f({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 0, 1, 2, 0, -2, 2, 3, 4, 5, 6, 7 });
    A a;
    a.t(q)
        .t(w)
        .t(latent)
        .i("M", 7)
        .i("heads", 2)
        .i("head_dim", 2)
        .i("latent_dim", 3)
        .s("mode", "absorb")
        .s("dtype", "bf16");
    REQUIRE_EQ(run("mla_project", a), RAD_OK);
    CHECK_EQ(latent.at(0), -2);
    CHECK_EQ(latent.at(1), -1);
    CHECK_EQ(latent.at(2), 0);
    CHECK_EQ(latent.at(3), -1);
    CHECK_EQ(latent.at(4), 0);
    CHECK_EQ(latent.at(5), 1);
    A b;
    b.t(latent)
        .t(w)
        .t(out)
        .i("M", 7)
        .i("heads", 2)
        .i("head_dim", 2)
        .i("latent_dim", 3)
        .s("mode", "expand")
        .s("dtype", "bf16");
    REQUIRE_EQ(run("mla_project", b), RAD_OK);
    CHECK_EQ(out.at(0), -22);
    CHECK_EQ(out.at(1), -31);
    CHECK_EQ(out.at(2), 2);
    CHECK_EQ(out.at(3), 2);
    auto flatten = [](RadTensor& t) {
        t.rank = 2;
        t.shape[1] *= t.shape[2];
        t.stride[1] = t.stride[2];
    };
    flatten(a.ts[0]);
    flatten(a.ts[2]);
    REQUIRE_EQ(run("mla_project", a), RAD_OK);
    flatten(b.ts[0]);
    flatten(b.ts[2]);
    REQUIRE_EQ(run("mla_project", b), RAD_OK);
    CHECK_EQ(out.at(0), -22);
    CHECK_EQ(out.at(1), -31);
    CHECK_EQ(out.at(2), 2);
    CHECK_EQ(out.at(3), 2);
}

TEST(moe_gather_preserves_f32_mixture_weights_between_bf16_expert_outputs) {
    T experts(RAD_BF16, { 2, 1 }), weights(RAD_F32, { 1, 2 }), sorted(RAD_I32, { 2 }),
        out(RAD_BF16, { 1, 1 });
    experts.f({ 1024, -1024 });
    weights.f({ 1.0f + 1.0f / 8192, 1 });
    sorted.n({ 0, 1 });
    A a;
    a.t(experts)
        .t(weights)
        .t(sorted)
        .tnull()
        .tnull()
        .t(out)
        .i("M", 1)
        .i("n", 1)
        .i("top_k", 2)
        .s("dtype", "bf16");
    REQUIRE_EQ(run("moe_gather", a), RAD_OK);
    CHECK_EQ(out.at(0), 0.125f);
}

TEST(dsa_query_normalizes_the_rotation_and_rounds_power_of_two_scales) {
    T x(RAD_BF16, { 1, 4 }), q(RAD_F8E4M3, { 1, 4 }), scale(RAD_F32, { 1, 1 });
    A a;
    a.t(x).t(q).t(scale).i("M", 7).i("n", 4).s("dtype", "bf16");
    x.f({ 3, 4, 7, 7 });
    REQUIRE_EQ(run("dsa_query", a), RAD_OK);
    CHECK_EQ(scale.at(0), 1.0f / 32);
    CHECK_EQ(q.at(0), 320);
    CHECK_EQ(q.at(1), -16);
    CHECK_EQ(q.at(2), -112);
    CHECK_EQ(q.at(3), -16);
    for (float v : { 0.0f, 224.0f, 225.0f }) {
        x.f({ v, v, v, v });
        REQUIRE_EQ(run("dsa_query", a), RAD_OK);
        CHECK_EQ(scale.at(0), v == 0 ? std::ldexp(1.0f, -22) : v == 224 ? 1.0f : 2.0f);
        CHECK_EQ(q.at(0), v == 0 ? 0 : v == 224 ? 448 : 224);
        for (int d = 1; d < 4; ++d) CHECK_EQ(q.at(d), 0);
    }
}

TEST(f32_projection_preserves_small_weight_differences_and_output_bits) {
    T x(RAD_BF16, { 1, 2 }), w(RAD_F32, { 2, 2 }), y(RAD_F32, { 1, 2 });
    x.f({ 1, -1 });
    w.f({ 1.0001f, 1.0f, 2.0002f, 2.0f });
    A a;
    a.t(x).t(w).t(y).i("M", 7).i("N", 2).i("K", 2).s("dtype", "bf16");
    REQUIRE_EQ(run("gemm_nt_f32", a), RAD_OK);
    CHECK_EQ(y.at(0), 1.0001f - 1.0f);
    CHECK_EQ(y.at(1), 2.0002f - 2.0f);
}

TEST(dsa_pool_writes_paged_codes_and_scales_and_the_scorer_reads_them) {
    T kg(RAD_BF16, { 12, 256 }), ape(RAD_F32, { 4, 128 }), q(RAD_F8E4M3, { 3, 128 }),
        scales(RAD_F32, { 3, 1 });
    T pages(RAD_I32, { 3 }), cache(RAD_U8, { 4, 1, 4, 34 });
    std::vector<float> input(12 * 256, 0);
    for (int r = 0; r < 12; ++r)
        for (int d = 0; d < 128; ++d) input[r * 256 + d] = 1;
    kg.f(input);
    ape.f(std::vector<float>(512, 0));
    pages.n({ 2, -1, 9 });
    std::fill(cache.mem.begin(), cache.mem.end(), 0xa5);
    A pool;
    pool.t(kg)
        .t(ape)
        .t(q)
        .t(scales)
        .t(pages)
        .t(cache)
        .i("M", 3)
        .i("n", 128)
        .i("ratio", 4)
        .s("dtype", "bf16");
    REQUIRE_EQ(run("dsa_pool", pool), RAD_OK);
    for (int i = 0; i < 4 * 136; ++i)
        if (i >= 2 * 136 && i < 2 * 136 + 128)
            CHECK_EQ(cache.mem[i], q.mem[i - 2 * 136]);
        else if (i < 2 * 136 + 128 || i >= 2 * 136 + 132)
            CHECK_EQ(cache.mem[i], 0xa5);
    float stored = 0;
    std::memcpy(&stored, cache.mem.data() + 2 * 136 + 128, sizeof(stored));
    CHECK_EQ(stored, scales.at(0));
    T query(RAD_F8E4M3, { 1, 32, 128 }), qs(RAD_F32, { 1, 32 }), gate(RAD_F32, { 1, 32 });
    query.f(std::vector<float>(32 * 128, 1));
    qs.f(std::vector<float>(32, 1));
    gate.f(std::vector<float>(32, 1));
    T bt(RAD_I32, { 1, 3 }), cu(RAD_I32, { 2 }), nc(RAD_I32, { 1 }), out(RAD_F32, { 1, 3 });
    bt.n({ 2, -1, 9 });
    cu.n({ 0, 1 });
    nc.n({ 3 });
    A score;
    score.t(query)
        .t(qs)
        .t(gate)
        .t(cache)
        .tnull()
        .t(bt)
        .t(cu)
        .t(nc)
        .t(out)
        .i("M", 1)
        .i("n", 128)
        .i("heads", 32)
        .i("blocks", 3)
        .i("seqs", 1)
        .s("dtype", "fp8_e4m3");
    REQUIRE_EQ(run("dsa_score", score), RAD_OK);
    CHECK_EQ(out.at(0), 5.5f);
    CHECK(std::isinf(out.at(1)) && out.at(1) < 0);
    CHECK(std::isinf(out.at(2)) && out.at(2) < 0);
}

TEST(dsa_pool_uses_channel_gates_bf16_boundaries_and_power_of_two_scales) {
    T kg(RAD_BF16, { 2, 8 }), ape(RAD_F32, { 2, 4 }), q(RAD_F8E4M3, { 1, 4 }),
        scale(RAD_F32, { 1, 1 });
    kg.f({ 0, 2, 4, 6, 0, 0, 0, 0, 4, 6, 8, 10, 0, 0, 0, 0 });
    ape.f({ 0, 0, 0, 0, std::log(3.0f), 0, std::log(3.0f), -std::log(3.0f) });
    A a;
    a.t(kg).t(ape).t(q).t(scale).i("M", 7).i("n", 4).i("ratio", 2).s("dtype", "bf16");
    REQUIRE_EQ(run("dsa_pool", a), RAD_OK);
    // Pooled [3,4,7,7]; H4/sqrt(4) = [10.5,-.5,-3.5,-.5].
    // 10.5*32=336 is the FP8 midpoint 320/352, rounded to even 320.
    CHECK_EQ(scale.at(0), 1.0f / 32);
    CHECK_EQ(q.at(0), 320);
    CHECK_EQ(q.at(1), -16);
    CHECK_EQ(q.at(2), -112);
    CHECK_EQ(q.at(3), -16);
    ape.f(std::vector<float>(8, 0));
    for (float key : { 0.0f, 224.0f, 225.0f }) {
        kg.f({ key, key, key, key, 0, 0, 0, 0, key, key, key, key, 0, 0, 0, 0 });
        REQUIRE_EQ(run("dsa_pool", a), RAD_OK);
        CHECK_EQ(scale.at(0), key == 0 ? std::ldexp(1.0f, -22) : key == 224 ? 1.0f : 2.0f);
        CHECK_EQ(q.at(0), key == 0 ? 0 : key == 224 ? 448 : 224);
        for (int i = 1; i < 4; ++i) CHECK_EQ(q.at(i), 0);
    }
}

TEST(swiglu_clamping_is_asymmetric_and_precedes_the_activation) {
    T gu(RAD_F32, { 1, 12 }), y(RAD_F32, { 1, 6 });
    gu.f({ -20, -2, 0, 2, 10, 20, -20, 20, 20, -20, 20, 20 });
    A a;
    a.t(gu).t(y).i("M", 9).i("n", 6).d("limit", 10).s("dtype", "f32");
    REQUIRE_EQ(run("swiglu_clamped", a), RAD_OK);
    const float gate[] = { -20, -2, 0, 2, 10, 10 }, up[] = { -10, 10, 10, -10, 10, 10 };
    for (int i = 0; i < 6; ++i)
        CHECK_NEAR(y.at(i), (gate[i] / (1 + std::exp(-gate[i]))) * up[i], 1e-5);
    CHECK(y.at(0) > 0 && y.at(0) < 1e-5); // a negative gate must not clamp to -10
    CHECK_EQ(y.at(4), y.at(5));
    A bad;
    bad.t(gu).t(y).i("M", 1).i("n", 6).d("limit", 0).s("dtype", "f32");
    CHECK_EQ(run("swiglu_clamped", bad), RAD_E_SHAPE);
}

TEST(swiglu_step_clips_the_activation_not_the_gate) {
    T gu(RAD_F32, { 1, 8 }), y(RAD_F32, { 1, 4 });
    gu.f({ -20, 2, 4.02f, 20, 9, 9, 9, -9 });
    A a;
    a.t(gu).t(y).i("M", 9).i("n", 4).d("limit", 4).s("dtype", "f32");
    REQUIRE_EQ(run("swiglu_step", a), RAD_OK);
    auto silu = [](float g) { return g / (1 + std::exp(-g)); };
    CHECK_NEAR(y.at(0), silu(-20) * 4, 1e-6);   // a negative activation is not clipped
    CHECK_NEAR(y.at(1), silu(2) * 4, 1e-5);
    CHECK_NEAR(y.at(2), silu(4.02f) * 4, 1e-5); // silu(4.02) is under 4: unclipped
    CHECK_EQ(y.at(3), -16.0f);                  // silu(20) clips to exactly 4, up to -4
    T gc(RAD_F32, { 1, 4 });
    A c;
    c.t(gu).t(gc).i("M", 9).i("n", 4).d("limit", 4).s("dtype", "f32");
    REQUIRE_EQ(run("swiglu_clamped", c), RAD_OK);
    CHECK(gc.at(3) > y.at(3));                  // the gate-first form stops at silu(4) * -4
}

TEST(group_router_selects_only_inside_the_best_groups) {
    T lg(RAD_F32, { 1, 8 }), bias(RAD_F32, { 8 }), ids(RAD_I32, { 1, 2 }), w(RAD_F32, { 1, 2 });
    auto logit = [](float p) { return std::log(p / (1 - p)); };
    // Group sums of the top two: 1.0, 1.5, 0.86, 1.2 -> groups 1 and 3 are kept, so the single
    // best experts (0 at 0.9, 4 at 0.85) are out and 2 and 3 are chosen.
    lg.f({ logit(0.9f), logit(0.1f), logit(0.8f), logit(0.7f), logit(0.85f), logit(0.01f),
           logit(0.6f), logit(0.6f) });
    bias.f({ 0, 0, 0, 0, 0, 0, 0, 0 });
    A a;
    a.t(lg).t(bias).t(ids).t(w).i("M", 1).i("n_expert", 8).i("top_k", 2).i("n_group", 4)
        .i("topk_group", 2).i("norm", 1).d("route_scale", 2.5).s("dtype", "f32");
    REQUIRE_EQ(run("router_sigmoid_group_topk", a), RAD_OK);
    CHECK_EQ(ids.at(0), 2);
    CHECK_EQ(ids.at(1), 3);
    CHECK_NEAR(w.at(0), 2.5 * 0.8 / 1.5, 1e-5);
    CHECK_NEAR(w.at(1), 2.5 * 0.7 / 1.5, 1e-5);
    // The bias ranks groups and experts and never weights them.
    bias.f({ 0, 0, 0, 0, 0, 0, 1, 0 });
    REQUIRE_EQ(run("router_sigmoid_group_topk", a), RAD_OK);
    CHECK_EQ(ids.at(0), 6);
    CHECK_EQ(ids.at(1), 2);
    CHECK_NEAR(w.at(0), 2.5 * 0.6 / 1.4, 1e-5);
    A bad;
    bad.t(lg).t(bias).t(ids).t(w).i("M", 1).i("n_expert", 8).i("top_k", 2).i("n_group", 3)
        .i("topk_group", 2).i("norm", 1).d("route_scale", 2.5).s("dtype", "f32");
    CHECK_EQ(run("router_sigmoid_group_topk", bad), RAD_E_SHAPE);
}

TEST(paged_latent_attention_is_causal_adds_the_rotary_part_and_gates_per_head) {
    // One sequence of three tokens, the last two queried (positions 1 and 2). Token 0 and 1 sit
    // in page 1, token 2 in page 0; a row is the latent (2) then the rotary part (2).
    T q(RAD_F32, { 2, 1, 2 }), qr(RAD_F32, { 2, 1, 2 }), kv(RAD_F32, { 2, 1, 2, 4 });
    T bt(RAD_I32, { 1, 2 }), su(RAD_I32, { 1 }), cu(RAD_I32, { 2 }), out(RAD_F32, { 2, 1, 2 });
    q.f({ 0, 0, 0, 0 });
    qr.f({ 0, 0, 0, 0 });
    kv.f({ 5, 6, 0, 0, 77, 77, 77, 77, 1, 2, 1, 0, 3, 4, 0, 0 });
    bt.n({ 1, 0 });
    su.n({ 3 });
    cu.n({ 0, 2 });
    A a;
    a.t(q).t(qr).t(kv).t(bt).t(su).t(cu).tnull().t(out).i("M", 9).i("heads", 1)
        .i("latent_dim", 2).i("rope_dim", 2).i("block_size", 2).d("scale", 1).s("dtype", "f32");
    REQUIRE_EQ(run("mla_attn_paged", a), RAD_OK);
    CHECK_NEAR(out.at(0), 2, 1e-6);   // position 1 sees tokens 0 and 1
    CHECK_NEAR(out.at(1), 3, 1e-6);
    CHECK_NEAR(out.at(2), 3, 1e-6);   // position 2 sees all three
    CHECK_NEAR(out.at(3), 4, 1e-6);
    // The rotary part alone scores token 0 at ln 3 and token 1 at 0: weights 3/4 and 1/4.
    qr.f({ std::log(3.0f), 0, 0, 0 });
    REQUIRE_EQ(run("mla_attn_paged", a), RAD_OK);
    CHECK_NEAR(out.at(0), 1.5, 1e-6);
    CHECK_NEAR(out.at(1), 2.5, 1e-6);
    // A gate logit of 0 halves the head.
    T gate(RAD_F32, { 2, 1 });
    gate.f({ 0, 0 });
    A g;
    g.t(q).t(qr).t(kv).t(bt).t(su).t(cu).t(gate).t(out).i("M", 9).i("heads", 1)
        .i("latent_dim", 2).i("rope_dim", 2).i("block_size", 2).d("scale", 1).s("dtype", "f32");
    REQUIRE_EQ(run("mla_attn_paged", g), RAD_OK);
    CHECK_NEAR(out.at(0), 0.75, 1e-6);
    CHECK_NEAR(out.at(1), 1.25, 1e-6);
    CHECK_NEAR(out.at(2), 1.5, 1e-6);
}

TEST(sigmoid_router_uses_correction_only_for_selection_and_stable_ties) {
    T lg(RAD_F32, { 1, 4 }), bias(RAD_F32, { 4 }), ids(RAD_I32, { 1, 2 }), w(RAD_F32, { 1, 2 });
    lg.f({ -std::log(3.0f), std::log(3.0f), 0, 0 });
    bias.f({ 1, -1, 0, 0 });
    for (int norm : { 0, 1 }) {
        A a;
        a.t(lg)
            .t(bias)
            .t(ids)
            .t(w)
            .i("M", 9)
            .i("n_expert", 4)
            .i("top_k", 2)
            .i("norm", norm)
            .d("route_scale", 2.5)
            .s("dtype", "f32");
        REQUIRE_EQ(run("router_sigmoid_topk", a), RAD_OK);
        CHECK_EQ(ids.at(0), 0);
        CHECK_EQ(ids.at(1), 2);
        CHECK_NEAR(w.at(0), norm ? 2.5 / 3 : 0.625, 1e-6);
        CHECK_NEAR(w.at(1), norm ? 5.0 / 3 : 1.25, 1e-6);
    }
    lg.f({ 0, 0, 0, 0 });
    bias.f({ 0, 0, 0, 0 });
    A tie;
    tie.t(lg)
        .t(bias)
        .t(ids)
        .t(w)
        .i("M", 1)
        .i("n_expert", 4)
        .i("top_k", 2)
        .i("norm", 1)
        .d("route_scale", 2.5)
        .s("dtype", "f32");
    REQUIRE_EQ(run("router_sigmoid_topk", tie), RAD_OK);
    CHECK_EQ(ids.at(0), 0);
    CHECK_EQ(ids.at(1), 1);
    CHECK_EQ(w.at(0), 1.25f);
    CHECK_EQ(w.at(1), 1.25f);
}

TEST(grouped_mxfp4_preserves_e2m1_codes_and_e8m0_scales) {
    T input(RAD_F8E4M3, { 2, 64 });
    std::vector<float> values(128, 1.0f);
    for (int i = 64; i < 128; ++i) values[(size_t)i] = 2;
    input.f(values);
    T as(RAD_F32, { 2, 1 });
    as.f({ 0.5f, 0.5f });
    T even(RAD_FP4E2M1, { 1, 1, 64 });
    even.raw(std::vector<uint8_t>(32, 0x22));
    T odd(RAD_FP4E2M1, { 1, 1, 64 });
    odd.raw(std::vector<uint8_t>(32, 0x33));
    T es(RAD_U8, { 1, 1, 2 });
    es.raw({ 127, 128 });
    T os(RAD_U8, { 1, 1, 2 });
    os.raw({ 126, 127 });
    T sorted(RAD_I32, { 2 });
    sorted.n({ 1, 0 });
    T offsets(RAD_I32, { 3 });
    offsets.n({ 0, 1, 2 });
    T out(RAD_BF16, { 2, 1 });
    A a;
    a.t(input)
        .t(as)
        .t(even)
        .t(es)
        .t(sorted)
        .t(offsets)
        .t(odd)
        .t(os)
        .t(out)
        .i("M", 2)
        .i("N", 1)
        .i("K", 64)
        .i("n_expert", 2)
        .i("top_k", 1)
        .i("group", 64)
        .s("a_order", "token")
        .s("dtype", "mxfp4a8");
    REQUIRE_EQ(run("moe_gemm_q", a), RAD_OK);
    /* E2M1 0x2=1, 0x3=1.5. The second half of each row has its own power-of-two scale. */
    CHECK_EQ(out.at(0), 96.0f);
    CHECK_EQ(out.at(1), 36.0f);
    a.strs.emplace_back("sorted");
    a.ps[6].sval = a.strs.back().c_str();
    REQUIRE_EQ(run("moe_gemm_q", a), RAD_OK);
    CHECK_EQ(out.at(0), 48.0f);
    CHECK_EQ(out.at(1), 72.0f);
    /* Each expert may have a separate address, including host-resident weights. */
    void* wp0[] = { even.t.data };
    void* wp1[] = { odd.t.data };
    void* sp0[] = { es.t.data };
    void* sp1[] = { os.t.data };
    const int operands[] = { 2, 3, 6, 7 };
    void* tables[] = { wp0, sp0, wp1, sp1 };
    for (int i = 0; i < 4; ++i) {
        a.ts[(size_t)operands[i]].data = tables[i];
        a.ts[(size_t)operands[i]].stride[0] = 0;
    }
    a.ts[3].dtype = a.ts[7].dtype = RAD_E8M0;
    REQUIRE_EQ(run("moe_gemm_q", a), RAD_OK);
    CHECK_EQ(out.at(0), 48.0f);
    CHECK_EQ(out.at(1), 72.0f);
}

/* ================================================================== the plugin surface */
TEST(plugin_surface) {
    CHECK(api().abi_version != nullptr);
    CHECK_EQ((long long)api().abi_version(), (long long)RAD_ABI_VERSION);
    const RadPluginInfo* pi = api().info();
    CHECK(pi != nullptr);
    CHECK_EQ((long long)pi->kind, (long long)RAD_PLUGIN_KERNEL);
    CHECK(std::strcmp(pi->name, "libref") == 0);

    /* Every kernel row is host domain at priority 0 and has a launch and a schema. Ref sits last
     * in the hierarchy and must never outbid anything. */
    for (int i = 0; i < api().kernel_count(); ++i) {
        const RadKernelInfo* k = api().kernel_at(i);
        CHECK(k->launch != nullptr);
        CHECK_EQ(k->domain, RAD_DOMAIN_HOST);
        CHECK_EQ(k->priority, 0);
        if (!schema(k->op)) fprintf(stderr, "    no schema for op '%s'\n", k->op);
        CHECK(schema(k->op) != nullptr);
    }
}

/* The conventional vocabulary, spelled out, so a missing op is named. An op leaving ref's list no
 * longer has a fallback or an oracle, which must be a decision, not an oversight (docs/OPS.md). */
TEST(op_vocabulary) {
    static const char* kOps[] = {
        "rmsnorm", "rmsnorm_p", "rmsnorm_add", "hc_enter", "hc_read", "hc_write", "hc_project",
        "mtp_enter",
        "mhc_pre", "mhc_pre_partial", "mhc_pre_finish", "mhc_post", "mhc_contract", "swiglu_clamped", "router_sigmoid_topk", "dsa_pool", "dsa_query", "dsa_score", "mla_attn", "mla_project", "gemm_nt_f32",
        "swiglu_step", "router_sigmoid_group_topk", "mla_attn_paged",
        "kda_update", "ssd_update",
        "layernorm", "add", "mul", "mul_weight", "silu_mul", "gelu_mul", "act_mul", "seg_mean",
        "seg_softmax_pool",
        "seg_mean_bf16_sum",
        "pool_rows", "pool_cols", "point_fourier", "relu", "clamp", "conv2d", "conv2d_unfold",
        "conv1d_dw",
        "rel_sinusoid",
        "seg_sinusoid", "attn_local_rel", "gelu", "silu", "sigmoid", "logit_softcap", "scale",
        "fill",
        "gather_rows", "scatter_rows", "ngram_ids", "ple_gate", "ple_conv", "causal_conv1d",
        /* Mamba-2's conv1d (Nemotron-3.5-Bolt): ple_conv with a bias inside the silu. */
        "ple_conv_bias",
        "embed_lookup_q", "cast",
        "scale_rows",
        "softmax",
        "quant_act_i8", "quant_act_i8g", "quant_act_fp8", "quant_cols_fp8", "had_quant_act_i8",
        "had_quant_act_i8g",
        "gram_accum",
        /* The fused fp8 forms. A host run needs every op covered, so these keep a host-backend run
         * from failing with "site 'host' is unavailable for this band". */
        "rmsnorm_quant_fp8", "gated_quant_fp8", "gemm_nt_q_gated", "dflash_select",
        "dflash_markov_greedy", "dflash_markov_greedy_f32",
        /* Tree verification (rad_drafter_tree): the drafter's tree and the attention over it. */
        "dflash_tree", "mtp_tree_grow", "attn_paged_tree", "attn_paged_anchor_window",
        "rmsnorm_had_quant_i8",
        "gated_had_quant_i8",
        "dequant", "dequant_w4",
        "gemm_nt", "gemm_nt_bias", "gemm_nt_q", "gemm_nt_q_bias", "grid_embed",
        /* Runtime LoRA's two halves, issued by the core beside a trunk GEMM. */
        "lora_shrink", "lora_expand", "lora_expand_mm",
        "grid_embed_aa", "grid_embed_cpe",
        "attn_paged", "attn_dense", "kv_store", "kv_store_streamed", "rope", "rope_table",
        "attn_paged_sink", "attn_paged_tree_sel",
        "qk_norm_rope", "qsa_block_key",
        /* libr4d's row-form names, served by the base op's row. */
        "rmsnorm_rows", "rope_rows",
        /* The rest of the QSA indexer and DFlash2's convolution: not fusions, so a model that
         * issues them has no other host path. */
        "qsa_score", "qsa_select", "qsa_tail_store", "qsa_tail_compact", "qsa_work", "kv_resolve",
        "dflash_conv",
        "router_topk", "moe_scatter", "router_topk_scatter", "router_gemm_topk_scatter",
        "moe_gemm", "moe_gemm_q", "moe_gemm_q_mod4", "moe_gather",
        "row_topk",
        "row_topk_merge", "pair_argmax", "logit_rerank",
        "gdn_conv_prep", "gdn_conv_update", "gdn_kkt_solve", "gdn_chunk_scan",
        "gdn_recurrent_update", "gdn_conv_recurrent_update", "gdn_gated_rmsnorm",
        "kv_window_compact",
        "all_reduce", "all_gather",
        "embed_lookup", "logits_gemm",
        "sample_penalties", "sample_dry", "sample_ngram_ban", "sample_temp", "sample_topk",
        "sample_topp", "sample_minp", "sample_typical", "sample_xtc", "sample_mask", "sample_pick",
        "sample_argmax", "sample_merge_topk", "sample_logprobs",
        /* The image generator's (arch/qwenimage21): a DiT's modulation and gates, the per-lane
         * sampler state, the VAE's convolutions and shortcuts, attention over a cached prefix. */
        "ln_mod", "gate_add", "timestep_embed", "gather_rows2", "lane_gather", "lane_store",
        "randn_lanes", "flow_euler", "chan_affine", "conv2d_nhwc", "im2col_nhwc", "l2norm_chan",
        "d2s_add", "s2d_avg_add", "to_rgba8", "gelu_tanh", "scatter_add_rows", "attn_prefix_self",
        /* SDXL's (arch/sdxl): CLIP-L's activation and the k-diffusion sampler over a
         * discrete-time denoiser. */
        "gelu_quick", "kd_input", "kd_step",
        /* The music generator's (arch/minimaxmusic3): the codec's convolutions and activation, the
         * guided frame sampler and the frame embeddings, the window states of the latent
         * denoiser, and the index helpers that feed them. */
        "conv1d", "conv_t1d", "snake", "cfg_topk_sample", "rvq_embed", "frame_mix_store",
        "end_status", "resample_nearest", "fourier_embed", "win_input", "win_prep", "win_finish",
        "chan_split", "interleave2", "iota", "cu_repeat", "int_affine", "frame_store",
        /* Fish Audio drama-3's (arch/fishomni): the repetition-aware choice between two draws,
         * and its codec encoder's residual quantiser step. */
        "ras_pick", "vq_encode",
        /* The video generator's (MiniMax-H3): its DiT's modulated norm and sampler step, its
         * decoder's patch unfold, tiles, stitch and positions, and the vocoder's alias-free
         * activation. */
        "rms_mod", "flow_x0_step", "flow_res_multistep", "patch_unfold",
        "vtile_gather",
        "vtile_stitch", "grid_pos3", "snake_aa",
        /* Its step-distilled form's video sparse attention: the tile selection and the
         * block-sparse attention over it. */
        "vsa_select", "attn_vsa",
        /* Its latent upscaler: the 3-D and temporal convolutions, the group norm, the resize,
         * the fold back into patch rows and the re-noising. */
        "conv3d_ndhwc", "group_norm", "tconv_dw", "resize_bilinear", "patch_fold",
        "flow_renoise", "cond_aug",
        /* Its encoder's tiling for one-frame media bigger than a tile: the plan, the cut and
         * the moments' blend back into the latent canvas. */
        "vtile_plan", "vtile_crop", "vtile_join",
        /* The joint image generators' (Krea 2, FLUX.3): attention over separate key bounds, a
         * per-layer table added to every row, a learned mix of consecutive rows, and the VAE's
         * channel statistics. */
        "attn_cross", "table_add", "mix_rows", "chan_stats",
        /* LTX-2's: its rotary over fractional coordinates, the per-head gate and the guided
         * sampler step. */
        "rope_split", "head_gate", "ltx_guide",
        /* Its VAE's norm, resampling and patches; its vocoder's spectrogram and resampler. */
        "pixel_norm", "d2s3d", "s2d3d", "pix_patch", "stft_logmel", "bwe_up",
        /* Its VAE decoded in tiles: the boxes, a tile's latent, the blended stitch. */
        "tile_grid", "tile_crop", "tile_stitch",
        /* LTX-2.5's diffusion decoder: its rotary's voxel coordinates and its neighborhood
         * attention. */
        "vol_index", "attn_na3d",
        /* Unlimited-OCR's SAM tower: windowed attention with decomposed relative position. */
        "sam_attn",
        /* A pipelined prefill's first-piece KV length, a device length less a device offset. */
        "sub_i32",
        "position_scale", "embed_attn", "ministral_rope", "pixtral_rope", "pixtral_merge",
        /* LiLiCorr's candidate correlator (Nemotron-3.5-Bolt's drafter). */
        "lili_cand", "lili_attn", "lili_factor", "lili_walk",
    };
    for (const char* op : kOps) {
        if (!kern(op)) fprintf(stderr, "    missing op '%s'\n", op);
        CHECK(kern(op) != nullptr);
    }
    CHECK_EQ(api().kernel_count(), (int)(sizeof(kOps) / sizeof(kOps[0])));

    /* The structural constraint on the paged block, and only that; everything else must match
     * anything. attn_paged_tree, attn_paged_anchor_window and attn_paged_sink take attn_paged's. */
    CHECK_EQ(kern("attn_paged")->n_constraints, 1);
    CHECK(std::strcmp(kern("attn_paged")->constraints[0].key, "block_size") == 0);
    CHECK_EQ(kern("attn_paged_tree")->n_constraints, 1);
    CHECK_EQ(kern("attn_paged_sink")->n_constraints, 1);
    CHECK(std::strcmp(kern("attn_paged_sink")->constraints[0].key, "block_size") == 0);
    CHECK_EQ(kern("kv_store")->n_constraints, 1);
    CHECK_EQ(kern("gemm_nt")->n_constraints, 0);
    CHECK_EQ(kern("gdn_chunk_scan")->n_constraints, 0);
    int constrained = 0;
    for (int i = 0; i < api().kernel_count(); ++i) constrained += api().kernel_at(i)->n_constraints > 0;
    CHECK_EQ(constrained, 5);
}

/* ================================================================== the numerics
 * Hand-written bit patterns, not round-trips: a round-trip through a broken pair of converters
 * passes. */
TEST(dtype_bf16) {
    CHECK_EQ(bf16_to_f32(0x3f80), 1.0f);
    CHECK_EQ(bf16_to_f32(0xbf80), -1.0f);
    CHECK_EQ(bf16_to_f32(0x4000), 2.0f);
    CHECK_EQ((long long)f32_to_bf16(1.0f), 0x3f80LL);
    CHECK_EQ((long long)f32_to_bf16(-2.0f), 0xc000LL);

    /* Round to nearest, ties to even. 0x3f808000 is exactly halfway between bf16 0x3f80 (even)
     * and 0x3f81, so it rounds down; 0x3f818000 is halfway between 0x3f81 (odd) and 0x3f82, so it
     * rounds up. */
    float half_even, half_odd, above;
    uint32_t u;
    u = 0x3f808000u; std::memcpy(&half_even, &u, 4);
    u = 0x3f818000u; std::memcpy(&half_odd, &u, 4);
    u = 0x3f808001u; std::memcpy(&above, &u, 4);
    CHECK_EQ((long long)f32_to_bf16(half_even), 0x3f80LL);
    CHECK_EQ((long long)f32_to_bf16(half_odd),  0x3f82LL);
    CHECK_EQ((long long)f32_to_bf16(above),     0x3f81LL);

    CHECK_EQ((long long)f32_to_bf16(INFINITY), 0x7f80LL);
    CHECK((f32_to_bf16(std::nanf("")) & 0x7f80u) == 0x7f80u);
    CHECK((f32_to_bf16(std::nanf("")) & 0x007fu) != 0u);   /* still a NaN, not an infinity */
}

TEST(dtype_f16) {
    CHECK_EQ(f16_to_f32(0x3c00), 1.0f);
    CHECK_EQ(f16_to_f32(0xc000), -2.0f);
    CHECK_EQ(f16_to_f32(0x0400), std::ldexp(1.0f, -14));    /* smallest normal */
    CHECK_EQ(f16_to_f32(0x0001), std::ldexp(1.0f, -24));    /* smallest subnormal */
    CHECK_EQ((long long)f32_to_f16(1.0f), 0x3c00LL);
    CHECK_EQ((long long)f32_to_f16(std::ldexp(1.0f, -24)), 0x0001LL);
    CHECK_EQ((long long)f32_to_f16(70000.0f), 0x7c00LL);    /* overflow -> infinity */
    CHECK_EQ(f16_to_f32(f32_to_f16(0.5f)), 0.5f);
}

TEST(dtype_fp8) {
    CHECK_EQ(fp8e4m3_to_f32(0x38), 1.0f);                   /* e=7 m=0 */
    CHECK_EQ(fp8e4m3_to_f32(0xb8), -1.0f);
    CHECK_EQ(fp8e4m3_to_f32(0x7e), 448.0f);                 /* the largest finite */
    CHECK_EQ((long long)f32_to_fp8e4m3(1.0f), 0x38LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(448.0f), 0x7eLL);
    CHECK_EQ((long long)f32_to_fp8e4m3(1e30f), 0x7eLL);     /* saturates, never NaN */
    CHECK_EQ(fp8e5m2_to_f32(0x3c), 1.0f);
    /* e5m2 rounds once, to nearest even, from f32. 1.125 is the tie between 0x3c (1.0) and 0x3d
     * (1.25) and goes to the even code; anything above it goes up. Rounding through f16 first
     * would land 1.125 + 2^-13 on the f16 tie and send it down. */
    CHECK_EQ((long long)f32_to_fp8e5m2(1.125f), 0x3cLL);
    CHECK_EQ((long long)f32_to_fp8e5m2(1.125f + std::ldexp(1.0f, -13)), 0x3dLL);

    /* The band under the smallest subnormal. 2^-9 is code 0x01 and 2^-10 is the tie between it
     * and zero: the tie goes to the even code, anything above goes up, and only what is below
     * flushes. */
    CHECK_EQ(fp8e4m3_to_f32(0x01), std::ldexp(1.0f, -9));
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(1.0f, -10)), 0x00LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::nextafter(std::ldexp(1.0f, -10), 1.0f)), 0x01LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(1.5f, -10)), 0x01LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(-std::ldexp(1.5f, -10)), 0x81LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::nextafter(std::ldexp(1.0f, -10), 0.0f)), 0x00LL);
    CHECK_EQ((long long)f32_to_fp8e4m3(std::ldexp(3.0f, -10)), 0x02LL);   /* tie 1|2 -> even */
}

/* The e4m3 encoder against an independent rounding written from the format's definition: the 127
 * non-negative finite codes decoded with ldexp, nearest in double, ties to the even code, at or
 * past 448 saturating. Every rounding boundary sits at or above f32 mantissa bit 12, so the sweep
 * covers every sign, exponent and top-eleven-mantissa value, and under them the low twelve bits
 * around zero and around the halfway point: every tie, one ulp either side, and every code. */
TEST(dtype_fp8_encode_rounds_to_nearest_even) {
    double tab[127];
    for (int c = 0; c < 127; ++c) {
        const int e = (c >> 3) & 15, m = c & 7;
        tab[c] = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    }
    auto want = [&](float f) -> uint8_t {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
        if (std::isnan(f)) return (uint8_t)(s | 0x7fu);
        const double a = std::fabs((double)f);
        if (a >= 448.0) return (uint8_t)(s | 0x7eu);
        int lo = 0, hi = 126;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (tab[mid] <= a) lo = mid; else hi = mid;
        }
        const double dl = a - tab[lo], dh = tab[hi] - a;
        const int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
        return (uint8_t)(s | c);
    };
    static const uint32_t kLow[] = { 0x000u, 0x001u, 0x7ffu, 0x800u, 0x801u, 0xfffu };
    long long bad = 0, n = 0;
    uint32_t first = 0;
    for (uint32_t hi = 0; hi < (1u << 20); ++hi) {           /* sign, exponent, top 11 mantissa */
        for (uint32_t lo : kLow) {
            const uint32_t u = (hi << 12) | lo;
            float f;
            std::memcpy(&f, &u, 4);
            ++n;
            if (f32_to_fp8e4m3(f) != want(f) && bad++ == 0) first = u;
        }
    }
    if (bad) fprintf(stderr, "    first mismatch at f32 bits 0x%08x\n", first);
    CHECK_EQ(bad, 0LL);
    CHECK_EQ(n, 6LL << 20);
}

TEST(dtype_packed) {
    /* w4: element 2i in the low nibble, value = (n ^ 8) - 8, plain 4-bit two's complement: codes
     * 0..7 are 0..7 and codes 8..15 are -8..-1. */
    uint8_t b = 0x21;                     /* low 0x1 -> 1, high 0x2 -> 2 */
    CHECK_EQ(w4_code(&b, 0), 1.0f);
    CHECK_EQ(w4_code(&b, 1), 2.0f);
    uint8_t c = 0xf8;                     /* low 0x8 -> -8, high 0xf -> -1 */
    CHECK_EQ(w4_code(&c, 0), -8.0f);
    CHECK_EQ(w4_code(&c, 1), -1.0f);

    /* w2 is the same construction one radix down: 0->0, 1->1, 2->-2, 3->-1. */
    uint8_t d = 0xe4;                     /* 0b11_10_01_00, codes 0,1,2,3 from the low end */
    CHECK_EQ(w2_code(&d, 0), 0.0f);
    CHECK_EQ(w2_code(&d, 1), 1.0f);
    CHECK_EQ(w2_code(&d, 2), -2.0f);
    CHECK_EQ(w2_code(&d, 3), -1.0f);

    uint8_t m = 0xa2;                     /* low 0x2 = +1.0, high 0xa = -1.0 */
    CHECK_EQ(mxfp4_code(&m, 0), 1.0f);
    CHECK_EQ(mxfp4_code(&m, 1), -1.0f);
    CHECK_EQ(e8m0_to_f32(127), 1.0f);
    CHECK_EQ(e8m0_to_f32(128), 2.0f);
    CHECK_EQ(e8m0_to_f32(126), 0.5f);
}

/* ================================================================== elementwise and norms */
TEST(op_rmsnorm) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T w(RAD_F32, {4});    w.f({1, 1, 1, 1});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(w).t(y).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm", a));
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386128 */
    const float s = 1.0f / 2.7386128f;
    CHECK_NEAR(y.at(0), 1 * s, 1e-6);
    CHECK_NEAR(y.at(3), 4 * s, 1e-6);

    /* wadd is GemmaRMSNorm's `1 +`: a zero-centred gain of 0 must behave as a gain of 1. */
    T w0(RAD_F32, {4}); w0.f({0, 0, 0, 0});
    T y2(RAD_F32, {1, 4});
    A b; b.t(x).t(w0).t(y2).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32").d("wadd", 1);
    CHECK_OK(run("rmsnorm", b));
    CHECK_NEAR(y2.at(3), 4 * s, 1e-6);
}

/* The weightless rmsnorm: rmsnorm with no gain, so the row divides by its rms and stops. The
 * literals are op_rmsnorm's, since a plain norm is rmsnorm with gain one. */
TEST(op_rmsnorm_p) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(y).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm_p", a));
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386128 */
    const float s = 1.0f / 2.7386128f;
    CHECK_NEAR(y.at(0), 1 * s, 1e-6);
    CHECK_NEAR(y.at(3), 4 * s, 1e-6);

    /* A non-positive n is not a normalisation, so it is refused rather than answered. */
    A bb; bb.t(x).t(y).i("M", 1).i("n", 0).d("eps", 0).s("dtype", "f32");
    CHECK(run("rmsnorm_p", bb) != RAD_OK);
}

TEST(op_rmsnorm_add) {
    T x(RAD_F32, {1, 2});   x.f({1, 3});
    T res(RAD_F32, {1, 2}); res.f({1, 1});
    T w(RAD_F32, {2});      w.f({1, 1});
    T y(RAD_F32, {1, 2}), ro(RAD_F32, {1, 2});
    A a; a.t(x).t(res).t(w).t(y).t(ro).i("M", 1).i("n", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm_add", a));
    /* residual_out = [2, 4]; rms = sqrt((4+16)/2) = sqrt(10) = 3.1622777 */
    CHECK_EQ(ro.at(0), 2.0f);
    CHECK_EQ(ro.at(1), 4.0f);
    CHECK_NEAR(y.at(0), 2.0f / 3.1622777f, 1e-6);
    CHECK_NEAR(y.at(1), 4.0f / 3.1622777f, 1e-6);
}

TEST(op_layernorm) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T w(RAD_F32, {4});    w.f({1, 1, 1, 1});
    T b(RAD_F32, {4});    b.f({0, 0, 0, 0});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(w).t(b).t(y).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("layernorm", a));
    /* mean 2.5, biased var 1.25, sd 1.1180340 */
    CHECK_NEAR(y.at(0), -1.5f / 1.1180340f, 1e-6);
    CHECK_NEAR(y.at(3),  1.5f / 1.1180340f, 1e-6);

    /* The bias operand is optional; a null tensor is not an error. */
    T y2(RAD_F32, {1, 4});
    A c; c.t(x).t(w).tnull().t(y2).i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("layernorm", c));
    CHECK_NEAR(y2.at(0), -1.5f / 1.1180340f, 1e-6);
}

TEST(op_add_mul) {
    T p(RAD_F32, {2, 2}); p.f({1, 2, 3, 4});
    T q(RAD_F32, {2, 2}); q.f({10, 20, 30, 40});
    T y(RAD_F32, {2, 2});
    A a; a.t(p).t(q).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", a));
    CHECK_EQ(y.at(0), 11.0f);
    CHECK_EQ(y.at(3), 44.0f);

    A b; b.t(p).t(q).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("mul", b));
    CHECK_EQ(y.at(0), 10.0f);
    CHECK_EQ(y.at(3), 160.0f);

    /* One row broadcasts along rows: the bias-add case and the only broadcast there is. */
    T bias(RAD_F32, {2}); bias.f({100, 200});
    A c; c.t(p).t(bias).t(y).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", c));
    CHECK_EQ(y.at(0), 101.0f);
    CHECK_EQ(y.at(3), 204.0f);
}

/* HYV3's two already-rounded arms must be added in FP32, then narrowed once. BF16 add is
 * exactly that, without materialising three FP32 scratch planes. Exercise every BF16 bit pattern,
 * including subnormals, ties, signed zero and nonfinite values, against the former cast chain. */
TEST(bf16_add_matches_explicit_fp32_combine) {
    constexpr int64_t M = 32, N = 2048;
    T routed(RAD_BF16, {M, N}), shared(RAD_BF16, {M, N});
    T rf(RAD_F32, {M, N}), sf(RAD_F32, {M, N}), sum(RAD_F32, {M, N});
    T chain(RAD_BF16, {M, N}), folded(RAD_BF16, {M, N});
    for (uint32_t i = 0; i < M * N; ++i) {
        const uint16_t r = (uint16_t)i, s = (uint16_t)(i * 40503u + 73u);
        std::memcpy(routed.mem.data() + i * 2, &r, 2);
        std::memcpy(shared.mem.data() + i * 2, &s, 2);
    }
    A r; r.t(routed).t(rf).i("M", M).i("n", N).s("from", "bf16").s("to", "f32");
    A s; s.t(shared).t(sf).i("M", M).i("n", N).s("from", "bf16").s("to", "f32");
    A a; a.t(rf).t(sf).t(sum).i("M", M).i("n", N).s("dtype", "f32");
    A c; c.t(sum).t(chain).i("M", M).i("n", N).s("from", "f32").s("to", "bf16");
    A f; f.t(routed).t(shared).t(folded).i("M", M).i("n", N).s("dtype", "bf16");
    CHECK_OK(run("cast", r)); CHECK_OK(run("cast", s));
    CHECK_OK(run("add", a)); CHECK_OK(run("cast", c)); CHECK_OK(run("add", f));
    CHECK_EQ(std::memcmp(chain.mem.data(), folded.mem.data(), M * N * 2), 0);
}

TEST(w4a16_bf16_activation_matches_the_former_explicit_f16_cast) {
    constexpr int64_t M = 1, N = 16, K = 128;
    T bf16(RAD_BF16, {M, K}), f16(RAD_F16, {M, K});
    T codes(RAD_U4, {N, K}), scales(RAD_F16, {N, 1}), zeros(RAD_U8, {N, 1});
    T out_bf16(RAD_BF16, {M, N}), out_f16(RAD_BF16, {M, N});
    const float edge = std::ldexp(1.0f + 1.0f / 128.0f, -20);
    const uint16_t b = f32_to_bf16(edge);
    const uint16_t f = f32_to_f16(bf16_to_f32(b));
    for (int64_t k = 0; k < K; ++k) {
        std::memcpy((uint8_t*)bf16.t.data + k * 2, &b, 2);
        std::memcpy((uint8_t*)f16.t.data + k * 2, &f, 2);
    }
    std::fill(codes.mem.begin(), codes.mem.end(), 0x11); // every u4 code is one
    scales.f(std::vector<float>((size_t)N, 1.0f));
    zeros.n(std::vector<int64_t>((size_t)N, 0));

    RadEncoding enc = rad_enc_affine(RAD_U4, RAD_F16, 1, K);
    rad_enc_add_plane(&enc, "zero", RAD_U8, 1, K);
    const int selected[] = { 1, 2 };
    RadTensor planes[] = { scales.t, zeros.t };
    RadLayout layout{};
    REQUIRE_EQ(rad_planes_layout(&enc, selected, planes, 2, &layout), RAD_OK);
    std::vector<uint8_t> joined((size_t)layout.bytes);
    REQUIRE_EQ(rad_planes_relayout(&enc, selected, planes, 2, joined.data(), layout.bytes), RAD_OK);
    RadTensor scale_zero{};
    scale_zero.data = joined.data(); scale_zero.dtype = layout.dtype; scale_zero.rank = 1;
    scale_zero.shape[0] = layout.bytes; scale_zero.stride[0] = 1;

    const RadKernelInfo* gemm = kern("gemm_nt_q");
    REQUIRE(gemm && gemm->launch);
    RadParam params[] = { { "M", RAD_P_INT, M }, { "N", RAD_P_INT, N },
                          { "K", RAD_P_INT, K }, { "group", RAD_P_INT, K },
                          { "dtype", RAD_P_STR, 0, 0, "w4a16_asym" } };
    auto launch = [&](T& activation, T& output) {
        RadTensor ts[7]{};
        ts[0] = activation.t; ts[2] = codes.t; ts[3] = scale_zero; ts[4] = output.t;
        RadArgs args{}; args.t = ts; args.n_t = 7; args.p = params; args.n_p = 5;
        args.world_size = 1;
        return gemm->launch(&args, nullptr);
    };
    REQUIRE_EQ(launch(f16, out_f16), RAD_OK);
    REQUIRE_EQ(launch(bf16, out_bf16), RAD_OK);
    CHECK_EQ(std::memcmp(out_f16.t.data, out_bf16.t.data, (size_t)M * N * 2), 0);
    CHECK_EQ(*(const uint16_t*)out_bf16.t.data, (uint16_t)0x3900);
}

/* add's `div` divides the sum after its rounding to y's dtype: 1 + 0.01953125 rounds to bf16
 * 1.015625 (a tie, to even), and 1.015625 / 3 to 0.337890625; the unrounded sum's third would be
 * 0.33984375. In f32 the quotient is f32(4 / 3). */
TEST(add_div_divides_the_rounded_sum) {
    T a1(RAD_BF16, {1, 1}); a1.f({1});
    T b1(RAD_BF16, {1, 1}); b1.f({0.01953125f});
    T y1(RAD_BF16, {1, 1});
    A a; a.t(a1).t(b1).t(y1).i("M", 1).i("n", 1).s("dtype", "bf16").d("div", 3.0);
    CHECK_OK(run("add", a));
    CHECK_EQ(y1.at(0), 0.337890625f);

    T a2(RAD_F32, {1, 3}); a2.f({1, 2, 4});
    T b2(RAD_F32, {1, 3}); b2.f({2, 2, 2});
    T y2(RAD_F32, {1, 3});
    A b; b.t(a2).t(b2).t(y2).i("M", 1).i("n", 3).s("dtype", "f32").d("div", 3.0);
    CHECK_OK(run("add", b));
    CHECK_EQ(y2.at(0), 1.0f); CHECK_EQ(y2.at(1), 4.0f / 3.0f); CHECK_EQ(y2.at(2), 2.0f);

    A c; c.t(a2).t(b2).t(y2).i("M", 1).i("n", 3).s("dtype", "f32").d("div", 0.0);
    CHECK_EQ(run("add", c), RAD_E_INVAL);
}

TEST(op_silu_mul_and_unary) {
    T gu(RAD_F32, {1, 4}); gu.f({1, 2, 3, 4});      /* gate [1,2], up [3,4] */
    T y(RAD_F32, {1, 2});
    A a; a.t(gu).t(y).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("silu_mul", a));
    CHECK_NEAR(y.at(0), 0.73105858f * 3.0f, 1e-6);
    CHECK_NEAR(y.at(1), 1.76159416f * 4.0f, 1e-6);

    T x(RAD_F32, {2}); x.f({0, 1});
    T g(RAD_F32, {2}), s(RAD_F32, {2});
    A b; b.t(x).t(g).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("gelu", b));
    CHECK_EQ(g.at(0), 0.0f);
    CHECK_NEAR(g.at(1), 0.84134475f, 1e-6);          /* the exact erf gelu, not tanh */
    A c; c.t(x).t(s).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("silu", c));
    CHECK_NEAR(s.at(1), 0.73105858f, 1e-6);
}

/* Gemma's gated FFN, on paper: silu_mul's fused layout (gate the first half, up the second) but the
 * gate runs the tanh GELU, not silu. gelu_tanh matches the gemm_nt_bias tanh epilogue and differs
 * from the unary `gelu` (exact erf); the check at 1.0 pins that difference: erf gelu(1) is
 * 0.84134..., the tanh form 0.84119.... */
TEST(op_gelu_mul) {
    T gu(RAD_F32, {1, 4}); gu.f({1, 2, 3, 4});      /* gate [1,2], up [3,4] */
    T y(RAD_F32, {1, 2});
    A a; a.t(gu).t(y).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("gelu_mul", a));
    CHECK_NEAR(y.at(0), 0.84119199f * 3.0f, 1e-6);   /* gelu_tanh(1) * up */
    CHECK_NEAR(y.at(1), 1.95459769f * 4.0f, 1e-6);   /* gelu_tanh(2) * up */
    /* The negative side, where tanh gelu and erf gelu drift furthest. */
    T gx(RAD_F32, {1, 4}); gx.f({-2, -1, 3, 4});     /* gate [-2,-1], up [3,4] */
    T gy(RAD_F32, {1, 2});
    A b; b.t(gx).t(gy).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("gelu_mul", b));
    CHECK_NEAR(gy.at(0), -0.04540231f * 3.0f, 1e-6); /* gelu_tanh(-2) * up */
    CHECK_NEAR(gy.at(1), -0.15880801f * 4.0f, 1e-6); /* gelu_tanh(-1) * up */
}

/* The final-logit softcap, on paper. `cap` at 30 (Gemma's value), and inputs reduced to the five
 * ratios the formula can take: 0, cap/2, both signs of cap, and 2 cap. Every ratio is exact in
 * f32, so the literals are tanh of a small integer, and the check is the f32 rounding of the
 * product only. */
TEST(op_logit_softcap) {
    T x(RAD_F32, {5}); x.f({ 0.0f, 15.0f, 30.0f, -30.0f, 60.0f });
    T y(RAD_F32, {5});
    A a; a.t(x).t(y).i("M", 1).i("n", 5).d("cap", 30.0).s("dtype", "f32");
    CHECK_OK(run("logit_softcap", a));
    CHECK_EQ(y.at(0), 0.0f);
    CHECK_NEAR(y.at(1), 30.0f * 0.4621171573f, 1e-5);   /* tanh(1/2) */
    CHECK_NEAR(y.at(2), 30.0f * 0.7615941560f, 1e-5);   /* tanh(1)   */
    CHECK_NEAR(y.at(3), -30.0f * 0.7615941560f, 1e-5);  /* tanh(-1)  */
    CHECK_NEAR(y.at(4), 30.0f * 0.9640275801f, 1e-5);   /* tanh(2)   */

    /* A non-positive cap is a division by zero in every lane, so it is refused, not answered. */
    T y2(RAD_F32, {5});
    A b; b.t(x).t(y2).i("M", 1).i("n", 5).d("cap", 0.0).s("dtype", "f32");
    CHECK(run("logit_softcap", b) != RAD_OK);
}

/* The per-tensor scalar scale. `s` is a [1] weight broadcast over every element, so the answer is
 * a literal times the literal. A scalar with no element is a shape error, not a broadcast of zero
 * (a zero would be a plausible number). */
TEST(op_scale) {
    T x(RAD_F32, {5}); x.f({ 0.0f, 2.0f, 4.0f, -6.0f, 8.0f });
    T s(RAD_F32, {1}); s.f({ 3.0f });
    T y(RAD_F32, {5});
    A a; a.t(x).t(s).t(y).i("M", 1).i("n", 5).s("dtype", "f32");
    CHECK_OK(run("scale", a));
    CHECK_EQ(y.at(0), 0.0f);
    CHECK_EQ(y.at(1), 6.0f);
    CHECK_EQ(y.at(2), 12.0f);
    CHECK_EQ(y.at(3), -18.0f);
    CHECK_EQ(y.at(4), 24.0f);

    /* A scalar with no element is a shape error, not a broadcast of zero. */
    T s0(RAD_F32, {0});
    T y2(RAD_F32, {5});
    A b; b.t(x).t(s0).t(y2).i("M", 1).i("n", 5).s("dtype", "f32");
    CHECK(run("scale", b) != RAD_OK);
}

/* The sharded draft head's merge at R = 1, three rows of four pairs, one rule each. Row 0 ties on
 * 2.0 between ids 9 and 3 (lower id wins, whatever the slot) beside a pad carrying the largest
 * value (a pad is a negative id and is dropped); row 1 is all pads (-1 and -inf); row 2 is all
 * negative (the largest still wins). `idx` is written at a pitch of two and the gaps are left
 * alone. */
TEST(op_pair_argmax) {
    struct P { int32_t id; float v; };
    const P rows[3][4] = {
        { { 5, 1.0f }, { 9, 2.0f }, { 3, 2.0f }, { -1, 7.0f } },
        { { -1, 9.0f }, { -1, 9.0f }, { -1, 9.0f }, { -1, 9.0f } },
        { { 7, -3.0f }, { 2, -4.0f }, { 11, -1.0f }, { 4, -1.5f } },
    };
    T pairs(RAD_F32, {3, 4, 2});
    float* pf = (float*)pairs.t.data;
    for (int m = 0; m < 3; ++m)
        for (int j = 0; j < 4; ++j) {
            std::memcpy(&pf[(m * 4 + j) * 2], &rows[m][j].id, 4);
            pf[(m * 4 + j) * 2 + 1] = rows[m][j].v;
        }
    T idx(RAD_I32, {3, 2}); idx.n({ 100, 100, 100, 100, 100, 100 });
    idx.t.shape[1] = 1;                     /* [3, 1] at a row pitch of 2 */
    T val(RAD_F32, {3});
    A a; a.t(pairs).t(idx).t(val).i("M", 3).i("n", 4).s("dtype", "f32");
    CHECK_OK(run("pair_argmax", a));
    CHECK_EQ(idx.iat(0), 3);   CHECK_EQ(val.at(0), 2.0f);
    CHECK_EQ(idx.iat(2), -1);  CHECK(std::isinf(val.at(1)) && val.at(1) < 0);
    CHECK_EQ(idx.iat(4), 11);  CHECK_EQ(val.at(2), -1.0f);
    CHECK_EQ(idx.iat(1), 100); CHECK_EQ(idx.iat(3), 100);

    /* `val` is optional, and the ids do not depend on it. */
    T idx2(RAD_I32, {3});
    A b; b.t(pairs).t(idx2).tnull().i("M", 3).i("n", 4).s("dtype", "f32");
    CHECK_OK(run("pair_argmax", b));
    CHECK_EQ(idx2.iat(0), 3); CHECK_EQ(idx2.iat(1), -1); CHECK_EQ(idx2.iat(2), 11);
}

/* The quantised gather. The E4M3 codes are chosen so their values are exact literals: 0x38 is 1.0,
 * 0x40 is 2.0, 0x3c is 1.5, 0xb8 is -1.0 (see abi/rad_plugin.h's decoder). */
TEST(op_embed_lookup_q) {
    T wte(RAD_F8E4M3, {4, 2});
    wte.raw({ 0x38, 0x40,      /* row 0:  1.0,  2.0 */
              0x3c, 0xb8,      /* row 1:  1.5, -1.0 */
              0x00, 0x38,      /* row 2:  0.0,  1.0 */
              0x40, 0x40 });   /* row 3:  2.0,  2.0 */
    T sc(RAD_BF16, {1}); sc.f({ 0.25f });
    T tok(RAD_I32, {3}); tok.n({ 1, 3, -1 });
    T x(RAD_BF16, {3, 2});

    A a; a.t(tok).t(wte).t(sc).t(x)
         .i("M", 3).i("n_embd", 2).i("n_vocab", 4).s("dtype", "bf16");
    CHECK_OK(run("embed_lookup_q", a));

    CHECK_EQ(x.at(0), 0.375f);      /*  1.5 * 0.25 */
    CHECK_EQ(x.at(1), -0.25f);      /* -1.0 * 0.25 */
    CHECK_EQ(x.at(2), 0.5f);        /*  2.0 * 0.25 */
    CHECK_EQ(x.at(3), 0.5f);
    /* A negative id is padding and writes a zero row, as in embed_lookup; a vocab-sharded table
     * depends on it: every rank writes zeros for ids outside its slice and the following
     * all_reduce sums to the right row. */
    CHECK_EQ(x.at(4), 0.0f);
    CHECK_EQ(x.at(5), 0.0f);

    /* An id at the table's row count is refused rather than read, since it would read whatever
     * the container put after the table. */
    T bad(RAD_I32, {1}); bad.n({ 4 });
    T y(RAD_BF16, {1, 2});
    A b; b.t(bad).t(wte).t(sc).t(y)
         .i("M", 1).i("n_embd", 2).i("n_vocab", 4).s("dtype", "bf16");
    CHECK(run("embed_lookup_q", b) != RAD_OK);

    /* Rank 0 of a vocab-parallel split also has offset 0, and there the same id is another rank's
     * token: a zero row, not a refusal. */
    T y0(RAD_BF16, {1, 2}); y0.f({ 9.0f, 9.0f });
    A c; c.t(bad).t(wte).t(sc).t(y0)
         .i("M", 1).i("n_embd", 2).i("n_vocab", 4).s("dtype", "bf16");
    CHECK_OK(run("embed_lookup_q", c, 0, 2));
    CHECK_EQ(y0.at(0), 0.0f);
    CHECK_EQ(y0.at(1), 0.0f);
}

/* PLE's gate, on paper. n = 2 and hc = 2, the smallest shape showing both asymmetries of the op:
 * the rsqrt is per n-wide stream while the gain spans hc*n, and the single `v` row is gated
 * differently into each stream.
 *
 * Every norm is exactly 1: a stream of (1, -1) or (1, 1) has mean square 1, so with eps 0 the
 * rsqrt is 1 and the normed value is the input times its gain, which makes the op hand-checkable.
 *
 *   stream 0:  K = (1, -1),  Q = (1, 1)  ->  dot 0       -> s = 0, and sign(0) = 0 so it stays 0
 *   stream 1:  K = (1,  1),  Q = (1, 1)  ->  dot 2       -> 2/sqrt(2) = sqrt(2), then sqrt again
 *
 * Stream 0 is the case a clamp gets wrong: clamping the magnitude to 1e-6 and re-signing the
 * clamped value gives 1e-3 where torch gives 0. */
TEST(op_ple_gate) {
    T k(RAD_F32, {1, 4}); k.f({ 1, -1,  1, 1 });
    T q(RAD_F32, {1, 4}); q.f({ 1,  1,  1, 1 });
    T v(RAD_F32, {1, 2}); v.f({ 4, 8 });
    T wk(RAD_F32, {4}); wk.f({ 1, 1, 1, 1 });
    T wq(RAD_F32, {4}); wq.f({ 1, 1, 1, 1 });
    T wc(RAD_F32, {4}); wc.f({ 1, 1, 1, 1 });
    T gv(RAD_F32, {1, 4}), gvn(RAD_F32, {1, 4});

    A a; a.t(k).t(q).t(v).t(wk).t(wq).t(wc).t(gv).t(gvn)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("ple_gate", a));

    /* Stream 0: sigmoid(0) = 0.5 exactly. */
    CHECK_NEAR(gv.at(0), 0.5f * 4.0f, 1e-6);
    CHECK_NEAR(gv.at(1), 0.5f * 8.0f, 1e-6);
    /* Stream 1: s = 2/sqrt(2) = 1.41421356, then sqrt -> 1.18920712, sigmoid -> 0.76659. */
    const float g1 = 1.0f / (1.0f + std::exp(-1.18920712f));
    CHECK_NEAR(gv.at(2), g1 * 4.0f, 1e-5);
    CHECK_NEAR(gv.at(3), g1 * 8.0f, 1e-5);

    /* The convolution branch does not see the gate at all, a property of the model: gv[c] is the
     * scalar g[c] times one shared v row, and norm_conv's rms is taken per n-wide stream, so the
     * scalar divides back out. gvn[c][d] is v[d]/rms(v) times its gain, whatever the gate did. A
     * kernel whose `gvn` varies with the gate has fused something it should not have. This holds
     * only in exact arithmetic (gv is narrowed before the norm reads it), so it is an invariant to
     * check, not a shortcut: computing gvn from v directly would differ in the last bits.
     *
     * Stream 0 holds (2, 4), so rms = sqrt(10) and the normed row is (2, 4)/sqrt(10). */
    const float r0 = 1.0f / std::sqrt((4.0f + 16.0f) / 2.0f);
    CHECK_NEAR(gvn.at(0), 2.0f * r0, 1e-5);
    CHECK_NEAR(gvn.at(1), 4.0f * r0, 1e-5);
    CHECK_NEAR(gvn.at(2), gvn.at(0), 1e-5);
    CHECK_NEAR(gvn.at(3), gvn.at(1), 1e-5);

    /* wadd is the `1 +` of Qwen4ExpTextRMSNorm: a zero-centred gain must behave as a gain of 1. */
    T w0(RAD_F32, {4}); w0.f({ 0, 0, 0, 0 });
    T gv2(RAD_F32, {1, 4}), gvn2(RAD_F32, {1, 4});
    A b; b.t(k).t(q).t(v).t(w0).t(w0).t(w0).t(gv2).t(gvn2)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32").d("wadd", 1);
    CHECK_OK(run("ple_gate", b));
    for (int i = 0; i < 4; ++i) CHECK_NEAR(gv2.at(i), gv.at(i), 1e-6);

    /* The sign survives. Flipping stream 1's query makes the dot -2, and the gate must land below
     * a half rather than at zero, which is what a signed root is for. */
    T qn(RAD_F32, {1, 4}); qn.f({ 1, 1, -1, -1 });
    T gv3(RAD_F32, {1, 4}), gvn3(RAD_F32, {1, 4});
    A c; c.t(k).t(qn).t(v).t(wk).t(wq).t(wc).t(gv3).t(gvn3)
         .i("M", 1).i("n", 2).i("hc", 2).d("eps", 0).s("dtype", "f32");
    CHECK_OK(run("ple_gate", c));
    const float gneg = 1.0f / (1.0f + std::exp(1.18920712f));
    CHECK_NEAR(gv3.at(2), gneg * 4.0f, 1e-5);
    CHECK(gv3.at(2) > 0.0f);            /* below half, not zero */
}

/* PLE's dilated convolution. Width 3 at dilation 2, so the taps are 4, 2 and 0 timesteps back and
 * the history is (3-1)*2 = 4: small enough to check by hand, wide enough that a kernel ignoring
 * the dilation (reading 2, 1, 0) is caught.
 *
 * The weights are (100, 10, 1), so the accumulator reads out as a decimal number: taps (a, b, c)
 * accumulate to 100a + 10b + c and any misread tap changes a digit. silu is applied after, so
 * the expectation is silu of that number.
 */
/* Mamba-2's conv1d: ple_conv_bias at dilation one and with no residual. The bias
 * must sit inside the activation, so silu(b + conv) and not b + silu(conv). */
TEST(op_ple_conv_bias_is_inside_the_silu) {
    auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
    T x(RAD_F32, {3, 2}); x.f({ 1, 10, 2, 20, 3, 30 });
    T w(RAD_F32, {2, 2}); w.f({ 1, 1, 0.5f, -1 });
    T cs(RAD_F32, {1, 2, 1}); cs.f({ 4, 8 });
    T cu(RAD_I32, {2}); cu.n({ 0, 3 });
    T ci(RAD_I32, {1}); ci.n({ 0 });
    T hi(RAD_I32, {1}); hi.n({ 1 });
    T y(RAD_F32, {3, 2});
    T b(RAD_F32, {2}); b.f({ -2, 3 });
    A a; a.t(x).t(w).t(cs).t(cu).tnull().t(ci).t(hi).tnull().t(b).t(y)
         .i("M", 3).i("n", 2).i("width", 2).i("dilation", 1).s("dtype", "f32");
    CHECK_OK(run("ple_conv_bias", a));
    /* channel 0: taps (prev, cur) * (1, 1) - 2; channel 1: (prev, cur) * (0.5, -1) + 3 */
    CHECK_NEAR(y.at(0), silu(4 + 1 - 2.0f), 1e-5);
    CHECK_NEAR(y.at(1), silu(0.5f * 8 - 10 + 3.0f), 1e-5);
    CHECK_NEAR(y.at(2), silu(1 + 2 - 2.0f), 1e-5);
    CHECK_NEAR(y.at(5), silu(0.5f * 20 - 30 + 3.0f), 1e-5);
    /* the window keeps the inputs, not the biased sums */
    CHECK_EQ(cs.at(0), 3.0f);
    CHECK_EQ(cs.at(1), 30.0f);
    T bad(RAD_F32, {3});
    a.ts[8] = bad.t;
    CHECK_EQ(run("ple_conv_bias", a), RAD_E_SHAPE);
}

TEST(op_ple_conv) {
    auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
    const int n = 1, wid = 3, dil = 2, hist = (wid - 1) * dil;   /* 4 */

    /* ---- cold: the window in front of the sequence is zeros, not the slot's contents --------- */
    {
        T x(RAD_F32, {6, 1}); x.f({ 1, 2, 3, 4, 5, 6 });
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, hist}); cs.f({ 7, 7, 7, 7 });   /* a recycled slot's leftovers */
        T cu(RAD_I32, {2}); cu.n({ 0, 6 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 0 });                  /* cold */
        T y(RAD_F32, {6, 1});
        A a; a.t(x).t(w).t(cs).t(cu).tnull().t(ci).t(hi).tnull().t(y)
             .i("M", 6).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));
        /* t=0 reads x[-4], x[-2], x[0] = 0, 0, 1  -> 1
         * t=1 reads x[-3], x[-1], x[1] = 0, 0, 2  -> 2
         * t=2 reads x[-2], x[0],  x[2] = 0, 1, 3  -> 13
         * t=4 reads x[0],  x[2],  x[4] = 1, 3, 5  -> 135  <- every tap in the sequence */
        CHECK_NEAR(y.at(0), silu(1.0f), 1e-4);
        CHECK_NEAR(y.at(1), silu(2.0f), 1e-4);
        CHECK_NEAR(y.at(2), silu(13.0f), 1e-4);
        CHECK_NEAR(y.at(4), silu(135.0f), 1e-4);
        /* The state left behind is the last `hist` inputs, at offset zero. */
        CHECK_EQ(cs.at(0), 3.0f);
        CHECK_EQ(cs.at(1), 4.0f);
        CHECK_EQ(cs.at(2), 5.0f);
        CHECK_EQ(cs.at(3), 6.0f);
    }

    /* ---- warm: the same six tokens as two steps must equal them as one ----------------------- */
    {
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, 4}); cs.f({ 0, 0, 0, 0 });
        T cu(RAD_I32, {2}); cu.n({ 0, 4 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T cold(RAD_I32, {1}); cold.n({ 0 });
        T warm(RAD_I32, {1}); warm.n({ 1 });

        T x1(RAD_F32, {4, 1}); x1.f({ 1, 2, 3, 4 });
        T y1(RAD_F32, {4, 1});
        A a; a.t(x1).t(w).t(cs).t(cu).tnull().t(ci).t(cold).tnull().t(y1)
             .i("M", 4).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));

        T x2(RAD_F32, {2, 1}); x2.f({ 5, 6 });
        T cu2(RAD_I32, {2}); cu2.n({ 0, 2 });
        T y2(RAD_F32, {2, 1});
        A b; b.t(x2).t(w).t(cs).t(cu2).tnull().t(ci).t(warm).tnull().t(y2)
             .i("M", 2).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", b));
        /* The continuation's t=0 is the whole sequence's t=4, which reads 1, 3, 5 -> 135. A cache
         * off by one fails here: with history width-1 = 2 instead of (width-1)*dilation = 4 it
         * would keep only (5, 6) and read zeros where the 1 belongs. */
        CHECK_NEAR(y2.at(0), silu(135.0f), 1e-4);
        CHECK_NEAR(y2.at(1), silu(246.0f), 1e-4);
    }

    /* ---- `resid` is added after the silu, and `num_accepted` shifts the read ----------------- */
    {
        T x(RAD_F32, {1, 1}); x.f({ 1 });
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        /* A window of six for a read window of four: two slots of speculative slack. */
        T cs(RAD_F32, {1, 1, 6}); cs.f({ 9, 9, 2, 0, 3, 0 });
        T cu(RAD_I32, {2}); cu.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 1 });
        T na(RAD_I32, {1}); na.n({ 3 });                  /* off = 2 */
        T rs(RAD_F32, {1, 1}); rs.f({ 1000 });
        T y(RAD_F32, {1, 1});
        A a; a.t(x).t(w).t(cs).t(cu).t(rs).t(ci).t(hi).t(na).t(y)
             .i("M", 1).i("n", n).i("width", wid).i("dilation", dil).s("dtype", "f32");
        CHECK_OK(run("ple_conv", a));
        /* off = num_accepted - 1 = 2, so the window is cs[2..5] = (2, 0, 3, 0) and t=0 reads
         * hist[0]=2, hist[2]=3, x[0]=1 -> 231. Read at offset 0 it would have been 9,2,1 = 921. */
        CHECK_NEAR(y.at(0), 1000.0f + silu(231.0f), 1e-3);
    }
}

/* SigLIP2's position resample, against torch's own F.interpolate(bilinear, align_corners=False,
 * antialias=True) of the same 4 x 4 x 2 table: shrinking to 2 x 3 (the filter widens), growing to
 * 6 x 5 (half-pixel taps, edges clamped) and to 1 x 1 (the whole table, averaged). */
TEST(op_grid_embed_aa_is_torchs_antialiased_bilinear) {
    const int side = 4, n = 2;
    T tab(RAD_F32, {side * side, n});
    tab.f({ 0.0f, -0.5f, 0.25f, 1.25f, 0.5f, 0.0f, 0.75f, 1.75f, 1.0f, 2.0f, 1.25f, 0.75f, 1.5f, 2.5f,
            1.75f, 1.25f, 2.0f, 1.5f, 2.25f, 3.25f, 2.5f, 2.0f, 2.75f, 3.75f, 3.0f, 4.0f, 3.25f,
            2.75f, 3.5f, 4.5f, 3.75f, 3.25f });
    struct G { int h, w; std::vector<float> want; };
    const G cases[] = {
        { 2, 3, { 0.789286f, 0.996429f, 1.089286f, 1.339286f, 1.389286f, 1.682143f, 2.360714f,
                  2.653571f, 2.660714f, 2.910714f, 2.960714f, 3.167857f } },
        { 6, 5, { 0.0f, -0.5f, 0.175f, 0.725f, 0.375f, 0.625f, 0.575f, 0.525f, 0.75f, 1.75f, 0.5f,
                  0.75f, 0.675f, 0.925f, 0.875f, 1.125f, 1.075f, 1.325f, 1.25f, 1.5f, 1.166667f,
                  1.916667f, 1.341667f, 1.391667f, 1.541667f, 1.791667f, 1.741667f, 2.191667f,
                  1.916667f, 1.666667f, 1.833333f, 1.583333f, 2.008333f, 2.458334f, 2.208333f,
                  2.458333f, 2.408334f, 2.458333f, 2.583333f, 3.333334f, 2.5f, 2.75f, 2.675f,
                  2.925f, 2.875f, 3.125f, 3.075f, 3.325f, 3.25f, 3.5f, 3.0f, 4.0f, 3.175f, 3.125f,
                  3.375f, 3.625f, 3.575f, 4.125f, 3.75f, 3.25f } },
        { 1, 1, { 1.875f, 2.125f } },
    };
    for (const G& g : cases) {
        const int M = g.h * g.w;
        T x(RAD_F32, {M, n});
        std::vector<float> zero((size_t)(M * n), 0.0f);
        x.f(zero);
        T crd(RAD_I32, {4, M});
        std::vector<int64_t> c((size_t)(4 * M));
        for (int r = 0; r < M; ++r) {
            c[(size_t)r] = r / g.w;
            c[(size_t)(M + r)] = r % g.w;
            c[(size_t)(2 * M + r)] = g.h;
            c[(size_t)(3 * M + r)] = g.w;
        }
        crd.n(c);
        A a; a.t(x).t(tab).t(crd).i("M", M).i("n", n).i("side", side).s("dtype", "f32");
        CHECK_OK(run("grid_embed_aa", a));
        for (int i = 0; i < M * n; ++i) CHECK_NEAR(x.at(i), g.want[(size_t)i], 2e-5);
    }
}

/* RADIO's cropped position embedding, against torch's F.interpolate(bilinear, align_corners=False)
 * onto a d x d square cropped to the grid (values from torch 2.11). A 3 x 5 grid upsamples the
 * 4 x 4 table to 5 x 5 and keeps three rows; a 2 x 2 grid downsamples it, which antialias would
 * change, so the two cases tell this op from grid_embed_aa. */
TEST(op_grid_embed_cpe_is_torchs_bilinear_on_a_square_cropped) {
    const int side = 4, n = 2;
    T tab(RAD_F32, {side * side, n});
    tab.f({ -5.0f, -1.75f, 2.0f, 5.25f, -2.0f, 1.25f, 5.0f, -2.75f, 1.0f, 4.25f, -3.0f, 0.25f,
            4.0f, -3.75f, 0.0f, 3.25f, -4.0f, -0.75f, 3.0f, -4.75f, -1.0f, 2.25f, -5.0f, -1.75f,
            2.0f, 5.25f, -2.0f, 1.25f, 5.0f, -2.75f, 1.0f, 4.25f });
    struct G { int h, w; std::vector<float> want; };
    const G cases[] = {
        { 3, 5, { -5.0f, -1.75f, -0.1f, 3.15f, 0.0f, 3.25f, 0.1f, 0.05f, 5.0f, -2.75f, -0.8f,
                  2.45f, -1.29f, 1.96f, 0.35f, -0.25f, 1.99f, -1.14f, 1.5f, 1.45f, -1.5f, 1.75f,
                  -0.45f, -1.05f, 0.75f, -1.5f, 0.3f, -0.3f, -2.5f, 0.75f } },
        { 2, 2, { -1.25f, 2.0f, 1.75f, -0.5f, -0.25f, 0.25f, 0.0f, 0.5f } },
    };
    for (const G& g : cases) {
        const int M = g.h * g.w;
        T x(RAD_F32, {M, n});
        std::vector<float> zero((size_t)(M * n), 0.0f);
        x.f(zero);
        T crd(RAD_I32, {4, M});
        std::vector<int64_t> c((size_t)(4 * M));
        for (int r = 0; r < M; ++r) {
            c[(size_t)r] = r / g.w;
            c[(size_t)(M + r)] = r % g.w;
            c[(size_t)(2 * M + r)] = g.h;
            c[(size_t)(3 * M + r)] = g.w;
        }
        crd.n(c);
        A a; a.t(x).t(tab).t(crd).i("M", M).i("n", n).i("side", side).s("dtype", "f32");
        CHECK_OK(run("grid_embed_cpe", a));
        for (int i = 0; i < M * n; ++i) CHECK_NEAR(x.at(i), g.want[(size_t)i], 2e-5);
    }
    /* The rows ahead of a segment's patches: row coordinate -1, the column naming the prefix row.
     * Two of them, then the 2 x 2 grid's (1, 0), which still takes the resampled table. */
    T pre(RAD_F32, {2, n}); pre.f({ 10.0f, 11.0f, 20.0f, 21.0f });
    T x(RAD_F32, {3, n}); x.f({ 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f });
    T crd(RAD_I32, {4, 3}); crd.n({ -1, -1, 1,   1, 0, 0,   2, 2, 2,   2, 2, 2 });
    A a; a.t(x).t(tab).t(crd).t(pre).i("M", 3).i("n", n).i("side", side).i("n_prefix", 2)
            .s("dtype", "f32");
    CHECK_OK(run("grid_embed_cpe", a));
    CHECK_NEAR(x.at(0), 20.5f, 1e-6); CHECK_NEAR(x.at(1), 21.5f, 1e-6);   /* 0.5 + prefix row 1 */
    CHECK_NEAR(x.at(2), 10.0f, 1e-6); CHECK_NEAR(x.at(3), 11.0f, 1e-6);   /* prefix row 0 */
    CHECK_NEAR(x.at(4), -0.25f, 2e-5); CHECK_NEAR(x.at(5), 0.25f, 2e-5);  /* torch's (1, 0) */
}

/* LFM2's short convolution: ple_conv's window at dilation one, with no activation. */
TEST(op_causal_conv1d) {
    const int n = 1, wid = 3;
    /* Cold: zeros in front of the sequence, and the last width-1 inputs left at offset zero. */
    {
        T x(RAD_F32, {4, 1}); x.f({ 1, 2, 3, 4 });
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, 2}); cs.f({ 7, 7 });          /* a recycled slot's leftovers */
        T cu(RAD_I32, {2}); cu.n({ 0, 4 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 0 });
        T y(RAD_F32, {4, 1});
        A a; a.t(x).t(w).t(cs).t(cu).t(ci).t(hi).tnull().t(y)
             .i("M", 4).i("n", n).i("width", wid).s("dtype", "f32");
        CHECK_OK(run("causal_conv1d", a));
        CHECK_NEAR(y.at(0), 1.0f, 1e-5);                 /* 0, 0, 1 -- no silu */
        CHECK_NEAR(y.at(1), 12.0f, 1e-5);                /* 0, 1, 2 */
        CHECK_NEAR(y.at(3), 234.0f, 1e-5);               /* 2, 3, 4 */
        CHECK_EQ(cs.at(0), 3.0f);
        CHECK_EQ(cs.at(1), 4.0f);
    }
    /* Warm, as two chunks, then a decode step read at num_accepted - 1. */
    {
        T w(RAD_F32, {1, 3}); w.f({ 100, 10, 1 });
        T cs(RAD_F32, {1, 1, 4}); cs.f({ 0, 0, 0, 0 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T cold(RAD_I32, {1}); cold.n({ 0 });
        T warm(RAD_I32, {1}); warm.n({ 1 });
        T x1(RAD_F32, {3, 1}); x1.f({ 1, 2, 3 });
        T cu1(RAD_I32, {2}); cu1.n({ 0, 3 });
        T y1(RAD_F32, {3, 1});
        A a; a.t(x1).t(w).t(cs).t(cu1).t(ci).t(cold).tnull().t(y1)
             .i("M", 3).i("n", n).i("width", wid).s("dtype", "f32");
        CHECK_OK(run("causal_conv1d", a));
        T x2(RAD_F32, {1, 1}); x2.f({ 4 });
        T cu2(RAD_I32, {2}); cu2.n({ 0, 1 });
        T y2(RAD_F32, {1, 1});
        A b; b.t(x2).t(w).t(cs).t(cu2).t(ci).t(warm).tnull().t(y2)
             .i("M", 1).i("n", n).i("width", wid).s("dtype", "f32");
        CHECK_OK(run("causal_conv1d", b));
        CHECK_NEAR(y2.at(0), 234.0f, 1e-5);              /* the one-pass t=3 */
        /* A verify of two drafts (5, 6) behind it, then a step that kept one of them. */
        T x3(RAD_F32, {2, 1}); x3.f({ 5, 6 });
        T cu3(RAD_I32, {2}); cu3.n({ 0, 2 });
        T na1(RAD_I32, {1}); na1.n({ 1 });
        T y3(RAD_F32, {2, 1});
        A c; c.t(x3).t(w).t(cs).t(cu3).t(ci).t(warm).t(na1).t(y3)
             .i("M", 2).i("n", n).i("width", wid).s("dtype", "f32");
        CHECK_OK(run("causal_conv1d", c));
        CHECK_NEAR(y3.at(0), 345.0f, 1e-5);
        CHECK_NEAR(y3.at(1), 456.0f, 1e-5);
        T x4(RAD_F32, {1, 1}); x4.f({ 9 });
        T cu4(RAD_I32, {2}); cu4.n({ 0, 1 });
        T na2(RAD_I32, {1}); na2.n({ 1 });               /* only 5 was kept: history is (4, 5) */
        T y4(RAD_F32, {1, 1});
        A d; d.t(x4).t(w).t(cs).t(cu4).t(ci).t(warm).t(na2).t(y4)
             .i("M", 1).i("n", n).i("width", wid).s("dtype", "f32");
        CHECK_OK(run("causal_conv1d", d));
        CHECK_NEAR(y4.at(0), 459.0f, 1e-5);
    }
}

/* PLE's n-gram ids, on the real checkpoint constants.
 *
 * The multipliers, vocab sizes and offsets below are the bytes of
 * `model.language_model.layers.1.ple.ple_embedding.*` in Qwen3.8-Flash-Next, read from the
 * safetensors: a synthetic multiplier would not reach the 10^13 magnitude that makes the products
 * fill 63 bits, which is where a hash goes wrong.
 *
 * The expectations are properties, not a second copy of the formula. What is checked is what the
 * reference's `_shift_right_ignore_eos` and its head blocking say, read off the Python rather than
 * ref_ngram.cpp:
 *
 *   - a 2-gram head depends on t0 and t1 and not on t2; a 3-gram head depends on all three;
 *   - an EOS at p-1 makes both t1 and t2 fall back to eos, so the row equals one whose history is
 *     entirely eos;
 *   - an EOS at p-2 alone makes t2 fall back but leaves t1, and since t2 was that eos anyway the
 *     row is unchanged (the identity that reduces the reference's cummax to two compares);
 *   - the first rows of a sequence read `prev`, and later ones read the batch;
 *   - two sequences in one batch do not see each other's tokens.
 */
TEST(op_ngram_ids) {
    const int64_t kEos = 248044, kHeads = 16, kNgram = 3;
    const int64_t mult[3] = { 23703573157769LL, 20109073645365LL, 8052911324071LL };
    const int64_t vsz[16] = { 20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                              20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                              20000153, 20000159, 20000161, 20000171 };
    int64_t offs[16];
    offs[0] = 0;
    for (int i = 1; i < 16; ++i) offs[i] = offs[i - 1] + vsz[i - 1];

    T tm(RAD_I64, {3});  tm.n({ mult[0], mult[1], mult[2] });
    T tv(RAD_I64, {16}); T to(RAD_I64, {16});
    { std::vector<int64_t> a(vsz, vsz + 16), b(offs, offs + 16); tv.n(a); to.n(b); }

    /* Every id lands inside its head's slice of the 320-million-row table: what makes the offsets
     * meaningful, and what a harness drawing small vocab sizes would miss. */
    auto in_slice = [&](const T& ids, int64_t row, int64_t h) {
        const int64_t v = ids.iat(row * kHeads + h);
        return v >= offs[h] && v < offs[h] + vsz[h];
    };

    /* `prev` is the warm contents of a one-slot state window, oldest first. The state is INOUT, so
     * each call gets its own; a shared one would carry the previous case's tail forward. */
    auto run4 = [&](std::vector<int64_t> tok, std::vector<int64_t> prev, T& ids) {
        const int64_t M = (int64_t)tok.size();
        T tt(RAD_I32, {M}); tt.n(tok);
        T ts(RAD_I32, {1, 2}); ts.n(prev);
        T tc(RAD_I32, {2}); tc.n({0, M});
        T ci(RAD_I32, {1}); ci.n({0});
        T hi(RAD_I32, {1}); hi.n({1});            /* warm: read the window as given */
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", M).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        return run("ngram_ids", a);
    };

    /* --- the head blocks: 0..7 are the 2-gram, 8..15 the 3-gram ------------------------------ */
    T base(RAD_I32, {1, kHeads}), t2_changed(RAD_I32, {1, kHeads}),
      t1_changed(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { 700, 22 }, base));           /* t0=11 t1=22 t2=700 */
    CHECK_OK(run4({ 11 }, { 701, 22 }, t2_changed));     /* only t2 differs */
    CHECK_OK(run4({ 11 }, { 700, 23 }, t1_changed));     /* only t1 differs */
    for (int64_t h = 0; h < 8; ++h) {
        CHECK_EQ(base.iat(h), t2_changed.iat(h));        /* a 2-gram head cannot see t2 */
        CHECK(base.iat(h) != t1_changed.iat(h));
        CHECK(in_slice(base, 0, h));
    }
    for (int64_t h = 8; h < 16; ++h) {
        CHECK(base.iat(h) != t2_changed.iat(h));         /* a 3-gram head must */
        CHECK(in_slice(base, 0, h));
    }

    /* --- the EOS rule ------------------------------------------------------------------------ */
    T all_eos(RAD_I32, {1, kHeads}), eos_at_p1(RAD_I32, {1, kHeads}),
      eos_at_p2(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { kEos, kEos }, all_eos));
    CHECK_OK(run4({ 11 }, { 700, kEos }, eos_at_p1));    /* t1 is the eos: t2 must follow it */
    CHECK_OK(run4({ 11 }, { kEos, 22 }, eos_at_p2));     /* t2 is the eos: t1 survives */
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(all_eos.iat(h), eos_at_p1.iat(h));
    /* An eos at p-2 changes nothing, because the value it substitutes is the value there. */
    T eos_p2_ref(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { kEos, 22 }, eos_p2_ref));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(eos_at_p2.iat(h), eos_p2_ref.iat(h));

    /* --- `prev` vs the batch: a row deep enough in the step must not read `prev` at all ------- */
    T deep_a(RAD_I32, {3, kHeads}), deep_b(RAD_I32, {3, kHeads});
    CHECK_OK(run4({ 5, 6, 11 }, { 900, 901 }, deep_a));
    CHECK_OK(run4({ 5, 6, 11 }, { 902, 903 }, deep_b));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(deep_a.iat(2 * kHeads + h), deep_b.iat(2 * kHeads + h));
    CHECK(deep_a.iat(0) != deep_b.iat(0));               /* but row 0 does */
    /* Row 2 of [5, 6, 11] sees exactly the history row 0 of [11] sees with prev = [5, 6]. */
    T from_prev(RAD_I32, {1, kHeads});
    CHECK_OK(run4({ 11 }, { 5, 6 }, from_prev));
    for (int64_t h = 0; h < 16; ++h) CHECK_EQ(deep_a.iat(2 * kHeads + h), from_prev.iat(h));

    /* --- two sequences in one batch do not see each other, and each reads its own slot --------- */
    {
        T tt(RAD_I32, {4}); tt.n({ 5, 6, 11, 11 });      /* seq0 = [5,6,11], seq1 = [11] */
        T ts(RAD_I32, {2, 2}); ts.n({ 900, 901, 5, 6 });
        T tc(RAD_I32, {3}); tc.n({ 0, 3, 4 });
        T ci(RAD_I32, {2}); ci.n({ 0, 1 });
        T hi(RAD_I32, {2}); hi.n({ 1, 1 });
        T ids(RAD_I32, {4, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        /* seq1's only row has a window of [5, 6] and token 11, the same history as `from_prev`,
         * although tokens 5 and 6 also sit in the batch two rows earlier. */
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(3 * kHeads + h), from_prev.iat(h));
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(2 * kHeads + h), from_prev.iat(h));
        /* Each slot was written with its own tail, at offset zero: seq0 ran [5,6,11] so its window
         * becomes (6, 11); seq1 ran [11] against (5, 6) so its window becomes (6, 11) too, by a
         * different route (one value carried from the old window, one from `tok`). */
        CHECK_EQ(ts.iat(0), 6);   CHECK_EQ(ts.iat(1), 11);
        CHECK_EQ(ts.iat(2), 6);   CHECK_EQ(ts.iat(3), 11);
    }

    /* --- a cold slot is EOS, not zero, and not what the slot last held ------------------------ */
    {
        T tt(RAD_I32, {1}); tt.n({ 11 });
        T ts(RAD_I32, {1, 2}); ts.n({ 700, 22 });        /* a recycled slot's leftovers */
        T tc(RAD_I32, {2}); tc.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 0 });                 /* cold */
        T ids(RAD_I32, {1, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids)
             .i("M", 1).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        /* Identical to a warm slot whose window is already all EOS, which is what "cold means eos"
         * means. Zero would be a real token and would hash elsewhere. */
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(h), all_eos.iat(h));
    }

    /* --- `num_accepted` shifts the read inside a deeper window -------------------------------- */
    {
        T tt(RAD_I32, {1}); tt.n({ 11 });
        T ts(RAD_I32, {1, 4}); ts.n({ 900, 901, 5, 6 }); /* two slots of speculative slack */
        T tc(RAD_I32, {2}); tc.n({ 0, 1 });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 1 });
        T na(RAD_I32, {1}); na.n({ 3 });                 /* off = 2, so the window is (5, 6) */
        T ids(RAD_I32, {1, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).t(na).t(ids)
             .i("M", 1).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        for (int64_t h = 0; h < 16; ++h) CHECK_EQ(ids.iat(h), from_prev.iat(h));
        /* The write still lands at offset zero, not at the offset it read from. */
        CHECK_EQ(ts.iat(0), 6);   CHECK_EQ(ts.iat(1), 11);
    }

    /* --- `ahead_ids`: the prompt that follows, hashed as the step that runs it will ------------
     *
     * Step one is two sequences, [5, 6] and [11, 42], and `tok` carries three more tokens of the
     * last one, [7, 8, 9]. Their ids must equal the ids step two computes when it runs [7, 8, 9]
     * against the window step one leaves, and that window and the other sequence must be what they
     * would have been without `ahead_ids`. */
    {
        T tt(RAD_I32, {7}); tt.n({ 5, 6, 11, 42, 7, 8, 9 });
        T ts(RAD_I32, {2, 2}); ts.n({ 900, 901, 902, 903 });
        T tc(RAD_I32, {3}); tc.n({ 0, 2, 4 });
        T ci(RAD_I32, {2}); ci.n({ 0, 1 });
        T hi(RAD_I32, {2}); hi.n({ 1, 1 });
        T ids(RAD_I32, {4, kHeads}), ahead(RAD_I32, {3, kHeads});
        A a; a.t(tt).t(ts).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids).t(ahead)
             .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a));
        CHECK_EQ(ts.iat(0), 5);   CHECK_EQ(ts.iat(1), 6);    /* seq0's window: its own tail */
        CHECK_EQ(ts.iat(2), 11);  CHECK_EQ(ts.iat(3), 42);   /* seq1's: its own, not the ahead */

        /* The same step without `ahead_ids`: the same ids for the step's own rows. */
        T tt0(RAD_I32, {4}); tt0.n({ 5, 6, 11, 42 });
        T ts0(RAD_I32, {2, 2}); ts0.n({ 900, 901, 902, 903 });
        T ids0(RAD_I32, {4, kHeads});
        A a0; a0.t(tt0).t(ts0).t(tc).t(tm).t(tv).t(to).t(ci).t(hi).tnull().t(ids0)
              .i("M", 4).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a0));
        for (int64_t i = 0; i < 4 * kHeads; ++i) CHECK_EQ(ids.iat(i), ids0.iat(i));

        /* Step two: seq1 runs [7, 8, 9] against the window step one left. */
        T tt2(RAD_I32, {3}); tt2.n({ 7, 8, 9 });
        T ts2(RAD_I32, {1, 2}); ts2.n({ ts.iat(2), ts.iat(3) });
        T tc2(RAD_I32, {2}); tc2.n({ 0, 3 });
        T ci2(RAD_I32, {1}); ci2.n({ 0 });
        T hi2(RAD_I32, {1}); hi2.n({ 1 });
        T ids2(RAD_I32, {3, kHeads});
        A a2; a2.t(tt2).t(ts2).t(tc2).t(tm).t(tv).t(to).t(ci2).t(hi2).tnull().t(ids2)
              .i("M", 3).i("heads", kHeads).i("ngram", kNgram).i("eos", kEos);
        CHECK_OK(run("ngram_ids", a2));
        for (int64_t i = 0; i < 3 * kHeads; ++i) CHECK_EQ(ahead.iat(i), ids2.iat(i));
    }
}

/* ============================================================ the PLE windows across a verify
 *
 * Definition: a step's outputs depend only on the committed history and its own tokens. So row t
 * of any step must equal the last row of one cold prefill over (committed tokens ++ the step's
 * first t + 1 tokens). A verify of n that kept k commits k and the next `num_accepted` is k; a
 * prefill chunk commits everything and the next `num_accepted` is 1. The window is also checked
 * directly: after a verify, offset k - 1 holds the last committed values for every possible k;
 * after a prefill chunk, offset zero does.
 *
 * No single layout passes both kinds of step: tail-at-zero gives a next step that kept one token
 * a history ending at the last rejected draft; the shifted layout alone gives a short prefill
 * chunk's next step a history ending at the chunk's first token. The chains include a two-token
 * prefill chunk, verifies accepting every count, and a check that the rejected drafts' history
 * really differs. */
namespace {

/* The n-gram op's constants: the real checkpoint's multipliers and a slice of its vocabulary. */
struct NgK {
    static constexpr int64_t kEos = 248044, kHeads = 16, kNgram = 3, kCtx = 2;
    T tm{RAD_I64, {3}}, tv{RAD_I64, {16}}, to{RAD_I64, {16}};
    NgK() {
        tm.n({ 23703573157769LL, 20109073645365LL, 8052911324071LL });
        std::vector<int64_t> v, o;
        int64_t acc = 0;
        for (int i = 0; i < 16; ++i) {
            v.push_back(20000003 + 10 * i);
            o.push_back(acc);
            acc += v.back();
        }
        tv.n(v);
        to.n(o);
    }
};

/* One launch over several sequences, sequence i on state slot `slots[i]`. `nacc` null means the
 * operand is absent (the prefill form), not a row of ones. */
int ng_launch(const NgK& K, const std::vector<std::vector<int64_t>>& seqs, T& state,
              const std::vector<int64_t>& slots, const std::vector<int64_t>& hini,
              const std::vector<int64_t>* nacc, T& ids) {
    std::vector<int64_t> tok, cu{ 0 };
    for (const auto& s : seqs) {
        tok.insert(tok.end(), s.begin(), s.end());
        cu.push_back((int64_t)tok.size());
    }
    const int64_t M = (int64_t)tok.size(), S = (int64_t)seqs.size();
    T tt(RAD_I32, {M}); tt.n(tok);
    T tc(RAD_I32, {S + 1}); tc.n(cu);
    T ci(RAD_I32, {S}); ci.n(slots);
    T hi(RAD_I32, {S}); hi.n(hini);
    T na(RAD_I32, {S});
    if (nacc) na.n(*nacc);
    A a; a.t(tt).t(state).t(tc).t(K.tm).t(K.tv).t(K.to).t(ci).t(hi);
    if (nacc) a.t(na); else a.tnull();
    a.t(ids).i("M", M).i("heads", NgK::kHeads).i("ngram", NgK::kNgram).i("eos", NgK::kEos);
    return run("ngram_ids", a);
}

/* The definition: the last row of one cold prefill over `hist`. */
std::vector<int64_t> ng_fresh(const NgK& K, const std::vector<int64_t>& hist) {
    const int64_t L = (int64_t)hist.size();
    T st(RAD_I32, {1, NgK::kCtx});
    T ids(RAD_I32, {L, NgK::kHeads});
    std::vector<int64_t> row((size_t)NgK::kHeads, -1);
    if (ng_launch(K, { hist }, st, { 0 }, { 0 }, nullptr, ids) != RAD_OK) return row;
    for (int64_t h = 0; h < NgK::kHeads; ++h) row[(size_t)h] = ids.iat((L - 1) * NgK::kHeads + h);
    return row;
}

std::vector<int64_t> ng_row(const T& ids, int64_t r) {
    std::vector<int64_t> row((size_t)NgK::kHeads);
    for (int64_t h = 0; h < NgK::kHeads; ++h) row[(size_t)h] = ids.iat(r * NgK::kHeads + h);
    return row;
}

/* The last `n` values of `v`, cold-padded on the left with `cold`. */
template <class V>
std::vector<V> last_n(const std::vector<V>& v, int64_t n, V cold) {
    std::vector<V> out((size_t)n, cold);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t src = (int64_t)v.size() - n + i;
        if (src >= 0) out[(size_t)i] = v[(size_t)src];
    }
    return out;
}

/* A step of a chain: `n` tokens, `k` of them kept. A prefill keeps all of them. */
struct ChainStep { bool prefill; int n; int k; };
const ChainStep kChain[] = {
    { true, 3, 3 }, { true, 2, 2 },                    /* a cold chunk, then a short warm one */
    { false, 4, 1 }, { false, 4, 4 }, { false, 4, 2 },  /* verifies of 1 + n_spec = 4 */
    { false, 1, 1 },                                    /* a decode that drafted nothing */
    { false, 4, 3 }, { false, 3, 1 }, { false, 4, 2 }, { false, 2, 2 },
};
constexpr int64_t kNspec = 3;

}  /* namespace */

TEST(ngram_window_follows_the_committed_history) {
    const NgK K;
    const int64_t W = NgK::kCtx + kNspec;       /* what RAD_KV_CONV sizes: ctx + n_spec */
    T st(RAD_I32, {1, W});
    st.n(std::vector<int64_t>((size_t)W, 777)); /* a recycled slot's leftovers */

    uint64_t lcg = 0x5EEDull;
    auto draw = [&]() {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        return (int64_t)((lcg >> 33) % 150000) + 1;
    };

    std::vector<int64_t> committed, wrong;   /* `wrong` keeps every draft, rejected or not */
    int64_t nacc_next = 1;
    bool cold = true, after_partial = false;
    int step_no = 0;
    for (const ChainStep& cs : kChain) {
        std::vector<int64_t> tok;
        for (int i = 0; i < cs.n; ++i) tok.push_back(draw());
        if (step_no == 6) tok[1] = NgK::kEos;     /* the EOS rule inside a verify, once */

        T ids(RAD_I32, {cs.n, NgK::kHeads});
        const std::vector<int64_t> na{ nacc_next };
        CHECK_OK(ng_launch(K, { tok }, st, { 0 }, { cold ? 0 : 1 }, cs.prefill ? nullptr : &na,
                           ids));

        /* Every row against the definition. */
        for (int t = 0; t < cs.n; ++t) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + t + 1);
            CHECK(ng_row(ids, t) == ng_fresh(K, h));
        }
        /* The sensitivity check: the history keeping the rejected drafts gives a different
         * answer, so agreement above says which history was read. */
        if (after_partial) {
            std::vector<int64_t> h = wrong;
            h.push_back(tok[0]);
            CHECK(ng_row(ids, 0) != ng_fresh(K, h));
        }

        /* The window left behind, at every offset the next step could read. */
        const int kmax = cs.prefill ? 1 : cs.n;
        for (int kk = 1; kk <= kmax; ++kk) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + (cs.prefill ? cs.n : kk));
            const std::vector<int64_t> want = last_n<int64_t>(h, NgK::kCtx, NgK::kEos);
            for (int64_t i = 0; i < NgK::kCtx; ++i) CHECK_EQ(st.iat(kk - 1 + i), want[(size_t)i]);
        }

        committed.insert(committed.end(), tok.begin(), tok.begin() + cs.k);
        wrong.insert(wrong.end(), tok.begin(), tok.end());
        after_partial = !cs.prefill && cs.k < cs.n;
        if (!after_partial) wrong = committed;
        nacc_next = cs.prefill ? 1 : cs.k;
        cold = false;
        ++step_no;
    }

    /* One case written out: history {1, 2, 3}, a verify of {4, 100, 200, 300} that kept only the
     * 4. The next step's two ids in front must be (3, 4), not (200, 300), the tail of rejected
     * drafts. */
    {
        T s2(RAD_I32, {1, W});
        T i1(RAD_I32, {3, NgK::kHeads}), i2(RAD_I32, {4, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 1, 2, 3 } }, s2, { 0 }, { 0 }, nullptr, i1));
        const std::vector<int64_t> one{ 1 };
        CHECK_OK(ng_launch(K, { { 4, 100, 200, 300 } }, s2, { 0 }, { 1 }, &one, i2));
        CHECK_EQ(s2.iat(0), 3LL);
        CHECK_EQ(s2.iat(1), 4LL);
        T i3(RAD_I32, {1, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 9 } }, s2, { 0 }, { 1 }, &one, i3));
        CHECK(ng_row(i3, 0) == ng_fresh(K, { 1, 2, 3, 4, 9 }));
    }

    /* Two sequences in one launch, on swapped slots and with different acceptances: each reads its
     * own window at its own offset. */
    {
        T s2(RAD_I32, {2, W});
        const std::vector<int64_t> A0{ 11, 12, 13 }, B0{ 21, 22, 23, 24 };
        T i1(RAD_I32, {7, NgK::kHeads});
        CHECK_OK(ng_launch(K, { A0, B0 }, s2, { 1, 0 }, { 0, 0 }, nullptr, i1));
        const std::vector<int64_t> A1{ 31, 32, 33, 34 }, B1{ 41, 42, 43 };
        const std::vector<int64_t> ones{ 1, 1 };
        T i2(RAD_I32, {7, NgK::kHeads});
        CHECK_OK(ng_launch(K, { A1, B1 }, s2, { 1, 0 }, { 1, 1 }, &ones, i2));
        const std::vector<int64_t> kept{ 2, 3 };  /* A kept 31, 32; B kept 41, 42, 43 */
        T i3(RAD_I32, {2, NgK::kHeads});
        CHECK_OK(ng_launch(K, { { 50 }, { 60 } }, s2, { 1, 0 }, { 1, 1 }, &kept, i3));
        CHECK(ng_row(i3, 0) == ng_fresh(K, { 11, 12, 13, 31, 32, 50 }));
        CHECK(ng_row(i3, 1) == ng_fresh(K, { 21, 22, 23, 24, 41, 42, 43, 60 }));
    }

    /* A decode step whose window is too shallow for the step it must leave behind is refused, not
     * truncated: ctx + n_spec slots hold a step of 1 + n_spec ids and no more. */
    {
        T s2(RAD_I32, {1, W});
        T ids(RAD_I32, {5, NgK::kHeads});
        const std::vector<int64_t> one{ 1 };
        CHECK_EQ(ng_launch(K, { { 1, 2, 3, 4, 5 } }, s2, { 0 }, { 1 }, &one, ids), RAD_E_SHAPE);
    }
}

namespace {

/* ple_conv at width 3 and dilation 2 (history 4) over two channels, f32 throughout so the
 * comparison against the definition is exact rather than a tolerance. */
struct PcK {
    static constexpr int64_t kN = 2, kWid = 3, kDil = 2, kHist = (kWid - 1) * kDil;
    T w{RAD_F32, {kN, kWid}};
    PcK() { w.f({ 0.5f, -0.25f, 1.0f, 0.125f, 0.75f, -0.5f }); }
};

/* A token's input row: a function of the token, so a history of tokens is a history of rows. */
float pc_x(int64_t tok, int64_t c) { return (float)((tok * 37 + c * 11) % 101) * 0.0625f - 3.0f; }

int pc_launch(const PcK& K, const std::vector<std::vector<int64_t>>& seqs, T& state,
              const std::vector<int64_t>& slots, const std::vector<int64_t>& hini,
              const std::vector<int64_t>* nacc, T& y) {
    std::vector<int64_t> cu{ 0 };
    std::vector<float> xv;
    for (const auto& s : seqs) {
        for (int64_t t : s)
            for (int64_t c = 0; c < PcK::kN; ++c) xv.push_back(pc_x(t, c));
        cu.push_back(cu.back() + (int64_t)s.size());
    }
    const int64_t M = cu.back(), S = (int64_t)seqs.size();
    T x(RAD_F32, {M, PcK::kN}); x.f(xv);
    T tc(RAD_I32, {S + 1}); tc.n(cu);
    T ci(RAD_I32, {S}); ci.n(slots);
    T hi(RAD_I32, {S}); hi.n(hini);
    T na(RAD_I32, {S});
    if (nacc) na.n(*nacc);
    A a; a.t(x).t(K.w).t(state).t(tc).tnull().t(ci).t(hi);
    if (nacc) a.t(na); else a.tnull();
    a.t(y).i("M", M).i("n", PcK::kN).i("width", PcK::kWid).i("dilation", PcK::kDil)
     .s("dtype", "f32");
    return run("ple_conv", a);
}

std::vector<float> pc_fresh(const PcK& K, const std::vector<int64_t>& hist) {
    const int64_t L = (int64_t)hist.size();
    T st(RAD_F32, {1, PcK::kN, PcK::kHist});
    T y(RAD_F32, {L, PcK::kN});
    std::vector<float> row((size_t)PcK::kN, NAN);
    if (pc_launch(K, { hist }, st, { 0 }, { 0 }, nullptr, y) != RAD_OK) return row;
    for (int64_t c = 0; c < PcK::kN; ++c) row[(size_t)c] = y.at((L - 1) * PcK::kN + c);
    return row;
}

std::vector<float> pc_row(const T& y, int64_t r) {
    std::vector<float> row((size_t)PcK::kN);
    for (int64_t c = 0; c < PcK::kN; ++c) row[(size_t)c] = y.at(r * PcK::kN + c);
    return row;
}

}  /* namespace */

TEST(ple_conv_window_follows_the_committed_history) {
    const PcK K;
    const int64_t W = PcK::kHist + kNspec;      /* (width-1)*dilation + n_spec */
    T st(RAD_F32, {1, PcK::kN, W});
    st.f(std::vector<float>((size_t)(PcK::kN * W), 9.0f));

    uint64_t lcg = 0xC0FFEEull;
    auto draw = [&]() {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        return (int64_t)((lcg >> 33) % 1000);
    };

    std::vector<int64_t> committed, wrong;
    int64_t nacc_next = 1;
    bool cold = true, after_partial = false;
    for (const ChainStep& cs : kChain) {
        std::vector<int64_t> tok;
        for (int i = 0; i < cs.n; ++i) tok.push_back(draw());

        T y(RAD_F32, {cs.n, PcK::kN});
        const std::vector<int64_t> na{ nacc_next };
        CHECK_OK(pc_launch(K, { tok }, st, { 0 }, { cold ? 0 : 1 }, cs.prefill ? nullptr : &na, y));

        for (int t = 0; t < cs.n; ++t) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + t + 1);
            CHECK(pc_row(y, t) == pc_fresh(K, h));
        }
        if (after_partial) {
            std::vector<int64_t> h = wrong;
            h.push_back(tok[0]);
            CHECK(pc_row(y, 0) != pc_fresh(K, h));
        }

        const int kmax = cs.prefill ? 1 : cs.n;
        for (int kk = 1; kk <= kmax; ++kk) {
            std::vector<int64_t> h = committed;
            h.insert(h.end(), tok.begin(), tok.begin() + (cs.prefill ? cs.n : kk));
            const std::vector<int64_t> want = last_n<int64_t>(h, PcK::kHist, -1);
            for (int64_t c = 0; c < PcK::kN; ++c)
                for (int64_t i = 0; i < PcK::kHist; ++i) {
                    const int64_t tk = want[(size_t)i];
                    CHECK_EQ(st.at(c * W + kk - 1 + i), tk < 0 ? 0.0f : pc_x(tk, c));
                }
        }

        committed.insert(committed.end(), tok.begin(), tok.begin() + cs.k);
        wrong.insert(wrong.end(), tok.begin(), tok.end());
        after_partial = !cs.prefill && cs.k < cs.n;
        if (!after_partial) wrong = committed;
        nacc_next = cs.prefill ? 1 : cs.k;
        cold = false;
    }

    /* Two sequences in one launch, swapped slots, different acceptances. */
    {
        T s2(RAD_F32, {2, PcK::kN, W});
        const std::vector<int64_t> A0{ 11, 12, 13, 14, 15 }, B0{ 21, 22 };
        T y1(RAD_F32, {7, PcK::kN});
        CHECK_OK(pc_launch(K, { A0, B0 }, s2, { 1, 0 }, { 0, 0 }, nullptr, y1));
        const std::vector<int64_t> A1{ 31, 32, 33, 34 }, B1{ 41, 42, 43, 44 };
        const std::vector<int64_t> ones{ 1, 1 };
        T y2(RAD_F32, {8, PcK::kN});
        CHECK_OK(pc_launch(K, { A1, B1 }, s2, { 1, 0 }, { 1, 1 }, &ones, y2));
        const std::vector<int64_t> kept{ 1, 3 };
        T y3(RAD_F32, {2, PcK::kN});
        CHECK_OK(pc_launch(K, { { 50 }, { 60 } }, s2, { 1, 0 }, { 1, 1 }, &kept, y3));
        CHECK(pc_row(y3, 0) == pc_fresh(K, { 11, 12, 13, 14, 15, 31, 50 }));
        CHECK(pc_row(y3, 1) == pc_fresh(K, { 21, 22, 41, 42, 43, 60 }));
    }

    /* Too shallow for the window a decode step of this length leaves: refused, not truncated. */
    {
        T s2(RAD_F32, {1, PcK::kN, W});
        T y(RAD_F32, {5, PcK::kN});
        const std::vector<int64_t> one{ 1 };
        CHECK_EQ(pc_launch(K, { { 1, 2, 3, 4, 5 } }, s2, { 0 }, { 1 }, &one, y), RAD_E_SHAPE);
    }
}

/* ============================================================ the PLE ops over a verify tree
 *
 * Definition (docs/OPS.md, "Tree verification of the PLE ops"): a node's row is what a sequential
 * decode of its root-to-node path computes in the same kernel. So every node of an explicit tree is
 * compared, bit for bit, against the last row of a chain over that path run against a copy of the
 * same committed window; and the window the tree step leaves is compacted by hand along an
 * accepted path (what kv_window_compact does, `off` = ngram - 2 for the token window and
 * (width-1)*dilation - 1 for the convolution's) and read by a following step, which must equal a
 * fresh decode of the committed history. */
namespace {

/* Rows of `par` from the root to node r, in order. */
std::vector<int64_t> tree_path(const std::vector<int64_t>& par, int64_t r) {
    std::vector<int64_t> p;
    for (int64_t c = r; c >= 0; c = par[(size_t)c] == -2 ? c - 1 : par[(size_t)c]) p.push_back(c);
    return std::vector<int64_t>(p.rbegin(), p.rend());
}

T clone_t(const T& s) {
    T c(s.t.dtype, std::vector<int64_t>(s.t.shape, s.t.shape + s.t.rank));
    c.mem = s.mem;
    return c;
}

/* Trees over a block of nine rows: a fork at the root, a three-deep branch, siblings at every
 * depth, and a deep chain beyond the dilated conv's deepest tap (4 rows at width 3, dilation 2). */
const std::vector<std::vector<int64_t>>& tree_shapes() {
    static const std::vector<std::vector<int64_t>> v = {
        { -1, 0, 1, 1, 0, 4, 5, 6, 0 },
        { -1, 0, 0, 0, 0, 0, 0, 0, 0 },                 /* a star: every leaf at depth 1 */
        { -1, 0, 1, 2, 3, 4, 5, 6, 7 },                 /* a chain: the plain op */
        { -1, 0, 1, 2, 3, 4, 5, 3, 7 },                 /* deep, with a fork at 3 */
        { -1, -2, -2, -2, -2, -2, -2, -2, -2 },         /* ordinary causal rows: a chain */
    };
    return v;
}

int ng_tree_launch(const NgK& K, const std::vector<int64_t>& tok, const std::vector<int64_t>* par,
                   T& state, int64_t hini, int64_t nacc, T& ids) {
    const int64_t M = (int64_t)tok.size();
    T tt(RAD_I32, {M}); tt.n(tok);
    T tc(RAD_I32, {2}); tc.n({ 0, M });
    T ci(RAD_I32, {1}); ci.n({ 0 });
    T hi(RAD_I32, {1}); hi.n({ hini });
    T na(RAD_I32, {1}); na.n({ nacc });
    T sp(RAD_I32, {M});
    A b; b.t(tt).t(state).t(tc).t(K.tm).t(K.tv).t(K.to).t(ci).t(hi).t(na).t(ids).tnull();
    if (par) { sp.n(*par); b.t(sp); }
    b.i("M", M).i("heads", NgK::kHeads).i("ngram", NgK::kNgram).i("eos", NgK::kEos);
    return run("ngram_ids", b);
}

std::vector<int64_t> ng_row2(const T& ids, int64_t r) {
    std::vector<int64_t> row((size_t)NgK::kHeads);
    for (int64_t h = 0; h < NgK::kHeads; ++h) row[(size_t)h] = ids.iat(r * NgK::kHeads + h);
    return row;
}

}  /* namespace */

TEST(ngram_ids_tree_rows_equal_their_sequential_paths) {
    const NgK K;
    const int64_t M = 9, W = NgK::kCtx - 1 + M;
    /* Token streams: plain, and with the EOS inside the tree. */
    const std::vector<std::vector<int64_t>> toks = {
        { 100, 101, 102, 103, 104, 105, 106, 107, 108 },
        { 100, NgK::kEos, 102, 103, 104, NgK::kEos, 106, 107, 108 },
    };
    const std::vector<std::vector<int64_t>> prevs = { { 51, 52 }, { 51, NgK::kEos } };
    for (const auto& par : tree_shapes())
        for (const auto& tok : toks)
            for (const auto& pv : prevs)
                for (int64_t hini = 0; hini <= 1; ++hini) {
                    /* Read at entry 2 (num_accepted = 3) of a window with stale slack around it. */
                    T st(RAD_I32, {1, W});
                    st.n({ 900, 901, pv[0], pv[1], 902, 903, 904, 905, 906, 907 });
                    const T st0 = clone_t(st);
                    T ids(RAD_I32, {M, NgK::kHeads});
                    CHECK_OK(ng_tree_launch(K, tok, &par, st, hini, 3, ids));

                    for (int64_t r = 0; r < M; ++r) {
                        std::vector<int64_t> pt;
                        for (int64_t q : tree_path(par, r)) pt.push_back(tok[(size_t)q]);
                        T sc = clone_t(st0);
                        T pids(RAD_I32, {(int64_t)pt.size(), NgK::kHeads});
                        CHECK_OK(ng_tree_launch(K, pt, nullptr, sc, hini, 3, pids));
                        CHECK(ng_row2(ids, r) == ng_row2(pids, (int64_t)pt.size() - 1));
                    }

                    /* The window is in row layout: the history shifted by one, then token t at
                     * ctx - 1 + t whatever its parent. */
                    CHECK_EQ(st.iat(0), hini ? pv[1] : NgK::kEos);
                    for (int64_t t = 0; t < M; ++t)
                        CHECK_EQ(st.iat(NgK::kCtx - 1 + t), tok[(size_t)t]);
                }

    /* An absent spec_parent and a chain one are the same op, bit for bit. */
    for (const int shape : { 2, 4 }) {
        const std::vector<int64_t> tok = toks[0];
        T s0(RAD_I32, {1, W}), s1(RAD_I32, {1, W});
        s0.n({ 900, 901, 51, 52, 1, 2, 3, 4, 5, 6 });
        s1.mem = s0.mem;
        T i0(RAD_I32, {M, NgK::kHeads}), i1(RAD_I32, {M, NgK::kHeads});
        CHECK_OK(ng_tree_launch(K, tok, nullptr, s0, 1, 3, i0));
        CHECK_OK(ng_tree_launch(K, tok, &tree_shapes()[(size_t)shape], s1, 1, 3, i1));
        CHECK(i0.mem == i1.mem);
        CHECK(s0.mem == s1.mem);
    }
}

TEST(ngram_ids_tree_window_compacts_to_the_committed_path) {
    const NgK K;
    const int64_t M = 9, W = NgK::kCtx - 1 + M, off = NgK::kNgram - 2;
    const std::vector<int64_t> par = tree_shapes()[0];
    const std::vector<int64_t> tok = { 100, 101, 102, 103, 104, 105, 106, 107, 108 };
    /* Accept every root-to-node path, including the bare anchor (k = 1). */
    for (int64_t last = 0; last < M; ++last) {
        const std::vector<int64_t> path = tree_path(par, last);
        const int64_t k = (int64_t)path.size();
        T st(RAD_I32, {1, W});
        st.n({ 900, 901, 51, 52, 1, 2, 3, 4, 5, 6 });
        T ids(RAD_I32, {M, NgK::kHeads});
        CHECK_OK(ng_tree_launch(K, tok, &par, st, 1, 3, ids));
        /* kv_window_compact: entry off + path[t] -> off + t, ascending. */
        for (int64_t t = 1; t < k; ++t)
            st_int(st.t.dtype, st.t.data, off + t, st.iat(off + path[(size_t)t]));

        /* The next step: one token, read at num_accepted = k. */
        std::vector<int64_t> hist = { 51, 52 };
        for (int64_t q : path) hist.push_back(tok[(size_t)q]);
        hist.push_back(77);
        T nid(RAD_I32, {1, NgK::kHeads});
        CHECK_OK(ng_tree_launch(K, { 77 }, nullptr, st, 1, k, nid));
        CHECK(ng_row2(nid, 0) == ng_fresh(K, hist));
    }
}

TEST(ngram_ids_tree_refuses_what_it_cannot_serve) {
    const NgK K;
    const int64_t M = 4, W = NgK::kCtx - 1 + M;
    const std::vector<int64_t> tok = { 1, 2, 3, 4 };
    T ids(RAD_I32, {M, NgK::kHeads});
    {   /* a child before its parent */
        T st(RAD_I32, {1, W});
        const std::vector<int64_t> bad = { -1, 2, 0, 0 };
        CHECK_EQ(ng_tree_launch(K, tok, &bad, st, 1, 1, ids), RAD_E_INVAL);
    }
    {   /* without num_accepted there is no row layout to compact */
        T st(RAD_I32, {1, W});
        const std::vector<int64_t> good = { -1, 0, 0, 1 };
        T tt(RAD_I32, {M}); tt.n(tok);
        T tc(RAD_I32, {2}); tc.n({ 0, M });
        T ci(RAD_I32, {1}); ci.n({ 0 });
        T hi(RAD_I32, {1}); hi.n({ 1 });
        T sp(RAD_I32, {M}); sp.n(good);
        A a; a.t(tt).t(st).t(tc).t(K.tm).t(K.tv).t(K.to).t(ci).t(hi).tnull().t(ids).tnull().t(sp)
             .i("M", M).i("heads", NgK::kHeads).i("ngram", NgK::kNgram).i("eos", NgK::kEos);
        CHECK_EQ(run("ngram_ids", a), RAD_E_INVAL);
    }
}

namespace {

int pc_tree_launch(const PcK& K, const std::vector<int64_t>& tok, const std::vector<int64_t>* par,
                   T& state, int64_t hini, int64_t nacc, T& y) {
    const int64_t M = (int64_t)tok.size();
    std::vector<float> xv;
    for (int64_t t : tok)
        for (int64_t c = 0; c < PcK::kN; ++c) xv.push_back(pc_x(t, c));
    T x(RAD_F32, {M, PcK::kN}); x.f(xv);
    T tc(RAD_I32, {2}); tc.n({ 0, M });
    T ci(RAD_I32, {1}); ci.n({ 0 });
    T hi(RAD_I32, {1}); hi.n({ hini });
    T na(RAD_I32, {1}); na.n({ nacc });
    T sp(RAD_I32, {M});
    A a; a.t(x).t(K.w).t(state).t(tc).tnull().t(ci).t(hi).t(na).t(y);
    if (par) { sp.n(*par); a.t(sp); }
    a.i("M", M).i("n", PcK::kN).i("width", PcK::kWid).i("dilation", PcK::kDil).s("dtype", "f32");
    return run("ple_conv", a);
}

}  /* namespace */

TEST(ple_conv_tree_rows_equal_their_sequential_paths) {
    const PcK K;
    const int64_t M = 9, W = PcK::kHist - 1 + M;
    /* A warm window from a committed prefill of six tokens, or a cold one. */
    for (const auto& par : tree_shapes())
        for (int64_t hini = 0; hini <= 1; ++hini) {
            T st(RAD_F32, {1, PcK::kN, W});
            {
                T y0(RAD_F32, {6, PcK::kN});
                CHECK_OK(pc_launch(K, { { 11, 12, 13, 14, 15, 16 } }, st, { 0 }, { 0 }, nullptr, y0));
            }
            const T st0 = clone_t(st);
            const std::vector<int64_t> tok = { 21, 22, 23, 24, 25, 26, 27, 28, 29 };
            T y(RAD_F32, {M, PcK::kN});
            CHECK_OK(pc_tree_launch(K, tok, &par, st, hini, 1, y));

            for (int64_t r = 0; r < M; ++r) {
                std::vector<int64_t> pt;
                for (int64_t q : tree_path(par, r)) pt.push_back(tok[(size_t)q]);
                T sc = clone_t(st0);
                T py(RAD_F32, {(int64_t)pt.size(), PcK::kN});
                CHECK_OK(pc_tree_launch(K, pt, nullptr, sc, hini, 1, py));
                CHECK(pc_row(y, r) == pc_row(py, (int64_t)pt.size() - 1));
            }
            /* Row layout: input t at slot hist - 1 + t of every channel. */
            for (int64_t c = 0; c < PcK::kN; ++c)
                for (int64_t t = 0; t < M; ++t)
                    CHECK_EQ(st.at(c * W + PcK::kHist - 1 + t), pc_x(tok[(size_t)t], c));
        }
}

TEST(ple_conv_tree_window_compacts_to_the_committed_path) {
    const PcK K;
    const int64_t M = 9, W = PcK::kHist - 1 + M, off = PcK::kHist - 1;
    const std::vector<int64_t> par = tree_shapes()[0];
    const std::vector<int64_t> tok = { 21, 22, 23, 24, 25, 26, 27, 28, 29 };
    const std::vector<int64_t> committed = { 11, 12, 13, 14, 15, 16 };
    for (int64_t last = 0; last < M; ++last) {
        const std::vector<int64_t> path = tree_path(par, last);
        const int64_t k = (int64_t)path.size();
        T st(RAD_F32, {1, PcK::kN, W});
        T y0(RAD_F32, {6, PcK::kN});
        CHECK_OK(pc_launch(K, { committed }, st, { 0 }, { 0 }, nullptr, y0));
        T y(RAD_F32, {M, PcK::kN});
        CHECK_OK(pc_tree_launch(K, tok, &par, st, 1, 1, y));
        /* kv_window_compact: entry off + path[t] -> off + t, every channel, ascending. */
        for (int64_t c = 0; c < PcK::kN; ++c)
            for (int64_t t = 1; t < k; ++t)
                st_dt(st.t.dtype, st.t.data, c * W + off + t, st.at(c * W + off + path[(size_t)t]));

        std::vector<int64_t> hist = committed;
        for (int64_t q : path) hist.push_back(tok[(size_t)q]);
        hist.push_back(77);
        T ny(RAD_F32, {1, PcK::kN});
        CHECK_OK(pc_tree_launch(K, { 77 }, nullptr, st, 1, k, ny));
        CHECK(pc_row(ny, 0) == pc_fresh(K, hist));
    }
}

TEST(ple_conv_tree_refuses_what_it_cannot_serve) {
    const PcK K;
    const int64_t M = 4, W = PcK::kHist - 1 + M;
    const std::vector<int64_t> tok = { 1, 2, 3, 4 };
    T y(RAD_F32, {M, PcK::kN});
    {
        T st(RAD_F32, {1, PcK::kN, W});
        const std::vector<int64_t> bad = { -1, 2, 0, 0 };
        CHECK_EQ(pc_tree_launch(K, tok, &bad, st, 1, 1, y), RAD_E_INVAL);
    }
    {   /* no num_accepted */
        T st(RAD_F32, {1, PcK::kN, W});
        const std::vector<int64_t> good = { -1, 0, 0, 1 };
        std::vector<float> xv((size_t)(M * PcK::kN), 1.0f);
        T x(RAD_F32, {M, PcK::kN}); x.f(xv);
        T tc(RAD_I32, {2}); tc.n({ 0, M });
        T sp(RAD_I32, {M}); sp.n(good);
        A a; a.t(x).t(K.w).t(st).t(tc).tnull().tnull().tnull().tnull().t(y).t(sp)
             .i("M", M).i("n", PcK::kN).i("width", PcK::kWid).i("dilation", PcK::kDil)
             .s("dtype", "f32");
        CHECK_EQ(run("ple_conv", a), RAD_E_INVAL);
    }
}

TEST(op_cast) {
    T x(RAD_F32, {4}); x.f({2.5f, -1.5f, 200.0f, 1.4f});
    T y(RAD_I8, {4});
    A a; a.t(x).t(y).i("M", 1).i("n", 4).s("from", "f32").s("to", "i8");
    CHECK_OK(run("cast", a));
    CHECK_EQ(y.iat(0), 2LL);      /* ties to even */
    CHECK_EQ(y.iat(1), -2LL);
    CHECK_EQ(y.iat(2), 127LL);    /* saturates rather than wrapping */
    CHECK_EQ(y.iat(3), 1LL);

    /* Integer to integer must not pass through f32: 2^25 + 1 is not representable there. */
    T big(RAD_I64, {1}); big.n({33554433});
    T out(RAD_I32, {1});
    A b; b.t(big).t(out).i("M", 1).i("n", 1).s("from", "i64").s("to", "i32");
    CHECK_OK(run("cast", b));
    CHECK_EQ(out.iat(0), 33554433LL);
}

TEST(op_softmax) {
    T x(RAD_F32, {1, 2}); x.f({0.0f, 0.69314718f});   /* ln 2 */
    T y(RAD_F32, {1, 2});
    A a; a.t(x).t(y).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("softmax", a));
    CHECK_NEAR(y.at(0), 1.0f / 3.0f, 1e-6);
    CHECK_NEAR(y.at(1), 2.0f / 3.0f, 1e-6);
}

/* ================================================================== quantisation */
/* The column quant_act_i8 stores at byte j of a row, as libr4d's quantiser computes it: the middle
 * two dwords of every sixteen-byte step exchanged. */
static int64_t step_src(int64_t j) {
    return (j & ~15LL) + 8 * ((j & 7) >> 2) + 4 * ((j & 15) >> 3) + (j & 3);
}

TEST(op_quant_act_i8) {
    /* Row 0 runs -1.75 .. 2 in quarters: amax 2, scale 2/127, codes round(x * 63.5) with ties to
     * even. Row 1 is zero: the scale floors at 1e-8 / 127 and every code is 0. */
    const int64_t n = 16;
    std::vector<float> xv((size_t)(2 * n), 0.0f);
    for (int64_t c = 0; c < n; ++c) xv[(size_t)c] = 0.25f * (float)(c - 7);
    T x(RAD_F32, {2, n}); x.f(xv);
    T q(RAD_I8, {2, n}), s(RAD_F32, {2, 1});
    /* `asum` at group 8: [2 groups][16 rows], rows 2..15 padding the kernel leaves alone. */
    T as(RAD_F32, {2, 16}); as.f(std::vector<float>(32, -1.0f));
    A a; a.t(x).t(q).t(s).t(as).i("M", 2).i("n", n).i("group", 8);
    CHECK_OK(run("quant_act_i8", a));
    CHECK_NEAR(s.at(0), 2.0f / 127.0f, 1e-9);
    CHECK_NEAR(s.at(1), 1e-8f / 127.0f, 1e-15);
    int64_t bad = 0, sum[2] = { 0, 0 };
    for (int64_t j = 0; j < n; ++j) {
        const int64_t want = (int64_t)std::nearbyintf(xv[(size_t)step_src(j)] * 63.5f);
        bad += q.iat(j) != want;
        bad += q.iat(n + j) != 0;
        sum[step_src(j) / 8] += want;
    }
    CHECK_EQ(bad, 0LL);
    CHECK_EQ(as.at(0), (float)sum[0]);
    CHECK_EQ(as.at(16), (float)sum[1]);
    CHECK_EQ(as.at(1), 0.0f);       /* row 1's sums */
    CHECK_EQ(as.at(2), -1.0f);      /* padding, not written */

    /* A row that is not whole sixteen-byte steps has no step order. */
    T x3(RAD_F32, {1, 8}), q3(RAD_I8, {1, 8}), s3(RAD_F32, {1, 1});
    A b; b.t(x3).t(q3).t(s3).i("M", 1).i("n", 8).i("group", 8);
    CHECK_EQ(run("quant_act_i8", b), RAD_E_SHAPE);
}

TEST(op_quant_act_i8g) {
    T x(RAD_F32, {1, 4}); x.f({1, -2, 0.5f, 2});
    T q(RAD_I8, {1, 4}), s(RAD_F32, {1, 1});
    A a; a.t(x).t(q).t(s).i("M", 1).i("n", 4).i("group", 4);
    CHECK_OK(run("quant_act_i8g", a));
    /* amax 2, so q = round(x * 63.5): 63.5 -> 64 (ties to even), -127, 31.75 -> 32, 127. */
    CHECK_EQ(q.iat(0), 64LL);
    CHECK_EQ(q.iat(1), -127LL);
    CHECK_EQ(q.iat(2), 32LL);
    CHECK_EQ(q.iat(3), 127LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f, 1e-9);

    /* Two groups of two, each with its own scale; an all-zero group gets scale 1. */
    T x2(RAD_F32, {1, 4}); x2.f({1, -2, 0, 0});
    T q2(RAD_I8, {1, 4}), s2(RAD_F32, {1, 2});
    A b; b.t(x2).t(q2).t(s2).i("M", 1).i("n", 4).i("group", 2);
    CHECK_OK(run("quant_act_i8g", b));
    CHECK_NEAR(s2.at(0), 2.0f / 127.0f, 1e-9);
    CHECK_EQ(s2.at(1), 1.0f);
    CHECK_EQ(q2.iat(0), 64LL);
    CHECK_EQ(q2.iat(3), 0LL);
}

TEST(op_had_quant_act_i8) {
    /* One rotation block of two: the butterfly is [a+b, a-b]. */
    T x(RAD_F32, {1, 2}); x.f({1, 3});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(x).t(q).t(s).i("M", 1).i("n", 2).i("group", 2);
    CHECK_OK(run("had_quant_act_i8", a));
    /* rotated [4, -2], amax 4, q = round(v * 127/4) = [127, -63.5 -> -64] */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), -64LL);
    /* the scale carries the 1/sqrt(group) the rotation left out */
    CHECK_NEAR(s.at(0), 4.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_rmsnorm_had_quant_i8) {
    /* Operand 1 is the optional `residual`; absent here, exercised below. */
    T x(RAD_F32, {1, 2}); x.f({1, 1});
    T w(RAD_F32, {2});    w.f({1, 1});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(x).tnull().t(w).t(q).t(s).i("M", 1).i("n", 2).d("eps", 0).i("group", 2);
    CHECK_OK(run("rmsnorm_had_quant_i8", a));
    /* rms 1, normed [1,1], rotated [2,0], amax 2 */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), 0LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f / 1.41421356f, 1e-9);
}

/* The fused_add_rms_norm form. The residual add happens first and is written back (which makes
 * `residual` inout and is what the next layer's shortcut reads), and `out_bf16` carries the
 * post-norm, pre-rotation tensor for a layer that also has a bf16 consumer of it. */
TEST(op_rmsnorm_had_quant_i8_residual) {
    T x(RAD_F32, {1, 2});   x.f({1, 1});
    T res(RAD_F32, {1, 2}); res.f({1, 1});
    T w(RAD_F32, {2});      w.f({1, 1});
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1}), ob(RAD_F32, {1, 2});
    A a; a.t(x).t(res).t(w).t(q).t(s).t(ob).i("M", 1).i("n", 2).d("eps", 0).i("group", 2);
    CHECK_OK(run("rmsnorm_had_quant_i8", a));
    /* residual becomes [2, 2]; rms 2; normed [1, 1]; rotated [2, 0]. */
    CHECK_EQ(res.at(0), 2.0f);
    CHECK_EQ(res.at(1), 2.0f);
    CHECK_NEAR(ob.at(0), 1.0f, 1e-6);
    CHECK_NEAR(ob.at(1), 1.0f, 1e-6);
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), 0LL);
    CHECK_NEAR(s.at(0), 2.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_gated_had_quant_i8) {
    /* `b` absent is the packed form: `a` is [M, 2n], gate first half, up second. */
    T gu(RAD_F32, {1, 4}); gu.f({0, 0, 5, 7});     /* gate [0,0], up [5,7] */
    T q(RAD_I8, {1, 2}), s(RAD_F32, {1});
    A a; a.t(gu).tnull().t(q).t(s).i("M", 1).i("n", 2).i("group", 2).s("act", "sigmoid");
    CHECK_OK(run("gated_had_quant_i8", a));
    /* sigmoid(0)*[5,7] = [2.5, 3.5]; rotated [6, -1]; amax 6 */
    CHECK_EQ(q.iat(0), 127LL);
    CHECK_EQ(q.iat(1), (long long)std::nearbyintf(-1.0f * 127.0f / 6.0f));
    CHECK_NEAR(s.at(0), 6.0f / 127.0f / 1.41421356f, 1e-9);

    /* The split form: the same numbers from two operands, with `mode` as libr4d's integer spelling
     * of the same activation. The two readings must agree, or the schema's union of them would
     * mean two different ops. */
    T ga(RAD_F32, {1, 2}); ga.f({0, 0});
    T up(RAD_F32, {1, 2}); up.f({5, 7});
    T q2(RAD_I8, {1, 2}), s2(RAD_F32, {1});
    A b; b.t(ga).t(up).t(q2).t(s2).i("M", 1).i("n", 2).i("group", 2).i("mode", 1);
    CHECK_OK(run("gated_had_quant_i8", b));
    CHECK_EQ(q2.iat(0), 127LL);
    CHECK_EQ(q2.iat(1), (long long)std::nearbyintf(-1.0f * 127.0f / 6.0f));
    CHECK_NEAR(s2.at(0), 6.0f / 127.0f / 1.41421356f, 1e-9);
}

TEST(op_dequant) {
    T q(RAD_I8, {1, 4}); q.n({1, 2, 3, 4});
    T s(RAD_F32, {1, 2}); s.f({0.5f, 2.0f});
    T y(RAD_F32, {1, 4});
    A a; a.t(q).t(s).t(y).i("M", 1).i("n", 4).i("group", 2).s("dtype", "i8");
    CHECK_OK(run("dequant", a));
    CHECK_EQ(y.at(0), 0.5f);
    CHECK_EQ(y.at(1), 1.0f);
    CHECK_EQ(y.at(2), 6.0f);
    CHECK_EQ(y.at(3), 8.0f);
}

/* ================================================================== gemm */
TEST(op_gemm_nt) {
    /* 2x3 times a transposed 4x3, checkable by hand. */
    T a_(RAD_F32, {2, 3}); a_.f({1, 2, 3, 4, 5, 6});
    T b_(RAD_F32, {4, 3}); b_.f({1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1});
    T y(RAD_F32, {2, 4});
    A a; a.t(a_).t(b_).t(y).i("M", 2).i("N", 4).i("K", 3).s("dtype", "f32");
    CHECK_OK(run("gemm_nt", a));
    CHECK_EQ(y.at(0), 1.0f); CHECK_EQ(y.at(1), 2.0f);
    CHECK_EQ(y.at(2), 3.0f); CHECK_EQ(y.at(3), 6.0f);
    CHECK_EQ(y.at(4), 4.0f); CHECK_EQ(y.at(5), 5.0f);
    CHECK_EQ(y.at(6), 6.0f); CHECK_EQ(y.at(7), 15.0f);

    T bias(RAD_F32, {4}); bias.f({1, 2, 3, 4});
    T y2(RAD_F32, {2, 4});
    A b; b.t(a_).t(b_).t(bias).tnull().t(y2).i("M", 2).i("N", 4).i("K", 3).s("dtype", "f32");
    CHECK_OK(run("gemm_nt_bias", b));
    CHECK_EQ(y2.at(0), 2.0f);
    CHECK_EQ(y2.at(7), 19.0f);

    /* The same GEMM in bf16: every value here is exact in bf16, so the answer is exact and the
     * bf16 path is checked by equality, not tolerance. */
    T ab(RAD_BF16, {2, 3}); ab.f({1, 2, 3, 4, 5, 6});
    T bb(RAD_BF16, {4, 3}); bb.f({1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1});
    T yb(RAD_BF16, {2, 4});
    A c; c.t(ab).t(bb).t(yb).i("M", 2).i("N", 4).i("K", 3).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", c));
    CHECK_EQ(yb.at(3), 6.0f);
    CHECK_EQ(yb.at(7), 15.0f);
}

/* Runtime LoRA, by hand. Every value is exact in its dtype, so the answers are equalities. Which
 * rows are touched is under test as much as what lands in them: a row whose slot is -1 or S is a
 * base row, its `t` row keeps the sentinel shrink found there and its `y` row keeps its bits, a -0
 * included (which `y + 0` would turn into +0). */
TEST(op_lora_shrink) {
    /* `a` is a column slice: columns 1..4 of a [3, 6] buffer, row stride 6. */
    T aw(RAD_BF16, {3, 6});
    aw.f({9, 1, 2, 3, 4, 9,   9, 5, 6, 7, 8, 9,   9, 1, 1, 1, 1, 9});
    T a_ = aw;                                   /* the slice shares aw's bytes */
    a_.t.data = (uint16_t*)aw.mem.data() + 1;
    a_.t.shape[1] = 4;
    T sl(RAD_I32, {3}); sl.n({1, -1, 0});
    T la(RAD_BF16, {2, 2, 4});
    la.f({1, 0, 0, 0,   0, 1, 0, 0,               /* slot 0: j0, j1 */
          1, 1, 1, 1,   1, -1, 0, 0.5f});         /* slot 1 */
    T t(RAD_F32, {3, 2}); t.f({99, 99, 99, 99, 99, 99});
    A a; a.t(a_).tnull().t(sl).t(la).t(t)
          .i("M", 3).i("K", 4).i("R", 2).i("S", 2).i("group", 0).s("dtype", "bf16");
    CHECK_OK(run("lora_shrink", a));
    CHECK_EQ(t.at(0), 10.0f);                    /* 1+2+3+4 */
    CHECK_EQ(t.at(1), 1.0f);                     /* 1-2+0+4*0.5 */
    CHECK_EQ(t.at(2), 99.0f);                    /* base row: not written */
    CHECK_EQ(t.at(3), 99.0f);
    CHECK_EQ(t.at(4), 1.0f);                     /* slot 0 picks a[2][0], a[2][1] */
    CHECK_EQ(t.at(5), 1.0f);

    /* An E4M3 activation with an f32 scale every two columns: x = code * its group's scale. */
    T q(RAD_F8E4M3, {1, 4}); q.f({1, 2, 0.5f, -1});
    T qs(RAD_F32, {1, 2});   qs.f({2.0f, 0.25f});
    T sq(RAD_I32, {1});      sq.n({0});
    T lq(RAD_BF16, {1, 1, 4}); lq.f({1, 1, 1, 1});
    T tq(RAD_F32, {1, 1});
    A b; b.t(q).t(qs).t(sq).t(lq).t(tq)
          .i("M", 1).i("K", 4).i("R", 1).i("S", 1).i("group", 2).s("dtype", "fp8_e4m3");
    CHECK_OK(run("lora_shrink", b));
    CHECK_EQ(tq.at(0), 5.875f);                  /* 2 + 4 + 0.125 - 0.25 */

    /* A code with no scale has no value: refused, never read as scale one. */
    A c; c.t(q).tnull().t(sq).t(lq).t(tq)
          .i("M", 1).i("K", 4).i("R", 1).i("S", 1).i("group", 0).s("dtype", "fp8_e4m3");
    CHECK_EQ(run("lora_shrink", c), RAD_E_INVAL);
    /* A `group` the stream does not state is a caller describing other bytes. */
    A d; d.t(q).t(qs).t(sq).t(lq).t(tq)
          .i("M", 1).i("K", 4).i("R", 1).i("S", 1).i("group", 4).s("dtype", "fp8_e4m3");
    CHECK_EQ(run("lora_shrink", d), RAD_E_SHAPE);
}

TEST(op_lora_expand) {
    T t(RAD_F32, {4, 2}); t.f({10, 1,   5, 5,   1, 1,   7, 7});
    T sl(RAD_I32, {4});   sl.n({1, -1, 0, 2});   /* 2 == S: a base row like -1 */
    T lb(RAD_BF16, {2, 3, 2});
    lb.f({1, 0,   0, 1,   1, 1,                   /* slot 0: n0, n1, n2 */
          2, 0,   0, 0.5f, 1, -1});               /* slot 1 */
    T sc(RAD_F32, {2, 3}); sc.f({1, 1, 1,   0.5f, 2, 1});
    /* `y` is a column slice too: columns 0..2 of a [4, 5] buffer. */
    T yw(RAD_BF16, {4, 5});
    yw.f({1, 1, 1, 9, 9,   -0.0f, 3, 7, 9, 9,   0, 0, 0, 9, 9,   -0.0f, -0.0f, 5, 9, 9});
    T y = yw;
    y.t.shape[1] = 3;
    A a; a.t(t).t(sl).t(lb).t(sc).t(y).i("M", 4).i("N", 3).i("R", 2).i("S", 2).s("dtype", "bf16");
    CHECK_OK(run("lora_expand", a));
    const uint16_t* yr = (const uint16_t*)yw.mem.data();
    CHECK_EQ(yw.at(0), 11.0f);                   /* 1 + 0.5 * (10*2 + 1*0) */
    CHECK_EQ(yw.at(1), 2.0f);                    /* 1 + 2 * (1*0.5) */
    CHECK_EQ(yw.at(2), 10.0f);                   /* 1 + 1 * (10 - 1) */
    CHECK_EQ((long long)yr[5], 0x8000LL);        /* base row: -0 keeps its sign bit */
    CHECK_EQ(yw.at(6), 3.0f);
    CHECK_EQ(yw.at(7), 7.0f);
    CHECK_EQ(yw.at(10), 1.0f);                   /* slot 0 over t[2] = (1, 1) */
    CHECK_EQ(yw.at(11), 1.0f);
    CHECK_EQ(yw.at(12), 2.0f);
    CHECK_EQ((long long)yr[15], 0x8000LL);       /* slot S: a base row */
    CHECK_EQ((long long)yr[16], 0x8000LL);
    CHECK_EQ(yw.at(17), 5.0f);
    for (int r = 0; r < 4; ++r) {                /* the slice's neighbours are not touched */
        CHECK_EQ(yw.at(r * 5 + 3), 9.0f);
        CHECK_EQ(yw.at(r * 5 + 4), 9.0f);
    }
}

TEST(op_gemm_nt_q) {
    /* int8 x int8 with a per-row activation scale and a per-row weight scale. */
    T a_(RAD_I8, {2, 2}); a_.n({1, 2, 3, 4});
    T as(RAD_F32, {2});   as.f({0.5f, 1.0f});
    T b_(RAD_I8, {2, 2}); b_.n({1, 1, 2, 0});
    T bs(RAD_F32, {2});   bs.f({2.0f, 1.0f});
    T y(RAD_F32, {2, 2});
    A a; a.t(a_).t(as).t(b_).t(bs).t(y)
          .i("M", 2).i("N", 2).i("K", 2).i("group", 2).s("dtype", "i8a8");
    CHECK_OK(run("gemm_nt_q", a));
    CHECK_EQ(y.at(0), 3.0f);    /* (1*1 + 2*1) * 0.5 * 2 */
    CHECK_EQ(y.at(1), 1.0f);    /* (1*2 + 2*0) * 0.5 * 1 */
    CHECK_EQ(y.at(2), 14.0f);   /* (3*1 + 4*1) * 1.0 * 2 */
    CHECK_EQ(y.at(3), 6.0f);    /* (3*2 + 4*0) * 1.0 * 1 */

    /* A 4-bit weight in the packed layout ref_common.h documents: element 2i in the low nibble,
     * value = (code ^ 8) - 8. Row 0 is byte 0x21 -> codes 1, 2 -> values 1, 2; row 1 is 0x0f ->
     * codes 0xf, 0x0 -> values -1, 0. At w4a8 the activation is quant_act_i8's, sixteen columns a
     * step; only columns 0 and 1 are non-zero, which the step order leaves in place. */
    std::vector<int64_t> a16(32, 0);
    a16[0] = 1; a16[1] = 2; a16[16] = 3; a16[17] = 4;
    T a4(RAD_I8, {2, 16}); a4.n(a16);
    std::vector<uint8_t> w16(16, 0);
    w16[0] = 0x21; w16[8] = 0x0f;
    T bw(RAD_I4, {2, 16}); bw.raw(w16);
    T bws(RAD_F32, {2});  bws.f({1.0f, 1.0f});
    T y2(RAD_F32, {2, 2});
    A b; b.t(a4).t(as).t(bw).t(bws).t(y2)
          .i("M", 2).i("N", 2).i("K", 16).i("group", 16).s("dtype", "w4a8");
    CHECK_OK(run("gemm_nt_q", b));
    CHECK_EQ(y2.at(0), 2.5f);   /* (1*1 + 2*2) * 0.5 */
    CHECK_EQ(y2.at(1), -0.5f);  /* (1*-1 + 2*0) * 0.5 */
    CHECK_EQ(y2.at(2), 11.0f);  /* (3*1 + 4*2) * 1.0 */
    CHECK_EQ(y2.at(3), -3.0f);

    /* mxfp4: e2m1 elements with an e8m0 block exponent. Codes 0x2 (+1) and 0x4 (+2); the byte is
     * 0x42 and the shared exponent 128 doubles both. */
    T bm(RAD_FP4E2M1, {1, 2}); bm.raw({0x42});
    T bms(RAD_U8, {1});      bms.n({128});
    T y3(RAD_F32, {2, 1});
    A c; c.t(a_).t(as).t(bm).t(bms).t(y3)
          .i("M", 2).i("N", 1).i("K", 2).i("group", 2).s("dtype", "mxfp4a8");
    CHECK_OK(run("gemm_nt_q", c));
    CHECK_EQ(y3.at(0), (1 * 2.0f + 2 * 4.0f) * 0.5f);
    CHECK_EQ(y3.at(1), (3 * 2.0f + 4 * 4.0f) * 1.0f);
}

/* quant_act_i8 into gemm_nt_q at w8a8, the pair a w8a8 linear issues: the GEMM must read the codes
 * back out of the step order. Integer sums are exact in f32, so only the two scale products
 * round. */
TEST(op_gemm_nt_q_reads_quant_act_i8s_step_order) {
    const int64_t M = 3, N = 5, K = 32;
    uint32_t rng = 11;
    auto nxt = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 9; };
    std::vector<float> xv((size_t)(M * K));
    for (auto& v : xv) v = 0.01f * (float)((int)(nxt() % 401) - 200);
    T x(RAD_F32, {M, K}); x.f(xv);
    T q(RAD_I8, {M, K}), s(RAD_F32, {M, 1});
    A a; a.t(x).t(q).t(s).tnull().i("M", M).i("n", K).i("group", K);
    CHECK_OK(run("quant_act_i8", a));

    std::vector<int64_t> wv((size_t)(N * K));
    for (auto& v : wv) v = (int64_t)(nxt() % 255) - 127;
    T w(RAD_I8, {N, K}); w.n(wv);
    T ws(RAD_F32, {N}); ws.f({ 0.5f, 0.25f, 1.0f, 2.0f, 0.125f });
    T y(RAD_F32, {M, N});
    A b; b.t(q).t(s).t(w).t(ws).t(y)
          .i("M", M).i("N", N).i("K", K).i("group", K).s("dtype", "w8a8");
    CHECK_OK(run("gemm_nt_q", b));

    int64_t bad = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += q.iat(m * K + step_src(k)) * wv[(size_t)(n * K + k)];
            const float want = (float)acc * s.at(m) * ws.at(n);
            bad += std::fabs(y.at(m * N + n) - want) > 1e-6f * std::fabs(want) + 1e-6f;
        }
    CHECK_EQ(bad, 0LL);
}

/* A weight whose row parts meet their own copy of the activation: parts = 2 rows [0, 16) and
 * [16, 32) over K = 128, the copies side by side in one row of `a`. The parted product must equal
 * each part's rows against its own copy, computed here as two plain calls. */
TEST(op_gemm_nt_q_parts_read_their_own_activation_copy) {
    const int64_t M = 3, N = 32, K = 128, P = 2;
    uint32_t rng = 7;
    auto nxt = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 9; };
    T a2(RAD_F8E4M3, {M, P * K});
    std::vector<float> av((size_t)(M * P * K));
    for (auto& v : av) v = (float)((int)(nxt() % 15) - 7) * 0.25f;
    a2.f(av);
    T as2(RAD_F32, {M, P});
    std::vector<float> asv((size_t)(M * P));
    for (auto& v : asv) v = 0.01f * (float)(1 + nxt() % 9);
    as2.f(asv);
    T b_(RAD_FP4E2M1, {N, K});
    for (int64_t i = 0; i < N * K / 2; ++i) b_.mem[(size_t)i] = (uint8_t)nxt();
    T bs(RAD_U8, {N, K / 32});
    for (int64_t i = 0; i < N * K / 32; ++i) bs.mem[(size_t)i] = (uint8_t)(124 + nxt() % 5);

    T y(RAD_BF16, {M, N});
    A a; a.t(a2).t(as2).t(b_).t(bs).t(y).tnull().tnull()
          .i("M", M).i("N", N).i("K", K).i("group", 32).s("dtype", "mxfp4a8").i("pb1", 16);
    CHECK_OK(run("gemm_nt_q", a));

    for (int64_t p = 0; p < P; ++p) {
        T ap(RAD_F8E4M3, {M, K}), asp(RAD_F32, {M, 1});
        std::vector<float> v1((size_t)(M * K)), v2((size_t)M);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) v1[(size_t)(m * K + k)] = av[(size_t)(m * P * K + p * K + k)];
            v2[(size_t)m] = asv[(size_t)(m * P + p)];
        }
        ap.f(v1);
        asp.f(v2);
        T bp(RAD_FP4E2M1, {N / P, K}), bsp(RAD_U8, {N / P, K / 32});
        std::memcpy(bp.mem.data(), b_.mem.data() + p * (N / P) * K / 2, (size_t)((N / P) * K / 2));
        std::memcpy(bsp.mem.data(), bs.mem.data() + p * (N / P) * K / 32, (size_t)((N / P) * K / 32));
        T yp(RAD_BF16, {M, N / P});
        A c; c.t(ap).t(asp).t(bp).t(bsp).t(yp).tnull().tnull()
              .i("M", M).i("N", N / P).i("K", K).i("group", 32).s("dtype", "mxfp4a8");
        CHECK_OK(run("gemm_nt_q", c));
        int bad = 0;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N / P; ++n)
                if (y.at(m * N + p * (N / P) + n) != yp.at(m * (N / P) + n)) ++bad;
        CHECK_EQ(bad, 0);
    }
}

/* A block grid whose last block is partial, on both axes. K = 300 at 128 is scales over 128, 128
 * and 44 columns; N = 129 is two row blocks (the second of one row), N = 300 three. With every
 * code 1, output n is the sum over column groups of width times scale[n / 128][g]. The block is
 * not extent / block count (100 at 300), and a scale stream with fewer rows than the operand is
 * still a block grid, not a per-row stream to read past. Checked with and without `group`. */
TEST(op_gemm_nt_q_partial_block_grid) {
    const int64_t K = 300, G = 128, gk = 3;
    const int64_t width[3] = { 128, 128, 44 };
    for (int64_t N : { 129, 300 }) {
        const int64_t gn = (N + G - 1) / G;
        T a_(RAD_F32, {1, K}); a_.f(std::vector<float>((size_t)K, 1.0f));
        T b_(RAD_F32, {N, K}); b_.f(std::vector<float>((size_t)(N * K), 1.0f));
        std::vector<float> sv;
        for (int64_t r = 0; r < gn; ++r)
            for (int64_t g = 0; g < gk; ++g) sv.push_back((float)(1 + 10 * r + g));
        T bs(RAD_F32, {gn, gk}); bs.f(sv);
        for (int with_group = 0; with_group < 2; ++with_group) {
            T y(RAD_F32, {1, N});
            A a; a.t(a_).tnull().t(b_).t(bs).t(y).i("M", 1).i("N", N).i("K", K).s("dtype", "f32");
            if (with_group) a.i("group", G);
            CHECK_OK(run("gemm_nt_q", a));
            for (int64_t n = 0; n < N; ++n) {
                float want = 0.0f;
                for (int64_t g = 0; g < gk; ++g)
                    want += (float)width[g] * sv[(size_t)((n / G) * gk + g)];
                CHECK_EQ(y.at(n), want);
            }
        }
    }
}

/* ================================================================== the fp8 fused quantisers */

/* gemm_nt_q_gated against the pair it replaces (gemm_nt_q then gated_quant_fp8), which libref and
 * libr4d both claim it is byte-identical to. The paths meet only at `quant_dot` (the pair through
 * the compile-time-dispatched gemm_quant, the fold through runtime-dispatched quant_dot), so
 * agreement is evidence. The per-op oracle cannot make this comparison, since the container stores
 * an fp8a8 weight in libr4d's WMMA fragment order; here both sides are row-major. Shapes are the
 * smallest the fold admits (a 128-column scale group, a 128x128 weight-scale block). */
TEST(op_gemm_nt_q_gated_is_the_pair) {
    const int64_t M = 3, K = 256, nout = 256, N = 2 * nout;
    const int64_t kb = K / 128, nb = N / 128, ng = nout / 128;

    /* One stream for both planes: a shared generator keeps the weight and activation uncorrelated,
     * which a per-tensor pattern does not. */
    uint32_t rs = 0x9e3779b9u;
    auto rnd = [&]() {
        rs = rs * 1664525u + 1013904223u;
        return (float)((int32_t)((rs >> 13) & 0xffu) - 128) / 64.0f;
    };

    T a_(RAD_F8E4M3, {M, K});
    for (int64_t i = 0; i < M * K; ++i) st_dt(RAD_F8E4M3, a_.t.data, i, rnd() * 2.0f);
    T b_(RAD_F8E4M3, {N, K});
    for (int64_t i = 0; i < N * K; ++i) st_dt(RAD_F8E4M3, b_.t.data, i, rnd());
    T as(RAD_F32, {M, kb});
    for (int64_t i = 0; i < M * kb; ++i) st_dt(RAD_F32, as.t.data, i, 0.5f + 0.25f * (float)i);
    T bs(RAD_BF16, {nb, kb});
    for (int64_t i = 0; i < nb * kb; ++i) st_dt(RAD_BF16, bs.t.data, i, 0.75f + 0.125f * (float)i);

    /* The fold: no [M, N] plane is ever written. */
    T q1(RAD_F8E4M3, {M, nout}), s1(RAD_F32, {M, ng}), y1(RAD_BF16, {M, nout});
    A f; f.t(a_).t(as).t(b_).t(bs).t(q1).t(s1).t(y1)
          .i("M", M).i("N", N).i("K", K).i("group", 128)
          .s("dtype", "fp8a8").s("act", "silu");
    CHECK_OK(run("gemm_nt_q_gated", f));

    /* The pair, over the plane the fold does not write. `N` is the weight's rows on the GEMM and
     * `n` is half of it on the quantiser, the parameter a caller coming from gemm_nt_q gets wrong;
     * r4d_fuse.cpp's decomposition of this op names it for the same reason. */
    T y(RAD_BF16, {M, N});
    A g; g.t(a_).t(as).t(b_).t(bs).t(y)
          .i("M", M).i("N", N).i("K", K).i("group", 128).s("dtype", "fp8a8");
    CHECK_OK(run("gemm_nt_q", g));

    T q2(RAD_F8E4M3, {M, nout}), s2(RAD_F32, {M, ng}), y2(RAD_BF16, {M, nout});
    A h; h.t(y).t(q2).t(s2).t(y2)
          .i("M", M).i("n", nout).i("group", 128).s("dtype", "bf16").s("act", "silu");
    CHECK_OK(run("gated_quant_fp8", h));

    /* Two planes of zeros would agree, so the operands are asserted live first: an all-zero group
     * takes scale 1 and codes 0 by fp8_row's rule, which is what an unwritten output looks like. */
    int64_t nz = 0;
    for (int64_t i = 0; i < M * nout; ++i) if (q1.mem[(size_t)i]) ++nz;
    REQUIRE(nz > M * nout / 2);

    int64_t bad_q = 0, bad_s = 0, bad_y = 0;
    for (int64_t i = 0; i < M * nout; ++i) {
        if (q1.mem[(size_t)i] != q2.mem[(size_t)i]) ++bad_q;
        if (y1.mem[(size_t)(2 * i)]     != y2.mem[(size_t)(2 * i)] ||
            y1.mem[(size_t)(2 * i + 1)] != y2.mem[(size_t)(2 * i + 1)]) ++bad_y;
    }
    for (int64_t i = 0; i < M * ng; ++i) if (s1.at(i) != s2.at(i)) ++bad_s;
    CHECK_EQ(bad_q, (int64_t)0);
    CHECK_EQ(bad_s, (int64_t)0);
    CHECK_EQ(bad_y, (int64_t)0);

    /* The gate half is the first n weight rows and the up half the second, so swapping them gives a
     * different answer; without this the case would pass on an implementation reading either. */
    T q3(RAD_F8E4M3, {M, nout}), s3(RAD_F32, {M, ng});
    T yswap(RAD_BF16, {M, N});
    for (int64_t m = 0; m < M; ++m)
        for (int64_t i = 0; i < nout; ++i) {
            yswap.mem[(size_t)((m * N + i) * 2)]          = y.mem[(size_t)((m * N + nout + i) * 2)];
            yswap.mem[(size_t)((m * N + i) * 2 + 1)]      = y.mem[(size_t)((m * N + nout + i) * 2 + 1)];
            yswap.mem[(size_t)((m * N + nout + i) * 2)]   = y.mem[(size_t)((m * N + i) * 2)];
            yswap.mem[(size_t)((m * N + nout + i) * 2 + 1)] = y.mem[(size_t)((m * N + i) * 2 + 1)];
        }
    A k; k.t(yswap).t(q3).t(s3)
          .i("M", M).i("n", nout).i("group", 128).s("dtype", "bf16").s("act", "silu");
    CHECK_OK(run("gated_quant_fp8", k));
    int64_t differs = 0;
    for (int64_t i = 0; i < M * nout; ++i) if (q1.mem[(size_t)i] != q3.mem[(size_t)i]) ++differs;
    CHECK(differs > M * nout / 4);
}

/* gated_quant_fp8 into an int8 `q` (a shared expert's gate in front of its int8 down projection,
 * ActFP8::q8_fed) writes quant_act_i8g's codes over the bf16 product, as libr4d's kernel does:
 * scale amax / 127 per group, not E4M3's amax / 448 saturated into int8. Checked against the pair
 * it replaces, the product's bf16 plane through quant_act_i8g. */
TEST(op_gated_quant_fp8_into_int8_is_quant_act_i8g) {
    const int64_t M = 3, n = 256, ng = n / 128;
    T gu(RAD_BF16, {M, 2 * n});
    uint32_t rs = 0x2545f491u;
    for (int64_t i = 0; i < M * 2 * n; ++i) {
        rs = rs * 1664525u + 1013904223u;
        st_dt(RAD_BF16, gu.t.data, i, (float)((int32_t)((rs >> 12) & 0x3ffu) - 512) / 128.0f);
    }
    T q(RAD_I8, {M, n}), s(RAD_F32, {M, ng}), y(RAD_BF16, {M, n});
    A a; a.t(gu).t(q).t(s).t(y).i("M", M).i("n", n).i("group", 128).s("dtype", "bf16")
          .s("act", "silu");
    CHECK_OK(run("gated_quant_fp8", a));

    T q2(RAD_I8, {M, n}), s2(RAD_F32, {M, ng});
    A b; b.t(y).t(q2).t(s2).i("M", M).i("n", n).i("group", 128);
    CHECK_OK(run("quant_act_i8g", b));
    int64_t bad = 0, top = 0;
    for (int64_t i = 0; i < M * n; ++i) {
        bad += q.iat(i) != q2.iat(i);
        top += q.iat(i) == 127 || q.iat(i) == -127;
    }
    for (int64_t i = 0; i < M * ng; ++i) bad += s.at(i) != s2.at(i);
    CHECK_EQ(bad, (int64_t)0);
    /* One code a group sits at the amax; saturating E4M3 codes would pin many more. */
    CHECK(top >= M * ng && top < M * n / 8);
}

TEST(op_vocab_edges) {
    T wte(RAD_F32, {3, 2}); wte.f({1, 2, 3, 4, 5, 6});
    T tok(RAD_I32, {3});    tok.n({2, 0, -1});
    T x(RAD_F32, {3, 2});
    A a; a.t(tok).t(wte).t(x).i("M", 3).i("n_embd", 2).i("n_vocab", 3).s("dtype", "f32");
    CHECK_OK(run("embed_lookup", a));
    CHECK_EQ(x.at(0), 5.0f); CHECK_EQ(x.at(1), 6.0f);
    CHECK_EQ(x.at(2), 1.0f); CHECK_EQ(x.at(3), 2.0f);
    CHECK_EQ(x.at(4), 0.0f); CHECK_EQ(x.at(5), 0.0f);   /* a negative id is padding */

    /* A runtime gather scale multiplies the rows that are found; a missed row stays zero, not zero
     * times the scale. Without it, the lookup is the plain gather (the first block). */
    T xs(RAD_F32, {3, 2});
    A as; as.t(tok).t(wte).t(xs).i("M", 3).i("n_embd", 2).i("n_vocab", 3)
          .s("dtype", "f32").d("mult", 2.0);
    CHECK_OK(run("embed_lookup", as));
    CHECK_EQ(xs.at(0), 10.0f); CHECK_EQ(xs.at(1), 12.0f);
    CHECK_EQ(xs.at(2), 2.0f);  CHECK_EQ(xs.at(3), 4.0f);
    CHECK_EQ(xs.at(4), 0.0f);  CHECK_EQ(xs.at(5), 0.0f);

    /* An id at or above n_vocab fails the launch rather than returning a plausible row. */
    T bad(RAD_I32, {1}); bad.n({7});
    T x2(RAD_F32, {1, 2});
    A b; b.t(bad).t(wte).t(x2).i("M", 1).i("n_embd", 2).i("n_vocab", 3).s("dtype", "f32");
    CHECK(run("embed_lookup", b) == RAD_E_INVAL);

    T h(RAD_F32, {1, 2});   h.f({1, 2});
    T lm(RAD_F32, {3, 2});  lm.f({1, 0, 0, 1, 1, 1});
    T lg(RAD_F32, {1, 3});
    A c; c.t(h).t(lm).t(lg).i("M", 1).i("n_vocab", 3).i("n_embd", 2).s("dtype", "f32");
    CHECK_OK(run("logits_gemm", c));
    CHECK_EQ(lg.at(0), 1.0f); CHECK_EQ(lg.at(1), 2.0f); CHECK_EQ(lg.at(2), 3.0f);
}

/* ================================================================== attention */
TEST(op_attn_paged) {
    /* Two query tokens, two query heads over one kv head, head_dim 2, one block of 2 slots.
     *   slot 0: K = [1,0]  V = [1,0]
     *   slot 1: K = [0,1]  V = [0,1]
     * Query head 0 takes q = [1,0] then [0,1]; head 1 takes q = [0,1] then [1,0]. */
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {2, 2, 2});
    q.f({ 1, 0,   0, 1,      /* token 0: head0 [1,0], head1 [0,1] */
          0, 1,   1, 0 });   /* token 1: head0 [0,1], head1 [1,0] */
    T kv(RAD_F32, {1, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,       /* slot 0: K then V */
           0, 1, 0, 1 });    /* slot 1 */
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1});    su.n({2});
    T out(RAD_F32, {2, 2, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", a));

    /* Token 0 sits at position 0 and sees key 0 only, so the softmax is trivial and both heads
     * return V0 = [1, 0]. */
    CHECK_NEAR(out.at(0), 1.0f, 1e-6); CHECK_NEAR(out.at(1), 0.0f, 1e-6);
    CHECK_NEAR(out.at(2), 1.0f, 1e-6); CHECK_NEAR(out.at(3), 0.0f, 1e-6);

    /* Token 1 sees both keys. Head 0, q = [0,1]: scores 0 and 1/sqrt(2); the weights are
     * exp(-0.70710678) = 0.49306869 and 1, summing to 1.49306869. */
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(out.at(4), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(5), 1.0f / sum, 1e-6);
    /* Head 1, q = [1,0]: the mirror image. */
    CHECK_NEAR(out.at(6), 1.0f / sum, 1e-6);
    CHECK_NEAR(out.at(7), w0 / sum, 1e-6);

    /* A window of 1 leaves token 1 seeing only its own key. */
    T out2(RAD_F32, {2, 2, 2});
    A b; b.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out2)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 1).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", b));
    CHECK_NEAR(out2.at(4), 0.0f, 1e-6);
    CHECK_NEAR(out2.at(5), 1.0f, 1e-6);
}

TEST(op_attn_paged_sink) {
    /* op_attn_paged's cache and queries, non-causal, with a sink logit a head: token 0 now sees both
     * keys too, and each sink adds exp(sink) to the denominator only. Head 0, token 0, q = [1,0]:
     * scores 1/sqrt(2) and 0, so weights 1, exp(-1/sqrt(2)) and exp(sink0 - 1/sqrt(2)). */
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {2, 2, 2});
    q.f({ 1, 0,   0, 1,
          0, 1,   1, 0 });
    T kv(RAD_F32, {1, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,
           0, 1, 0, 1 });
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1});    su.n({2});
    T sk(RAD_F32, {2});    sk.f({ 0.5f, 3.0f });     /* head 1's sink lies above every score */
    T out(RAD_F32, {2, 2, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(sk).t(out)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 0).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged_sink", a));
    const float w = std::exp(-sc);
    const float d0 = 1.0f + w + std::exp(0.5f - sc);          /* head 0, either token */
    CHECK_NEAR(out.at(0), 1.0f / d0, 1e-6); CHECK_NEAR(out.at(1), w / d0, 1e-6);
    CHECK_NEAR(out.at(4), w / d0, 1e-6);    CHECK_NEAR(out.at(5), 1.0f / d0, 1e-6);
    const float d1 = 1.0f + w + std::exp(3.0f - sc);          /* head 1 */
    CHECK_NEAR(out.at(2), w / d1, 1e-6);    CHECK_NEAR(out.at(3), 1.0f / d1, 1e-6);
    CHECK_NEAR(out.at(6), 1.0f / d1, 1e-6); CHECK_NEAR(out.at(7), w / d1, 1e-6);

    /* A sink far below every score changes nothing: attn_paged's values, causal. */
    T sk2(RAD_F32, {2}); sk2.f({ -1e30f, -1e30f });
    T o1(RAD_F32, {2, 2, 2}), o2(RAD_F32, {2, 2, 2});
    A b; b.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(o1)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", b));
    A c; c.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(sk2).t(o2)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 2).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged_sink", c));
    for (int i = 0; i < 8; ++i) CHECK_NEAR(o2.at(i), o1.at(i), 1e-7);
}

TEST(op_attn_paged_ragged) {
    /* A ragged batch, as a serving step really looks: sequence 0 prefills two tokens while
     * sequence 1 decodes one. total_q is 3 and n_seq is 2, so without cu_seqlens this batch cannot
     * be expressed (the op would have to assume 1.5 query rows each).
     *
     * One query head over one KV head, head_dim 2, one 2-slot block per sequence, both sequences
     * holding the same keys: slot 0 K = V = [1,0], slot 1 K = V = [0,1]. */
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {3, 1, 2});
    q.f({ 1, 0,        /* seq 0 token 0 */
          0, 1,        /* seq 0 token 1 */
          0, 1 });     /* seq 1, its only row, at position 1 */
    T kv(RAD_F32, {2, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,   0, 1, 0, 1,      /* block 0 */
           1, 0, 1, 0,   0, 1, 0, 1 });   /* block 1 */
    T bt(RAD_I32, {2, 1}); bt.n({0, 1});
    T su(RAD_I32, {2});    su.n({2, 2});
    T cu(RAD_I32, {3});    cu.n({0, 2, 3});
    T out(RAD_F32, {3, 1, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().t(cu).t(out)
          .i("q_len", 2).i("head_dim", 2).i("gqa", 1).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", a));

    /* Sequence 0's first token is at position 0 and sees key 0 only. */
    CHECK_NEAR(out.at(0), 1.0f, 1e-6); CHECK_NEAR(out.at(1), 0.0f, 1e-6);

    /* Its second is at position 1 and sees both. */
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(out.at(2), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(3), 1.0f / sum, 1e-6);

    /* Sequence 1 carries one row, so its position is ctx-1 = 1, not ctx-q_len+0 for the batch's
     * q_len of 2; with the batch maximum it would sit at position 0 and return V0 = [1,0]. */
    CHECK_NEAR(out.at(4), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(5), 1.0f / sum, 1e-6);
}

/* A run longer than its context: four query rows over two keys, as an MTP head's history pass
 * right after a one-token prompt is (core/sched/advance_impl.h ADV_HIST ends the context at the
 * last committed position). Rows sit at seqused - q_len + j = -2, -1, 0, 1: the first two see no
 * key and are zero, not NaN, and the last two are as if the run were two rows. */
TEST(op_attn_paged_anchor_window_is_exact_at_512_history_and_shared_across_rows) {
    constexpr int qlen = 7, ctx = 520, bs = 16, seqs = 2, window = 513;
    const int blocks = (ctx + bs - 1) / bs;
    T q(RAD_F32, {seqs * qlen, 1, 1});
    T kv(RAD_F32, {seqs * blocks, 1, bs, 2});
    for (int s = 0; s < seqs; ++s) {
        const int slot_base = s * blocks * bs;
        const float before = s == 0 ? 500.0f : 700.0f;  /* anchor - 513: masked */
        const float edge = s == 0 ? 10.0f : 20.0f;       /* anchor - 512: visible */
        const float future = s == 0 ? 1000.0f : 2000.0f;
        st_dt(RAD_F32, kv.t.data, (int64_t)(slot_base + 0) * 2 + 1, before);
        st_dt(RAD_F32, kv.t.data, (int64_t)(slot_base + 1) * 2 + 1, edge);
        st_dt(RAD_F32, kv.t.data, (int64_t)(slot_base + ctx - 1) * 2 + 1, future);
    }
    T bt(RAD_I32, {seqs, blocks});
    std::vector<int64_t> block_ids(seqs * blocks);
    for (int s = 0; s < seqs; ++s)
        for (int b = 0; b < blocks; ++b) block_ids[s * blocks + b] = s * blocks + b;
    bt.n(block_ids);
    T su(RAD_I32, {seqs}); su.n({ctx, ctx});
    T cu(RAD_I32, {seqs + 1}); cu.n({0, qlen, 2 * qlen});
    T out(RAD_F32, {seqs * qlen, 1, 1});
    for (int causal : {0, 1}) {
        A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().t(cu).t(out)
              .i("q_len", qlen).i("head_dim", 1).i("gqa", 1).i("block_size", bs)
              .i("causal", causal).i("window", window).s("q_dtype", "f32").s("kv_dtype", "f32");
        CHECK_OK(run("attn_paged_anchor_window", a));
        const float denom = causal ? 513.0f : 519.0f;
        const float expected0 = causal ? 10.0f / denom : 1010.0f / denom;
        const float expected1 = causal ? 20.0f / denom : 2020.0f / denom;
        for (int i = 0; i < qlen; ++i) {
            CHECK_NEAR(out.at(i), expected0, 1e-6);
            CHECK_NEAR(out.at(qlen + i), expected1, 1e-6);
        }
    }
}

TEST(op_attn_paged_rows_before_the_context) {
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {4, 1, 2});
    q.f({ 1, 0,   0, 1,   1, 0,   0, 1 });
    T kv(RAD_F32, {1, 1, 2, 4});
    kv.f({ 1, 0, 1, 0,
           0, 1, 0, 1 });
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1});    su.n({2});
    T out(RAD_F32, {4, 1, 2});
    A a; a.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out)
          .i("q_len", 4).i("head_dim", 2).i("gqa", 1).i("block_size", 2)
          .i("causal", 1).i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_paged", a));
    for (int e = 0; e < 4; ++e) CHECK_EQ(out.at(e), 0.0f);
    /* Row 2 at position 0 sees key 0; row 3 at position 1 sees both, q = [0,1]. */
    CHECK_NEAR(out.at(4), 1.0f, 1e-6); CHECK_NEAR(out.at(5), 0.0f, 1e-6);
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(out.at(6), w0 / sum, 1e-6);
    CHECK_NEAR(out.at(7), 1.0f / sum, 1e-6);
}

/* attn_paged_tree_sel on an explicit tree. Five rows of one sequence over two kv heads (gqa 2,
 * head_dim 8, page 4):
 *
 *   committed pages  physical blocks 3 and 5   (row 2 selected only block 3)
 *   region           rb = 8, block table entries 2 and 3 -> physical blocks 1 and 0, so region
 *                    key k lives in block (k < 4 ? 1 : 0), slot k % 4
 *
 *   row 0 anchor              vmask 0b00001   sees region key 0
 *   row 1 child of 0          vmask 0b00011   0, 1
 *   row 2 child of 0          vmask 0b00101   0, 2            (a sibling of row 1)
 *   row 3 child of 1          vmask 0b01011   0, 1, 3
 *   row 4 child of 3          vmask 0b11011   0, 1, 3, 4      (the path crosses a page)
 *
 * Two oracles. The test's own double-precision softmax over each row's explicit key list, and
 * the claim the op exists for: a row computes what a sequential decode of its root->node path
 * computes. The second is built by laying the path's keys contiguously in a fresh page and
 * running attn_paged on a one-row sequence of (selected pages, that page), compared bit for bit.
 * Keys are filled from a fixed stream, V included, so a wrong key or an off-by-one slot moves
 * the output. */
static void tree_sel_case(uint32_t kvdt) {
    const int64_t HD = 8, G = 2, KVH = 2, NH = G * KVH, BS = 4, NB = 6, R = 5, PITCH = 3;
    const bool i8 = kvdt == RAD_I8;
    const int64_t row_el = i8 ? 2 * HD + 4 : 2 * HD;
    const int64_t esz = i8 ? 1 : (int64_t)dt_bytes(kvdt, 1);
    uint32_t rng = 12345u;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 9) & 0x7fff) / 16384.0f - 1.0f; };

    T kv(kvdt, {NB, KVH, BS, row_el});
    for (int64_t tok = 0; tok < NB * KVH * BS; ++tok) {
        if (i8) {
            int8_t* c = (int8_t*)kv.t.data + tok * row_el;
            for (int64_t d = 0; d < 2 * HD; ++d) c[d] = (int8_t)(int)std::lround(rnd() * 100.0f);
            for (int h = 0; h < 2; ++h) {
                const uint16_t s = f32_to_f16(0.01f + 0.01f * (rnd() + 1.0f));
                std::memcpy((uint8_t*)c + 2 * HD + 2 * h, &s, 2);
            }
        } else {
            for (int64_t d = 0; d < row_el; ++d) st_dt(kvdt, kv.t.data, tok * row_el + d, rnd());
        }
    }
    T q(RAD_F32, {R, NH, HD});
    for (int64_t i = 0; i < R * NH * HD; ++i) st_dt(RAD_F32, q.t.data, i, rnd() * 2.0f);

    const int64_t selv[R][PITCH] = { {3, 5, 0}, {3, 5, 0}, {3, 0, 0}, {3, 5, 0}, {5, 3, 0} };
    const int64_t nsv[R] = { 2, 2, 1, 2, 2 };
    const int64_t vmv[R] = { 0x01, 0x03, 0x05, 0x0b, 0x1b };
    T sel(RAD_I32, {R, PITCH}), nsel(RAD_I32, {R}), vm(RAD_I64, {R});
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < PITCH; ++i) st_int(RAD_I32, sel.t.data, r * PITCH + i, selv[r][i]);
        st_int(RAD_I32, nsel.t.data, r, nsv[r]);
        st_int(RAD_I64, vm.t.data, r, vmv[r]);
    }
    T bt(RAD_I32, {1, 4}); bt.n({0, 0, 1, 0});
    T rb(RAD_I32, {1});    rb.n({8});
    T out(RAD_F32, {R, NH, HD});
    auto params = [&](A& a, int64_t qlen, bool tree) {
        a.i("q_len", qlen).i("head_dim", HD).i("gqa", G).i("block_size", BS).i("causal", 1)
         .i("window", 0).s("q_dtype", "f32").s("kv_dtype", kvdt == RAD_F32 ? "f32" : i8 ? "i8" : "bf16")
         .i("n_head", NH);
        if (tree) a.i("sel_pitch", PITCH);
    };
    A a; a.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out).t(vm).t(bt).t(rb);
    params(a, R, true);
    REQUIRE_EQ(run("attn_paged_tree_sel", a), RAD_OK);

    auto key_base = [&](int64_t blk, int64_t kh, int64_t slot) { return ((blk * KVH + kh) * BS + slot); };
    auto kval = [&](int64_t tok, int64_t d, int half) -> double {
        if (i8) {
            const int8_t* c = (const int8_t*)kv.t.data + tok * row_el;
            uint16_t s; std::memcpy(&s, (const uint8_t*)c + 2 * HD + 2 * half, 2);
            return (double)c[half * HD + d] * (double)f16_to_f32(s);
        }
        return (double)kv.at(tok * row_el + half * HD + d);
    };
    /* Region key k's physical (block, slot). */
    auto region = [&](int k) { return std::pair<int64_t, int64_t>(k < 4 ? 1 : 0, k % 4); };

    for (int r = 0; r < R; ++r) {
        std::vector<std::pair<int64_t, int64_t>> keys;
        for (int i = 0; i < nsv[r]; ++i)
            for (int t = 0; t < BS; ++t) keys.push_back({ selv[r][i], t });
        for (int k = 0; k < 8; ++k) if ((vmv[r] >> k) & 1) keys.push_back(region(k));
        for (int h = 0; h < NH; ++h) {
            const int64_t kh = h / G;
            std::vector<double> sc;
            double mx = -1e300;
            for (auto& kk : keys) {
                double dot = 0;
                for (int d = 0; d < HD; ++d)
                    dot += (double)q.at((r * NH + h) * HD + d) * kval(key_base(kk.first, kh, kk.second), d, 0);
                sc.push_back(dot / std::sqrt((double)HD));
                mx = std::max(mx, sc.back());
            }
            double sum = 0;
            std::vector<double> acc(HD, 0.0);
            for (size_t j = 0; j < keys.size(); ++j) {
                const double w = std::exp(sc[j] - mx);
                sum += w;
                for (int d = 0; d < HD; ++d)
                    acc[d] += w * kval(key_base(keys[j].first, kh, keys[j].second), d, 1);
            }
            for (int d = 0; d < HD; ++d)
                CHECK_NEAR(out.at((r * NH + h) * HD + d), acc[d] / sum, 2e-5);
        }
    }

    /* The sequential decode of each row's root->node path: the path's region keys laid in order in
     * block 4 (the cache's spare page), the selected pages before it, one query row, causal. */
    for (int r = 0; r < R; ++r) {
        T kv2(kvdt, {NB, KVH, BS, row_el});
        std::memcpy(kv2.t.data, kv.t.data, (size_t)(NB * KVH * BS * row_el * esz));
        int slot = 0;
        for (int k = 0; k < 8; ++k) {
            if (!((vmv[r] >> k) & 1)) continue;
            const auto rk = region(k);
            for (int64_t kh = 0; kh < KVH; ++kh)
                std::memcpy((uint8_t*)kv2.t.data + (size_t)(key_base(4, kh, slot) * row_el * esz),
                            (const uint8_t*)kv.t.data +
                                (size_t)(key_base(rk.first, kh, rk.second) * row_el * esz),
                            (size_t)(row_el * esz));
            ++slot;
        }
        T tab(RAD_I32, {1, PITCH + 1}), su(RAD_I32, {1});
        for (int i = 0; i < nsv[r]; ++i) st_int(RAD_I32, tab.t.data, i, selv[r][i]);
        st_int(RAD_I32, tab.t.data, nsv[r], 4);
        su.n({ nsv[r] * BS + slot });
        T q1(RAD_F32, {1, NH, HD}), o1(RAD_F32, {1, NH, HD});
        std::memcpy(q1.t.data, (const float*)q.t.data + r * NH * HD, sizeof(float) * NH * HD);
        A b; b.t(q1).t(kv2).t(tab).t(su).tnull().tnull().tnull().t(o1);
        params(b, 1, false);
        REQUIRE_EQ(run("attn_paged", b), RAD_OK);
        for (int e = 0; e < NH * HD; ++e) CHECK_EQ(o1.at(e), out.at(r * NH * HD + e));
    }

    /* Per-sequence rows: the same rows split over two sequences by seq_cu, the second with its own
     * region base and table, give the same answers for the rows they share. */
    {
        T bt2(RAD_I32, {2, 4}); bt2.n({0, 0, 1, 0, 0, 0, 1, 0});
        T rb2(RAD_I32, {2});    rb2.n({8, 8});
        T cu(RAD_I32, {3});     cu.n({0, 2, 5});
        T out2(RAD_F32, {R, NH, HD});
        A c; c.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out2).t(vm).t(bt2).t(rb2).t(cu);
        params(c, R, true);
        REQUIRE_EQ(run("attn_paged_tree_sel", c), RAD_OK);
        for (int e = 0; e < R * NH * HD; ++e) CHECK_EQ(out2.at(e), out.at(e));
        /* A region base of 4 shifts which block-table entries the bits name: key 0 is now entry 1
         * (physical block 0), so row 0 no longer reads block 1. */
        T rb3(RAD_I32, {2}); rb3.n({4, 8});
        T out3(RAD_F32, {R, NH, HD});
        A d; d.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out3).t(vm).t(bt2).t(rb3).t(cu);
        params(d, R, true);
        REQUIRE_EQ(run("attn_paged_tree_sel", d), RAD_OK);
        bool moved = false, same = true;
        for (int e = 0; e < NH * HD; ++e) moved |= out3.at(e) != out.at(e);
        for (int e = 2 * NH * HD; e < R * NH * HD; ++e) same &= out3.at(e) == out.at(e);
        CHECK(moved);
        CHECK(same);
    }

    /* Refusals: descales, a page count past the pitch, an unaligned region base, a bit past the
     * table. */
    {
        T ds(RAD_F32, {1, KVH});
        A e1; e1.t(q).t(kv).t(sel).t(nsel).t(ds).tnull().tnull().t(out).t(vm).t(bt).t(rb);
        params(e1, R, true);
        CHECK_EQ(run("attn_paged_tree_sel", e1), RAD_E_UNSUPPORTED);
        T ns2(RAD_I32, {R}); ns2.n({2, 2, 4, 2, 2});
        A e2; e2.t(q).t(kv).t(sel).t(ns2).tnull().tnull().tnull().t(out).t(vm).t(bt).t(rb);
        params(e2, R, true);
        CHECK_EQ(run("attn_paged_tree_sel", e2), RAD_E_INVAL);
        T rb4(RAD_I32, {1}); rb4.n({6});
        A e3; e3.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out).t(vm).t(bt).t(rb4);
        params(e3, R, true);
        CHECK_EQ(run("attn_paged_tree_sel", e3), RAD_E_INVAL);
        T vm2(RAD_I64, {R}); vm2.n({1, 1, 1, 1, (int64_t)1 << 20});
        A e4; e4.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out).t(vm2).t(bt).t(rb);
        params(e4, R, true);
        CHECK_EQ(run("attn_paged_tree_sel", e4), RAD_E_INVAL);
    }
}

TEST(op_attn_paged_tree_sel_f32) { tree_sel_case(RAD_F32); }
TEST(op_attn_paged_tree_sel_bf16) { tree_sel_case(RAD_BF16); }
TEST(op_attn_paged_tree_sel_i8) { tree_sel_case(RAD_I8); }

TEST(op_attn_dense) {
    const float sc = 1.0f / 1.41421356f;
    T q(RAD_F32, {2, 1, 2}); q.f({1, 0, 0, 1});
    T k(RAD_F32, {2, 1, 2}); k.f({1, 0, 0, 1});
    T v(RAD_F32, {2, 1, 2}); v.f({1, 0, 0, 1});
    T cu(RAD_I32, {2});      cu.n({0, 2});
    T o(RAD_F32, {2, 1, 2});
    A a; a.t(q).t(k).t(v).t(cu).t(o)
          .i("M", 2).i("head_dim", 2).i("gqa", 1).i("causal", 0).s("dtype", "f32");
    CHECK_OK(run("attn_dense", a));
    const float w0 = std::exp(-sc), sum = w0 + 1.0f;
    CHECK_NEAR(o.at(0), 1.0f / sum, 1e-6);
    CHECK_NEAR(o.at(1), w0 / sum, 1e-6);
    CHECK_NEAR(o.at(2), w0 / sum, 1e-6);
    CHECK_NEAR(o.at(3), 1.0f / sum, 1e-6);
}

/* The window is two-sided when the attention is: with window 2 a key one position away on either
 * side is visible and one two away is not. Three tokens, unit queries and keys, values naming the
 * key they came from: token 0 sees keys 0..1, token 1 sees 0..2, token 2 sees 1..2. */
TEST(op_attn_dense_window) {
    T q(RAD_F32, {3, 1, 1}); q.f({0, 0, 0});           /* zero scores: a plain mean of the keys */
    T k(RAD_F32, {3, 1, 1}); k.f({0, 0, 0});
    T v(RAD_F32, {3, 1, 1}); v.f({1, 10, 100});
    T cu(RAD_I32, {2});      cu.n({0, 3});
    T o(RAD_F32, {3, 1, 1});
    A a; a.t(q).t(k).t(v).t(cu).t(o)
          .i("M", 3).i("head_dim", 1).i("gqa", 1).i("causal", 0).i("window", 2)
          .s("dtype", "f32");
    CHECK_OK(run("attn_dense", a));
    CHECK_NEAR(o.at(0), 11.0f / 2, 1e-5);
    CHECK_NEAR(o.at(1), 111.0f / 3, 1e-5);
    CHECK_NEAR(o.at(2), 110.0f / 2, 1e-5);
    /* Causal with the same window: token 2 sees keys 1..2 and nothing after itself. */
    T o2(RAD_F32, {3, 1, 1});
    A c; c.t(q).t(k).t(v).t(cu).t(o2)
          .i("M", 3).i("head_dim", 1).i("gqa", 1).i("causal", 1).i("window", 2)
          .s("dtype", "f32");
    CHECK_OK(run("attn_dense", c));
    CHECK_NEAR(o2.at(0), 1.0f, 1e-5);
    CHECK_NEAR(o2.at(1), 11.0f / 2, 1e-5);
    CHECK_NEAR(o2.at(2), 110.0f / 2, 1e-5);
}

/* Blocks cut a segment into windows that do not see each other, counted from the segment's start;
 * cu_div reads the bounds in units finer than the rows. Five tokens over two segments [0, 2) and
 * [2, 5) as cu {0, 4, 10} at cu_div 2, blocks of 2: token 2 and 3 see each other, token 4 alone. */
TEST(op_attn_dense_block) {
    T q(RAD_F32, {5, 1, 1}); q.f({0, 0, 0, 0, 0});
    T k(RAD_F32, {5, 1, 1}); k.f({0, 0, 0, 0, 0});
    T v(RAD_F32, {5, 1, 1}); v.f({1, 10, 100, 1000, 10000});
    T cu(RAD_I32, {3});      cu.n({0, 4, 10});
    T o(RAD_F32, {5, 1, 1});
    A a; a.t(q).t(k).t(v).t(cu).t(o)
          .i("M", 5).i("head_dim", 1).i("gqa", 1).i("causal", 0).i("block", 2).i("cu_div", 2)
          .s("dtype", "f32");
    CHECK_OK(run("attn_dense", a));
    CHECK_NEAR(o.at(0), 11.0f / 2, 1e-4);
    CHECK_NEAR(o.at(1), 11.0f / 2, 1e-4);
    CHECK_NEAR(o.at(2), 1100.0f / 2, 1e-3);
    CHECK_NEAR(o.at(3), 1100.0f / 2, 1e-3);
    CHECK_NEAR(o.at(4), 10000.0f, 1e-2);
}

/* act_mul over two column slices of one fused row, the way an MLP's gate_up reaches it. */
TEST(op_act_mul) {
    T gu(RAD_F32, {2, 4}); gu.f({ 1, -2, 3, 4,   0.5f, 0, -1, 2 });
    T g(RAD_F32, {2, 2}), u(RAD_F32, {2, 2});
    g.t.data = gu.t.data;                         g.t.stride[0] = 4;
    u.t.data = (void*)((float*)gu.t.data + 2);    u.t.stride[0] = 4;
    T y(RAD_F32, {2, 2});
    auto gt = [](float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); };
    A a; a.t(g).t(u).t(y).i("M", 2).i("n", 2).s("act", "gelu_tanh").s("dtype", "f32");
    CHECK_OK(run("act_mul", a));
    CHECK_NEAR(y.at(0), gt(1) * 3, 1e-5);
    CHECK_NEAR(y.at(1), gt(-2) * 4, 1e-5);
    CHECK_NEAR(y.at(2), gt(0.5f) * -1, 1e-5);
    CHECK_NEAR(y.at(3), 0.0f, 1e-6);
    A bad; bad.t(g).t(u).t(y).i("M", 2).i("n", 2).s("act", "tanh").s("dtype", "f32");
    CHECK(run("act_mul", bad) < 0);
}

/* act_mul's relu over one tensor in both operands: relu(x)^2, zero below. */
TEST(op_act_mul_relu_of_one_tensor_squares_it) {
    T x(RAD_F32, {1, 4}); x.f({ 3, -2, 0.5f, 0 });
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(x).t(y).i("M", 1).i("n", 4).s("act", "relu").s("dtype", "f32");
    CHECK_OK(run("act_mul", a));
    CHECK_NEAR(y.at(0), 9.0f, 1e-6);
    CHECK_NEAR(y.at(1), 0.0f, 1e-6);
    CHECK_NEAR(y.at(2), 0.25f, 1e-6);
    CHECK_NEAR(y.at(3), 0.0f, 1e-6);
}

/* seg_mean: one row per segment, an empty segment a zero row. */
TEST(op_seg_mean) {
    T x(RAD_F32, {4, 2}); x.f({ 1, 2,  3, 4,  5, 6,  7, 9 });
    T cu(RAD_I32, {4});    cu.n({0, 3, 3, 4});
    T y(RAD_F32, {3, 2});
    A a; a.t(x).t(cu).t(y).i("M", 4).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("seg_mean", a));
    CHECK_NEAR(y.at(0), 3.0f, 1e-6); CHECK_NEAR(y.at(1), 4.0f, 1e-6);
    CHECK_EQ(y.at(2), 0.0f);         CHECK_EQ(y.at(3), 0.0f);
    CHECK_NEAR(y.at(4), 7.0f, 1e-6); CHECK_NEAR(y.at(5), 9.0f, 1e-6);
}

/* seg_softmax_pool: a segment's rows summed with the softmax of their scores as weights. Scores 0
 * and ln 3 weigh two rows 1/4 and 3/4; an empty segment is a zero row. */
TEST(op_seg_softmax_pool) {
    T sc(RAD_F32, {4});   sc.f({ 0.0f, 1.0986123f, 5.0f, 2.0f });
    T v(RAD_F32, {4, 2}); v.f({ 1, 2,  5, 6,  7, 9,  3, 3 });
    T cu(RAD_I32, {4});   cu.n({0, 2, 2, 4});
    T y(RAD_F32, {3, 2});
    A a; a.t(sc).t(v).t(cu).t(y).i("M", 4).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("seg_softmax_pool", a));
    CHECK_NEAR(y.at(0), 4.0f, 1e-5);   CHECK_NEAR(y.at(1), 5.0f, 1e-5);
    CHECK_EQ(y.at(2), 0.0f);           CHECK_EQ(y.at(3), 0.0f);
    /* Scores 5 and 2: weights 1 / (1 + e^-3) and the rest. */
    CHECK_NEAR(y.at(4), 6.8103f, 1e-3); CHECK_NEAR(y.at(5), 8.7155f, 1e-3);
}

/* relu and clamp: the plain element rules. */
TEST(op_relu_clamp) {
    T x(RAD_F32, {1, 3}); x.f({ -3.0f, 0.5f, 4.0f });
    T y(RAD_F32, {1, 3});
    A a; a.t(x).t(y).i("M", 1).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("relu", a));
    CHECK_EQ(y.at(0), 0.0f); CHECK_EQ(y.at(1), 0.5f); CHECK_EQ(y.at(2), 4.0f);
    T lo(RAD_F32, {1}); lo.f({ -1.0f });
    T hi(RAD_F32, {1}); hi.f({ 2.0f });
    A c; c.t(x).t(lo).t(hi).t(y).i("M", 1).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("clamp", c));
    CHECK_EQ(y.at(0), -1.0f); CHECK_EQ(y.at(1), 0.5f); CHECK_EQ(y.at(2), 2.0f);
}

/* rmsnorm's mult scales the normed row; act_mul's sigmoid is a GLU gate. */
TEST(op_rmsnorm_mult_act_sigmoid) {
    T x(RAD_F32, {1, 2}); x.f({ 3, 4 });
    T y(RAD_F32, {1, 2});
    A a; a.t(x).tnull().t(y).i("M", 1).i("n", 2).d("eps", 0.0).d("mult", 0.5).s("dtype", "f32");
    CHECK_OK(run("rmsnorm", a));
    const float r = 0.5f / std::sqrt(12.5f);
    CHECK_NEAR(y.at(0), 3.0f * r, 1e-6); CHECK_NEAR(y.at(1), 4.0f * r, 1e-6);
    T g(RAD_F32, {1, 2}); g.f({ 0.0f, 2.0f });
    T u(RAD_F32, {1, 2}); u.f({ 3.0f, 1.0f });
    A b; b.t(g).t(u).t(y).i("M", 1).i("n", 2).s("act", "sigmoid").s("dtype", "f32");
    CHECK_OK(run("act_mul", b));
    CHECK_NEAR(y.at(0), 1.5f, 1e-6);
    CHECK_NEAR(y.at(1), 1.0f / (1.0f + std::exp(-2.0f)), 1e-6);
}

/* conv2d: a 3x3 stride-2 window of ones over (time, width), each segment padded at its own
 * edges, and the row mask. */
TEST(op_conv2d) {
    T x(RAD_F32, {4, 2}); x.f({ 1, 2,  3, 4,  5, 6,  7, 8 });
    T w(RAD_F32, {1, 1, 3, 3}); w.f({ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    T cu(RAD_I32, {2}); cu.n({ 0, 4 });
    T y(RAD_F32, {2, 1});
    auto args = [&](A& a) {
        a.i("M", 4).i("width", 2).i("cin", 1).i("cout", 1).i("kh", 3).i("kw", 3).i("stride", 2)
         .i("pad", 1).s("dtype", "f32");
    };
    A a; a.t(x).t(w).t(cu).tnull().t(y); args(a);
    CHECK_OK(run("conv2d", a));
    CHECK_NEAR(y.at(0), 1 + 2 + 3 + 4, 1e-6);          /* rows -1..1, cols -1..1 */
    CHECK_NEAR(y.at(1), 3 + 4 + 5 + 6 + 7 + 8, 1e-6);  /* rows 1..3 */
    T valid(RAD_I32, {4}); valid.n({ 1, 1, 0, 1 });
    A b; b.t(x).t(w).t(cu).t(valid).t(y); args(b);
    CHECK_OK(run("conv2d", b));
    CHECK_NEAR(y.at(1), 3 + 4 + 7 + 8, 1e-6);          /* row 2 masked */
    T cu2(RAD_I32, {3}); cu2.n({ 0, 2, 4 });
    A c; c.t(x).t(w).t(cu2).tnull().t(y); args(c);
    CHECK_OK(run("conv2d", c));
    CHECK_NEAR(y.at(0), 1 + 2 + 3 + 4, 1e-6);
    CHECK_NEAR(y.at(1), 5 + 6 + 7 + 8, 1e-6);          /* row 1 is the other segment's */
}

/* conv2d's epilogue and the time-last kernel. A 3x3 kernel whose taps say where they are, over a
 * single input value at (time 1, width 1): output (0, 0) reads it through tap (time 1, width 1)
 * either way, and output (1, 0) at stride 1 reads it through tap (0, 1) -- time first -- or the
 * tap at kernel index [1][0] when the checkpoint's kernel is [width][time]. The bias lands before
 * the rounding and gelu after it. */
TEST(op_conv2d_bias_gelu_time_last) {
    T x(RAD_F32, {3, 3}); x.f({ 0, 0, 0,  0, 1, 0,  0, 0, 0 });
    T w(RAD_F32, {1, 1, 3, 3}); w.f({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    T cu(RAD_I32, {2}); cu.n({ 0, 3 });
    T bias(RAD_F32, {1}); bias.f({ 0.5f });
    T y(RAD_F32, {3, 3});
    auto args = [&](A& a, int tlast, const char* act) {
        a.i("M", 3).i("width", 3).i("cin", 1).i("cout", 1).i("kh", 3).i("kw", 3).i("stride", 1)
         .i("pad", 1).s("dtype", "f32").i("t_last", tlast).s("act", act);
    };
    A a; a.t(x).t(w).t(cu).tnull().t(y).t(bias); args(a, 0, "none");
    CHECK_OK(run("conv2d", a));
    CHECK_NEAR(y.at(0), 9 + 0.5f, 1e-6);               /* tap [2][2] */
    CHECK_NEAR(y.at(1), 8 + 0.5f, 1e-6);               /* tap [2][1]: time 2, width 1 */
    CHECK_NEAR(y.at(3), 6 + 0.5f, 1e-6);               /* tap [1][2]: time 1, width 2 */
    A b; b.t(x).t(w).t(cu).tnull().t(y).t(bias); args(b, 1, "none");
    CHECK_OK(run("conv2d", b));
    CHECK_NEAR(y.at(1), 6 + 0.5f, 1e-6);               /* time 2, width 1 is [width 1][time 2] */
    CHECK_NEAR(y.at(3), 8 + 0.5f, 1e-6);
    A c; c.t(x).t(w).t(cu).tnull().t(y).t(bias); args(c, 0, "gelu");
    CHECK_OK(run("conv2d", c));
    const float v = 9.5f;
    CHECK_NEAR(y.at(0), 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f))), 1e-5);
    CHECK_NEAR(y.at(8), 0.5f * 1.5f * (1.0f + std::erf(1.5f / std::sqrt(2.0f))), 1e-5);
}

/* conv2d_unfold times the weight is conv2d: two segments, a masked row, stride 2, both kernel
 * layouts, against conv2d itself over the same input. */
TEST(op_conv2d_unfold_then_a_gemm_is_conv2d) {
    const int M = 8, W = 5, cin = 3, cout = 4, K = cin * 9;
    T x(RAD_F32, {M, W * cin});
    std::vector<float> xv;
    for (int i = 0; i < M * W * cin; ++i) xv.push_back(std::sin(0.37f * (float)i + 0.1f));
    x.f(xv);
    T w(RAD_F32, {cout, cin, 3, 3});
    std::vector<float> wv;
    for (int i = 0; i < cout * K; ++i) wv.push_back(std::cos(0.23f * (float)i));
    w.f(wv);
    T cu(RAD_I32, {3}); cu.n({ 0, 4, 8 });
    T valid(RAD_I32, {M}); valid.n({ 1, 1, 0, 1, 1, 1, 1, 0 });
    const int To = 4, Wo = 3;
    for (int tl : { 0, 1 }) {
        T y(RAD_F32, {To, Wo * cout});
        A a; a.t(x).t(w).t(cu).t(valid).t(y)
              .i("M", M).i("width", W).i("cin", cin).i("cout", cout).i("kh", 3).i("kw", 3)
              .i("stride", 2).i("pad", 1).s("dtype", "f32").i("t_last", tl);
        CHECK_OK(run("conv2d", a));
        T u(RAD_F32, {To * Wo, K});
        A b; b.t(x).t(cu).t(valid).t(u)
              .i("M", M).i("width", W).i("cin", cin).i("kh", 3).i("kw", 3).i("stride", 2)
              .i("pad", 1).s("dtype", "f32").i("t_last", tl);
        CHECK_OK(run("conv2d_unfold", b));
        for (int r = 0; r < To * Wo; ++r)
            for (int co = 0; co < cout; ++co) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += u.at(r * K + k) * wv[(size_t)(co * K + k)];
                CHECK_NEAR(s, y.at(r * cout + co), 1e-4);
            }
    }
}

/* seg_sinusoid: positions restart at each segment's start and every `period` rows inside it. */
TEST(op_seg_sinusoid) {
    T cu(RAD_I32, {3}); cu.n({ 0, 5, 7 });
    T y(RAD_F32, {7, 4});
    A a; a.t(cu).t(y).i("M", 7).i("n", 4).i("period", 3).s("dtype", "f32");
    CHECK_OK(run("seg_sinusoid", a));
    const int pos[7] = { 0, 1, 2, 0, 1, 0, 1 };
    const float inc = std::log(10000.0f) / 1.0f;
    for (int r = 0; r < 7; ++r)
        for (int i = 0; i < 2; ++i) {
            const float ang = (float)pos[r] * std::exp((float)i * -inc);
            CHECK_NEAR(y.at(r * 4 + i), std::sin(ang), 1e-6);
            CHECK_NEAR(y.at(r * 4 + 2 + i), std::cos(ang), 1e-6);
        }
}

/* conv1d_dw: causal, the taps before a segment's start read as zero. */
TEST(op_conv1d_dw) {
    T x(RAD_F32, {4, 1}); x.f({ 1, 1, 1, 1 });
    T w(RAD_F32, {1, 1, 3}); w.f({ 1, 2, 3 });
    T cu(RAD_I32, {3}); cu.n({ 0, 2, 4 });
    T y(RAD_F32, {4, 1});
    A a; a.t(x).t(w).t(cu).t(y).i("M", 4).i("n", 1).i("k", 3).s("dtype", "f32");
    CHECK_OK(run("conv1d_dw", a));
    CHECK_NEAR(y.at(0), 3.0f, 1e-6); CHECK_NEAR(y.at(1), 5.0f, 1e-6);
    CHECK_NEAR(y.at(2), 3.0f, 1e-6); CHECK_NEAR(y.at(3), 5.0f, 1e-6);
}

/* rel_sinusoid: sin then cos over the timescales. */
TEST(op_rel_sinusoid) {
    T y(RAD_F32, {2, 4});
    A a; a.t(y).i("M", 2).i("n", 4).d("max_timescale", 10000.0).s("dtype", "f32");
    CHECK_OK(run("rel_sinusoid", a));
    const float inv1 = std::exp(-std::log(10000.0f));
    CHECK_EQ(y.at(0), 0.0f); CHECK_EQ(y.at(2), 1.0f);
    CHECK_NEAR(y.at(4), std::sin(1.0f), 1e-6); CHECK_NEAR(y.at(5), std::sin(inv1), 1e-9);
    CHECK_NEAR(y.at(6), std::cos(1.0f), 1e-6); CHECK_NEAR(y.at(7), std::cos(inv1), 1e-6);
}

/* attn_local_rel: the window, the relative term and the softcap, against the formula. */
TEST(op_attn_local_rel) {
    const int Tn = 3, D = 2, L = 1;
    T q(RAD_F32, {Tn, D}); q.f({ 0.5f, -1.0f, 1.0f, 0.25f, -0.5f, 2.0f });
    T k(RAD_F32, {Tn, D}); k.f({ 1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.5f });
    T v(RAD_F32, {Tn, D}); v.f({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    T rk(RAD_F32, {L + 1, D}); rk.f({ 0.1f, 0.2f, -0.3f, 0.4f });
    T pds(RAD_F32, {D}); pds.f({ 0.0f, 1.0f });
    T cu(RAD_I32, {2}); cu.n({ 0, Tn });
    T o(RAD_F32, {Tn, D});
    A a; a.t(q).t(k).t(v).t(rk).t(pds).t(cu).t(o).i("M", Tn).i("head_dim", D).i("n_head", 1)
          .i("left", L).i("right", 0).d("softcap", 5.0).s("dtype", "f32");
    CHECK_OK(run("attn_local_rel", a));
    const float qs = (float)(1.0 / std::sqrt((double)D) / std::log(2.0));
    const float ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    for (int i = 0; i < Tn; ++i) {
        float lg[2], p[2], qf[2];
        for (int d = 0; d < D; ++d) {
            const float x = pds.at(d);
            qf[d] = q.at(i * D + d) * qs * std::log1p(std::exp(x));
        }
        const int j0 = i - L > 0 ? i - L : 0;
        float mx = -1e30f, sum = 0.0f;
        for (int j = j0; j <= i; ++j) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d)
                s += qf[d] * k.at(j * D + d) * ks + qf[d] * rk.at((i - j) * D + d);
            lg[j - j0] = 5.0f * std::tanh(s / 5.0f);
            mx = lg[j - j0] > mx ? lg[j - j0] : mx;
        }
        for (int j = j0; j <= i; ++j) { p[j - j0] = std::exp(lg[j - j0] - mx); sum += p[j - j0]; }
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int j = j0; j <= i; ++j) acc += p[j - j0] / sum * v.at(j * D + d);
            CHECK_NEAR(o.at(i * D + d), acc, 1e-5);
        }
    }
}

/* pool_rows: the mean of every k rows, rounded, then the multiplier. */
TEST(op_pool_rows) {
    T x(RAD_F32, {4, 2}); x.f({ 1, 2,  3, 4,  5, 6,  7, 9 });
    T y(RAD_F32, {2, 2});
    A a; a.t(x).t(y).i("M", 4).i("n", 2).i("k", 2).d("mult", 2.0).s("dtype", "f32");
    CHECK_OK(run("pool_rows", a));
    CHECK_NEAR(y.at(0), 4.0f, 1e-6);  CHECK_NEAR(y.at(1), 6.0f, 1e-6);
    CHECK_NEAR(y.at(2), 12.0f, 1e-6); CHECK_NEAR(y.at(3), 15.0f, 1e-6);
    A bad; bad.t(x).t(y).i("M", 4).i("n", 2).i("k", 3).s("dtype", "f32");
    CHECK(run("pool_rows", bad) < 0);
}

/* pool_cols: the mean of every k columns of a row, rounded, then the multiplier. */
TEST(op_pool_cols) {
    T x(RAD_F32, {2, 4}); x.f({ 1, 2,  3, 4,  5, 6,  7, 9 });
    T y(RAD_F32, {2, 2});
    A a; a.t(x).t(y).i("M", 2).i("n", 4).i("k", 2).d("mult", 2.0).s("dtype", "f32");
    CHECK_OK(run("pool_cols", a));
    CHECK_NEAR(y.at(0), 3.0f, 1e-6);  CHECK_NEAR(y.at(1), 7.0f, 1e-6);
    CHECK_NEAR(y.at(2), 11.0f, 1e-6); CHECK_NEAR(y.at(3), 16.0f, 1e-6);
    A bad; bad.t(x).t(y).i("M", 2).i("n", 4).i("k", 3).s("dtype", "f32");
    CHECK(run("pool_cols", bad) < 0);
}

/* point_fourier: [x | sin(x 2^j) | cos(x 2^j)], sines and cosines each value-major then octave;
 * with `pi`, the octaves times pi. */
TEST(op_point_fourier) {
    T x(RAD_F32, {2, 2}); x.f({ 0.5f, -0.25f,  1.0f, 0.0f });
    T y(RAD_F32, {2, 10});
    A a; a.t(x).t(y).i("M", 2).i("n", 2).i("freqs", 2).s("dtype", "f32");
    CHECK_OK(run("point_fourier", a));
    const float v[2][2] = { { 0.5f, -0.25f }, { 1.0f, 0.0f } };
    for (int r = 0; r < 2; ++r)
        for (int i = 0; i < 2; ++i) {
            CHECK_EQ(y.at(r * 10 + i), v[r][i]);
            for (int j = 0; j < 2; ++j) {
                const float t = v[r][i] * (float)(1 << j);
                CHECK_NEAR(y.at(r * 10 + 2 + i * 2 + j), std::sin(t), 1e-6);
                CHECK_NEAR(y.at(r * 10 + 6 + i * 2 + j), std::cos(t), 1e-6);
            }
        }
    A b; b.t(x).t(y).i("M", 2).i("n", 2).i("freqs", 2).i("pi", 1).s("dtype", "f32");
    CHECK_OK(run("point_fourier", b));
    CHECK_NEAR(y.at(3), std::sin(0.5f * 2.0f * 3.14159265f), 1e-6);
    A bad; bad.t(x).t(y).i("M", 2).i("n", 2).i("freqs", 3).s("dtype", "f32");
    CHECK(run("point_fourier", bad) < 0);
}

/* rmsnorm with no gain: a unit gain. */
TEST(op_rmsnorm_no_gain) {
    T x(RAD_F32, {1, 2}); x.f({ 3, 4 });
    T y(RAD_F32, {1, 2});
    A a; a.t(x).tnull().t(y).i("M", 1).i("n", 2).d("eps", 0.0).s("dtype", "f32");
    CHECK_OK(run("rmsnorm", a));
    const float r = 1.0f / std::sqrt(12.5f);
    CHECK_NEAR(y.at(0), 3.0f * r, 1e-6); CHECK_NEAR(y.at(1), 4.0f * r, 1e-6);
}

/* qk_norm_rope `blocks`: each section rotates its own channel block by its own component. */
TEST(op_qk_norm_rope_blocks) {
    /* one token, one q head and one k head of 8 channels; sections "2 2", comps "1 0". */
    T qkv(RAD_F32, {1, 24});
    std::vector<float> v(24);
    for (int i = 0; i < 24; ++i) v[(size_t)i] = 0.1f * (float)(i + 1);
    qkv.f(v);
    T pos(RAD_I32, {2, 1}); pos.n({ 3, 5 });
    T qw(RAD_F32, {8}); qw.f({ 1, 1, 1, 1, 1, 1, 1, 1 });
    T kw(RAD_F32, {8}); kw.f({ 1, 1, 1, 1, 1, 1, 1, 1 });
    A a; a.t(qkv).t(pos).t(qw).t(kw).i("M", 1).i("head_dim", 8).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 100.0).d("eps", 0.0).s("mode", "blocks").s("sections", "2 2").s("comps", "1 0");
    CHECK_OK(run("qk_norm_rope", a));
    /* The q head: normed, then block 0 (channels 0..3) turned by component 1 (5) and block 1
     * (channels 4..7) by component 0 (3), pairs (j, j+2) at theta^(-j/2). */
    float h[8], ss = 0.0f;
    for (int i = 0; i < 8; ++i) { h[i] = 0.1f * (float)(i + 1); ss += h[i] * h[i]; }
    const float sc = 1.0f / std::sqrt(ss / 8.0f);
    for (int i = 0; i < 8; ++i) h[i] *= sc;
    const float p[2] = { 5.0f, 3.0f };
    for (int c = 0; c < 2; ++c)
        for (int j = 0; j < 2; ++j) {
            const float ang = p[c] / std::pow(100.0f, (float)j / 2.0f);
            const float x0 = h[4 * c + j], x1 = h[4 * c + 2 + j];
            CHECK_NEAR(qkv.at(4 * c + j), x0 * std::cos(ang) - x1 * std::sin(ang), 1e-5);
            CHECK_NEAR(qkv.at(4 * c + 2 + j), x0 * std::sin(ang) + x1 * std::cos(ang), 1e-5);
        }
}

/* mul_weight: a one-element weight scales every element, a one-row weight every column. */
TEST(op_mul_weight) {
    T x(RAD_F32, {2, 3}); x.f({ 1, 2, 3, 4, 5, 6 });
    T s(RAD_F32, {1});    s.f({ 0.5f });
    T y(RAD_F32, {2, 3});
    A a; a.t(x).t(s).t(y).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("mul_weight", a));
    for (int i = 0; i < 6; ++i) CHECK_NEAR(y.at(i), 0.5f * (float)(i + 1), 1e-6);
    T r(RAD_F32, {3});    r.f({ 1, 10, 100 });
    A b; b.t(x).t(r).t(y).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("mul_weight", b));
    CHECK_NEAR(y.at(4), 50.0f, 1e-6);
    T bad(RAD_F32, {2});  bad.f({ 1, 2 });
    A c; c.t(x).t(bad).t(y).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK(run("mul_weight", c) < 0);
}

/* v_norm: the v heads are RMS-normed with unit gain and not rotated; q and k as before. */
TEST(op_qk_norm_rope_v_norm) {
    T qkv(RAD_F32, {1, 6}); qkv.f({ 3, 4,   1, 1,   6, 8 });   /* one q, one k, one v head of 2 */
    T pos(RAD_I32, {1});    pos.n({ 0 });                       /* position 0: no rotation */
    T qw(RAD_F32, {2});     qw.f({ 1, 1 });
    T kw(RAD_F32, {2});     kw.f({ 2, 2 });
    A a; a.t(qkv).t(pos).t(qw).t(kw)
          .i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 10000.0).d("eps", 0.0).i("v_norm", 1);
    CHECK_OK(run("qk_norm_rope", a));
    const float r = 1.0f / std::sqrt(12.5f);
    CHECK_NEAR(qkv.at(0), 3 * r, 1e-5);   CHECK_NEAR(qkv.at(1), 4 * r, 1e-5);
    CHECK_NEAR(qkv.at(2), 2.0f, 1e-5);    CHECK_NEAR(qkv.at(3), 2.0f, 1e-5);
    const float rv = 1.0f / std::sqrt(50.0f);
    CHECK_NEAR(qkv.at(4), 6 * rv, 1e-5);  CHECK_NEAR(qkv.at(5), 8 * rv, 1e-5);
}

TEST(op_kv_store) {
    T k(RAD_F32, {2, 1, 2}); k.f({1, 2, 5, 6});
    T v(RAD_F32, {2, 1, 2}); v.f({3, 4, 7, 8});
    T sm(RAD_I32, {2});      sm.n({1, -1});         /* the second token has no slot */
    T kv(RAD_F32, {1, 1, 2, 4});
    A a; a.t(k).t(v).t(sm).t(kv)
          .i("M", 2).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2).s("kv_dtype", "f32");
    CHECK_OK(run("kv_store", a));
    /* K then V inside slot 1 of block 0. */
    CHECK_EQ(kv.at(4), 1.0f); CHECK_EQ(kv.at(5), 2.0f);
    CHECK_EQ(kv.at(6), 3.0f); CHECK_EQ(kv.at(7), 4.0f);
    /* Slot 0 was never mapped and must be untouched. */
    CHECK_EQ(kv.at(0), 0.0f); CHECK_EQ(kv.at(3), 0.0f);
}

/* The int8 cache: each head's K and V as codes against its own f16 scale, the head's absmax / 127,
 * after the codes; and attention reads a key back as code times scale. */
TEST(op_kv_store_int8_and_the_attention_that_reads_it) {
    T k(RAD_BF16, {1, 1, 2}); k.f({ 254, -64 });
    T v(RAD_BF16, {1, 1, 2}); v.f({ 31.75f, 8 });
    T sm(RAD_I32, {1});       sm.n({ 1 });
    T kv(RAD_I8, {1, 1, 2, 8});
    A a; a.t(k).t(v).t(sm).t(kv)
          .i("M", 1).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2).s("kv_dtype", "i8");
    CHECK_OK(run("kv_store", a));
    /* Position 1: K scale 2.0 (f16 0x4000), V scale 0.25 (0x3400). */
    const int want[8] = { 127, -32, 127, 32, 0x00, 0x40, 0x00, 0x34 };
    for (int i = 0; i < 8; ++i) CHECK_EQ((int)kv.at(8 + i), want[i]);
    for (int i = 0; i < 8; ++i) CHECK_EQ(kv.at(i), 0.0f);

    /* One query over that one key: the output is the key's V, 127 * 0.25 and 32 * 0.25. */
    T q(RAD_BF16, {1, 1, 2}); q.f({ 1, 1 });
    T bt(RAD_I32, {1, 1});    bt.n({ 0 });
    T su(RAD_I32, {1});       su.n({ 2 });
    T out(RAD_BF16, {1, 1, 2});
    /* a window of one: the query at position 1 sees the key at position 1 alone */
    A b; b.t(q).t(kv).t(bt).t(su).tnull().tnull().tnull().t(out)
          .i("q_len", 1).i("head_dim", 2).i("gqa", 1).i("block_size", 2).i("causal", 1)
          .i("window", 1).s("q_dtype", "bf16").s("kv_dtype", "i8").d("scale", 1.0);
    CHECK_OK(run("attn_paged", b));
    CHECK_NEAR(out.at(0), 31.75f, 1e-3); CHECK_NEAR(out.at(1), 8.0f, 1e-3);
}

/* The engine hands k and v a column slice, not a dense plane. Without de-interleaving, the KV
 * halves are declared over the fused qkv buffer, [tokens, kv_heads * head_dim] with the whole
 * projection's row stride. Deriving the stride from the shape puts every token after the first
 * inside the previous token's query rows: a cache that looks written, passes extent checks, and
 * holds the wrong keys. */
TEST(op_kv_store_reads_the_row_stride_and_not_the_shape) {
    /* A [3, 6] projection: columns 0..1 are the query, 2..3 the key, 4..5 the value. */
    T qkv(RAD_F32, {3, 6});
    qkv.f({ 90, 91,  1,  2,  3,  4,
            92, 93,  5,  6,  7,  8,
            94, 95,  9, 10, 11, 12 });
    T kslice(RAD_F32, {3, 2}), vslice(RAD_F32, {3, 2});
    kslice.t.data = (void*)((float*)qkv.t.data + 2); kslice.t.stride[0] = 6;
    vslice.t.data = (void*)((float*)qkv.t.data + 4); vslice.t.stride[0] = 6;

    T sm(RAD_I32, {3}); sm.n({0, 1, 2});
    T kv(RAD_F32, {1, 1, 4, 4});          /* one block, one kv head, four slots, K|V of width 2 */
    A a; a.t(kslice).t(vslice).t(sm).t(kv)
          .i("M", 3).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 4).s("kv_dtype", "f32");
    CHECK_OK(run("kv_store", a));
    for (int64_t t = 0; t < 3; ++t) {
        CHECK_EQ(kv.at(t * 4 + 0), 1.0f + 4 * (float)t);   /* k */
        CHECK_EQ(kv.at(t * 4 + 1), 2.0f + 4 * (float)t);
        CHECK_EQ(kv.at(t * 4 + 2), 3.0f + 4 * (float)t);   /* v */
        CHECK_EQ(kv.at(t * 4 + 3), 4.0f + 4 * (float)t);
    }
    /* Nothing reached the query columns and nothing reached the unmapped fourth slot. */
    CHECK_EQ(qkv.at(0), 90.0f);
    CHECK_EQ(kv.at(12), 0.0f); CHECK_EQ(kv.at(15), 0.0f);
}

/* kv_store_streamed: the group's host mirror is written for every token with a slot, and the ring
 * slot only when the residency map says the block is resident. The two planes are the same paged
 * layout, so a token's bytes land at the same offset in either.
 *
 * The case that decides the design: block 0 is NOT resident and block 1 is, and one token lands in
 * each. Block 1's token must reach both copies (the ring's copy of a resident block would otherwise
 * fall behind, and the resolve does not re-copy a resident block); block 0's must reach the mirror
 * only, leaving the ring untouched until a resolve copies the whole block in. */
TEST(op_kv_store_streamed_writes_the_mirror_always_and_the_ring_when_resident) {
    /* The int8 cache at head_dim 2, one kv head, block_size 2: a position is [K codes 2][V codes 2]
     * [K scale f16][V scale f16] = 8 bytes, a block row 16. Each head's larger element is 127 times
     * a power of two, so its scale is that power exactly and its codes the elements over it. */
    T k(RAD_BF16, {2, 1, 2}); k.f({ 127, 64, 254, -2 });
    T v(RAD_BF16, {2, 1, 2}); v.f({ -127, 2, 4, 508 });
    T sm(RAD_I32, {2});       sm.n({ 0, 3 });        /* block 0 offset 0, block 1 offset 1 */
    T res(RAD_I32, {2});      res.n({ -1, 0 });      /* block 0 absent, block 1 in ring slot 0 */
    T host(RAD_U8, {2, 16});
    T kv(RAD_I8, {1, 1, 2, 8});                      /* one ring slot, one block deep */
    A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
          .i("M", 2).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
          .s("kv_dtype", "i8");
    CHECK_OK(run("kv_store_streamed", a));
    /* Token 0, block 0, position 0, mirror only: scales 1.0 and 1.0 (f16 0x3C00). */
    const int t0[8] = { 127, 64, 129 /* -127 */, 2, 0x00, 0x3C, 0x00, 0x3C };
    for (int i = 0; i < 8; ++i) CHECK_EQ((int)host.at(i), t0[i]);
    /* Token 1, block 1, position 1: scales 2.0 (0x4000) and 4.0 (0x4400); both planes, at the same
     * in-block offset, so the same bytes. */
    const int t1[8] = { 127, 255 /* -1 */, 1, 127, 0x00, 0x40, 0x00, 0x44 };
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ((int)host.at(24 + i), t1[i]);
        CHECK_EQ((int)(uint8_t)(int8_t)kv.at(8 + i), t1[i]);
    }
    /* Nothing else: block 0 has no slot, so its ring stays as it was, and block 1's position 0 is
     * another token's. */
    for (int i = 0; i < 8; ++i) CHECK_EQ(kv.at(i), 0.0f);
    for (int i = 8; i < 24; ++i) CHECK_EQ(host.at(i), 0.0f);
}

/* The bf16 streamed store: both planes hold the values themselves, two bytes an element, exactly
 * as kv_store_bf16 writes them. */
TEST(op_kv_store_streamed_bf16_writes_the_values_into_both_planes) {
    T k(RAD_BF16, {2, 1, 2}); k.f({ 1, 2, 5, 6 });
    T v(RAD_BF16, {2, 1, 2}); v.f({ 3, 4, 7, 8 });
    T sm(RAD_I32, {2});       sm.n({ 0, 3 });        /* block 0 offset 0, block 1 offset 1 */
    T res(RAD_I32, {2});      res.n({ -1, 0 });      /* block 0 absent, block 1 in ring slot 0 */
    T host(RAD_U8, {2, 16});                         /* 1 kv head, 2 positions, 2*hd=4, bf16 */
    T kv(RAD_BF16, {1, 1, 2, 4});                    /* one ring slot, one block deep */
    A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
          .i("M", 2).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
          .s("kv_dtype", "bf16").d("k_scale", 0.5).d("v_scale", 2.0);
    CHECK_OK(run("kv_store_streamed", a));
    auto at16 = [](const std::vector<uint8_t>& m, size_t i) {
        return (unsigned)((unsigned)m[i] | ((unsigned)m[i + 1] << 8));
    };
    /* Token 0, block 0: mirror only, the values themselves. */
    CHECK_EQ(at16(host.mem, 0), (unsigned)f32_to_bf16(1.0f));
    CHECK_EQ(at16(host.mem, 2), (unsigned)f32_to_bf16(2.0f));
    CHECK_EQ(at16(host.mem, 4), (unsigned)f32_to_bf16(3.0f));
    CHECK_EQ(at16(host.mem, 6), (unsigned)f32_to_bf16(4.0f));
    /* Token 1, block 1 position 1: both planes at the same in-block offset. */
    CHECK_EQ(at16(host.mem, 24), (unsigned)f32_to_bf16(5.0f));
    CHECK_EQ(at16(host.mem, 26), (unsigned)f32_to_bf16(6.0f));
    CHECK_EQ(at16(host.mem, 28), (unsigned)f32_to_bf16(7.0f));
    CHECK_EQ(at16(host.mem, 30), (unsigned)f32_to_bf16(8.0f));
    for (size_t i = 0; i < 8; ++i) CHECK_EQ(kv.mem[i], 0);       /* block 0 wrote no slot */
    for (size_t i = 0; i < 8; ++i) CHECK_EQ((int)kv.mem[8 + i], (int)host.mem[24 + i]);
}

/* A block absent from the ring is written to the mirror alone, whatever the map holds: this is the
 * state every token of a block's first step is in, and the resolve copies the whole block in. */
TEST(op_kv_store_streamed_a_block_with_no_slot_reaches_only_the_mirror) {
    T k(RAD_BF16, {1, 1, 2}); k.f({ 127, 1 });
    T v(RAD_BF16, {1, 1, 2}); v.f({ 2, -254 });
    T sm(RAD_I32, {1});       sm.n({ 3 });           /* block 1, position 1 */
    T res(RAD_I32, {2});      res.n({ -1, -1 });
    T host(RAD_U8, {2, 16});
    T kv(RAD_I8, {1, 1, 2, 8});
    A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
          .i("M", 1).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
          .s("kv_dtype", "i8");
    CHECK_OK(run("kv_store_streamed", a));
    const int want[8] = { 127, 1, 1, 129 /* -127 */, 0x00, 0x3C, 0x00, 0x40 };
    for (int i = 0; i < 8; ++i) CHECK_EQ((int)host.at(24 + i), want[i]);
    for (int i = 0; i < 16; ++i) CHECK_EQ(kv.at(i), 0.0f);      /* the ring is untouched */
}

/* The refusals libref makes before its first write: a slot naming a block past the mirror, a
 * residency entry outside [-1, slots), and a mirror row wider than the block. Nothing is written on
 * any of them - a half-written mirror is what the next step's copies would believe. */
TEST(op_kv_store_streamed_refuses_a_state_fault_before_writing) {
    T k(RAD_BF16, {1, 1, 2}); k.f({ 1, 2 });
    T v(RAD_BF16, {1, 1, 2}); v.f({ 3, 4 });

    /* A map entry naming a slot the ring does not have. */
    {
        T sm(RAD_I32, {1});  sm.n({ 0 });
        T res(RAD_I32, {2}); res.n({ 1, -1 });       /* slot 1 of a one-slot ring */
        T host(RAD_U8, {2, 16});
        T kv(RAD_I8, {1, 1, 2, 8});
        A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
              .i("M", 1).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
              .s("kv_dtype", "i8");
        CHECK_EQ(run("kv_store_streamed", a), RAD_E_INVAL);
        for (int i = 0; i < 16; ++i) CHECK_EQ(host.at(i), 0.0f);
    }
    /* A slot naming a block past the mirror's extent. */
    {
        T sm(RAD_I32, {1});  sm.n({ 4 });            /* block 2 of a two-block mirror */
        T res(RAD_I32, {2}); res.n({ -1, -1 });
        T host(RAD_U8, {2, 16});
        T kv(RAD_I8, {1, 1, 2, 8});
        A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
              .i("M", 1).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
              .s("kv_dtype", "i8");
        CHECK_EQ(run("kv_store_streamed", a), RAD_E_SHAPE);
        for (int i = 0; i < 16; ++i) CHECK_EQ(host.at(i), 0.0f);
    }
    /* A mirror row that is not the block's byte pitch: a copy into it would spill over the next
     * block. */
    {
        T sm(RAD_I32, {1});  sm.n({ 0 });
        T res(RAD_I32, {2}); res.n({ -1, -1 });
        T host(RAD_U8, {2, 16});
        host.t.stride[0] = 20;                       /* a padded row */
        T kv(RAD_I8, {1, 1, 2, 8});
        A a; a.t(k).t(v).t(sm).t(kv).t(res).t(host)
              .i("M", 1).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2)
              .s("kv_dtype", "i8");
        CHECK_EQ(run("kv_store_streamed", a), RAD_E_STRIDE);
    }
}

TEST(op_rope) {
    /* head_dim 2 means one rotary pair with inv_freq 1, so the angle is the position itself. */
    T qkv(RAD_F32, {1, 6}); qkv.f({1, 0,  1, 0,  9, 9});   /* q, k, v */
    T pos(RAD_I32, {1});    pos.n({1});
    A a; a.t(qkv).t(pos).i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 10000).d("scale", 1).s("mode", "neox");
    CHECK_OK(run("rope", a));
    CHECK_NEAR(qkv.at(0), std::cos(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(1), std::sin(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(2), std::cos(1.0f), 1e-6);
    CHECK_NEAR(qkv.at(3), std::sin(1.0f), 1e-6);
    CHECK_EQ(qkv.at(4), 9.0f);          /* the value heads are left alone */
    CHECK_EQ(qkv.at(5), 9.0f);
}

/* Static YaRN, in double: cos and sin of rotary pair i at one position. The inverse frequency
 * theta^(-2i/rot) is blended toward inv/factor over the linear ramp [low, high] as vLLM's
 * YaRNScalingRotaryEmbedding blends it, times the magnitude. The bounds are vLLM's correction range
 * at factor 2 over 262144 trained positions, theta 1e7 (tests/yarn_test.cpp holds yarn_from to the
 * same integers), so these and the formula are the whole of the vLLM agreement.
 *
 * `tol` is the f32 evaluation's allowance: 8 ulp of the angle (powf, the reciprocal, the blend and
 * the product each cost about one) plus the rounding of the result. It is a few 1e-5 where the
 * ramp sits at position 524287, loose only in the fastest pairs there, which YaRN leaves unscaled
 * and no f32 rotary places better than its angle's own ulp. */
struct YarnRef { double c, s, tol; };
YarnRef yarn_ref(double pos, int i, int rot, double low, double high) {
    const double theta = 1e7, factor = 2.0, magnitude = 1.0 + 0.1 * std::log(2.0);
    const double inv  = std::pow(theta, -2.0 * i / rot);
    const double ramp = std::min(1.0, std::max(0.0, (i - low) / (high == low ? 0.001 : high - low)));
    const double a    = pos * (inv / factor * ramp + inv * (1.0 - ramp));
    return { std::cos(a) * magnitude, std::sin(a) * magnitude,
             magnitude * (std::fabs(a) * 0x1p-20 + 0x1p-21) };
}

TEST(yarn_rotary_matches_a_double_yarn_through_524287) {
    const std::vector<int64_t> positions = { 0, 1, 4096, 262143, 262144, 500000, 524287 };
    const double magnitude = 1.0 + 0.1 * std::log(2.0);
    const struct { int rot; double low, high; } cases[] = { { 64, 14, 22 }, { 128, 28, 43 },
                                                            { 256, 56, 85 } };
    for (const auto& c : cases) {
        const int rot = c.rot, half = rot / 2, hd = rot + 16;
        T pos(RAD_I32, {(int64_t)positions.size()}); pos.n(positions);
        T qkv(RAD_F32, {(int64_t)positions.size(), hd * 3});
        std::vector<float> values(positions.size() * hd * 3, 9.0f);
        for (size_t m = 0; m < positions.size(); ++m)
            for (int h = 0; h < 2; ++h)
                for (int i = 0; i < rot; ++i) values[m * hd * 3 + h * hd + i] = i < half ? 1.0f : 0.0f;
        qkv.f(values);
        A direct; direct.t(qkv).t(pos).i("M", positions.size()).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 1)
            .i("rotary_dim", rot).d("theta", 1e7).d("scale", 1).s("mode", "neox")
            .d("yarn_factor", 2).d("yarn_low", c.low).d("yarn_high", c.high).d("yarn_magnitude", magnitude);
        CHECK_OK(run("rope", direct));
        /* The table is position-addressed: allocate its real extent, not a compact stand-in. */
        T table(RAD_F32, {524288, rot});
        A ta; ta.t(pos).t(table).i("M", positions.size()).i("rot", rot).d("theta", 1e7).d("scale", 1).s("dtype", "f32")
            .d("yarn_factor", 2).d("yarn_low", c.low).d("yarn_high", c.high).d("yarn_magnitude", magnitude);
        CHECK_OK(run("rope_table", ta));
        for (size_t m = 0; m < positions.size(); ++m) {
            for (int i = 0; i < half; ++i) {
                const YarnRef y = yarn_ref((double)positions[m], i, rot, c.low, c.high);
                /* A unit pair (1, 0) rotated is (cos, sin). */
                CHECK_NEAR(qkv.at(m * hd * 3 + i), y.c, y.tol);
                CHECK_NEAR(qkv.at(m * hd * 3 + half + i), y.s, y.tol);
                CHECK_EQ(table.at(positions[m] * rot + i), qkv.at(m * hd * 3 + i));
                CHECK_EQ(table.at(positions[m] * rot + half + i), qkv.at(m * hd * 3 + half + i));
                CHECK_EQ(qkv.at(m * hd * 3 + hd + i), qkv.at(m * hd * 3 + i));
            }
            for (int i = rot; i < hd; ++i) CHECK_EQ(qkv.at(m * hd * 3 + i), 9.0f);
            for (int i = 0; i < hd; ++i) CHECK_EQ(qkv.at(m * hd * 3 + 2 * hd + i), 9.0f);
        }
    }
}

/* The reference is not vacuous: the same op with the bounds one pair off, or without the
 * magnitude, lands outside yarn_ref's bound somewhere in the ramp. */
TEST(yarn_reference_bound_sees_a_shifted_ramp_and_a_missing_magnitude) {
    const int rot = 64, hd = 64;
    const double magnitude = 1.0 + 0.1 * std::log(2.0);
    for (int variant = 0; variant < 2; ++variant) {
        T pos(RAD_I32, {1}); pos.n({500000});
        T qkv(RAD_F32, {1, hd});
        std::vector<float> values((size_t)hd, 0.0f);
        for (int i = 0; i < rot / 2; ++i) values[(size_t)i] = 1.0f;
        qkv.f(values);
        A a; a.t(qkv).t(pos).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
            .i("rotary_dim", rot).d("theta", 1e7).d("scale", 1).s("mode", "neox")
            .d("yarn_factor", 2).d("yarn_low", variant == 0 ? 15 : 14)
            .d("yarn_high", variant == 0 ? 23 : 22).d("yarn_magnitude", variant == 0 ? magnitude : 1.0);
        CHECK_OK(run("rope", a));
        int outside = 0;
        for (int i = 0; i < rot / 2; ++i) {
            const YarnRef y = yarn_ref(500000.0, i, rot, 14, 22);
            outside += std::fabs(qkv.at(i) - y.c) > y.tol || std::fabs(qkv.at(rot / 2 + i) - y.s) > y.tol;
        }
        CHECK(outside > 0);
    }
}

/* QSA's compressed key. Identical unit keys pool to one and a unit gain with eps 0 norms to one,
 * so what is left is the rotation at the block's position: c - s, then s + c, then the unrotated
 * tail, held to yarn_ref with YaRN on. Then the op's three other jobs: the mean rounded to k's
 * dtype before the norm, `page` choosing the output row (-1 for none), and an [M, 3] position
 * list read per frequency under imrope. */
TEST(op_qsa_block_key) {
    const int64_t n = 128, ratio = 4, rot = 64;
    const double magnitude = 1.0 + 0.1 * std::log(2.0);
    {
        const std::vector<int64_t> blocks = { 0, 262144, 524287 };
        const int64_t M = (int64_t)blocks.size();
        T k(RAD_F32, {M * ratio, n}); k.f(std::vector<float>((size_t)(M * ratio * n), 1.0f));
        T w(RAD_F32, {n}); w.f(std::vector<float>((size_t)n, 1.0f));
        T pos(RAD_I32, {M}); pos.n(blocks);
        T out(RAD_F32, {M, n});
        A a; a.t(k).t(w).tnull().t(pos).t(out).i("M", M).i("n", n).i("ratio", ratio)
            .i("rotary_dim", rot).d("eps", 0).d("theta", 1e7).d("scale", 1).d("wadd", 0)
            .s("dtype", "f32").d("yarn_factor", 2).d("yarn_low", 14).d("yarn_high", 22)
            .d("yarn_magnitude", magnitude);
        CHECK_OK(run("qsa_block_key", a));
        for (int64_t b = 0; b < M; ++b)
            for (int64_t i = 0; i < n; ++i) {
                if (i >= rot) { CHECK_EQ(out.at(b * n + i), 1.0f); continue; }
                const YarnRef y = yarn_ref((double)blocks[(size_t)b], (int)(i % (rot / 2)), (int)rot,
                                           14, 22);
                CHECK_NEAR(out.at(b * n + i), i < rot / 2 ? y.c - y.s : y.c + y.s, 2 * y.tol + 1e-6);
            }
    }
    {   /* bf16 keys whose lane-0 mean, 1 + 2^-9, rounds to 1 before the norm reads it. */
        T k(RAD_BF16, {ratio, 2});
        k.f({ 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f + 0x1p-7f, 2.0f });
        T w(RAD_F32, {2}); w.f({ 1.0f, 1.0f });
        T out(RAD_F32, {1, 2});
        A a; a.t(k).t(w).tnull().tnull().t(out).i("M", 1).i("n", 2).i("ratio", ratio)
            .i("rotary_dim", 0).i("pos0", 0).d("eps", 0).d("theta", 1e7).d("wadd", 0)
            .s("dtype", "bf16");
        CHECK_OK(run("qsa_block_key", a));
        CHECK_NEAR(out.at(0), 1.0 / std::sqrt(2.5), 1e-6);
        CHECK_NEAR(out.at(1), 2.0 / std::sqrt(2.5), 1e-6);
    }
    {   /* page: block 0 to row 1, block 1 nowhere, block 2 to row 0. Block b's lanes are
         * (1, b + 1), so each normed row says which block it came from. */
        T k(RAD_F32, {3 * ratio, 2});
        std::vector<float> kv;
        for (int b = 0; b < 3; ++b)
            for (int r = 0; r < ratio; ++r) { kv.push_back(1.0f); kv.push_back((float)(b + 1)); }
        k.f(kv);
        T w(RAD_F32, {2}); w.f({ 1.0f, 1.0f });
        T page(RAD_I32, {3}); page.n({ 1, -1, 0 });
        T out(RAD_F32, {2, 2}); out.f({ 7, 7, 7, 7 });
        A a; a.t(k).t(w).t(page).tnull().t(out).i("M", 3).i("n", 2).i("ratio", ratio)
            .i("rotary_dim", 0).i("pos0", 0).d("eps", 0).d("theta", 1e7).d("wadd", 0)
            .s("dtype", "f32");
        CHECK_OK(run("qsa_block_key", a));
        CHECK_NEAR(out.at(0), 1.0 / std::sqrt(5.0), 1e-6);
        CHECK_NEAR(out.at(1), 3.0 / std::sqrt(5.0), 1e-6);
        CHECK_NEAR(out.at(2), 1.0, 1e-6);
        CHECK_NEAR(out.at(3), 1.0, 1e-6);
    }
    {   /* imrope with a [M, 3] list: equal components are the one-position rotation, and moving
         * component 1 moves frequency 1 (i % 3 == 1) and leaves frequency 0 alone. */
        T k(RAD_F32, {ratio, n}); k.f(std::vector<float>((size_t)(ratio * n), 1.0f));
        T w(RAD_F32, {n}); w.f(std::vector<float>((size_t)n, 1.0f));
        auto key = [&](std::vector<int64_t> p, bool multi) {
            T pos(RAD_I32, multi ? std::vector<int64_t>{1, 3} : std::vector<int64_t>{1}); pos.n(p);
            T out(RAD_F32, {1, n});
            A a; a.t(k).t(w).tnull().t(pos).t(out).i("M", 1).i("n", n).i("ratio", ratio)
                .i("rotary_dim", rot).d("eps", 0).d("theta", 1e7).d("wadd", 0).s("dtype", "f32")
                .s("mode", "imrope").s("sections", "11 11 10");
            CHECK_OK(run("qsa_block_key", a));
            std::vector<float> v((size_t)n);
            for (int64_t i = 0; i < n; ++i) v[(size_t)i] = out.at(i);
            return v;
        };
        const std::vector<float> one = key({ 5000 }, false), same = key({ 5000, 5000, 5000 }, true);
        const std::vector<float> moved = key({ 5000, 9000, 5000 }, true);
        for (int64_t i = 0; i < n; ++i) CHECK_EQ(same[(size_t)i], one[(size_t)i]);
        CHECK_EQ(moved[0], one[0]);
        CHECK(std::fabs(moved[1] - one[1]) > 1e-3);
    }
}

TEST(disabled_yarn_ignores_auxiliary_coefficients) {
    for (double magnitude : {2.0, (double)NAN}) {
        T pos(RAD_I32, {1}); pos.n({0});
        T qkv(RAD_F32, {1, 4}); qkv.f({1, 2, 3, 4});
        A direct; direct.t(qkv).t(pos).i("M", 1).i("head_dim", 4)
            .i("n_head", 1).i("n_head_kv", 0).d("theta", 10000).d("scale", 1)
            .s("mode", "neox").d("yarn_factor", 0).d("yarn_low", NAN)
            .d("yarn_high", NAN).d("yarn_magnitude", magnitude);
        CHECK_OK(run("rope", direct));
        for (int i = 0; i < 4; ++i) CHECK_EQ(qkv.at(i), (float)(i + 1));
        T table(RAD_F32, {1, 4});
        A ta; ta.t(pos).t(table).i("M", 1).i("rot", 4).d("theta", 10000)
            .d("scale", 1).s("dtype", "f32").d("yarn_factor", 0)
            .d("yarn_low", NAN).d("yarn_high", NAN).d("yarn_magnitude", magnitude);
        CHECK_OK(run("rope_table", ta));
        CHECK_EQ(table.at(0), 1.0f); CHECK_EQ(table.at(1), 1.0f);
        CHECK_EQ(table.at(2), 0.0f); CHECK_EQ(table.at(3), 0.0f);
    }
}

TEST(yarn_multimodal_positions_and_invalid_parameters) {
    const int rot = 64, hd = 80;
    T positions(RAD_I32, {3, 1}); positions.n({262144, 500000, 524287});
    T mixed(RAD_F32, {1, hd});
    std::vector<float> seed(hd, 9.0f);
    for (int i = 0; i < rot; ++i) seed[i] = i < rot/2 ? 1.0f : 0.0f;
    mixed.f(seed);
    auto params = [&](A& a, const char* mode) -> A& {
        return a.i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
            .i("rotary_dim", rot).d("theta", 1e7).d("scale", 1).s("mode", mode)
            .s("sections", "11 11 10").d("yarn_factor", 2).d("yarn_low", 14)
            .d("yarn_high", 22).d("yarn_magnitude", 1.069314718);
    };
    A ma; ma.t(mixed).t(positions); params(ma, "imrope");
    CHECK_OK(run("rope", ma));
    for (int component = 0; component < 3; ++component) {
        T plain(RAD_F32, {1, hd}); plain.f(seed);
        T pos(RAD_I32, {1}); pos.n({positions.iat(component)});
        A a; a.t(plain).t(pos); params(a, "neox");
        CHECK_OK(run("rope", a));
        for (int i = 0; i < 32; ++i) {
            const int selected = i % 3 == 1 && i < 33 ? 1 : (i % 3 == 2 && i < 30 ? 2 : 0);
            if (selected != component) continue;
            CHECK_EQ(mixed.at(i), plain.at(i)); CHECK_EQ(mixed.at(i+32), plain.at(i+32));
        }
    }
    for (int i = rot; i < hd; ++i) CHECK_EQ(mixed.at(i), 9.0f);
    T single(RAD_I32, {1}); single.n({1});
    A invalid; invalid.t(mixed).t(single); params(invalid, "axial");
    CHECK_EQ(run("rope", invalid), RAD_E_UNSUPPORTED);
    for (auto& p : invalid.ps) if (!std::strcmp(p.key, "yarn_factor")) p = RAD_F64("yarn_factor", NAN);
    CHECK_EQ(run("rope", invalid), RAD_E_INVAL);
}

/* The epilogue is the unfused chain: the biased product, then its activation, then the residual,
 * each rounded where a separate tensor would be. */
TEST(op_gemm_nt_bias_epilogue) {
    T a_(RAD_F32, {1, 2}); a_.f({1, -1});
    T b_(RAD_F32, {3, 2}); b_.f({1, 0,  0, 1,  2, 1});
    T bias(RAD_F32, {3}); bias.f({0.5f, 0.25f, -1.0f});
    T res(RAD_F32, {1, 3}); res.f({10, 20, 30});
    const float pre[3] = { 1.5f, -0.75f, 0.0f };      /* a . b + bias */
    for (const char* act : { "gelu", "gelu_tanh" }) {
        T y(RAD_F32, {1, 3});
        A e; e.t(a_).t(b_).t(bias).t(res).t(y).i("M", 1).i("N", 3).i("K", 2).s("dtype", "f32")
              .s("act", act);
        CHECK_OK(run("gemm_nt_bias", e));
        for (int n = 0; n < 3; ++n) {
            const float x = pre[n];
            const float g = !std::strcmp(act, "gelu")
                ? 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)))
                : 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            CHECK_NEAR(y.at(n), res.at(n) + g, 1e-5);
        }
    }
    /* bf16 rounds at each step: 1.5 * 1.5 is exact, but the residual sum 258 + 2.25 is not. */
    T a2(RAD_BF16, {1, 1}); a2.f({1.5f});
    T b2(RAD_BF16, {1, 1}); b2.f({1.5f});
    T bias2(RAD_BF16, {1}); bias2.f({0.0f});
    T res2(RAD_BF16, {1, 1}); res2.f({258.0f});
    T y2(RAD_BF16, {1, 1});
    A f; f.t(a2).t(b2).t(bias2).t(res2).t(y2).i("M", 1).i("N", 1).i("K", 1).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt_bias", f));
    CHECK_EQ(y2.at(0), rad_bf16_to_f32(rad_f32_to_bf16(258.0f + 2.25f)));
}

/* The multi-component rotary. With every component of every token equal, the interleaved form is
 * NeoX to the bit; with them different, each frequency reads the component its section names. */
TEST(op_rope_multi_component) {
    const int64_t hd = 12, rot = 12;           /* six frequencies: sections 2 2 2 */
    std::vector<float> row(3 * hd);
    for (int64_t i = 0; i < 3 * hd; ++i) row[(size_t)i] = 0.1f * (float)(i + 1);
    T ref(RAD_F32, {1, 3 * hd}); ref.f(row);
    T pos1(RAD_I32, {1}); pos1.n({7});
    A a; a.t(ref).t(pos1).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 100).d("scale", 1).s("mode", "neox").i("rotary_dim", rot);
    CHECK_OK(run("rope", a));
    for (const char* mode : { "imrope", "mrope" }) {
        T got(RAD_F32, {1, 3 * hd}); got.f(row);
        T pos3(RAD_I32, {3, 1}); pos3.n({7, 7, 7});
        A b; b.t(got).t(pos3).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 1)
              .d("theta", 100).d("scale", 1).s("mode", mode).i("rotary_dim", rot)
              .s("sections", "2 2 2");
        CHECK_OK(run("rope", b));
        for (int64_t i = 0; i < 3 * hd; ++i) CHECK_EQ(got.at(i), ref.at(i));
    }
    /* Distinct components: frequency i of the interleaved form reads t, h, w, t, h, w for sections
     * 2 2 2 (i % 3 == 1 and i < 6 is h, i % 3 == 2 and i < 6 is w). */
    {
        T got(RAD_F32, {1, 2 * hd}); got.f(std::vector<float>(2 * hd, 1.0f));
        T pos3(RAD_I32, {3, 1}); pos3.n({1, 2, 3});
        A b; b.t(got).t(pos3).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "imrope").i("rotary_dim", rot)
              .s("sections", "2 2 2");
        CHECK_OK(run("rope", b));
        const int comp[6] = { 0, 1, 2, 0, 1, 2 };
        const float p[3] = { 1, 2, 3 };
        for (int i = 0; i < 6; ++i) {
            const float inv = 1.0f / std::pow(100.0f, (float)(2 * i) / (float)rot);
            const float ang = p[comp[i]] * inv;
            CHECK_NEAR(got.at(i), std::cos(ang) - std::sin(ang), 1e-6);
            CHECK_NEAR(got.at(i + 6), std::sin(ang) + std::cos(ang), 1e-6);
        }
    }
    /* Axial: two sections of three, each restarting its ladder at theta^0. */
    {
        T got(RAD_F32, {1, 2 * hd}); got.f(std::vector<float>(2 * hd, 1.0f));
        T pos2(RAD_I32, {2, 1}); pos2.n({5, 9});
        A b; b.t(got).t(pos2).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "axial").i("rotary_dim", rot)
              .s("sections", "3 3");
        CHECK_OK(run("rope", b));
        for (int i = 0; i < 6; ++i) {
            const int c = i < 3 ? 0 : 1, j = i < 3 ? i : i - 3;
            const float inv = 1.0f / std::pow(100.0f, (float)(2 * j) / 6.0f);
            const float ang = (c ? 9.0f : 5.0f) * inv;
            CHECK_NEAR(got.at(i), std::cos(ang) - std::sin(ang), 1e-6);
        }
    }
}

/* The proportional ladder. With head_dim wider than rotary_dim the two ladders give frequency 1
 * different angles: `rotary` divides by rotary_dim, `head` by head_dim. They also pair different
 * channels: `rotary` is a complete rotary of rotary_dim's width, pair i = (i, i + rot/2), with the
 * tail past rot untouched, while `head` is HF's proportional RoPE
 * (_compute_proportional_rope_parameters + rotate_half over the whole head): pair i is
 * (i, i + head_dim/2) and every frequency past rot/2 is zero, so only rot/2 pairs turn.
 * Frequency 0 is theta^0 under both; frequency 1 pins the ladder, the partner index the pairing. */
TEST(op_rope_ladder) {
    const int64_t hd = 8, rot = 4;        /* two frequencies, a partial rotary (hd > rot) */
    const float pos = 7.0f;
    for (const char* lad : { "rotary", "head" }) {
        const bool head = lad[0] == 'h';
        const int64_t den = head ? hd : rot;
        const int64_t poff = head ? hd / 2 : rot / 2;   /* where pair i's partner sits */
        T got(RAD_F32, { 1, hd });
        got.f(std::vector<float>(hd, 1.0f));
        T p1(RAD_I32, { 1 }); p1.n({ 7 });
        A a; a.t(got).t(p1).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "neox").i("rotary_dim", rot)
              .s("ladder", lad);
        CHECK_OK(run("rope", a));
        std::vector<bool> turned(hd, false);
        for (int i = 0; i < rot / 2; ++i) {
            const float inv = 1.0f / std::pow(100.0f, (float)(2 * i) / (float)den);
            const float ang = pos * inv;
            CHECK_NEAR(got.at(i), std::cos(ang) - std::sin(ang), 1e-5);
            CHECK_NEAR(got.at(i + poff), std::sin(ang) + std::cos(ang), 1e-5);
            turned[i] = turned[i + poff] = true;
        }
        for (int d = 0; d < hd; ++d)
            if (!turned[d]) CHECK_EQ(got.at(d), 1.0f);   /* every other channel untouched */
    }
    /* absent ladder == rotary: models that never set it keep their angles bit for bit */
    {
        T got(RAD_F32, { 1, hd });
        got.f(std::vector<float>(hd, 1.0f));
        T p1(RAD_I32, { 1 }); p1.n({ 7 });
        A a; a.t(got).t(p1).i("M", 1).i("head_dim", hd).i("n_head", 1).i("n_head_kv", 0)
              .d("theta", 100).d("scale", 1).s("mode", "neox").i("rotary_dim", rot);
        CHECK_OK(run("rope", a));
        const float ang = pos * (1.0f / std::pow(100.0f, (float)(2 * 1) / (float)rot));
        CHECK_NEAR(got.at(1), std::cos(ang) - std::sin(ang), 1e-5);
    }
}

/* Aligned corners: a grid's first and last rows and columns sit exactly on the table's, so a
 * corner patch reads one table row whole; an interior one blends four. */
TEST(op_grid_embed) {
    const int64_t side = 3, n = 2;
    std::vector<float> tab((size_t)(side * side * n));
    for (int64_t r = 0; r < side * side; ++r)
        for (int64_t i = 0; i < n; ++i) tab[(size_t)(r * n + i)] = (float)(r * 10 + i);
    T table(RAD_F32, {side * side, n}); table.f(tab);
    /* Three rows of a 5 x 4 grid: the top-left corner, the bottom-right, and (2, 1). */
    T x(RAD_F32, {3, n}); x.f({0, 0,  100, 100,  0, 0});
    T crd(RAD_I32, {4, 3}); crd.n({0, 4, 2,   0, 3, 1,   5, 5, 5,   4, 4, 4});
    A a; a.t(x).t(table).t(crd).i("M", 3).i("n", n).i("side", side).s("dtype", "f32");
    CHECK_OK(run("grid_embed", a));
    CHECK_EQ(x.at(0), 0.0f);  CHECK_EQ(x.at(1), 1.0f);                /* table row 0 */
    CHECK_EQ(x.at(2), 180.0f); CHECK_EQ(x.at(3), 181.0f);             /* 100 + table row 8 */
    /* (2, 1) of 5 x 4 sits at table (1.0, 0.6667): rows 3 and 4 by 1/3 and 2/3. */
    const float want = (30.0f * (1.0f / 3.0f) + 40.0f * (2.0f / 3.0f));
    CHECK_NEAR(x.at(4), want, 1e-4);
}

TEST(op_qk_norm_rope) {
    /* At position 0 the rotation is the identity, so this isolates the per-head RMS norm. */
    T qkv(RAD_F32, {1, 6}); qkv.f({3, 4,  3, 4,  7, 7});
    T pos(RAD_I32, {1});    pos.n({0});
    T qw(RAD_F32, {2});     qw.f({1, 1});
    T kw(RAD_F32, {2});     kw.f({2, 2});
    A a; a.t(qkv).t(pos).t(qw).t(kw).i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 1)
          .d("theta", 10000).d("eps", 0);
    CHECK_OK(run("qk_norm_rope", a));
    const float rms = std::sqrt((9.0f + 16.0f) / 2.0f);   /* 3.5355339 */
    CHECK_NEAR(qkv.at(0), 3.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(1), 4.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(2), 2.0f * 3.0f / rms, 1e-6);
    CHECK_NEAR(qkv.at(3), 2.0f * 4.0f / rms, 1e-6);
    CHECK_EQ(qkv.at(4), 7.0f);
}

/* qk_norm_rope `axial`: NeoX pairs (i, i + rot/2), section c's frequencies turned by position
 * component c on its own ladder (one pair a section here, so theta^0 = 1). Positions are f32
 * [C, M] and read as they are. q's gain 2 plus wadd 1 scales its normed row by 3; k has no gain,
 * which is a unit gain without wadd. Every head is [1, -1, 1, -1] (rms 1); v is left alone. */
TEST(qk_norm_rope_axial_turns_each_section_by_its_component) {
    T qkv(RAD_F32, {1, 12}); qkv.f({1, -1, 1, -1,  1, -1, 1, -1,  7, 7, 7, 7});
    T pos(RAD_F32, {2, 1}); pos.f({0.5f, -1.25f});
    T qw(RAD_F32, {4}); qw.f({2, 2, 2, 2});
    A a; a.t(qkv).t(pos).t(qw).tnull().i("M", 1).i("head_dim", 4).i("n_head", 1)
          .i("n_head_kv", 1).d("theta", 100).d("eps", 0).d("wadd", 1).s("mode", "axial")
          .i("rotary_dim", 4).s("sections", "1 1");
    CHECK_OK(run("qk_norm_rope", a));
    const double ca = std::cos(0.5), sa = std::sin(0.5), cb = std::cos(-1.25), sb = std::sin(-1.25);
    /* q pairs (0, 2) at 0.5 over (3, 3) and (1, 3) at -1.25 over (-3, -3). */
    CHECK_NEAR(qkv.at(0), 3 * ca - 3 * sa, 1e-5);
    CHECK_NEAR(qkv.at(2), 3 * sa + 3 * ca, 1e-5);
    CHECK_NEAR(qkv.at(1), -3 * cb + 3 * sb, 1e-5);
    CHECK_NEAR(qkv.at(3), -3 * sb - 3 * cb, 1e-5);
    CHECK_NEAR(qkv.at(4), ca - sa, 1e-6);
    CHECK_NEAR(qkv.at(6), sa + ca, 1e-6);
    CHECK_NEAR(qkv.at(5), -cb + sb, 1e-6);
    CHECK_NEAR(qkv.at(7), -sb - cb, 1e-6);
    for (int i = 8; i < 12; ++i) CHECK_EQ(qkv.at(i), 7.0f);
}

/* Rank-1 positions in `axial` are every component at once, still on the section ladders: with
 * sections "1 1" both pairs turn by the position itself (2), where neox at theta 100 would turn
 * the second by 2 * 100^(-1/2) = 0.2. `rotary_dim` 0 is the norm alone and takes no positions:
 * [3, 4, 0, 0] / 2.5 and [2, 2, 2, 2] / 2. */
TEST(qk_norm_rope_axial_rank1_and_the_norm_alone) {
    T qkv(RAD_F32, {1, 4}); qkv.f({1, -1, 1, -1});
    T pos(RAD_I32, {1}); pos.n({2});
    A a; a.t(qkv).t(pos).tnull().tnull().i("M", 1).i("head_dim", 4).i("n_head", 1)
          .i("n_head_kv", 0).d("theta", 100).d("eps", 0).s("mode", "axial").i("rotary_dim", 4)
          .s("sections", "1 1");
    CHECK_OK(run("qk_norm_rope", a));
    const double c = std::cos(2.0), s = std::sin(2.0);
    CHECK_NEAR(qkv.at(0), c - s, 1e-6);  CHECK_NEAR(qkv.at(2), s + c, 1e-6);
    CHECK_NEAR(qkv.at(1), -c + s, 1e-6); CHECK_NEAR(qkv.at(3), -s - c, 1e-6);

    T q2(RAD_F32, {2, 4}); q2.f({3, 4, 0, 0,  2, 2, 2, 2});
    A b; b.t(q2).tnull().tnull().tnull().i("M", 2).i("head_dim", 4).i("n_head", 1)
          .i("n_head_kv", 0).d("theta", 100).d("eps", 0).i("rotary_dim", 0);
    CHECK_OK(run("qk_norm_rope", b));
    CHECK_NEAR(q2.at(0), 1.2f, 1e-6); CHECK_NEAR(q2.at(1), 1.6f, 1e-6);
    CHECK_EQ(q2.at(2), 0.0f); CHECK_EQ(q2.at(3), 0.0f);
    for (int i = 4; i < 8; ++i) CHECK_NEAR(q2.at(i), 1.0f, 1e-6);

    /* A rotation without positions is refused. */
    A c2; c2.t(q2).tnull().tnull().tnull().i("M", 2).i("head_dim", 4).i("n_head", 1)
           .i("n_head_kv", 0).d("theta", 100).d("eps", 0);
    CHECK_EQ(run("qk_norm_rope", c2), RAD_E_INVAL);
}

/* rope reads an f32 `positions` as it is, divided by `scale`: 0.75 / 0.5 = 1.5 and -0.3 / 0.5;
 * an integer one stays integer, 3 / 2 = 1.5. */
TEST(rope_reads_f32_positions_as_fractions) {
    T qkv(RAD_F32, {2, 2}); qkv.f({1, 0, 1, 0});
    T pos(RAD_F32, {2}); pos.f({0.75f, -0.3f});
    A a; a.t(qkv).t(pos).i("M", 2).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 0)
          .d("theta", 10000).d("scale", 0.5).s("mode", "neox");
    CHECK_OK(run("rope", a));
    CHECK_NEAR(qkv.at(0), std::cos(1.5), 1e-6); CHECK_NEAR(qkv.at(1), std::sin(1.5), 1e-6);
    CHECK_NEAR(qkv.at(2), std::cos(-0.6), 1e-6); CHECK_NEAR(qkv.at(3), std::sin(-0.6), 1e-6);

    T q2(RAD_F32, {1, 2}); q2.f({1, 0});
    T p2(RAD_I32, {1}); p2.n({3});
    A b; b.t(q2).t(p2).i("M", 1).i("head_dim", 2).i("n_head", 1).i("n_head_kv", 0)
          .d("theta", 10000).d("scale", 2.0).s("mode", "neox");
    CHECK_OK(run("rope", b));
    CHECK_NEAR(q2.at(0), std::cos(1.5), 1e-6); CHECK_NEAR(q2.at(1), std::sin(1.5), 1e-6);
}

/* ================================================================== mixture of experts */
TEST(op_router_topk) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 2, 3, 0});
    T ids(RAD_I32, {1, 2}), w(RAD_F32, {1, 2});
    A a; a.t(lg).t(ids).t(w).i("M", 1).i("n_expert", 4).i("top_k", 2).i("norm", 0)
          .s("dtype", "f32");
    CHECK_OK(run("router_topk", a));
    CHECK_EQ(ids.iat(0), 2LL);
    CHECK_EQ(ids.iat(1), 1LL);
    const float sum = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f + std::exp(-3.0f);
    CHECK_NEAR(w.at(0), 1.0f / sum, 1e-6);
    CHECK_NEAR(w.at(1), std::exp(-1.0f) / sum, 1e-6);

    /* Renormalised, the kept pair is the softmax of {3, 2}, i.e. sigmoid(1) and its complement, a
     * number this test can state exactly. */
    T ids2(RAD_I32, {1, 2}), w2(RAD_F32, {1, 2});
    A b; b.t(lg).t(ids2).t(w2).i("M", 1).i("n_expert", 4).i("top_k", 2).i("norm", 1)
          .s("dtype", "f32");
    CHECK_OK(run("router_topk", b));
    CHECK_NEAR(w2.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(w2.at(1), 0.26894142f, 1e-6);
}

TEST(op_router_gemm_topk_scatter) {
    /* THE GATE-ABSENT FORM: the plugin-owned shared arm issues the gate itself, so the fused op
     * resolves with `gate_b` and `gate` as NULL SLOTS. Two tokens, three columns, four experts,
     * top-two. */
    const int64_t M = 2, K = 3, ne = 4, k = 2;
    T x(RAD_F32, {M, K});   x.f({1, 0, 0,   0, 1, 0});
    T w(RAD_F32, {ne, K});  w.f({1, 0, 0,   0, 2, 0,   0, 0, 3,   1, 1, 0});
    T lg(RAD_F32, {M, ne});
    T ids(RAD_I32, {M, k}), ew(RAD_F32, {M, k});
    T sorted(RAD_I32, {M * k}), eoff(RAD_I32, {ne + 1}), ecnt(RAD_I32, {ne});
    A a; a.t(x).t(w).tnull().t(lg).t(ids).t(ew).t(sorted).t(eoff).t(ecnt).tnull()
          .i("M", M).i("K", K).i("n_expert", ne).i("n_local", ne).i("top_k", k).i("norm", 0)
          .s("dtype", "f32").i("expert_base", 0);
    CHECK_OK(run("router_gemm_topk_scatter", a));
    /* Row 0: logits {1, 0, 0, 1} -- experts 0 and 3 tie at 1, the LOWER id is selected first,
     * then 3; row 1: {0, 2, 0, 1} -- 1 at 2, then 3 at 1. */
    CHECK_EQ(ids.iat(0), 0LL); CHECK_EQ(ids.iat(1), 3LL);
    CHECK_EQ(ids.iat(2), 1LL); CHECK_EQ(ids.iat(3), 3LL);
    CHECK_NEAR(lg.at(3), 1.0f, 1e-6);
    CHECK_NEAR(lg.at(5), 2.0f, 1e-6);
    /* The sort: expert 0 has slot 0, expert 1 slot 2, expert 3 slots 1 and 3 -- sorted by pair. */
    CHECK_EQ(eoff.iat(0), 0LL); CHECK_EQ(eoff.iat(1), 1LL); CHECK_EQ(eoff.iat(2), 2LL);
    CHECK_EQ(eoff.iat(3), 2LL); CHECK_EQ(eoff.iat(4), 4LL);
    CHECK_EQ(sorted.iat(0), 0LL); CHECK_EQ(sorted.iat(1), 2LL);
    CHECK_EQ(sorted.iat(2), 1LL); CHECK_EQ(sorted.iat(3), 3LL);
    CHECK_NEAR(ew.at(0) + ew.at(1),
               (std::exp(0.0f) + std::exp(0.0f)) /
                   (2.0f * std::exp(0.0f) + 2.0f * std::exp(-1.0f)), 1e-5);

    /* THE GATE PRESENT: `gate` = x @ gate_b^T at N = 1, beside the router, over the same x. */
    T gb(RAD_F32, {1, K}); gb.f({2, 3, 0});
    T gate(RAD_F32, {M, 1});
    T lg2(RAD_F32, {M, ne});
    T ids2(RAD_I32, {M, k}), ew2(RAD_F32, {M, k});
    T sorted2(RAD_I32, {M * k}), eoff2(RAD_I32, {ne + 1}), ecnt2(RAD_I32, {ne});
    A b; b.t(x).t(w).t(gb).t(lg2).t(ids2).t(ew2).t(sorted2).t(eoff2).t(ecnt2).t(gate)
          .i("M", M).i("K", K).i("n_expert", ne).i("n_local", ne).i("top_k", k).i("norm", 1)
          .s("dtype", "f32").i("expert_base", 0);
    CHECK_OK(run("router_gemm_topk_scatter", b));
    CHECK_NEAR(gate.at(0), 2.0f, 1e-6);
    CHECK_NEAR(gate.at(1), 3.0f, 1e-6);
    CHECK_NEAR(ew2.at(0) + ew2.at(1), 1.0f, 1e-6);   /* renormalised to sum to one */
}

/* The verify expert budget (opt-in, lossy): router_gemm_topk_scatter with `budget`. Six experts, top
 * three, norm on; the router weight is the identity, so a row's logits are the row of x. Row 0 picks
 * {0, 1, 2}, row 1 {0, 3, 4} and row 2 {1, 4, 5}. Budget 4 with row 0 exact: the forced three, plus the
 * one restricted-row expert with the most mass, 4 (0.09 + 0.245). Rows 1 and 2 lose their slots on 3
 * and 5, and their kept weights are scaled back up to the row's whole mass. */
TEST(op_router_gemm_topk_scatter_budget) {
    const int64_t M = 3, K = 6, ne = 6, k = 3;
    T x(RAD_F32, {M, K});  x.f({5, 4, 3, 0, 0, 0,   5, 0, 0, 4, 3, 0,   0, 5, 0, 0, 4, 3});
    T w(RAD_F32, {ne, K}); w.f({1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1});
    auto go = [&](int64_t budget, T& ids, T& ew, T& sorted, T& eoff, T& ecnt) {
        T lg(RAD_F32, {M, ne});
        A a; a.t(x).t(w).tnull().t(lg).t(ids).t(ew).t(sorted).t(eoff).t(ecnt).tnull()
              .i("M", M).i("K", K).i("n_expert", ne).i("n_local", ne).i("top_k", k).i("norm", 1)
              .s("dtype", "f32").i("expert_base", 0);
        if (budget > 0) a.i("budget", budget).i("exact_rows", 1);
        return run("router_gemm_topk_scatter", a);
    };
    T ids(RAD_I32, {M, k}), ew(RAD_F32, {M, k}), sorted(RAD_I32, {M * k}), eoff(RAD_I32, {ne + 1}), ecnt(RAD_I32, {ne});
    CHECK_OK(go(4, ids, ew, sorted, eoff, ecnt));
    CHECK_EQ(ids.iat(3), 0LL); CHECK_EQ(ids.iat(4), 3LL); CHECK_EQ(ids.iat(5), 4LL);   /* ids unchanged */
    const float e5 = std::exp(5.f), e4 = std::exp(4.f), e3 = std::exp(3.f), tot = e5 + e4 + e3;
    CHECK_NEAR(ew.at(0), e5 / tot, 1e-5);                         /* row 0 exact */
    CHECK_NEAR(ew.at(3), e5 / (e5 + e3), 1e-5);                   /* row 1: experts 0 and 4 stay */
    CHECK_NEAR(ew.at(5), e3 / (e5 + e3), 1e-5);
    CHECK_NEAR(ew.at(7), e4 / (e5 + e4), 1e-5);                   /* row 2: experts 1 and 4 stay */
    CHECK_EQ(ecnt.iat(3), 0LL); CHECK_EQ(ecnt.iat(5), 0LL);       /* 3 and 5 are not read */
    CHECK_EQ(ecnt.iat(0), 2LL); CHECK_EQ(ecnt.iat(1), 2LL); CHECK_EQ(ecnt.iat(2), 1LL);
    CHECK_EQ(ecnt.iat(4), 2LL);
    CHECK_EQ(eoff.iat(6), 7LL);
    /* budget 3 = row 0's experts only: row 1 keeps expert 0 alone, row 2 expert 1 alone, each with the
     * row's whole mass (weight 1). */
    T ids2(RAD_I32, {M, k}), ew2(RAD_F32, {M, k}), sorted2(RAD_I32, {M * k}), eoff2(RAD_I32, {ne + 1}), ecnt2(RAD_I32, {ne});
    CHECK_OK(go(3, ids2, ew2, sorted2, eoff2, ecnt2));
    CHECK_NEAR(ew2.at(3), 1.0f, 1e-5); CHECK_NEAR(ew2.at(6), 1.0f, 1e-5);
    CHECK_EQ(eoff2.iat(6), 5LL);
    /* budget off and a huge budget: the same bytes. */
    T ids3(RAD_I32, {M, k}), ew3(RAD_F32, {M, k}), sorted3(RAD_I32, {M * k}), eoff3(RAD_I32, {ne + 1}), ecnt3(RAD_I32, {ne});
    CHECK_OK(go(0, ids3, ew3, sorted3, eoff3, ecnt3));
    T ids4(RAD_I32, {M, k}), ew4(RAD_F32, {M, k}), sorted4(RAD_I32, {M * k}), eoff4(RAD_I32, {ne + 1}), ecnt4(RAD_I32, {ne});
    CHECK_OK(go(100, ids4, ew4, sorted4, eoff4, ecnt4));
    for (int i = 0; i < M * k; ++i) { CHECK_NEAR(ew3.at(i), ew4.at(i), 0.0); CHECK_EQ(sorted3.iat(i), sorted4.iat(i)); }
}

/* The four-class grouped GEMM: expert e is entry e / 4 of class e % 4, each class at its own
 * width, and a narrower class's columns past its width are zero. Ten experts, so classes 0 and 1
 * hold three and 2 and 3 hold two (the shape of a layer with 510 quantised experts), and class 1
 * is the wide one, as on rank 1. Every value is a small integer, so the sums are exact. */
TEST(op_moe_gemm_q_mod4) {
    const int64_t K = 3, top_k = 2, M = 3, ne = 10;
    const int64_t Nq[4] = { 2, 4, 2, 2 };
    auto wv = [](int64_t e, int64_t n, int64_t k) { return (float)(e * 100 + n * 10 + k); };
    std::vector<T> w, ws;
    w.reserve(4);
    ws.reserve(4);
    for (int q = 0; q < 4; ++q) {
        const int64_t cnt = (ne + 3 - q) / 4;
        w.emplace_back(RAD_F32, std::vector<int64_t>{ cnt, Nq[q], K });
        std::vector<float> v;
        for (int64_t j = 0; j < cnt; ++j)
            for (int64_t n = 0; n < Nq[q]; ++n)
                for (int64_t k = 0; k < K; ++k) v.push_back(wv(q + 4 * j, n, k));
        w.back().f(v);
        /* One unit scale per expert: the arithmetic under test is which weight a row reads. */
        ws.emplace_back(RAD_F32, std::vector<int64_t>{ cnt, 1, 1 });
        ws.back().f(std::vector<float>((size_t)cnt, 1.0f));
    }
    T act(RAD_F32, {M, K});
    act.f({ 1, 2, 3,  4, 5, 6,  7, 8, 9 });
    T as(RAD_F32, {M, 1});
    as.f({ 1, 1, 1 });
    /* Rows by expert: 1 owns rows 0-1, 2 row 2, 6 row 3, 9 rows 4-5, the rest none. */
    const std::vector<int64_t> eoff = { 0, 0, 2, 3, 3, 3, 3, 4, 4, 4, 6 };
    const std::vector<int64_t> flat = { 0, 2, 1, 4, 3, 5 };
    T sorted(RAD_I32, {6}); sorted.n(flat);
    T off(RAD_I32, {ne + 1}); off.n(eoff);
    T y(RAD_F32, {6, 4});
    A a;
    a.t(act).t(as).t(w[0]).t(ws[0]).t(sorted).t(off).t(w[1]).t(ws[1]).t(w[2]).t(ws[2]).t(w[3])
     .t(ws[3]).t(y)
     .i("M", M).i("N", Nq[0]).i("K", K).i("N_1", Nq[1]).i("K_1", K).i("N_2", Nq[2]).i("K_2", K)
     .i("N_3", Nq[3]).i("K_3", K).i("n_expert", ne).i("top_k", top_k).s("a_order", "token")
     .s("dtype", "f32");
    CHECK_OK(run("moe_gemm_q_mod4", a));
    for (int64_t i = 0; i < 6; ++i) {
        int64_t e = 0;
        while (eoff[(size_t)(e + 1)] <= i) ++e;
        const int64_t tok = flat[(size_t)i] / top_k;
        for (int64_t n = 0; n < 4; ++n) {
            float want = 0.0f;
            if (n < Nq[e % 4])
                for (int64_t k = 0; k < K; ++k) want += act.at(tok * K + k) * wv(e, n, k);
            CHECK_EQ(y.at(i * 4 + n), want);
        }
    }
}

/* The weight scale grid is the one the dtype names, in both grouped GEMMs: 128x128 tiles for block
 * fp8; for four-bit experts one scale per row per K group of 128 (w4a8/w4a8h/w4nla8h), 64
 * (w4nl64a8h/w5nl64a8h) or 32 (w4nl32a8h). The oracle inverts a relaid weight into exactly this
 * description (core/runtime/oracle.cpp). The final case is one last-class expert at K = 128 whose
 * 64-wide halves scale by 2 and 3 under one 0.5 activation scale; any other grid reads differently. */
TEST(op_moe_gemm_q_scale_grid_by_dtype) {
    struct Want { const char* dt; int64_t rows, cols; };
    /* N 256, K 512, group 128; the last class at N 128, K 256. */
    const Want wants[] = { { "fp8a8", 2, 4 },      { "w4a8", 256, 4 },     { "w4a8h", 256, 4 },
                           { "w4nla8h", 256, 4 },  { "w4nl64a8h", 256, 8 }, { "w5nl64a8h", 256, 8 },
                           { "w4nl32a8h", 256, 16 }, { "mxfp4a8", 256, 16 } };
    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        const bool four = std::strcmp(op, "moe_gemm_q_mod4") == 0;
        const int last = four ? 11 : 7;
        for (const Want& w : wants) {
            A a;
            a.i("M", 2).i("N", 256).i("K", 512).i("n_expert", 8).i("top_k", 2).i("group", 128)
             .s("a_order", "token").s("dtype", w.dt);
            if (four) a.i("N_3", 128).i("K_3", 256);
            else      a.i("N_odd", 128).i("K_odd", 256);
            const RadArgs* r = a.get();
            RadOpdDesc d{}, dl{};
            REQUIRE_EQ(k->opd_shape(r->p, r->n_p, 3, &d), RAD_OK);
            REQUIRE_EQ(k->opd_shape(r->p, r->n_p, last, &dl), RAD_OK);
            CHECK_EQ(d.dtype, !std::strcmp(w.dt, "mxfp4a8") ? (uint32_t)RAD_U8 : (uint32_t)RAD_BF16);
            CHECK_EQ(d.rank, 3u);
            CHECK_EQ(d.shape[0], four ? 2LL : 4LL);
            if (d.shape[1] != w.rows || d.shape[2] != w.cols)
                std::fprintf(stderr, "    %s at %s: w_scale [%lld, %lld], want [%lld, %lld]\n", op,
                             w.dt, (long long)d.shape[1], (long long)d.shape[2],
                             (long long)w.rows, (long long)w.cols);
            CHECK_EQ(d.shape[1], w.rows);
            CHECK_EQ(d.shape[2], w.cols);
            /* the last class at its own geometry, on the same grid */
            CHECK_EQ(dl.shape[1], w.rows == 2 ? 1LL : 128LL);
            CHECK_EQ(dl.shape[2], w.cols / 2);
        }
    }

    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        const bool four = std::strcmp(op, "moe_gemm_q_mod4") == 0;
        const int ncls = four ? 4 : 2, ne = ncls, n_opd = four ? 13 : 9;
        auto params = [&](A& x) {
            x.i("M", 1).i("N", 2).i("K", 128).i("n_expert", ne).i("top_k", 1).i("group", 128)
             .s("a_order", "token").s("dtype", "w4nl64a8h");
        };
        A p;
        params(p);
        const RadArgs* pr = p.get();
        std::deque<T> opd;
        for (int i = 0; i < n_opd; ++i) {
            RadOpdDesc d{};
            REQUIRE_EQ(k->opd_shape(pr->p, pr->n_p, i, &d), RAD_OK);
            opd.emplace_back(d.dtype, std::vector<int64_t>(d.shape, d.shape + d.rank));
        }
        /* the classes' (codes, scale) operand pairs, class 0 first */
        std::vector<std::pair<int, int>> cls = { { 2, 3 }, { 6, 7 } };
        if (four) cls.insert(cls.end(), { { 8, 9 }, { 10, 11 } });
        for (int q = 0; q < ncls; ++q) {
            T& w = opd[(size_t)cls[(size_t)q].first];
            T& s = opd[(size_t)cls[(size_t)q].second];
            REQUIRE_EQ(s.t.shape[1], 2LL);
            REQUIRE_EQ(s.t.shape[2], 2LL);
            const bool hit = q == ncls - 1;
            std::vector<float> wv(256);
            for (int64_t n = 0; n < 2; ++n)
                for (int64_t kk = 0; kk < 128; ++kk)
                    wv[(size_t)(n * 128 + kk)] = hit ? (float)(n + 1) : 7.0f;
            w.f(wv);
            s.f(hit ? std::vector<float>{ 2, 3, 2, 3 } : std::vector<float>{ 1, 1, 1, 1 });
        }
        opd[0].f(std::vector<float>(128, 1.0f));
        opd[1].f({ 0.5f });
        opd[4].n({ 0 });
        /* every expert empty but the last one, which owns the one row */
        std::vector<int64_t> off((size_t)ne + 1, 0);
        off[(size_t)ne] = 1;
        opd[5].n(off);
        A a;
        for (int i = 0; i < n_opd; ++i) a.t(opd[(size_t)i]);
        params(a);
        CHECK_OK(run(op, a));
        const T& y = opd[(size_t)n_opd - 1];
        CHECK_EQ(y.at(0), 160.0f);                   /* 0.5 * (64*1*2 + 64*1*3) */
        CHECK_EQ(y.at(1), 320.0f);                   /* 0.5 * (64*2*2 + 64*2*3) */
    }

    /* A dtype naming none of those grids (another op's format, or no dtype) is declined for every
     * operand, scale planes included: a plane sized on a guessed grid has the wrong extent. */
    for (const char* op : { "moe_gemm_q", "moe_gemm_q_mod4" }) {
        const RadKernelInfo* k = kern(op);
        REQUIRE(k != nullptr && k->opd_shape != nullptr);
        for (const char* dt : { "w8a8", "invalid", "" }) {
            A a;
            a.i("M", 2).i("N", 256).i("K", 512).i("n_expert", 8).i("top_k", 2).i("group", 128)
             .s("a_order", "token");
            if (*dt) a.s("dtype", dt);
            const RadArgs* r = a.get();
            for (int opd : { 0, 2, 3 }) {
                RadOpdDesc d{};
                if (k->opd_shape(r->p, r->n_p, opd, &d) != RAD_E_UNSUPPORTED)
                    std::fprintf(stderr, "    %s at dtype '%s': operand %d described\n", op, dt,
                                 opd);
                CHECK_EQ(k->opd_shape(r->p, r->n_p, opd, &d), RAD_E_UNSUPPORTED);
            }
        }
    }
}

TEST(op_moe_chain) {
    /* Two tokens, two experts, top_k 2: every token visits every expert, so the whole chain is
     * checkable without routing subtleties. */
    T ids(RAD_I32, {2, 2}); ids.n({1, 0, 0, 1});
    T sorted(RAD_I32, {4}), off(RAD_I32, {3}), cnt(RAD_I32, {2});
    A a; a.t(ids).t(sorted).t(off).t(cnt).i("M", 2).i("n_expert", 2).i("top_k", 2);
    CHECK_OK(run("moe_scatter", a));
    CHECK_EQ(cnt.iat(0), 2LL); CHECK_EQ(cnt.iat(1), 2LL);
    CHECK_EQ(off.iat(0), 0LL); CHECK_EQ(off.iat(1), 2LL); CHECK_EQ(off.iat(2), 4LL);
    /* expert 0 owns flat slots 1 and 2; expert 1 owns 0 and 3, in ascending flat order. */
    CHECK_EQ(sorted.iat(0), 1LL); CHECK_EQ(sorted.iat(1), 2LL);
    CHECK_EQ(sorted.iat(2), 0LL); CHECK_EQ(sorted.iat(3), 3LL);

    T act(RAD_F32, {2, 2}); act.f({1, 2, 3, 4});
    T w(RAD_F32, {2, 1, 2}); w.f({1, 0,  0, 1});     /* expert 0 picks dim 0, expert 1 dim 1 */
    T ye(RAD_F32, {4, 1});
    A b; b.t(act).t(w).t(sorted).t(off).t(ye)
          .i("M", 2).i("N", 1).i("K", 2).i("n_expert", 2).i("top_k", 2).s("dtype", "f32");
    CHECK_OK(run("moe_gemm", b));
    CHECK_EQ(ye.at(0), 1.0f);   /* sorted[0] = flat 1 -> token 0, expert 0 -> a[0][0] */
    CHECK_EQ(ye.at(1), 3.0f);
    CHECK_EQ(ye.at(2), 2.0f);
    CHECK_EQ(ye.at(3), 4.0f);

    T ew(RAD_F32, {2, 2}); ew.f({0.5f, 0.5f, 0.25f, 0.75f});
    T y(RAD_F32, {2, 1});
    A c; c.t(ye).t(ew).t(sorted).tnull().tnull().t(y)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32");
    CHECK_OK(run("moe_gather", c));
    /* token 0: slot 0 is at sorted position 2 (value 2), slot 1 at position 0 (value 1). */
    CHECK_NEAR(y.at(0), 0.5f * 2.0f + 0.5f * 1.0f, 1e-6);
    CHECK_NEAR(y.at(1), 0.25f * 3.0f + 0.75f * 4.0f, 1e-6);

    /* The shared arm folded in: the same routed result plus `shared * act(gate)`, the
     * `scale_rows` this op's output would otherwise go to. Both or neither: one alone is an error,
     * since a gate with nothing to gate means a caller who thinks they folded and did not. */
    T sh(RAD_F32, {2, 1}); sh.f({10.0f, 20.0f});
    T sg(RAD_F32, {2, 1}); sg.f({0.5f, 0.25f});
    T yf(RAD_F32, {2, 1});
    A d; d.t(ye).t(ew).t(sorted).t(sh).t(sg).t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32").s("act", "none");
    CHECK_OK(run("moe_gather", d));
    CHECK_NEAR(yf.at(0), 10.0f * 0.5f + 1.5f, 1e-6);
    CHECK_NEAR(yf.at(1), 20.0f * 0.25f + 3.75f, 1e-6);

    /* `act` applies to the scalar, not the product: sigmoid(0) = 0.5 exactly. */
    T sg0(RAD_F32, {2, 1}); sg0.f({0.0f, 0.0f});
    A e; e.t(ye).t(ew).t(sorted).t(sh).t(sg0).t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32").s("act", "sigmoid");
    CHECK_OK(run("moe_gather", e));
    CHECK_NEAR(yf.at(0), 10.0f * 0.5f + 1.5f, 1e-6);
    CHECK_NEAR(yf.at(1), 20.0f * 0.5f + 3.75f, 1e-6);

    A f; f.t(ye).t(ew).t(sorted).t(sh).tnull().t(yf)
          .i("M", 2).i("n", 1).i("top_k", 2).s("dtype", "f32");
    CHECK_EQ(run("moe_gather", f), RAD_E_INVAL);
}

TEST(op_moe_scatter_expert_base) {
    /* Expert parallelism, which `expert_base` is for: four experts in the model, this rank owns
     * [2, 4), and the replicated router picks from all four. Slots routed to other ranks must be
     * dropped here and the rest sorted by local index, since the offsets, counts and weight table
     * downstream are all this rank's.
     *
     * Two tokens, top_k 2: flat slots 0..3 route to experts 3, 0, 2, 1. Only slots 0 (expert 3 ->
     * local 1) and 2 (expert 2 -> local 0) are ours. */
    T ids(RAD_I32, {2, 2}); ids.n({3, 0, 2, 1});
    T sorted(RAD_I32, {4}), off(RAD_I32, {3}), cnt(RAD_I32, {2});
    A a; a.t(ids).t(sorted).t(off).t(cnt)
          .i("M", 2).i("n_expert", 2).i("top_k", 2).i("expert_base", 2);
    CHECK_OK(run("moe_scatter", a));
    CHECK_EQ(cnt.iat(0), 1LL); CHECK_EQ(cnt.iat(1), 1LL);
    CHECK_EQ(off.iat(0), 0LL); CHECK_EQ(off.iat(1), 1LL); CHECK_EQ(off.iat(2), 2LL);
    CHECK_EQ(sorted.iat(0), 2LL);   /* local expert 0 (global 2) owns flat slot 2 */
    CHECK_EQ(sorted.iat(1), 0LL);   /* local expert 1 (global 3) owns flat slot 0 */
    /* The tail is -1, which tells the GEMM there is no row there. Two of the four slots belong to
     * another rank and this one computes nothing for them; its partial is summed with that rank's
     * by the block's end-of-block all-reduce. */
    CHECK_EQ(sorted.iat(2), -1LL); CHECK_EQ(sorted.iat(3), -1LL);

    /* base 0 with the full count is the single-rank case: every slot is kept. */
    T s2(RAD_I32, {4}), o2(RAD_I32, {5}), c2(RAD_I32, {4});
    A b; b.t(ids).t(s2).t(o2).t(c2).i("M", 2).i("n_expert", 4).i("top_k", 2);
    CHECK_OK(run("moe_scatter", b));
    CHECK_EQ(o2.iat(4), 4LL);
    CHECK_EQ(s2.iat(0), 1LL); CHECK_EQ(s2.iat(1), 3LL);
    CHECK_EQ(s2.iat(2), 2LL); CHECK_EQ(s2.iat(3), 0LL);
}

TEST(op_row_topk) {
    T x(RAD_F32, {1, 4}); x.f({3, 1, 3, 2});
    T idx(RAD_I32, {1, 2}), val(RAD_F32, {1, 2});
    A a; a.t(x).t(idx).t(val).i("M", 1).i("N", 4).i("R", 2).s("dtype", "f32");
    CHECK_OK(run("row_topk", a));
    CHECK_EQ(idx.iat(0), 0LL);      /* ties go to the lower column */
    CHECK_EQ(idx.iat(1), 2LL);
    CHECK_EQ(val.at(0), 3.0f);
    CHECK_EQ(val.at(1), 3.0f);
}

/* ================================================================== gated delta net */
TEST(op_gdn_conv_prep) {
    /* One sequence, two tokens, one head each of k and v, all width 1, conv width 2 with the
     * weight [0, 1], so the convolution is the identity on the current token and only the silu
     * acts on it. x is the fused row [q | k | v | a | b], five channels. */
    const int64_t conv_dim = 3, D = 5;
    T x(RAD_F32, {2, D});
    x.f({ 1, 1, 1, 0, 0,
          2, 2, 2, 0, 0 });
    T w(RAD_F32, {conv_dim, 2}); w.f({0, 1, 0, 1, 0, 1});
    T b(RAD_F32, {conv_dim});    b.f({0, 0, 0});
    T alog(RAD_F32, {1});        alog.f({0});
    T dtb(RAD_F32, {1});         dtb.f({0});
    T cs(RAD_F32, {1, conv_dim, 1});
    T cu(RAD_I32, {2});          cu.n({0, 2});
    T q(RAD_F32, {2, 1, 1}), k(RAD_F32, {2, 1, 1}), v(RAD_F32, {2, 1, 1});
    T g(RAD_F32, {2, 1}), beta(RAD_F32, {2, 1});
    /* Inputs first, outputs last: x, w, b, A_log, dt_bias, conv_state, cu, then the four optional
     * inputs (a, b_gate, cache_idx, has_init), then q, k, v, g, beta. `a` and `b_gate` are absent
     * here, so the gate and beta come out of the fused x row's last two columns. */
    A a; a.t(x).t(w).t(b).t(alog).t(dtb).t(cs).t(cu).tnull().tnull().tnull().tnull()
          .t(q).t(k).t(v).t(g).t(beta)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2).i("conv_width", 2);
    CHECK_OK(run("gdn_conv_prep", a));

    /* v is not normalised, so it is exactly silu of the input. */
    CHECK_NEAR(v.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(v.at(1), 1.76159416f, 1e-6);
    /* q and k are l2-normalised over a one-element head, so they are 1 up to the epsilon. */
    CHECK_NEAR(q.at(0), 1.0f, 1e-4);
    CHECK_NEAR(k.at(1), 1.0f, 1e-4);
    /* g = -exp(A_log) * softplus(a + dt_bias) = -ln 2, cumulative within the chunk. */
    CHECK_NEAR(g.at(0), -0.69314718f, 1e-6);
    CHECK_NEAR(g.at(1), -1.38629436f, 1e-6);
    CHECK_NEAR(beta.at(0), 0.5f, 1e-6);
    /* The conv state keeps the sequence's last conv_width-1 raw tokens. */
    CHECK_NEAR(cs.at(0), 2.0f, 1e-6);
    CHECK_NEAR(cs.at(2), 2.0f, 1e-6);

    /* The separate gate and beta projections. The same numbers must come out when `a` and
     * `b_gate` are named operands rather than the tail of x, or the schema's union of the two
     * readings would mean two ops. `cache_idx` names the slot and `has_init` 0 says it holds no
     * history; the conv weight [0, 1] multiplies the history by zero either way, so this case
     * proves the operand positions, not the zeroing. */
    T x2(RAD_F32, {2, conv_dim});
    x2.f({ 1, 1, 1,
           2, 2, 2 });
    T ta(RAD_F32, {2, 1});   ta.f({0, 0});
    T tb(RAD_F32, {2, 1});   tb.f({0, 0});
    T cidx(RAD_I32, {1});    cidx.n({0});
    T hini(RAD_U8, {1});     hini.n({0});
    T cs2(RAD_F32, {1, conv_dim, 1});
    for (int c = 0; c < conv_dim; ++c) st_dt(RAD_F32, cs2.t.data, c, 99.0f);   /* must be ignored */
    T q2(RAD_F32, {2, 1, 1}), k2(RAD_F32, {2, 1, 1}), v2(RAD_F32, {2, 1, 1});
    T g2(RAD_F32, {2, 1}), beta2(RAD_F32, {2, 1});
    A b2; b2.t(x2).t(w).t(b).t(alog).t(dtb).t(cs2).t(cu).t(ta).t(tb).t(cidx).t(hini)
            .t(q2).t(k2).t(v2).t(g2).t(beta2)
            .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2).i("conv_width", 2);
    CHECK_OK(run("gdn_conv_prep", b2));
    CHECK_NEAR(v2.at(0), 0.73105858f, 1e-6);
    CHECK_NEAR(v2.at(1), 1.76159416f, 1e-6);
    CHECK_NEAR(g2.at(0), -0.69314718f, 1e-6);
    CHECK_NEAR(g2.at(1), -1.38629436f, 1e-6);
    CHECK_NEAR(beta2.at(0), 0.5f, 1e-6);
}

TEST(op_gdn_conv_update_rollback) {
    /* The speculative rollback contract, the reason this op exists separately. conv_width 3, so the
     * state holds two history entries plus this step's tokens; a step of two candidate tokens
     * needs (width-2) + 2 = 3 slots. The state starts as [1, 2, 3] and num_accepted = 2 means the
     * read window is slots [1, 2] = [2, 3]. */
    const int64_t conv_dim = 3, D = 3, W = 3;
    T x(RAD_F32, {2, D});
    x.f({ 10, 10, 10,
          20, 20, 20 });
    T w(RAD_F32, {conv_dim, W}); w.f({1, 1, 1,  1, 1, 1,  1, 1, 1});
    T b(RAD_F32, {conv_dim});    b.f({0, 0, 0});
    T cs(RAD_F32, {1, conv_dim, 3});
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 3; ++i) st_dt(RAD_F32, cs.t.data, c * 3 + i, (float)(i + 1));
    T sidx(RAD_I32, {1}); sidx.n({0});
    T nacc(RAD_I32, {1}); nacc.n({2});
    T cu(RAD_I32, {2});   cu.n({0, 2});
    T q(RAD_F32, {2, 1, 1}), k(RAD_F32, {2, 1, 1}), v(RAD_F32, {2, 1, 1});
    A a; a.t(x).t(w).t(b).t(cs).t(sidx).t(nacc).t(cu).t(q).t(k).t(v)
          .i("q_len", 2).i("head_k", 1).i("head_v", 1).i("conv_width", 3);
    CHECK_OK(run("gdn_conv_update", a));

    /* Token 0's window is [state[1], state[2], x0] = [2, 3, 10] -> sum 15 -> silu(15). */
    CHECK_NEAR(v.at(0), 15.0f * (1.0f / (1.0f + std::exp(-15.0f))), 1e-4);
    /* Token 1's window is [state[2], x0, x1] = [3, 10, 20] -> 33. */
    CHECK_NEAR(v.at(1), 33.0f * (1.0f / (1.0f + std::exp(-33.0f))), 1e-3);

    /* The buffer has shifted down by one: [old[2], x0, x1]. Invariant: the last accepted token of
     * a step sits at slot (its index) + width - 2, so the next step reads at num_accepted - 1 and
     * finds exactly the right history. */
    CHECK_NEAR(cs.at(0), 3.0f, 1e-6);
    CHECK_NEAR(cs.at(1), 10.0f, 1e-6);
    CHECK_NEAR(cs.at(2), 20.0f, 1e-6);
}

TEST(op_gdn_kkt_solve) {
    /* One chunk of two tokens, one head, head_k 1, both keys [1]. Then
     *     M[1][0] = beta1 * (k1 . k0) * exp(g1 - g0) = 0.5
     * and the inverse of the unit lower triangular I + M is I - M. */
    T k(RAD_F32, {2, 1, 1}); k.f({1, 1});
    T beta(RAD_F32, {2, 1}); beta.f({1.0f, 0.5f});
    T g(RAD_F32, {2, 1});    g.f({0, 0});
    T cu(RAD_I32, {2});      cu.n({0, 2});
    T A_(RAD_F32, {2, 1, 2});
    A a; a.t(k).t(beta).t(g).t(cu).t(A_).i("M", 2).i("head_k", 1).i("chunk", 2);
    CHECK_OK(run("gdn_kkt_solve", a));
    CHECK_NEAR(A_.at(0), 1.0f, 1e-6);
    CHECK_NEAR(A_.at(1), 0.0f, 1e-6);
    CHECK_NEAR(A_.at(2), -0.5f, 1e-6);
    CHECK_NEAR(A_.at(3), 1.0f, 1e-6);
}

TEST(op_gdn_chunk_scan_matches_recurrence) {
    /* The cross-check. The chunked scan and the plain recurrence compute the same thing by
     * different routes (a triangular solve against a sequential update), so comparing them on one
     * input is worth more than either alone.
     *
     * Two tokens, one head, head_k = head_v = 1, no gating (g = 0, beta = 1), q = k = [1].
     * By hand, recurrently:  S=0; u0 = 1*(1 - 0) = 1, S = 1, o0 = S*q0 = 1;
     *                             u1 = 1*(2 - 1) = 1, S = 2, o1 = 2.
     * The chunked form must produce the same o and the same final state. */
    T q(RAD_F32, {2, 1, 1}); q.f({1, 1});
    T k(RAD_F32, {2, 1, 1}); k.f({1, 1});
    T v(RAD_F32, {2, 1, 1}); v.f({1, 2});
    T g(RAD_F32, {2, 1});    g.f({0, 0});
    T beta(RAD_F32, {2, 1}); beta.f({1, 1});
    T cu(RAD_I32, {2});      cu.n({0, 2});

    /* A comes from the kkt solve on the same input, so the two ops are chained as a real layer
     * chains them. */
    T A_(RAD_F32, {2, 1, 2});
    A s; s.t(k).t(beta).t(g).t(cu).t(A_).i("M", 2).i("head_k", 1).i("chunk", 2);
    CHECK_OK(run("gdn_kkt_solve", s));
    CHECK_NEAR(A_.at(2), -1.0f, 1e-6);          /* beta1 * k1.k0 * e^0 = 1 */

    T h0(RAD_F32, {1, 1, 1, 1});
    T o(RAD_F32, {2, 1, 1}), ht(RAD_F32, {1, 1, 1, 1});
    A a; a.t(q).t(k).t(v).t(A_).t(g).t(beta).t(h0).t(cu).t(o).t(ht)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2);
    CHECK_OK(run("gdn_chunk_scan", a));
    CHECK_NEAR(o.at(0), 1.0f, 1e-5);
    CHECK_NEAR(o.at(1), 2.0f, 1e-5);
    CHECK_NEAR(ht.at(0), 2.0f, 1e-5);

    /* A null h0 is a zero initial state, which is what a fresh sequence has. */
    T o2(RAD_F32, {2, 1, 1}), ht2(RAD_F32, {1, 1, 1, 1});
    A b; b.t(q).t(k).t(v).t(A_).t(g).t(beta).tnull().t(cu).t(o2).t(ht2)
          .i("M", 2).i("head_k", 1).i("head_v", 1).i("chunk", 2);
    CHECK_OK(run("gdn_chunk_scan", b));
    CHECK_NEAR(o2.at(1), 2.0f, 1e-5);
}

TEST(op_gdn_chunk_scan_matches_recurrence_gated) {
    /* The same cross-check with gating on and across a chunk boundary. The identity case above
     * cannot see that `g` arrives as a per-chunk inclusive cumsum (gdn_conv_prep writes it so;
     * kkt_solve and chunk_scan read differences), or that carried state decays by the whole chunk.
     * Six tokens at chunk 4 is two chunks, one partial, as in a real prefill. The expectation is a
     * sequential recurrence written out here. */
    const int64_t Tn = 6, C = 4, K = 2, V = 2;
    const float qd[Tn][K] = {{1.0f,0.0f},{0.5f,0.5f},{0.0f,1.0f},{1.0f,-0.5f},{0.25f,0.75f},{-1.0f,0.5f}};
    const float kd[Tn][K] = {{0.5f,0.5f},{1.0f,0.0f},{0.25f,-0.5f},{0.0f,1.0f},{0.75f,0.25f},{0.5f,-1.0f}};
    const float vd[Tn][V] = {{1.0f,2.0f},{-1.0f,0.5f},{0.25f,1.0f},{2.0f,-1.0f},{0.5f,0.5f},{1.0f,0.0f}};
    const float gr[Tn]    = {-0.1f, -0.4f, -0.2f, -0.7f, -0.3f, -0.5f};   /* raw log-decay */
    const float bt[Tn]    = { 0.9f,  0.3f,  0.6f,  0.2f,  0.8f,  0.5f};

    T q(RAD_F32, {Tn, 1, K}), k(RAD_F32, {Tn, 1, K}), v(RAD_F32, {Tn, 1, V});
    T g(RAD_F32, {Tn, 1}), beta(RAD_F32, {Tn, 1});
    std::vector<float> qf, kf, vf, gf, bf;
    for (int64_t t = 0; t < Tn; ++t) {
        for (int64_t d = 0; d < K; ++d) { qf.push_back(qd[t][d]); kf.push_back(kd[t][d]); }
        for (int64_t e = 0; e < V; ++e) vf.push_back(vd[t][e]);
        bf.push_back(bt[t]);
        /* the per-chunk inclusive cumsum, restarted at every chunk boundary */
        const float prev = (t % C == 0) ? 0.0f : gf.back();
        gf.push_back(prev + gr[t]);
    }
    q.f(qf); k.f(kf); v.f(vf); g.f(gf); beta.f(bf);
    T cu(RAD_I32, {2}); cu.n({0, (int)Tn});

    T A_(RAD_F32, {Tn, 1, C});
    A s; s.t(k).t(beta).t(g).t(cu).t(A_).i("M", Tn).i("head_k", K).i("chunk", C);
    CHECK_OK(run("gdn_kkt_solve", s));

    T o(RAD_F32, {Tn, 1, V}), ht(RAD_F32, {1, 1, V, K});
    A a; a.t(q).t(k).t(v).t(A_).t(g).t(beta).tnull().t(cu).t(o).t(ht)
          .i("M", Tn).i("head_k", K).i("head_v", V).i("chunk", C);
    CHECK_OK(run("gdn_chunk_scan", a));

    /* S *= e^g ; u = beta (v - S k) ; S += u k^T ; o = S q -- the delta rule, in order. */
    double S[V][K] = {{0, 0}, {0, 0}};
    for (int64_t t = 0; t < Tn; ++t) {
        const double dec = std::exp((double)gr[t]);
        double u[V];
        for (int64_t e = 0; e < V; ++e) {
            double sk = 0;
            for (int64_t d = 0; d < K; ++d) { S[e][d] *= dec; sk += S[e][d] * kd[t][d]; }
            u[e] = bt[t] * (vd[t][e] - sk);
        }
        for (int64_t e = 0; e < V; ++e) {
            double so = 0;
            for (int64_t d = 0; d < K; ++d) { S[e][d] += u[e] * kd[t][d]; so += S[e][d] * qd[t][d]; }
            CHECK_NEAR(o.at(t * V + e), (float)so, 2e-5);
        }
    }
    for (int64_t e = 0; e < V; ++e)
        for (int64_t d = 0; d < K; ++d)
            CHECK_NEAR(ht.at(e * K + d), (float)S[e][d], 2e-5);
}

TEST(op_gdn_recurrent_update) {
    /* One sequence, one token, one head, head_k 1 and head_v 2, so the gated norm over the head
     * is meaningful. A_log = 0 and a = 0 give g = -ln 2; b = 0 gives beta = 1/2. */
    T q(RAD_F32, {1, 1, 1}); q.f({1});
    T k(RAD_F32, {1, 1, 1}); k.f({1});
    T v(RAD_F32, {1, 1, 2}); v.f({3, 4});
    T av(RAD_F32, {1, 1});   av.f({0});
    T bv(RAD_F32, {1, 1});   bv.f({0});
    T alog(RAD_F32, {1});    alog.f({0});
    T dtb(RAD_F32, {1});     dtb.f({0});
    T state(RAD_F32, {2, 1, 2, 1});                   /* two slots */
    T sidx(RAD_I32, {1, 1}); sidx.n({1});
    T nacc(RAD_I32, {1});    nacc.n({1});
    T cu(RAD_I32, {2});      cu.n({0, 1});
    T z(RAD_F32, {1, 1, 2}); z.f({0, 0});
    T nw(RAD_F32, {2});      nw.f({1, 1});
    T o(RAD_F32, {1, 1, 2});
    /* `cu` sits between num_accepted and z, the same slot the sibling gdn_conv_update uses. It is
     * required: libr4d's kernel reads cu[n] before any null test, so a schema omitting it would
     * make the launch fault, not make it optional. */
    A a; a.t(q).t(k).t(v).t(av).t(bv).t(alog).t(dtb).t(state).t(sidx).t(nacc).t(cu).t(z).t(nw)
          .t(o)
          .i("q_len", 1).i("head_k", 1).i("head_v", 2).d("eps", 0).s("act", "sigmoid");
    CHECK_OK(run("gdn_recurrent_update", a));

    /* The state is written to slot state_idx[0][0] = 1: u = beta * v = [1.5, 2], and the state
     * update is u k^T with k l2-normalised to 1. */
    CHECK_NEAR(state.at(2), 1.5f, 1e-4);
    CHECK_NEAR(state.at(3), 2.0f, 1e-4);
    /* o_raw = S q = [1.5, 2]; the gated norm divides by sqrt(mean(o^2)) = sqrt(3.125) and
     * multiplies by sigmoid(0) = 0.5. */
    const float sc = 1.0f / std::sqrt(3.125f);
    CHECK_NEAR(o.at(0), 1.5f * sc * 0.5f, 1e-4);
    CHECK_NEAR(o.at(1), 2.0f * sc * 0.5f, 1e-4);
}

/* The fused op is defined as gdn_conv_update then gdn_recurrent_update. In f32, so the q/k/v
 * staging between them holds four-byte elements. */
TEST(op_gdn_conv_recurrent_update_is_the_pair) {
    const int64_t TT = 2, Hg = 1, K = 4, Hv = 2, V = 4, W = 3, CD = 2 * Hg * K + Hv * V;
    auto fill = [](T& t, float s) {
        int64_t n = 1;
        for (uint32_t d = 0; d < t.t.rank; ++d) n *= t.t.shape[d];
        for (int64_t i = 0; i < n; ++i) st_dt(t.t.dtype, t.t.data, i, s * (float)((i * 7) % 11 - 5));
    };
    T x(RAD_F32, {TT, CD}), w(RAD_F32, {CD, W});
    T cs1(RAD_F32, {1, CD, W - 2 + TT}), cs2(RAD_F32, {1, CD, W - 2 + TT});
    T av(RAD_F32, {TT, Hv}), bv(RAD_F32, {TT, Hv}), alog(RAD_F32, {Hv}), dtb(RAD_F32, {Hv});
    T st1(RAD_F32, {TT, Hv, V, K}), st2(RAD_F32, {TT, Hv, V, K});
    T z(RAD_F32, {TT, Hv, V}), nw(RAD_F32, {V});
    fill(x, 0.1f); fill(w, 0.2f); fill(cs1, 0.05f); fill(cs2, 0.05f); fill(av, 0.3f);
    fill(bv, 0.25f); fill(alog, 0.1f); fill(dtb, 0.1f); fill(st1, 0.02f); fill(st2, 0.02f);
    fill(z, 0.3f); fill(nw, 0.1f);
    T cidx(RAD_I32, {1});     cidx.n({0});
    T sidx(RAD_I32, {1, TT}); sidx.n({0, 1});
    T nacc(RAD_I32, {1});     nacc.n({1});
    T cu(RAD_I32, {2});       cu.n({0, TT});
    T o1(RAD_F32, {TT, Hv, V}), o2(RAD_F32, {TT, Hv, V});
    T q(RAD_F32, {TT, Hg, K}), k(RAD_F32, {TT, Hg, K}), v(RAD_F32, {TT, Hv, V});

    A f; f.t(x).t(w).tnull().t(cs1).t(cidx).t(av).t(bv).t(alog).t(dtb).t(st1).t(sidx).t(nacc)
          .t(cu).t(z).t(nw).t(o1)
          .i("q_len", TT).i("head_k", K).i("head_v", V).i("conv_width", W);
    CHECK_OK(run("gdn_conv_recurrent_update", f));

    A c; c.t(x).t(w).tnull().t(cs2).t(cidx).t(nacc).t(cu).t(q).t(k).t(v)
          .i("q_len", TT).i("head_k", K).i("head_v", V).i("conv_width", W);
    CHECK_OK(run("gdn_conv_update", c));
    A r; r.t(q).t(k).t(v).t(av).t(bv).t(alog).t(dtb).t(st2).t(sidx).t(nacc).t(cu).t(z).t(nw)
          .t(o2).i("q_len", TT).i("head_k", K).i("head_v", V);
    CHECK_OK(run("gdn_recurrent_update", r));

    for (int64_t i = 0; i < TT * Hv * V; ++i) CHECK_EQ(o1.at(i), o2.at(i));
    for (int64_t i = 0; i < TT * Hv * V * K; ++i) CHECK_EQ(st1.at(i), st2.at(i));
    for (int64_t i = 0; i < CD * (W - 2 + TT); ++i) CHECK_EQ(cs1.at(i), cs2.at(i));
}

TEST(op_gdn_gated_rmsnorm) {
    T x(RAD_F32, {1, 2}); x.f({3, 4});
    T z(RAD_F32, {1, 2}); z.f({0, 0});
    T w(RAD_F32, {2});    w.f({1, 1});
    T o(RAD_F32, {1, 2});
    A a; a.t(x).t(z).t(w).t(o).i("M", 1).i("channels", 2).d("eps", 0).s("act", "sigmoid");
    CHECK_OK(run("gdn_gated_rmsnorm", a));
    const float sc = 1.0f / std::sqrt(12.5f);
    CHECK_NEAR(o.at(0), 3.0f * sc * 0.5f, 1e-6);
    CHECK_NEAR(o.at(1), 4.0f * sc * 0.5f, 1e-6);
}

/* ================================================================== collectives */
TEST(op_collectives_world1) {
    T x(RAD_F32, {2}); x.f({1, 2});
    A a; a.t(x).i("world_size", 1).i("numel", 2).s("dtype", "f32").i("exact", 1);
    CHECK_OK(run("all_reduce", a));
    CHECK_EQ(x.at(0), 1.0f);        /* inout: the answer was already there */
    CHECK_EQ(x.at(1), 2.0f);

    T y(RAD_F32, {2});
    A b; b.t(x).t(y).i("world_size", 1).i("numel", 2).s("dtype", "f32");
    CHECK_OK(run("all_gather", b));
    CHECK_EQ(y.at(0), 1.0f);
    CHECK_EQ(y.at(1), 2.0f);
}

TEST(op_collectives_world2) {
    /* The fake multi-device host backend: two rank threads in one address space, as the engine
     * runs tensor parallel (spec §1). */
    T x0(RAD_F32, {2}); x0.f({1, 2});
    T x1(RAD_F32, {2}); x1.f({10, 20});
    T g0(RAD_F32, {4}), g1(RAD_F32, {4});
    int st[2] = { 0, 0 };

    auto rank_body = [&](int r, T& x, T& gy) {
        A a; a.t(x).i("world_size", 2).i("numel", 2).s("dtype", "f32").i("exact", 1);
        int s = run("all_reduce", a, r, 2);
        A b; b.t(x).t(gy).i("world_size", 2).i("numel", 2).s("dtype", "f32");
        int s2 = run("all_gather", b, r, 2);
        st[r] = (s < 0) ? s : s2;
    };
    std::thread t0(rank_body, 0, std::ref(x0), std::ref(g0));
    std::thread t1(rank_body, 1, std::ref(x1), std::ref(g1));
    t0.join();
    t1.join();
    CHECK_OK(st[0]);
    CHECK_OK(st[1]);

    CHECK_EQ(x0.at(0), 11.0f); CHECK_EQ(x0.at(1), 22.0f);
    CHECK_EQ(x1.at(0), 11.0f); CHECK_EQ(x1.at(1), 22.0f);
    /* After the reduce both ranks hold the same values, so the gather is four copies of them. */
    CHECK_EQ(g0.at(0), 11.0f); CHECK_EQ(g0.at(2), 11.0f);
    CHECK_EQ(g1.at(1), 22.0f); CHECK_EQ(g1.at(3), 22.0f);
}

/* ================================================================== sampling */
/* Every scalar is a field of the params row, not a parameter (see SP above and
 * abi/rad_sample.h). These cases check that arithmetic against the schema core/sample/sampler.h
 * issues against. */
TEST(op_sample_penalties) {
    T lg(RAD_F32, {1, 4}); lg.f({1, -1, 2, 0});
    T hist(RAD_I32, {1, 3}); hist.n({0, 0, 2});
    SP sp(1);
    sp[0].rep_penalty = 2.0f; sp[0].freq_penalty = 0.5f; sp[0].pres_penalty = 1.0f;
    sp[0].hist_len = 3; sp[0].flags = RAD_SP_PENALTY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_penalties", a));
    /* token 0 appears twice: 1/2 - 2*0.5 - 1 = -1.5. token 2 appears once: 2/2 - 0.5 - 1 = -0.5. */
    CHECK_NEAR(lg.at(0), -1.5f, 1e-6);
    CHECK_NEAR(lg.at(1), -1.0f, 1e-6);
    CHECK_NEAR(lg.at(2), -0.5f, 1e-6);
    CHECK_NEAR(lg.at(3), 0.0f, 1e-6);

    /* A negative logit is multiplied by the repetition penalty, never divided; dividing would make
     * it larger, the opposite of a penalty. */
    T l2(RAD_F32, {1, 2}); l2.f({-2, 0});
    T h2(RAD_I32, {1, 1}); h2.n({0});
    SP sp2(1);
    sp2[0].rep_penalty = 2.0f; sp2[0].hist_len = 1; sp2[0].flags = RAD_SP_PENALTY;
    A b; b.t(l2).t(sp2.operand()).t(h2).i("M", 1).i("n_vocab", 2);
    CHECK_OK(run("sample_penalties", b));
    CHECK_NEAR(l2.at(0), -4.0f, 1e-6);

    /* Without the flag the row is untouched whatever the floats say: the flag lets a kernel skip
     * a stage for one sequence of a mixed batch. */
    T l3(RAD_F32, {1, 2}); l3.f({-2, 0});
    SP sp3(1);
    sp3[0].rep_penalty = 2.0f; sp3[0].hist_len = 1;      /* flags left at 0 */
    A e; e.t(l3).t(sp3.operand()).t(h2).i("M", 1).i("n_vocab", 2);
    CHECK_OK(run("sample_penalties", e));
    CHECK_EQ(l3.at(0), -2.0f);
}

TEST(op_sample_dry) {
    /* History "a b c a b": the last two tokens (a b) reproduce positions 0-1, so the token that
     * followed there (c) would extend the repeat, and it is the one penalised. */
    T lg(RAD_F32, {1, 4}); lg.f({0, 0, 0, 0});
    T hist(RAD_I32, {1, 5}); hist.n({0, 1, 2, 0, 1});
    SP sp(1);
    sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
    sp[0].hist_len = 5; sp[0].flags = RAD_SP_DRY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", a));
    CHECK_NEAR(lg.at(2), -1.0f, 1e-6);     /* n == allowed_length, so base^0 = 1 */
    CHECK_EQ(lg.at(3), 0.0f);              /* nothing else is touched */
}

TEST(op_sample_temp_topk_argmax) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 3, 2, 3});
    SP sp(1); sp[0].temp = 0.5f;
    A a; a.t(lg).t(sp.operand()).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_temp", a));
    CHECK_EQ(lg.at(0), 2.0f);
    CHECK_EQ(lg.at(1), 6.0f);

    T ci(RAD_I32, {1, 2}), cv(RAD_F32, {1, 2});
    SP spk(1); spk[0].top_k = 2;
    A b; b.t(lg).t(spk.operand()).t(ci).t(cv).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", b));
    CHECK_EQ(ci.iat(0), 1LL);       /* the tie between index 1 and 3 goes to the lower first */
    CHECK_EQ(ci.iat(1), 3LL);
    CHECK_EQ(cv.at(0), 6.0f);

    /* A row's top_k narrower than the buffer leaves holes, which every later stage skips. */
    T ci2(RAD_I32, {1, 4}), cv2(RAD_F32, {1, 4});
    SP sp1(1); sp1[0].top_k = 1;
    A e; e.t(lg).t(sp1.operand()).t(ci2).t(cv2).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", e));
    CHECK_EQ(ci2.iat(0), 1LL);
    CHECK_EQ(ci2.iat(1), -1LL);
    CHECK(cv2.at(1) == -INFINITY);

    T tk(RAD_I32, {1});
    SP spa(1);
    A c; c.t(lg).t(spa.operand()).t(tk).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_argmax", c));
    CHECK_EQ(tk.iat(0), 1LL);

    /* temp <= 0 leaves the row untouched: greedy is sample_argmax, not a special temperature. */
    SP spz(1); spz[0].temp = 0.0f;
    A d; d.t(lg).t(spz.operand()).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_temp", d));
    CHECK_EQ(lg.at(1), 6.0f);
}

/* The raw distribution's summary: max, sum of exp(l - max), the k largest with the lower id first
 * on a tie, slots past the vocabulary as holes, global ids under an offset, and a row without the
 * flag left as it was. */
TEST(op_sample_logprobs) {
    T lg(RAD_F32, {2, 4}); lg.f({1, 3, 2, 3,  0, 0, 0, 0});
    T st(RAD_F32, {2, 2 + 2 * 5});
    st.f(std::vector<float>(2 * 12, 9.0f));
    SP sp(2); sp[0].flags = RAD_SP_LOGPROBS;
    A a; a.t(lg).t(sp.operand()).t(st).i("M", 2).i("n_vocab", 4).i("k", 5).i("vocab_off", 100);
    CHECK_OK(run("sample_logprobs", a));
    CHECK_EQ(st.at(0), 3.0f);
    CHECK_NEAR(st.at(1), std::exp(-2.0f) + 1.0f + std::exp(-1.0f) + 1.0f, 1e-6);
    CHECK_EQ(st.at(2), 101.0f); CHECK_EQ(st.at(3), 3.0f);     /* the tie: id 1 before id 3 */
    CHECK_EQ(st.at(4), 103.0f); CHECK_EQ(st.at(5), 3.0f);
    CHECK_EQ(st.at(6), 102.0f); CHECK_EQ(st.at(7), 2.0f);
    CHECK_EQ(st.at(8), 100.0f); CHECK_EQ(st.at(9), 1.0f);
    CHECK_EQ(st.at(10), -1.0f);                                /* four tokens, five slots */
    CHECK(st.at(11) == -INFINITY);
    CHECK_EQ(st.at(12), 9.0f);                                 /* row 1 asked for nothing */
}

TEST(op_sample_narrowing) {
    /* Two candidates whose probabilities are 3/4 and 1/4. */
    const float ln3 = 1.09861229f;
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({ln3, 0});
    SP sp(1); sp[0].top_p = 0.7f;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", a));
    CHECK_EQ(ci.iat(0), 5LL);
    CHECK_EQ(ci.iat(1), -1LL);              /* dropped: index -1, value -infinity */
    CHECK(cv.at(1) == -INFINITY);

    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({ln3, 0});
    SP sp2(1); sp2[0].top_p = 0.9f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", b));
    CHECK_EQ(ci2.iat(1), 7LL);              /* 0.75 alone does not reach 0.9 */

    T ci3(RAD_I32, {1, 2}); ci3.n({5, 7});
    T cv3(RAD_F32, {1, 2}); cv3.f({ln3, 0});
    SP sp3(1); sp3[0].min_p = 0.5f;
    A c; c.t(ci3).t(cv3).t(sp3.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_minp", c));
    CHECK_EQ(ci3.iat(1), -1LL);             /* 1/3 of the peak is below 0.5 */

    T ci4(RAD_I32, {1, 2}); ci4.n({5, 7});
    T cv4(RAD_F32, {1, 2}); cv4.f({ln3, 0});
    SP sp4(1); sp4[0].min_p = 0.2f;
    A d; d.t(ci4).t(cv4).t(sp4.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_minp", d));
    CHECK_EQ(ci4.iat(1), 7LL);

    /* An identity setting is a no-op rather than a narrowing: top_p 1 keeps everything. */
    T ci5(RAD_I32, {1, 2}); ci5.n({5, 7});
    T cv5(RAD_F32, {1, 2}); cv5.f({ln3, 0});
    SP sp5(1);
    A f; f.t(ci5).t(cv5).t(sp5.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", f));
    CHECK_EQ(ci5.iat(1), 7LL);
}

TEST(op_sample_xtc) {
    /* Two candidates both above the threshold, and probability 1 so the draw always fires: the
     * more likely one is dropped and the least likely above the threshold survives, the inversion
     * the stage exists for. */
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({0.0f, 0.0f});
    SP sp(1); sp[0].xtc_probability = 1.0f; sp[0].xtc_threshold = 0.4f; sp[0].flags = RAD_SP_XTC;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_xtc", a));
    CHECK_EQ(ci.iat(0), -1LL);
    CHECK_EQ(ci.iat(1), 7LL);

    /* One candidate above the threshold is a no-op: dropping it would leave the row no choice of
     * its own. */
    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({0.0f, -80.0f});
    SP sp2(1); sp2[0].xtc_probability = 1.0f; sp2[0].xtc_threshold = 0.4f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_xtc", b));
    CHECK_EQ(ci2.iat(0), 5LL);
    CHECK_EQ(ci2.iat(1), 7LL);
}

TEST(op_sample_typical) {
    /* A uniform distribution is entirely typical: every candidate's surprisal is the entropy, so
     * nothing is dropped whatever typical_p asks. */
    T ci(RAD_I32, {1, 4}); ci.n({1, 2, 3, 4});
    T cv(RAD_F32, {1, 4}); cv.f({0, 0, 0, 0});
    SP sp(1); sp[0].typical_p = 0.5f;
    A a; a.t(ci).t(cv).t(sp.operand()).i("M", 1).i("n_cand", 4);
    CHECK_OK(run("sample_typical", a));
    for (int i = 0; i < 4; ++i) CHECK(ci.iat(i) >= 0);

    /* A peaked row: logits {4, 0, 0} are probabilities {0.964, 0.018, 0.018}, entropy 0.180. The
     * peak's surprisal is 0.037 (distance 0.143) and the tail's 4.017 (distance 3.837). So the
     * peak is the typical one and the tail is dropped, the opposite of what "exclude the most
     * likely" suggests: typical sampling is a distance from the entropy, not a rank.
     *
     * typical_p 0.1 is reached by the peak alone (0.964), so the cut lands at its distance. */
    T ci2(RAD_I32, {1, 3}); ci2.n({1, 2, 3});
    T cv2(RAD_F32, {1, 3}); cv2.f({4.0f, 0.0f, 0.0f});
    SP sp2(1); sp2[0].typical_p = 0.1f;
    A b; b.t(ci2).t(cv2).t(sp2.operand()).i("M", 1).i("n_cand", 3);
    CHECK_OK(run("sample_typical", b));
    CHECK_EQ(ci2.iat(0), 1LL);
    CHECK_EQ(ci2.iat(1), -1LL);
    CHECK_EQ(ci2.iat(2), -1LL);
}

TEST(op_sample_mask) {
    T lg(RAD_F32, {1, 4}); lg.f({1, 2, 3, 4});
    T bm(RAD_U32, {1, 1}); bm.n({0x5});     /* a set bit means allowed: tokens 0 and 2 */
    SP sp(1); sp[0].mask_row = 0; sp[0].flags = RAD_SP_MASK;
    A a; a.t(lg).t(sp.operand()).t(bm).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_mask", a));
    CHECK_EQ(lg.at(0), 1.0f);
    CHECK(lg.at(1) == -INFINITY);
    CHECK_EQ(lg.at(2), 3.0f);
    CHECK(lg.at(3) == -INFINITY);

    /* mask_row -1 is an unconstrained sequence sharing the plane with a constrained one. */
    T l2(RAD_F32, {1, 4}); l2.f({1, 2, 3, 4});
    SP sp2(1); sp2[0].mask_row = -1; sp2[0].flags = RAD_SP_MASK;
    A b; b.t(l2).t(sp2.operand()).t(bm).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_mask", b));
    CHECK_EQ(l2.at(1), 2.0f);
}

TEST(op_sample_pick) {
    /* A degenerate distribution has one answer whatever the draw is. */
    T ci(RAD_I32, {1, 2}); ci.n({5, 7});
    T cv(RAD_F32, {1, 2}); cv.f({0.0f, -80.0f});
    T tk(RAD_I32, {1});
    SP sp(1); sp[0].seed = 12345; sp[0].pos = 0;
    A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_pick", a));
    CHECK_EQ(tk.iat(0), 5LL);

    /* Determinism: the same (seed, pos) gives the same token, every time. */
    T ci2(RAD_I32, {1, 2}); ci2.n({5, 7});
    T cv2(RAD_F32, {1, 2}); cv2.f({0.0f, 0.0f});
    int64_t first = -1;
    for (int rep = 0; rep < 4; ++rep) {
        T tk2(RAD_I32, {1});
        A b; b.t(ci2).t(cv2).t(sp.operand()).t(tk2).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", b));
        if (rep == 0) first = tk2.iat(0);
        CHECK_EQ(tk2.iat(0), first);
    }
    CHECK(first == 5 || first == 7);

    /* The stream is not constant: over a run of positions an even split produces both. */
    int saw5 = 0, saw7 = 0;
    for (int off = 0; off < 64; ++off) {
        SP spo(1); spo[0].seed = 12345; spo[0].pos = (uint64_t)off;
        T tk3(RAD_I32, {1});
        A c; c.t(ci2).t(cv2).t(spo.operand()).t(tk3).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", c));
        if (tk3.iat(0) == 5) ++saw5; else ++saw7;
    }
    CHECK(saw5 > 8);
    CHECK(saw7 > 8);

    /* A dropped candidate is skipped, never picked. */
    T ci3(RAD_I32, {1, 2}); ci3.n({-1, 7});
    T cv3(RAD_F32, {1, 2}); cv3.f({-INFINITY, 0.0f});
    T tk4(RAD_I32, {1});
    A d; d.t(ci3).t(cv3).t(sp.operand()).t(tk4).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_pick", d));
    CHECK_EQ(tk4.iat(0), 7LL);
}

TEST(op_sample_merge_topk) {
    /* Two ranks' candidate sets, already carrying global token ids, as (u32 id, f32 logit) pairs.
     * The merge is the k largest across both, descending. */
    T gt(RAD_U32, {1, 8});
    float* gf = (float*)gt.t.data;
    uint32_t* gi = (uint32_t*)gt.t.data;
    gi[0] = 11; gf[1] = 3.0f;      /* rank 0 */
    gi[2] = 12; gf[3] = 1.0f;
    gi[4] = 20; gf[5] = 4.0f;      /* rank 1 */
    gi[6] = 21; gf[7] = 2.0f;
    T ci(RAD_I32, {1, 3}), cv(RAD_F32, {1, 3});
    SP sp(1);
    A a; a.t(gt).t(sp.operand()).t(ci).t(cv).i("M", 1).i("n_cand", 3);
    CHECK_OK(run("sample_merge_topk", a));
    CHECK_EQ(ci.iat(0), 20LL); CHECK_EQ(cv.at(0), 4.0f);
    CHECK_EQ(ci.iat(1), 11LL); CHECK_EQ(cv.at(1), 3.0f);
    CHECK_EQ(ci.iat(2), 21LL); CHECK_EQ(cv.at(2), 2.0f);

    /* Holes (id 0xffffffff, logit -inf) are skipped: with fewer live candidates than n_cand the
     * tail is -1, also through an i64 cand_idx that would show a widened hole id. */
    T gh(RAD_U32, {1, 4});
    uint32_t* hi = (uint32_t*)gh.t.data;
    float* hf = (float*)gh.t.data;
    hi[0] = 11; hf[1] = 3.0f;
    hi[2] = 0xffffffffu; hf[3] = -INFINITY;
    T hci(RAD_I64, {1, 2}), hcv(RAD_F32, {1, 2});
    A h; h.t(gh).t(sp.operand()).t(hci).t(hcv).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_merge_topk", h));
    CHECK_EQ(hci.iat(0), 11LL);
    CHECK_EQ(hci.iat(1), -1LL);
}

/* ------------------------------------------------------------------ the history stages, per row */

/* Every row reads its own row of the history plane. hist_off is an offset within the row, so the
 * engine stages 0 for every row, and a second sampled row (another sequence, or a speculative
 * verify's next position) is penalised against its own tokens, not left alone. */
TEST(op_sample_history_stages_read_each_rows_own_row) {
    T lg(RAD_F32, {2, 4}); lg.f({1, 1, 1, 1,  1, 1, 1, 1});
    T hist(RAD_I32, {2, 3}); hist.n({0, 0, 0,  3, 3, 3});
    SP sp(2);
    for (size_t m = 0; m < 2; ++m) {
        sp[m].rep_penalty = 2.0f; sp[m].hist_len = 3; sp[m].flags = RAD_SP_PENALTY;
    }
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_penalties", a));
    CHECK_NEAR(lg.at(0), 0.5f, 1e-6);          /* row 0 penalises token 0 ... */
    CHECK_NEAR(lg.at(3), 1.0f, 1e-6);
    CHECK_NEAR(lg.at(4 + 3), 0.5f, 1e-6);      /* ... and row 1 token 3, from its own row */
    CHECK_NEAR(lg.at(4 + 0), 1.0f, 1e-6);

    /* The same for DRY: row 1's history "a b c a b" repeats, row 0's does not. */
    T l2(RAD_F32, {2, 4}); l2.f({0, 0, 0, 0,  0, 0, 0, 0});
    T h2(RAD_I32, {2, 5}); h2.n({0, 1, 2, 3, 0,  0, 1, 2, 0, 1});
    SP sd(2);
    for (size_t m = 0; m < 2; ++m) {
        sd[m].dry_multiplier = 1.0f; sd[m].dry_base = 2.0f; sd[m].dry_allowed_length = 2;
        sd[m].hist_len = 5; sd[m].flags = RAD_SP_DRY;
    }
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", b));
    for (int i = 0; i < 4; ++i) CHECK_EQ(l2.at(i), 0.0f);
    CHECK_NEAR(l2.at(4 + 2), -1.0f, 1e-6);
}

/* A split vocabulary: the history holds global ids and the row is this rank's shard, so a token's
 * column is its id less vocab_off, and an id another rank owns is not a column here at all. */
TEST(op_sample_history_stages_subtract_the_vocab_offset) {
    /* This rank holds ids 4..7. */
    T lg(RAD_F32, {1, 4}); lg.f({1, 1, 1, 1});
    T hist(RAD_I32, {1, 3}); hist.n({5, 1, 6});
    SP sp(1);
    sp[0].rep_penalty = 2.0f; sp[0].hist_len = 3; sp[0].flags = RAD_SP_PENALTY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4).i("vocab_off", 4);
    CHECK_OK(run("sample_penalties", a));
    CHECK_NEAR(lg.at(0), 1.0f, 1e-6);          /* id 4: not in the history */
    CHECK_NEAR(lg.at(1), 0.5f, 1e-6);          /* id 5 */
    CHECK_NEAR(lg.at(2), 0.5f, 1e-6);          /* id 6 */
    CHECK_NEAR(lg.at(3), 1.0f, 1e-6);          /* id 7; and id 1 is rank 0's, not column 1 */

    /* DRY: "4 5 7 4 5" repeats "4 5", so id 7 (column 3) is the continuation. */
    T l2(RAD_F32, {1, 4}); l2.f({0, 0, 0, 0});
    T h2(RAD_I32, {1, 5}); h2.n({4, 5, 7, 4, 5});
    SP sd(1);
    sd[0].dry_multiplier = 1.0f; sd[0].dry_base = 2.0f; sd[0].dry_allowed_length = 2;
    sd[0].hist_len = 5; sd[0].flags = RAD_SP_DRY;
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 1).i("n_vocab", 4).i("vocab_off", 4);
    CHECK_OK(run("sample_dry", b));
    CHECK_NEAR(l2.at(3), -1.0f, 1e-6);
    CHECK_EQ(l2.at(0), 0.0f);
    CHECK_EQ(l2.at(1), 0.0f);
    CHECK_EQ(l2.at(2), 0.0f);
}

/* repeat_last_n and dry_penalty_last_n are llama.cpp's: 0 is OFF, negative is the whole history,
 * n > 0 is the last n tokens. */
TEST(op_sample_history_window_of_zero_turns_the_stage_off) {
    auto penal = [](int32_t last_n) {
        T lg(RAD_F32, {1, 4}); lg.f({1, 1, 1, 1});
        T hist(RAD_I32, {1, 3}); hist.n({0, 1, 2});
        SP sp(1);
        sp[0].rep_penalty = 2.0f; sp[0].hist_len = 3; sp[0].penalty_last_n = last_n;
        sp[0].flags = RAD_SP_PENALTY;
        A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 4);
        CHECK_OK(run("sample_penalties", a));
        return std::vector<float>{ lg.at(0), lg.at(1), lg.at(2) };
    };
    const auto off = penal(0), all = penal(-1), tail = penal(2);
    CHECK_EQ(off[0], 1.0f); CHECK_EQ(off[1], 1.0f); CHECK_EQ(off[2], 1.0f);
    CHECK_NEAR(all[0], 0.5f, 1e-6); CHECK_NEAR(all[2], 0.5f, 1e-6);
    CHECK_EQ(tail[0], 1.0f);                   /* the window is the tail */
    CHECK_NEAR(tail[1], 0.5f, 1e-6); CHECK_NEAR(tail[2], 0.5f, 1e-6);

    T l2(RAD_F32, {1, 4}); l2.f({0, 0, 0, 0});
    T h2(RAD_I32, {1, 5}); h2.n({0, 1, 2, 0, 1});
    SP sd(1);
    sd[0].dry_multiplier = 1.0f; sd[0].dry_base = 2.0f; sd[0].dry_allowed_length = 2;
    sd[0].hist_len = 5; sd[0].dry_penalty_last_n = 0; sd[0].flags = RAD_SP_DRY;
    A b; b.t(l2).t(sd.operand()).t(h2).i("M", 1).i("n_vocab", 4);
    CHECK_OK(run("sample_dry", b));
    CHECK_EQ(l2.at(2), 0.0f);
}

/* DRY penalises a token once, by its longest repeat. History "a b x a b x a b": the suffix "a b"
 * also ends at position 1 (a repeat of 2) and at position 4 (a repeat of 5), and x follows both.
 * llama.cpp takes the longer: 1 * 2^(5-2) = 8; summing would give 9. */
TEST(op_sample_dry_penalises_each_token_once_by_its_longest_repeat) {
    T lg(RAD_F32, {1, 3}); lg.f({0, 0, 0});
    T hist(RAD_I32, {1, 8}); hist.n({0, 1, 2, 0, 1, 2, 0, 1});
    SP sp(1);
    sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
    sp[0].hist_len = 8; sp[0].flags = RAD_SP_DRY;
    A a; a.t(lg).t(sp.operand()).t(hist).i("M", 1).i("n_vocab", 3);
    CHECK_OK(run("sample_dry", a));
    CHECK_EQ(lg.at(2), -8.0f);
    CHECK_EQ(lg.at(0), 0.0f);
    CHECK_EQ(lg.at(1), 0.0f);
}

/* The breakers reach the kernel. A position the breaker plane marks holds a whole sequence
 * breaker, which DRY never penalises, and the row's dry_rep_limit caps how long a repeat may
 * count, down to "not at all" below allowed_length. */
TEST(op_sample_dry_honours_the_breaker_plane_and_the_repeat_limit) {
    auto dry = [](int32_t rep_limit, bool mark_x) {
        T lg(RAD_F32, {1, 3}); lg.f({0, 0, 0});
        T hist(RAD_I32, {1, 8}); hist.n({0, 1, 2, 0, 1, 2, 0, 1});
        T brk(RAD_U8, {1, 8});
        for (int i = 0; i < 8; ++i) st_int(RAD_U8, brk.t.data, i, 0);
        if (mark_x) { st_int(RAD_U8, brk.t.data, 2, 1); st_int(RAD_U8, brk.t.data, 5, 1); }
        SP sp(1);
        sp[0].dry_multiplier = 1.0f; sp[0].dry_base = 2.0f; sp[0].dry_allowed_length = 2;
        sp[0].hist_len = 8; sp[0].dry_rep_limit = rep_limit; sp[0].flags = RAD_SP_DRY;
        A a; a.t(lg).t(sp.operand()).t(hist).t(brk).i("M", 1).i("n_vocab", 3);
        CHECK_OK(run("sample_dry", a));
        return lg.at(2);
    };
    CHECK_EQ(dry(0, false), -8.0f);            /* the plane present and empty changes nothing */
    CHECK_EQ(dry(0, true), 0.0f);              /* x is a breaker */
    CHECK_EQ(dry(3, false), -2.0f);            /* the repeat of 5 counts as 3: 2^(3-2) */
    CHECK_EQ(dry(1, false), 0.0f);             /* below allowed_length nothing counts */
}

/* ------------------------------------------------------------------ greedy rows in a mixed batch */

/* A greedy row's token is the argmax's. In a batch mixing greedy and sampled rows the argmax and
 * the candidate chain both run over every row, so every candidate stage must treat a greedy row as
 * having no candidates, or the pick replaces its argmax with a draw. */
TEST(op_sample_candidate_stages_leave_a_greedy_row_alone) {
    T lg(RAD_F32, {2, 4}); lg.f({0, 5, 0, 0,  -80, -80, 0, -80});
    SP sp(2);
    sp[0].flags = RAD_SP_GREEDY;
    sp[0].seed = 1; sp[1].seed = 1;

    T ci(RAD_I32, {2, 4}), cv(RAD_F32, {2, 4});
    A k; k.t(lg).t(sp.operand()).t(ci).t(cv).i("M", 2).i("n_vocab", 4);
    CHECK_OK(run("sample_topk", k));
    for (int i = 0; i < 4; ++i) CHECK_EQ(ci.iat(i), -1LL);   /* row 0: no candidates */
    CHECK_EQ(ci.iat(4), 2LL);                                /* row 1: its own set */

    T tk(RAD_I32, {2}); tk.n({1, -7});                      /* the argmax already wrote row 0 */
    A p; p.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 2).i("n_cand", 4);
    CHECK_OK(run("sample_pick", p));
    CHECK_EQ(tk.iat(0), 1LL);
    CHECK_EQ(tk.iat(1), 2LL);

    /* The narrowing stages leave a greedy row's set as it is, holes or not. */
    T c2(RAD_I32, {1, 2}); c2.n({5, 7});
    T v2(RAD_F32, {1, 2}); v2.f({0.0f, 0.0f});
    SP sg(1); sg[0].flags = RAD_SP_GREEDY; sg[0].top_p = 0.3f;
    A n; n.t(c2).t(v2).t(sg.operand()).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_topp", n));
    CHECK_EQ(c2.iat(1), 7LL);

    /* The merge writes a greedy row an empty set. */
    T gt(RAD_U32, {1, 4});
    uint32_t* gi = (uint32_t*)gt.t.data;
    float* gf = (float*)gt.t.data;
    gi[0] = 11; gf[1] = 3.0f; gi[2] = 20; gf[3] = 4.0f;
    T mi(RAD_I32, {1, 2}), mv(RAD_F32, {1, 2});
    A g; g.t(gt).t(sg.operand()).t(mi).t(mv).i("M", 1).i("n_cand", 2);
    CHECK_OK(run("sample_merge_topk", g));
    CHECK_EQ(mi.iat(0), -1LL);
    CHECK_EQ(mi.iat(1), -1LL);
}

/* The pick's fall-through never lands on a masked candidate. top-k fills its width from a masked
 * row, so the grammar's forbidden tokens sit at -infinity at the tail of the set, and a draw that
 * walks past every comparison (float drift at the top of the range) takes the last candidate,
 * which must be the last one the distribution allows. A NaN logit makes the sum NaN and forces
 * the fall-through on every draw, which makes the case checkable. */
TEST(op_sample_pick_falls_through_to_an_allowed_candidate) {
    T ci(RAD_I32, {1, 3}); ci.n({5, 6, 7});
    T cv(RAD_F32, {1, 3}); cv.f({0.0f, NAN, -INFINITY});
    SP sp(1); sp[0].seed = 3;
    for (uint64_t pos = 0; pos < 8; ++pos) {
        sp[0].pos = pos;
        T tk(RAD_I32, {1});
        A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 3);
        CHECK_OK(run("sample_pick", a));
        CHECK_EQ(tk.iat(0), 5LL);
    }
}

/* The pick's draw is rad_sample.h's, on the pick's key: the same uniform the host reference and
 * every other implementation take for (seed, pos). Two candidates at 1/4 and 3/4 split exactly at
 * u = 0.25, so the token says which side the draw fell. */
TEST(op_sample_pick_draws_the_specified_uniform) {
    const float ln3 = 1.09861229f;
    int agree = 0;
    for (uint64_t pos = 0; pos < 64; ++pos) {
        T ci(RAD_I32, {1, 2}); ci.n({5, 7});
        T cv(RAD_F32, {1, 2}); cv.f({0.0f, ln3});
        SP sp(1); sp[0].seed = 99; sp[0].pos = pos;
        T tk(RAD_I32, {1});
        A a; a.t(ci).t(cv).t(sp.operand()).t(tk).i("M", 1).i("n_cand", 2);
        CHECK_OK(run("sample_pick", a));
        const float u = rad_sample_uniform01(99 ^ RAD_SAMPLE_KEY_PICK, pos);
        const int64_t want = u < 0.25f ? 5 : 7;
        if (std::fabs(u - 0.25f) > 1e-5f) { CHECK_EQ(tk.iat(0), want); ++agree; }
    }
    CHECK(agree > 60);
}

/* ================================================================== the n-gram ban */
/* Unlimited-OCR's SlidingWindowNoRepeatNgramProcessor, ported straight from
 * modeling_unlimitedocr.py:353-383 as the oracle's second formulation. `s` is the row's history
 * oldest first; `whitelist` is its `whitelist_token_ids`, removed from the banned set after it is
 * built (line 379). The result marks every global token id it bans. */
static std::vector<char> ref_ngram_banned(const std::vector<int32_t>& s, int n, int window,
                                          const std::vector<int32_t>& whitelist = {}) {
    std::vector<char> banned(256, 0);
    if (n <= 0 || (int64_t)s.size() < (int64_t)n) return banned;
    const int64_t len = (int64_t)s.size();
    const int64_t search_start = window > 0 ? std::max<int64_t>(0, len - window) : 0;
    const int64_t search_end   = len - n + 1;
    if (search_end <= search_start) return banned;
    std::vector<int32_t> prefix;
    if (n > 1) prefix.assign(s.begin() + (size_t)(len - n + 1), s.end());
    for (int64_t idx = search_start; idx < search_end; ++idx) {
        bool match = true;
        for (size_t j = 0; j < prefix.size(); ++j)
            if (s[(size_t)(idx + (int64_t)j)] != prefix[j]) { match = false; break; }
        if (!match) continue;
        const int32_t tok = s[(size_t)(idx + n - 1)];
        if (tok >= 0 && tok < (int32_t)banned.size()) banned[(size_t)tok] = 1;
    }
    for (int32_t id : whitelist)
        if (id >= 0 && id < (int32_t)banned.size()) banned[(size_t)id] = 0;
    return banned;
}

/* The op over one row: returns the banned global ids it wrote as -infinity, and whether every other
 * column is bit-for-bit its input. `whitelist` ids arrive as the bitmask operand (a set bit marks
 * an exempt column); an empty list passes an all-zero plane, and `no_operand` passes no 5th
 * operand at all -- the default serving form. */
static std::vector<int32_t> ref_ngram_op(const std::vector<int32_t>& s, int n, int window,
                                         int n_vocab, bool* untouched_out,
                                         const std::vector<int32_t>& whitelist = {},
                                         bool no_operand = false) {
    const int64_t H = 8;   /* the history plane's extent in these cases */
    T lg(RAD_F32, {1, n_vocab});
    std::vector<float> in((size_t)n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) in[(size_t)i] = 0.5f * (float)i - 1.0f;
    lg.f(in);
    T hist(RAD_I32, {1, H});
    std::vector<int64_t> hv((size_t)H, 0);
    for (size_t i = 0; i < s.size() && i < (size_t)H; ++i) hv[i] = s[i];
    hist.n(hv);

    SP sp(1);
    sp[0].hist_off = 0;
    sp[0].hist_len = (int32_t)std::min<size_t>(s.size(), (size_t)H);
    T ng(RAD_I32, {1, 2}); ng.n({ n, window });
    const int64_t words = (n_vocab + 31) / 32;
    T wl(RAD_I32, {1, words});
    std::vector<int64_t> wv((size_t)words, 0);
    for (int32_t id : whitelist)
        if (id >= 0 && id < n_vocab) wv[(size_t)(id >> 5)] |= 1LL << (id & 31);
    wl.n(wv);

    A a; a.t(lg).t(sp.operand()).t(hist).t(ng);
    if (!no_operand) a.t(wl);
    a.i("M", 1).i("n_vocab", n_vocab);
    if (run("sample_ngram_ban", a) != RAD_OK) return { -99 };

    std::vector<int32_t> banned;
    bool untouched = true;
    for (int64_t i = 0; i < n_vocab; ++i) {
        const float v = lg.at(i);
        if (std::isinf(v) && v < 0.0f) banned.push_back((int32_t)i);
        else if (std::memcmp(&v, &in[(size_t)i], 4) != 0) untouched = false;
    }
    if (untouched_out) *untouched_out = untouched;
    return banned;
}

/* The reference's banned set and the op's, over:
 *   - every binary history of length 0..6, n 1..4 and windows 0, 2, 3, 4 and 64. A binary
 *     alphabet makes repeats and the "abcabc" shapes dense, and lengths short of, equal to and
 *     longer than n cover the early exits;
 *   - 200 random histories over five ids, lengths 1..24, n 1..8 and windows 0..16, where the
 *     window boundary falls inside repeated runs;
 *   - 300 of those shapes repeated with a random whitelist of 0..32 ids (some outside the
 *     vocabulary), asserting `banned.difference_update(self.whitelist)`: the whitelist removes
 *     bans and bans nothing, and every column it spares keeps its input bits;
 *   - the named shapes: an n-gram whose last token is the window's last (idx == search_end - 1),
 *     an n-gram that starts at search_start while an earlier copy starts before it, n == 1 (the
 *     empty prefix, every window token banned) and n == 0 (off), each once with a whitelist. */
TEST(op_sample_ngram_ban_matches_the_reference_processor) {
    int64_t trials = 0, mismatches = 0, bans = 0, wl_trials = 0, wl_bans = 0;
    bool all_untouched = true;

    auto trial = [&](const std::vector<int32_t>& s_in, int n, int window,
                     const std::vector<int32_t>& whitelist = {}, bool wl_case = false) {
        /* The engine's plane holds the LAST H tokens of the sequence (build_step truncates the
         * oldest), so the oracle and the op must both see that span. */
        const std::vector<int32_t> s = s_in.size() > 8
                                           ? std::vector<int32_t>(s_in.end() - 8, s_in.end())
                                           : s_in;
        const int n_vocab = 8;
        bool untouched = true;
        const std::vector<int32_t> got = ref_ngram_op(s, n, window, n_vocab, &untouched, whitelist);
        if (got.size() == 1 && got[0] == -99) { ++mismatches; return; }
        const std::vector<char> want = ref_ngram_banned(s, n, window, whitelist);
        ++trials;
        bans += (int64_t)got.size();
        if (wl_case) { ++wl_trials; wl_bans += (int64_t)got.size(); }
        all_untouched = all_untouched && untouched;
        int64_t live = 0;
        for (int id = 0; id < n_vocab; ++id) live += want[(size_t)id] ? 1 : 0;
        if (live != (int64_t)got.size()) { ++mismatches; return; }
        for (int32_t id : got)
            if (id < 0 || id >= n_vocab || !want[(size_t)id]) { ++mismatches; return; }
    };

    /* Every binary history of length 0..6. */
    for (int64_t len = 0; len <= 6; ++len) {
        for (int64_t bits = 0; bits < (int64_t)1 << len; ++bits) {
            std::vector<int32_t> s;
            for (int64_t i = 0; i < len; ++i) s.push_back((int32_t)((bits >> i) & 1));
            for (int n = 1; n <= 4; ++n)
                for (int w : { 0, 2, 3, 4, 64 }) trial(s, n, w);
        }
    }

    /* Random histories over five ids. */
    uint32_t rs = 0x9e3779b9u;
    for (int it = 0; it < 200; ++it) {
        rs = rs * 1664525u + 1013904223u;
        const int64_t len = 1 + (int64_t)((rs >> 8) % 24u);
        std::vector<int32_t> s((size_t)len);
        for (int64_t i = 0; i < len; ++i) {
            rs = rs * 1664525u + 1013904223u;
            s[(size_t)i] = (int32_t)((rs >> 12) % 5u);
        }
        rs = rs * 1664525u + 1013904223u;
        const int n = 1 + (int)((rs >> 8) % 8u);
        rs = rs * 1664525u + 1013904223u;
        const int w = (int)((rs >> 8) % 17u);
        trial(s, n, w);
    }

    /* The same random shapes with random whitelists: 0..32 ids over a range wider than the
     * vocabulary, so exempt tokens, non-banned tokens and out-of-vocabulary ids all occur. */
    for (int it = 0; it < 300; ++it) {
        rs = rs * 1664525u + 1013904223u;
        const int64_t len = 1 + (int64_t)((rs >> 8) % 24u);
        std::vector<int32_t> s((size_t)len);
        for (int64_t i = 0; i < len; ++i) {
            rs = rs * 1664525u + 1013904223u;
            s[(size_t)i] = (int32_t)((rs >> 12) % 5u);
        }
        rs = rs * 1664525u + 1013904223u;
        const int n = 1 + (int)((rs >> 8) % 8u);
        rs = rs * 1664525u + 1013904223u;
        const int w = (int)((rs >> 8) % 17u);
        rs = rs * 1664525u + 1013904223u;
        const int k = (int)((rs >> 8) % 33u);           /* 0..32 ids, empty included */
        std::vector<int32_t> wl((size_t)k);
        for (int j = 0; j < k; ++j) {
            rs = rs * 1664525u + 1013904223u;
            wl[(size_t)j] = (int32_t)((rs >> 12) % 12u);  /* ids 8..11 are out of vocabulary */
        }
        trial(s, n, w, wl, true);
    }

    /* The named shapes. [4,1,2, 4,1,2, 4,1,2] with n 3 and window 6: the newest complete n-gram
     * starts exactly at search_end - 1, and the earlier copy at search_start bans the same token;
     * window 3 moves search_start so that only the spanning copy is searched. */
    trial({ 4, 1, 2, 4, 1, 2, 4, 1, 2 }, 3, 6);
    trial({ 4, 1, 2, 4, 1, 2, 4, 1, 2 }, 3, 3);
    trial({ 1, 1, 1, 1, 1, 1 }, 2, 4);        /* repeated tokens, every position matches */
    trial({ 1, 1, 1, 1, 1, 1 }, 1, 3);        /* n 1: the entire window is banned */
    trial({ 1, 1, 1, 1, 1, 1 }, 0, 3);        /* off */
    trial({ 7, 2, 7, 2, 7 }, 2, 0);           /* window 0: the whole span */
    /* The same shapes with whitelists: the banned token exempted (ban vanishes), a token that
     * would not be banned exempted (inert), an out-of-vocabulary id (inert) and n 0 with a
     * whitelist (still off). */
    trial({ 4, 1, 2, 4, 1, 2, 4, 1, 2 }, 3, 6, { 2 }, true);   /* a non-banned id: inert */
    trial({ 4, 1, 2, 4, 1, 2, 4, 1, 2 }, 3, 6, { 4 }, true);   /* the banned id: ban vanishes */
    trial({ 4, 1, 2, 4, 1, 2, 4, 1, 2 }, 3, 6, { 4, 2 }, true);
    trial({ 1, 1, 1, 1, 1, 1 }, 1, 3, { 1 }, true);        /* n 1, the whole window exempted */
    trial({ 1, 1, 1, 1, 1, 1 }, 1, 3, { 7 }, true);        /* out of vocabulary: inert */
    trial({ 1, 1, 1, 1, 1, 1 }, 0, 3, { 1 }, true);        /* off stays off with a whitelist */
    trial({ 7, 2, 7, 2, 7 }, 2, 0, { 7, 2 }, true);        /* every banned token exempted */

    /* The whitelist as a plain bitmask must agree with the same case with the operand absent
     * (the reference's empty whitelist): an all-zero plane bans exactly what no plane bans. */
    {
        bool u1 = true, u2 = true;
        const std::vector<int32_t> s = { 4, 1, 2, 4, 1, 2, 4, 1 };
        const std::vector<int32_t> a = ref_ngram_op(s, 3, 0, 8, &u1);
        const std::vector<int32_t> b = ref_ngram_op(s, 3, 0, 8, &u2, {}, true);
        CHECK(a == b);
        CHECK(!a.empty());
    }

    CHECK(trials > 2500);
    CHECK(mismatches == 0);
    CHECK(bans > 0);
    CHECK(wl_trials >= 300);
    /* The whitelist bites: random whitelists leave bans on the table AND remove some, so a
     * kernel that ignored the plane would fail the sweeps above, not merely agree with them. */
    CHECK(wl_bans > 0);
    /* Every column the reference does not ban is bit-for-bit its input: the op writes -infinity
     * and nothing else, so penalties and temperature see the same numbers. */
    CHECK(all_untouched);
}

/* Off is the default, and off must be invisible to greedy decoding: sample_ngram_ban writes
 * nothing for a row with ngram_ban_n 0 -- with a whitelist plane present as well, the reference's
 * empty-whitelist default -- so the argmax over the same row is the same token and the logits are
 * bit-identical. With the ban on, the outcome changes exactly when the ban covers the argmax's
 * token, and the whitelist moves it back. */
TEST(op_sample_ngram_ban_off_leaves_greedy_decoding_unchanged) {
    /* Token 5 is the clear maximum. Its id appears 4 times in the history, so both the ban and the
     * penalties have something to do; the ban is off here. */
    T hist(RAD_I32, {1, 8}); hist.n({ 5, 1, 5, 2, 5, 3, 5, 1 });

    auto greedy = [&](int n, int window, std::vector<float>* after, int wl_id = -1) {
        T lg(RAD_F32, {1, 8});
        lg.f({ 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 4.0f, 0.5f, 0.6f });
        SP sp(1);
        sp[0].hist_off = 0;
        sp[0].hist_len = 8;
        T ng(RAD_I32, {1, 2}); ng.n({ n, window });
        T wl(RAD_I32, {1, 1});
        wl.n({ wl_id >= 0 ? (int64_t)1 << wl_id : 0 });
        A a; a.t(lg).t(sp.operand()).t(hist).t(ng).t(wl).i("M", 1).i("n_vocab", 8);
        CHECK_OK(run("sample_ngram_ban", a));
        T tk(RAD_I32, {1});
        A b; b.t(lg).t(sp.operand()).t(tk).i("M", 1).i("n_vocab", 8);
        CHECK_OK(run("sample_argmax", b));
        if (after)
            for (int64_t i = 0; i < 8; ++i) after->push_back(lg.at(i));
        return tk.iat(0);
    };

    std::vector<float> off_logits;
    CHECK_EQ(greedy(0, 0, &off_logits, 5), 5);
    /* Bit-identical to the input: the stage did not run, whitelist plane present or not. */
    static const float kIn[8] = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 4.0f, 0.5f, 0.6f };
    for (int64_t i = 0; i < 8; ++i) CHECK(std::memcmp(&off_logits[(size_t)i], &kIn[i], 4) == 0);
    std::vector<float> off_plain;
    CHECK_EQ(greedy(0, 0, &off_plain), 5);
    CHECK(off_plain == off_logits);

    /* The history's prefix (5,1) is followed by token 5 at positions 0 and 2 in [5,1,5,2,5,3,5,1]:
     * with n 2 the ban covers 5, so greedy moves to the next best (token 7). */
    CHECK_EQ(greedy(2, 8, nullptr), 7);
    /* The whitelist removes 5 from the banned set (the reference's difference_update), and greedy
     * is back on the maximum. */
    CHECK_EQ(greedy(2, 8, nullptr, 5), 5);
    /* A ban that only names other tokens leaves the argmax alone: with n 2 and a one-token window,
     * search_start == search_end - 1 is impossible (window 1 < n), so nothing is banned. */
    CHECK_EQ(greedy(2, 1, nullptr), 5);
}

/* The host chain (core/sample/host_ref.cpp) reproduces the reference op for op: the banned set,
 * then the whitelist removed from it, then -inf -- and off (ngram_ban_n 0) leaves the logits
 * bit-identical through the greedy chain, whitelist plane present or not. */
TEST(host_ngram_ban_whitelist_matches_the_reference) {
    using namespace rad;
    const std::vector<float> in = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 4.0f, 0.5f, 0.6f };
    const std::vector<int32_t> hist = { 5, 1, 5, 2, 5, 3, 5, 1 };
    auto bit = [](std::vector<uint32_t>& v, int id) { v[(size_t)(id >> 5)] |= 1u << (uint32_t)(id & 31); };

    /* The bare ban: token 5 (the prefix (1) is followed by 5 in the window) gets -infinity, every
     * other candidate keeps its input bits. */
    {
        CandidateSet s;
        s.from_logits(in.data(), 8);
        host_ngram_ban(s, hist.data(), 8, 2, 8);
        int64_t banned = 0;
        for (const Candidate& c : s.c) {
            if (std::isinf(c.logit) && c.logit < 0.0f) { ++banned; CHECK_EQ(c.id, 5); }
            else CHECK(std::memcmp(&c.logit, &in[(size_t)c.id], 4) == 0);
        }
        CHECK_EQ(banned, 1LL);
    }
    /* The whitelist bit for the banned token: banned.difference_update, nothing written. */
    {
        std::vector<uint32_t> wl(1, 0);
        bit(wl, 5);
        CandidateSet s;
        s.from_logits(in.data(), 8);
        host_ngram_ban(s, hist.data(), 8, 2, 8, wl.data(), 1);
        for (const Candidate& c : s.c) CHECK(std::memcmp(&c.logit, &in[(size_t)c.id], 4) == 0);
    }
    /* A bit for a token that is not banned is inert; a bit past the plane marks nothing, so the
     * ban stands. */
    {
        std::vector<uint32_t> wl(1, 0);
        bit(wl, 3);
        CandidateSet s;
        s.from_logits(in.data(), 8);
        host_ngram_ban(s, hist.data(), 8, 2, 8, wl.data(), 1);
        bool five = false;
        for (const Candidate& c : s.c) {
            if (c.id == 5) { five = std::isinf(c.logit) && c.logit < 0.0f; continue; }
            CHECK(std::memcmp(&c.logit, &in[(size_t)c.id], 4) == 0);
        }
        CHECK(five);
    }
    /* Off is invisible to the whole greedy chain: ngram_ban_n 0 with the whitelist input set and
     * with it clear take the same token over bit-identical logits. */
    {
        std::vector<uint32_t> wl(1, 0);
        bit(wl, 5);
        SamplingParams sp;
        sp.temp = 0.0f;                       /* greedy */
        HostRefInput plain;
        plain.history = hist.data();
        plain.n_history = 8;
        HostRefInput with = plain;
        with.ngram_whitelist = wl.data();
        with.ngram_whitelist_words = 1;
        CandidateSet a, b;
        a.from_logits(in.data(), 8);
        b.from_logits(in.data(), 8);
        CHECK_EQ(host_sample_into(a, sp, plain), 5);
        CHECK_EQ(host_sample_into(b, sp, with), 5);
        for (size_t i = 0; i < a.c.size(); ++i) {
            CHECK(std::memcmp(&a.c[i].logit, &in[(size_t)a.c[i].id], 4) == 0);
            CHECK(std::memcmp(&b.c[i].logit, &in[(size_t)b.c[i].id], 4) == 0);
        }
    }
}

/* ================================================================== failure modes */
/* Nothing that did not do the work returns RAD_OK. The engine turns these into a named abort
 * rather than a silent fallback (spec §17). */
TEST(errors_are_values) {
    T x(RAD_F32, {4});
    A a; a.t(x).i("M", 1).i("n", 0).d("eps", 0).s("dtype", "f32");
    CHECK(run("rmsnorm", a) < 0);              /* n = 0 is a shape, not a no-op */

    A b; b.i("M", 1).i("n", 4).d("eps", 0).s("dtype", "f32");
    CHECK_EQ(run("rmsnorm", b), RAD_E_INVAL);  /* no operands at all */

    /* A rotation wider than the reference does in one frame says so by name instead of rotating a
     * narrower block. */
    T big(RAD_F32, {1, 8192});
    T q(RAD_I8, {1, 8192}), s(RAD_F32, {1});
    A c; c.t(big).t(q).t(s).i("M", 1).i("n", 8192).i("group", 8192);
    CHECK_EQ(run("had_quant_act_i8", c), RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()

/* scatter_rows: the three rules that distinguish it from the gather it mirrors, each because the
 * destination is live memory (a KV pool other sequences use) rather than a fresh buffer. */
TEST(op_scatter_rows_writes_only_the_rows_it_is_given) {
    T v(RAD_F32, {2, 3});   v.f({ 10, 11, 12,  20, 21, 22 });
    T ix(RAD_I32, {2});     ix.n({ 3, 0 });
    T x(RAD_F32, {4, 3});   x.f({ 1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4 });

    A a; a.t(v).t(ix).t(x).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("scatter_rows", a));

    CHECK_EQ(x.at(0), 20.0f);  CHECK_EQ(x.at(2), 22.0f);    /* row 0 <- v[1] */
    CHECK_EQ(x.at(9), 10.0f);  CHECK_EQ(x.at(11), 12.0f);   /* row 3 <- v[0] */
    /* The unnamed rows are untouched, the property the op is shaped around: rows 1 and 2 belong
     * to someone else, and a scatter that rewrote or zeroed them would silently destroy another
     * sequence's context. */
    CHECK_EQ(x.at(3), 2.0f);   CHECK_EQ(x.at(5), 2.0f);
    CHECK_EQ(x.at(6), 3.0f);   CHECK_EQ(x.at(8), 3.0f);
}

TEST(op_scatter_rows_skips_a_negative_index_rather_than_zeroing) {
    /* gather_rows reads a zero row for a negative index (the padding a short prefill chunk leaves
     * behind). The mirror here would be a zero fill, which against a live pool is data loss, so
     * the row is left alone. */
    T v(RAD_F32, {2, 2});   v.f({ 7, 7,  8, 8 });
    T ix(RAD_I32, {2});     ix.n({ -1, 1 });
    T x(RAD_F32, {3, 2});   x.f({ 5, 5,  6, 6,  9, 9 });

    A a; a.t(v).t(ix).t(x).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("scatter_rows", a));
    CHECK_EQ(x.at(0), 5.0f);   CHECK_EQ(x.at(1), 5.0f);   /* nothing written, not zeroed */
    CHECK_EQ(x.at(2), 8.0f);   CHECK_EQ(x.at(3), 8.0f);
    CHECK_EQ(x.at(4), 9.0f);   CHECK_EQ(x.at(5), 9.0f);
}

TEST(op_scatter_rows_refuses_an_index_past_the_destination) {
    /* The same rule as gather_rows, for the mirrored reason: reading past the operand would read
     * another sequence's state, and writing past it would corrupt one. */
    T v(RAD_F32, {1, 2});   v.f({ 1, 2 });
    T ix(RAD_I32, {1});     ix.n({ 3 });
    T x(RAD_F32, {3, 2});   x.f({ 0, 0,  0, 0,  0, 0 });
    A a; a.t(v).t(ix).t(x).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_EQ(run("scatter_rows", a), RAD_E_INVAL);
}

TEST(op_scatter_rows_round_trips_opaque_bytes_through_u8) {
    /* The dtype the tier traffic uses. A KV fragment is whatever the attention kernel wrote, so it
     * is described as bytes and never interpreted; gathered out and scattered back, every byte
     * must survive. u8 makes that checkable here: ref's float path carries 0..255 exactly, where a
     * bf16 NaN payload need not survive. */
    const int n = 8, rows = 4;
    T src(RAD_U8, {rows, n});
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i)
            st_dt(src.t.dtype, src.t.data, r * n + i, (float)((r * 37 + i * 11) & 0xff));

    T ix(RAD_I32, {2});   ix.n({ 2, 0 });
    T packed(RAD_U8, {2, n});
    A g; g.t(src).t(ix).t(packed).i("M", 2).i("n", n).s("dtype", "u8");
    CHECK_OK(run("gather_rows", g));

    T dst(RAD_U8, {rows, n});
    for (int i = 0; i < rows * n; ++i) st_dt(dst.t.dtype, dst.t.data, i, 0.0f);
    A s; s.t(packed).t(ix).t(dst).i("M", 2).i("n", n).s("dtype", "u8");
    CHECK_OK(run("scatter_rows", s));

    for (int i = 0; i < n; ++i) {
        CHECK_EQ(dst.at(2 * n + i), src.at(2 * n + i));   /* the two gathered rows came back */
        CHECK_EQ(dst.at(0 * n + i), src.at(0 * n + i));
        CHECK_EQ(dst.at(1 * n + i), 0.0f);                /* and the others are still untouched */
        CHECK_EQ(dst.at(3 * n + i), 0.0f);
    }
}

/* ================================================================== hyper-connections
 * docs/OPS.md "Hyper-connections". The expectations are the formulas evaluated in double from the
 * operands, so a mix that sums instead of averaging, a gate argument missing its `/ hc`, or a write
 * into the normed stream fails by far more than the tolerance (each case checks that its
 * alternative would). f32 operands keep libref's own roundings at f32 level. */
namespace {

struct HcCase {
    int64_t hc = 2, n = 4, lr = 2, rows = 2;
    std::vector<float> h, w, down, up, injw;
    float eps = 1e-6f, wadd = 0.25f;
    HcCase() {
        uint32_t s = 5;
        auto u = [&]() { s = s * 1664525u + 1013904223u; return (float)((int)(s >> 20) % 41 - 20) / 16.0f; };
        const int64_t hn = hc * n;
        for (int64_t i = 0; i < rows * hn; ++i) h.push_back(u() * (i % hn < n ? 1.0f : 3.0f));
        for (int64_t i = 0; i < hn; ++i) w.push_back(1.0f + u() / 8.0f);
        for (int64_t i = 0; i < lr * hn; ++i) down.push_back(u());
        for (int64_t i = 0; i < hn * lr; ++i) up.push_back(u());
        for (int64_t i = 0; i < hc * hn; ++i) injw.push_back(u());
    }
    /* N, the grouped norm of row r, in double. */
    std::vector<double> normed(int64_t r) const {
        const int64_t hn = hc * n;
        std::vector<double> N((size_t)hn);
        for (int64_t c = 0; c < hc; ++c) {
            double ss = 0;
            for (int64_t i = 0; i < n; ++i) ss += (double)h[r * hn + c * n + i] * h[r * hn + c * n + i];
            const double rs = 1.0 / std::sqrt(ss / n + eps);
            for (int64_t i = 0; i < n; ++i)
                N[c * n + i] = h[r * hn + c * n + i] * rs * (w[c * n + i] + wadd);
        }
        return N;
    }
    /* x of row r, with the three divisions by hc switchable to show what each one changes. */
    std::vector<double> mix(int64_t r, double down_div, double mean_div) const {
        const int64_t hn = hc * n;
        const std::vector<double> N = normed(r);
        std::vector<double> d((size_t)lr);
        for (int64_t j = 0; j < lr; ++j) {
            double acc = 0;
            for (int64_t i = 0; i < hn; ++i) acc += N[i] * down[j * hn + i];
            const double z = acc / down_div;
            d[j] = z / (1.0 + std::exp(-z));
        }
        std::vector<double> x((size_t)n, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t c = 0; c < hc; ++c) {
                double gv = 0;
                for (int64_t j = 0; j < lr; ++j) gv += d[j] * up[(c * n + i) * lr + j];
                x[i] += N[c * n + i] / (1.0 + std::exp(-gv));
            }
            x[i] /= mean_div;
        }
        return x;
    }
    std::vector<double> inject(int64_t r, double div) const {
        const int64_t hn = hc * n;
        const std::vector<double> N = normed(r);
        std::vector<double> g((size_t)hc);
        for (int64_t c = 0; c < hc; ++c) {
            double acc = 0;
            for (int64_t i = 0; i < hn; ++i) acc += N[i] * injw[c * hn + i];
            g[c] = 2.0 / (1.0 + std::exp(-acc / div));
        }
        return g;
    }
};

}  /* namespace */

/* hc_enter: the wide stream starts as hc copies of the embedding row. */
TEST(hc_enter_copies_x_into_every_stream) {
    T x(RAD_F32, {2, 3}); x.f({1, 2, 3, 4, 5, 6});
    T h(RAD_F32, {2, 6});
    A a; a.t(x).t(h).i("M", 2).i("n", 3).i("hc", 2).s("dtype", "f32");
    CHECK_OK(run("hc_enter", a));
    const float want[12] = { 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 };
    for (int i = 0; i < 12; ++i) CHECK_EQ(h.at(i), want[i]);
}

/* hc_read's x is the mean over the streams of the gated normed streams, and both gate arguments
 * carry their own / hc. The sum, or a missing division, would change x by tens of percent. */
TEST(hc_read_is_the_mean_of_the_gated_streams_with_each_gate_divided_by_hc) {
    const HcCase c;
    const int64_t hn = c.hc * c.n;
    T h(RAD_F32, {c.rows, hn}); h.f(c.h);
    T w(RAD_F32, {hn}); w.f(c.w);
    T dn(RAD_F32, {c.lr, hn}); dn.f(c.down);
    T up(RAD_F32, {hn, c.lr}); up.f(c.up);
    T iw(RAD_F32, {c.hc, hn}); iw.f(c.injw);
    T x(RAD_F32, {c.rows, c.n}), inj(RAD_F32, {c.rows, c.hc});
    A a; a.t(h).t(w).t(dn).t(up).t(iw).t(x).t(inj).tnull().tnull().tnull().tnull()
          .i("M", c.rows).i("n", c.n).i("hc", c.hc).i("lowrank", c.lr).d("eps", c.eps)
          .s("dtype", "f32").d("wadd", c.wadd);
    CHECK_OK(run("hc_read", a));
    double sum_err = 0, div_err = 0;
    for (int64_t r = 0; r < c.rows; ++r) {
        const std::vector<double> want = c.mix(r, (double)c.hc, (double)c.hc);
        const std::vector<double> as_sum = c.mix(r, (double)c.hc, 1.0);
        const std::vector<double> no_div = c.mix(r, 1.0, (double)c.hc);
        for (int64_t i = 0; i < c.n; ++i) {
            const float got = x.at(r * c.n + i);
            CHECK_NEAR(got, want[i], 1e-5 * (1.0 + std::fabs(want[i])));
            sum_err = std::max(sum_err, std::fabs(got - as_sum[i]));
            div_err = std::max(div_err, std::fabs(got - no_div[i]));
        }
        const std::vector<double> g = c.inject(r, (double)c.hc), g1 = c.inject(r, 1.0);
        for (int64_t s = 0; s < c.hc; ++s) {
            CHECK_NEAR(inj.at(r * c.hc + s), g[s], 1e-5);
            CHECK(std::fabs(inj.at(r * c.hc + s) - g1[s]) > 1e-3);
        }
    }
    CHECK(sum_err > 1e-2);     /* the case tells a mean from a sum */
    CHECK(div_err > 1e-3);     /* and a divided silu argument from an undivided one */
}

/* hc_write adds inj[s] * y into the stream it is given, unnormed and in place. */
TEST(hc_write_adds_into_the_unnormed_stream) {
    T y(RAD_F32, {1, 4}); y.f({1, 1, 1, 1});
    T inj(RAD_F32, {1, 2}); inj.f({0.5f, 2.0f});
    T h(RAD_F32, {1, 8}); h.f({1, 2, 3, 4, 5, 6, 7, 8});
    A a; a.t(y).t(inj).t(h).i("M", 1).i("n", 4).i("hc", 2).s("dtype", "f32");
    CHECK_OK(run("hc_write", a));
    const float want[8] = { 1.5f, 2.5f, 3.5f, 4.5f, 7, 8, 9, 10 };
    for (int i = 0; i < 8; ++i) CHECK_EQ(h.at(i), want[i]);
}

/* hc_project removes each stream's component along the layer's unit direction, and leaves a row
 * whose switch is 0 untouched. The second half is the property the feature leans on: a request that
 * did not ask for the vector computes exactly what it computes with no vector loaded. */
TEST(hc_project_removes_the_direction_per_stream_and_skips_off_rows) {
    /* Two rows, hc 2, n 2: v = (3,4)/5 is unit; the first stream of row 0 is entirely along it. */
    T h(RAD_F32, {2, 4}); h.f({ 0.6f, 0.8f, 1.0f, 0.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    T v(RAD_F32, {2});    v.f({ 0.6f, 0.8f });
    T on(RAD_I32, {2});   on.n({ 1, 0 });
    A a; a.t(h).t(v).t(on).i("M", 2).i("n", 2).i("hc", 2).d("scale", 1.0).s("dtype", "f32");
    CHECK_OK(run("hc_project", a));
    /* Row 0, stream 0: (0.6, 0.8) is v itself, dot 1, so it becomes (0, 0). */
    CHECK_NEAR(h.at(0), 0.0f, 1e-6);
    CHECK_NEAR(h.at(1), 0.0f, 1e-6);
    /* Row 0, stream 1: (1, 0) has dot 0.6, leaving (1, 0) - 0.6 * (0.6, 0.8). */
    CHECK_NEAR(h.at(2), 0.64f, 1e-6);
    CHECK_NEAR(h.at(3), -0.48f, 1e-6);
    /* Row 1 asked for the stock model: bit for bit what it was. */
    CHECK_EQ(h.at(4), 2.0f);
    CHECK_EQ(h.at(5), 3.0f);
    CHECK_EQ(h.at(6), 4.0f);
    CHECK_EQ(h.at(7), 5.0f);
}

/* With `group`, hc_read also emits x as block-scaled fp8 taken over the stored (narrowed) x: byte
 * for byte what quant_act_fp8 produces from the stored x. `rotate` adds the Hadamard-rotated twin
 * beside it and leaves the plain pair as it is. */
TEST(hc_read_fp8_forms_are_the_quantisers_of_the_stored_x) {
    const HcCase c;
    const int64_t hn = c.hc * c.n, g = 2;
    T h(RAD_F32, {c.rows, hn}); h.f(c.h);
    T w(RAD_F32, {hn}); w.f(c.w);
    T dn(RAD_F32, {c.lr, hn}); dn.f(c.down);
    T up(RAD_F32, {hn, c.lr}); up.f(c.up);
    T x(RAD_BF16, {c.rows, c.n});
    T q(RAD_F8E4M3, {c.rows, c.n}), s(RAD_F32, {c.rows, c.n / g});
    T rq(RAD_F8E4M3, {c.rows, c.n}), rs(RAD_F32, {c.rows, c.n / g});
    A a; a.t(h).t(w).t(dn).t(up).tnull().t(x).tnull().t(q).t(s).t(rq).t(rs)
          .i("M", c.rows).i("n", c.n).i("hc", c.hc).i("lowrank", c.lr).d("eps", c.eps)
          .s("dtype", "bf16").d("wadd", c.wadd).i("inject", 0).i("group", g).i("rotate", 1);
    CHECK_OK(run("hc_read", a));

    T q2(RAD_F8E4M3, {c.rows, c.n}), s2(RAD_F32, {c.rows, c.n / g});
    A b; b.t(x).t(q2).t(s2).i("M", c.rows).i("n", c.n).i("group", g).s("dtype", "bf16");
    CHECK_OK(run("quant_act_fp8", b));
    CHECK(std::memcmp(q.t.data, q2.t.data, (size_t)(c.rows * c.n)) == 0);
    CHECK(std::memcmp(s.t.data, s2.t.data, (size_t)(c.rows * c.n / g * 4)) == 0);
}

/* mtp_enter with `ngroup` hc (one rsqrt per sub-stream) and 1 (one over the wide row), on streams
 * of different scale: each matches its own formula and the two differ. */
TEST(mtp_enter_grouped_and_flat_norms_follow_their_formulas) {
    const int64_t n = 2, hc = 2, hn = 4;
    const std::vector<float> hv = { 1, 2, 6, -9 }, ev = { 3, -1 };
    const std::vector<float> wh = { 1, 2, 0.5f, 1 }, we = { 1, 0.5f };
    const std::vector<float> fch = { 1, 2, -1, 0.5f }, fce = { 0.25f, 1, 2, -3 };
    const float eps = 1e-6f, wadd = 0.0f;
    auto expect = [&](bool grouped, int64_t k, int64_t i) {
        double se = 0;
        for (float v : ev) se += (double)v * v;
        const double rse = 1.0 / std::sqrt(se / n + eps);
        double fe = 0;
        for (int64_t j = 0; j < n; ++j) fe += fce[i * n + j] * (ev[j] * rse * (we[j] + wadd));
        double ss = 0;
        const int64_t lo = grouped ? k * n : 0, hi = grouped ? k * n + n : hn;
        for (int64_t j = lo; j < hi; ++j) ss += (double)hv[j] * hv[j];
        const double rs = 1.0 / std::sqrt(ss / (double)(hi - lo) + eps);
        double fh = 0;
        for (int64_t j = 0; j < n; ++j) fh += fch[i * n + j] * (hv[k * n + j] * rs * (wh[k * n + j] + wadd));
        return fh + fe;
    };
    float got[2][4];
    for (int grouped = 0; grouped < 2; ++grouped) {
        T h(RAD_F32, {1, hn}); h.f(hv);
        T e(RAD_F32, {1, n}); e.f(ev);
        T w_h(RAD_F32, {hn}); w_h.f(wh);
        T w_e(RAD_F32, {n}); w_e.f(we);
        T f_h(RAD_F32, {n, n}); f_h.f(fch);
        T f_e(RAD_F32, {n, n}); f_e.f(fce);
        T x(RAD_F32, {1, hn});
        A a; a.t(h).t(e).t(w_h).t(w_e).t(f_h).t(f_e).t(x).i("M", 1).i("n", n).i("hc", hc)
              .d("eps", eps).s("dtype", "f32").i("ngroup", grouped ? hc : 1);
        CHECK_OK(run("mtp_enter", a));
        for (int64_t k = 0; k < hc; ++k)
            for (int64_t i = 0; i < n; ++i) {
                got[grouped][k * n + i] = x.at(k * n + i);
                CHECK_NEAR(x.at(k * n + i), expect(grouped != 0, k, i), 1e-5);
            }
    }
    CHECK(std::fabs(got[0][0] - got[1][0]) > 1e-2);
}

/* ================================================================== image generation
 * docs/OPS.md "Image generation", one case per op. Values are chosen so every rounding the
 * statement names is exact or lands on a known bf16 value. */

/* ln_mod: bf16(bf16(LN(x)) * bf16(1 + s[seq])), a gainless layer norm and its sequence's row. */
TEST(ln_mod_scales_each_rows_layer_norm_by_its_sequences_row) {
    T x(RAD_BF16, {2, 2}); x.f({1, 3, 0, 4});           /* both normalise to [-1, 1] */
    T s(RAD_BF16, {2, 2}); s.f({0.5f, -0.5f, 1, 0});
    T cu(RAD_I32, {3}); cu.n({0, 1, 2});
    T y(RAD_BF16, {2, 2});
    A a; a.t(x).t(s).t(cu).t(y).i("M", 2).i("n", 2).d("eps", 0.0).s("dtype", "bf16");
    CHECK_OK(run("ln_mod", a));
    CHECK_EQ(y.at(0), -1.5f); CHECK_EQ(y.at(1), 0.5f);
    CHECK_EQ(y.at(2), -2.0f); CHECK_EQ(y.at(3), 1.0f);
}

/* gate_add: x += bf16(bf16(tanh(g[seq])) * y). tanh(0) gates a row off; tanh(+-20) is +-1 in f32. */
TEST(gate_add_adds_y_through_its_sequences_tanh_gate) {
    T y(RAD_BF16, {2, 2}); y.f({5, 6, 7, 8});
    T g(RAD_BF16, {2, 2}); g.f({0, 0, 20, -20});
    T cu(RAD_I32, {3}); cu.n({0, 1, 2});
    T x(RAD_BF16, {2, 2}); x.f({1, 2, 3, 4});
    A a; a.t(y).t(g).t(cu).t(x).i("M", 2).i("n", 2).s("dtype", "bf16");
    CHECK_OK(run("gate_add", a));
    CHECK_EQ(x.at(0), 1.0f); CHECK_EQ(x.at(1), 2.0f);
    CHECK_EQ(x.at(2), 10.0f); CHECK_EQ(x.at(3), -4.0f);
}

/* gate_add with `act` none: x = bf16(x + bf16(g[seq] * y)), the gate as it is. An f32 gate of
 * 1.00390625 is not rounded first: 3 g = 3.01171875 rounds to 3.015625, where bf16(g) = 1 (a tie,
 * to even) would give 3. */
TEST(gate_add_act_none_adds_through_the_gate_itself) {
    T y(RAD_BF16, {2, 2}); y.f({5, 6, 7, 8});
    T g(RAD_BF16, {2, 2}); g.f({0.5f, -2, 0.25f, 3});
    T cu(RAD_I32, {3}); cu.n({0, 1, 2});
    T x(RAD_BF16, {2, 2}); x.f({1, 2, 3, 4});
    A a; a.t(y).t(g).t(cu).t(x).i("M", 2).i("n", 2).s("dtype", "bf16").s("act", "none");
    CHECK_OK(run("gate_add", a));
    CHECK_EQ(x.at(0), 3.5f); CHECK_EQ(x.at(1), -10.0f);
    CHECK_EQ(x.at(2), 4.75f); CHECK_EQ(x.at(3), 28.0f);

    T y1(RAD_BF16, {1, 1}); y1.f({3});
    T g1(RAD_F32, {1, 1}); g1.f({1.00390625f});
    T cu1(RAD_I32, {2}); cu1.n({0, 1});
    T x1(RAD_BF16, {1, 1}); x1.f({0});
    A b; b.t(y1).t(g1).t(cu1).t(x1).i("M", 1).i("n", 1).s("dtype", "bf16").s("act", "none");
    CHECK_OK(run("gate_add", b));
    CHECK_EQ(x1.at(0), 3.015625f);

    A c; c.t(y1).t(g1).t(cu1).t(x1).i("M", 1).i("n", 1).s("dtype", "bf16").s("act", "silu");
    CHECK_EQ(run("gate_add", c), RAD_E_INVAL);
}

/* timestep_embed: [cos | sin] of t' f_i, f_i = exp(-ln(max_period) i / (n/2)); with `round`, t' is
 * scale * bf16(bf16(t * scale) / scale), the value diffusers hands its transformer. */
TEST(timestep_embed_is_cos_then_sin_of_the_scaled_time) {
    T t(RAD_F32, {2}); t.f({0, 1});
    T y(RAD_F32, {2, 4});
    A a; a.t(t).t(y).i("M", 2).i("n", 4).d("scale", 1.0).d("max_period", 100.0).s("dtype", "f32");
    CHECK_OK(run("timestep_embed", a));
    CHECK_EQ(y.at(0), 1.0f); CHECK_EQ(y.at(1), 1.0f);
    CHECK_EQ(y.at(2), 0.0f); CHECK_EQ(y.at(3), 0.0f);
    CHECK_NEAR(y.at(4), std::cos(1.0), 1e-6); CHECK_NEAR(y.at(5), std::cos(0.1), 1e-6);
    CHECK_NEAR(y.at(6), std::sin(1.0), 1e-6); CHECK_NEAR(y.at(7), std::sin(0.1), 1e-6);

    /* 0.3 * 1000 = 300 is exact in bf16; 300 / 1000 rounds to bf16 0.30078125. */
    T t2(RAD_F32, {1}); t2.f({0.3f});
    T y2(RAD_F32, {1, 2});
    A b; b.t(t2).t(y2).i("M", 1).i("n", 2).d("scale", 1000.0).d("max_period", 100.0)
          .i("round", 1).s("dtype", "f32");
    CHECK_OK(run("timestep_embed", b));
    CHECK_NEAR(y2.at(0), std::cos(300.78125), 1e-5);
    CHECK_NEAR(y2.at(1), std::sin(300.78125), 1e-5);

    /* clamp_min raises a timestep below it to it before the embedding (the pinned references'
     * near-clean timestep). h = n/2 = 1, so f_0 = 1 and the row is [cos | sin](t'). Row 0's t is
     * below the floor and is embedded at it; row 1's is above and is embedded as it is. */
    T t3(RAD_F32, {2}); t3.f({0.05f, 1.0f});
    T y3(RAD_F32, {2, 2});
    A c; c.t(t3).t(y3).i("M", 2).i("n", 2).d("scale", 1.0).d("max_period", 100.0)
           .d("clamp_min", 0.999).s("dtype", "f32");
    CHECK_OK(run("timestep_embed", c));
    CHECK_NEAR(y3.at(0), std::cos(0.999), 1e-6); CHECK_NEAR(y3.at(1), std::sin(0.999), 1e-6);
    CHECK_NEAR(y3.at(2), std::cos(1.0), 1e-6);   CHECK_NEAR(y3.at(3), std::sin(1.0), 1e-6);
}

/* gather_rows2: idx >= 0 reads a, idx < 0 reads b[-1 - idx], and a row past either source is
 * zeros. */
TEST(gather_rows2_reads_two_sources_by_the_index_sign) {
    T a_(RAD_F32, {2, 2}); a_.f({1, 2, 3, 4});
    T b_(RAD_F32, {1, 2}); b_.f({5, 6});
    T idx(RAD_I32, {5}); idx.n({1, -1, 0, -2, 5});
    T y(RAD_F32, {5, 2}); y.f({9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
    A a; a.t(a_).t(b_).t(idx).t(y).i("M", 5).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("gather_rows2", a));
    const float want[10] = { 3, 4, 5, 6, 1, 2, 0, 0, 0, 0 };
    for (int i = 0; i < 10; ++i) CHECK_EQ(y.at(i), want[i]);
}

/* lane_gather copies entry e's rows out of lane lane[e] (`stride` rows apart); lane_store copies
 * one source into every listed lane and leaves the rest. */
TEST(lane_gather_and_lane_store_address_lanes_by_stride) {
    T src(RAD_F32, {8, 1}); src.f({0, 1, 2, 3, 4, 5, 6, 7});
    T lane(RAD_I32, {2}); lane.n({3, 1});
    T cu(RAD_I32, {3}); cu.n({0, 2, 3});
    T y(RAD_F32, {3, 1});
    A a; a.t(src).t(lane).t(cu).t(y).i("M", 3).i("n", 1).i("stride", 2).s("dtype", "f32");
    CHECK_OK(run("lane_gather", a));
    CHECK_EQ(y.at(0), 6.0f); CHECK_EQ(y.at(1), 7.0f); CHECK_EQ(y.at(2), 2.0f);

    /* An entry longer than its lane is refused. */
    T cu2(RAD_I32, {2}); cu2.n({0, 3});
    T y2(RAD_F32, {3, 1});
    A a2; a2.t(src).t(lane).t(cu2).t(y2).i("M", 3).i("n", 1).i("stride", 2).s("dtype", "f32");
    CHECK_EQ(run("lane_gather", a2), RAD_E_SHAPE);

    T s2(RAD_F32, {2, 1}); s2.f({10, 20});
    T l2(RAD_I32, {2}); l2.n({0, 2});
    T dst(RAD_F32, {8, 1});
    A b; b.t(s2).t(l2).t(dst).i("M", 2).i("n", 1).i("stride", 2).s("dtype", "f32");
    CHECK_OK(run("lane_store", b));
    const float want[8] = { 10, 20, 0, 0, 10, 20, 0, 0 };
    for (int i = 0; i < 8; ++i) CHECK_EQ(dst.at(i), want[i]);
}

/* randn_lanes: a first-step entry's lane rows become its seed's normal stream, element (k, c) the
 * stream's k * n + c, whatever lane or batch it is in; other entries' lanes are untouched. */
TEST(randn_lanes_is_the_seed_stream_independent_of_lane_and_batch) {
    const int64_t n = 2, st = 2;
    const uint64_t seed = 0x123456789abcull;
    T fl(RAD_I32, {2}); fl.n({1, 0});
    T sd(RAD_U32, {2, 2}); sd.n({(int64_t)(seed & 0xffffffffu), (int64_t)(seed >> 32), 7, 9});
    T lane(RAD_I32, {2}); lane.n({1, 0});
    T cu(RAD_I32, {3}); cu.n({0, 2, 3});
    T dst(RAD_BF16, {2 * st, n});
    A a; a.t(fl).t(sd).t(lane).t(cu).t(dst).i("M", 3).i("n", n).i("stride", st).s("dtype", "bf16");
    CHECK_OK(run("randn_lanes", a));
    for (int64_t k = 0; k < 2; ++k)
        for (int64_t c = 0; c < n; ++c) {
            CHECK_EQ(dst.at((st + k) * n + c),
                     round_dt(RAD_BF16, rad_philox_normal(seed, (uint64_t)(k * n + c))));
            CHECK_EQ(dst.at(k * n + c), 0.0f);           /* entry 1 is not a first step */
        }
    /* The same entry alone, in lane 0: the same values. */
    T fl1(RAD_I32, {1}); fl1.n({1});
    T sd1(RAD_U32, {1, 2}); sd1.n({(int64_t)(seed & 0xffffffffu), (int64_t)(seed >> 32)});
    T ln1(RAD_I32, {1}); ln1.n({0});
    T cu1(RAD_I32, {2}); cu1.n({0, 2});
    T d1(RAD_BF16, {st, n});
    A b; b.t(fl1).t(sd1).t(ln1).t(cu1).t(d1).i("M", 2).i("n", n).i("stride", st).s("dtype", "bf16");
    CHECK_OK(run("randn_lanes", b));
    for (int64_t i = 0; i < st * n; ++i) CHECK_EQ(d1.at(i), dst.at(st * n + i));
}

/* flow_euler: entry 0 is guided by entry 1 (pair 1) and previews, entry 1 is a negative
 * conditioning (pair -2) that moves no lane, entry 2 is unguided and repaints its kept region
 * (mask 0) at the last step (sigma_next 0), which is its source latent exactly. */
TEST(flow_euler_guides_previews_and_repaints_as_stated) {
    const int64_t n = 2, st = 2;
    T v(RAD_BF16, {3, n}); v.f({1, 2, 0.5f, 0.5f, 4, -4});
    T sg(RAD_F32, {3}); sg.f({1, 1, 1});
    T sgn(RAD_F32, {3}); sgn.f({0.5f, 0.5f, 0});
    T cfg(RAD_F32, {3}); cfg.f({3, 1, 1});
    T pair(RAD_I32, {3}); pair.n({1, -2, -1});
    T lane(RAD_I32, {3}); lane.n({0, 1, 2});
    T fl(RAD_I32, {3}); fl.n({4, 0, 2});
    T sd(RAD_U32, {3, 2}); sd.n({1, 2, 3, 4, 5, 6});
    T cu(RAD_I32, {4}); cu.n({0, 1, 2, 3});
    T inp(RAD_BF16, {3 * st, n}); inp.f({0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0});
    T mask(RAD_F32, {3 * st}); mask.f({1, 1, 1, 1, 0, 1});
    T state(RAD_BF16, {3 * st, n}); state.f({2, 2, 0, 0, 3, 3, 0, 0, 1, 1, 0, 0});
    T prev(RAD_BF16, {3 * st, n});
    A a; a.t(v).t(sg).t(sgn).t(cfg).t(pair).t(lane).t(fl).t(sd).t(cu).t(inp).t(mask).t(state)
          .t(prev).i("M", 3).i("n", n).i("stride", st).s("dtype", "bf16");
    CHECK_OK(run("flow_euler", a));
    /* Entry 0: v' = 0.5 + 3 * (v - 0.5) = [2, 5]; preview x - v' = [0, -3]; x' = x - 0.5 v'. */
    CHECK_EQ(prev.at(0), 0.0f); CHECK_EQ(prev.at(1), -3.0f);
    CHECK_EQ(state.at(0), 1.0f); CHECK_EQ(state.at(1), -0.5f);
    /* Entry 1's lane is unchanged; entry 2's kept row is its source. */
    CHECK_EQ(state.at(4), 3.0f); CHECK_EQ(state.at(5), 3.0f);
    CHECK_EQ(state.at(8), 7.0f); CHECK_EQ(state.at(9), 8.0f);
}

/* kd_input: entry 0's first step fills its lane with the seed's stream times sqrt(1 + sigma^2)
 * and writes x / sqrt(sigma^2 + 1) to its rows and its pair's (entry 1, pair -2, writes nothing of
 * its own); entry 2 reads its lane as it is. */
TEST(kd_input_starts_the_lane_and_scales_by_c_in) {
    const int64_t n = 2, st = 2;
    const uint64_t seed = 0xabcdef12345ull;
    T sg(RAD_F32, {3}); sg.f({3, 3, 1});
    T pair(RAD_I32, {3}); pair.n({1, -2, -1});
    T lane(RAD_I32, {3}); lane.n({0, 1, 2});
    T fl(RAD_I32, {3}); fl.n({1, 0, 0});
    T sd(RAD_U32, {3, 2}); sd.n({(int64_t)(seed & 0xffffffffu), (int64_t)(seed >> 32), 0, 0, 5, 0});
    T cu(RAD_I32, {4}); cu.n({0, 2, 4, 5});
    T state(RAD_F32, {3 * st, n}); state.f({0, 0, 0, 0, 7, 7, 7, 7, 4, -2, 9, 9});
    T y(RAD_F32, {5, n});
    A a; a.t(sg).t(pair).t(lane).t(fl).t(sd).t(cu).t(state).t(y).i("M", 5).i("n", n)
          .i("stride", st).s("dtype", "f32");
    CHECK_OK(run("kd_input", a));
    const float init = std::sqrt(1.0f + 9.0f), cin = 1.0f / std::sqrt(9.0f + 1.0f);
    for (int64_t k = 0; k < 2; ++k)
        for (int64_t c = 0; c < n; ++c) {
            const float x0 = rad_philox_normal(seed, (uint64_t)(k * n + c)) * init;
            CHECK_EQ(state.at(k * n + c), x0);
            CHECK_EQ(y.at(k * n + c), x0 * cin);
            CHECK_EQ(y.at((2 + k) * n + c), x0 * cin);
            CHECK_EQ(state.at((st + k) * n + c), 7.0f);   /* the parked entry's lane */
        }
    CHECK_NEAR(y.at(8), 4.0f / std::sqrt(2.0f), 1e-6); CHECK_NEAR(y.at(9), -2.0f / std::sqrt(2.0f), 1e-6);
    /* A repaint is refused. */
    T fl2(RAD_I32, {3}); fl2.n({2, 0, 0});
    A b; b.t(sg).t(pair).t(lane).t(fl2).t(sd).t(cu).t(state).t(y).i("M", 5).i("n", n)
          .i("stride", st).s("dtype", "f32");
    CHECK_EQ(run("kd_input", b), RAD_E_UNSUPPORTED);
}

/* kd_step, one element an entry: Euler guided against a parked entry with a preview; Euler
 * ancestral at sigma_next 0 and above it; DPM++ 2M's first step, its second-order step and its
 * last step; and a v-prediction model. */
TEST(kd_step_denoises_guides_and_steps_each_sampler) {
    const int64_t n = 1, st = 1;
    /* entry: 0 euler guided, 1 parked, 2 ancestral to 0, 3 ancestral, 4 dpm first, 5 dpm second,
     * 6 dpm last */
    T o(RAD_F32, {7, n}); o.f({0.5f, 0.25f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
    T sg(RAD_F32, {7}); sg.f({2, 2, 2, 2, 2, 2, 2});
    T sgn(RAD_F32, {7}); sgn.f({1, 1, 0, 1, 1, 1, 0});
    T sgp(RAD_F32, {7}); sgp.f({0, 0, 0, 0, 0, 4, 4});
    T cfg(RAD_F32, {7}); cfg.f({3, 1, 1, 1, 1, 1, 1});
    T pair(RAD_I32, {7}); pair.n({1, -2, -1, -1, -1, -1, -1});
    T lane(RAD_I32, {7}); lane.n({0, 1, 2, 3, 4, 5, 6});
    T fl(RAD_I32, {7}); fl.n({4, 0, 0, 0, 1, 0, 0});
    T sd(RAD_U32, {7, 2}); sd.n({0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 0, 0, 0, 0});
    T smp(RAD_I32, {7}); smp.n({0, 0, 1, 1, 2, 2, 2});
    T stp(RAD_I32, {7}); stp.n({0, 0, 0, 5, 0, 1, 2});
    T cu(RAD_I32, {8}); cu.n({0, 1, 2, 3, 4, 5, 6, 7});
    T state(RAD_F32, {7, n}); state.f({2, 9, 2, 2, 2, 2, 2});
    T hist(RAD_F32, {7, n}); hist.f({0, 0, 0, 0, 0, 0.5f, 0});
    T prev(RAD_F32, {7, n});
    A a; a.t(o).t(sg).t(sgn).t(sgp).t(cfg).t(pair).t(lane).t(fl).t(sd).t(smp).t(stp).t(cu)
          .t(state).t(hist).t(prev).i("M", 7).i("n", n).i("stride", st).s("pred", "eps")
          .s("dtype", "f32");
    CHECK_OK(run("kd_step", a));
    /* D = x - sigma o: 1 for the positive entry, 1.5 for its negative, guided 1.5 + 3 (1 - 1.5) =
     * 0; Euler: x + (1 - 2)(2 - 0) / 2 = 1. */
    CHECK_EQ(prev.at(0), 0.0f);
    CHECK_EQ(state.at(0), 1.0f);
    CHECK_EQ(hist.at(0), 0.0f);
    CHECK_EQ(state.at(1), 9.0f);                     /* parked */
    CHECK_EQ(state.at(2), 1.0f);                     /* ancestral to 0: D */
    {
        /* sigma_up = min(1, sqrt(1 (4 - 1) / 4)) = sqrt(0.75), sigma_down = sqrt(1 - 0.75). */
        const float sup = std::sqrt(0.75f), sdown = std::sqrt(1.0f - sup * sup);
        const float d = (2.0f - 1.0f) / 2.0f;
        const float want = 2.0f + d * (sdown - 2.0f) + rad_philox_normal_s(77, 6, 0) * sup;
        CHECK_NEAR(state.at(3), want, 1e-6);
    }
    /* DPM++ 2M: h = ln 2, so x' = x / 2 - expm1(-ln 2) D = 1 + 0.5 D' with D = 1. */
    CHECK_NEAR(state.at(4), 1.5f, 1e-6);
    /* r = (ln 4 - ln 2) / ln 2 = 1: D' = 1.5 D - 0.5 hist = 1.25. */
    CHECK_NEAR(state.at(5), 1.625f, 1e-6);
    CHECK_EQ(state.at(6), 1.0f);                     /* to sigma 0: D */
    CHECK_EQ(hist.at(5), 1.0f);

    /* v-prediction at sigma 1: D = x / 2 - o / sqrt(2), so o = sqrt(2) at x = 2 gives D = 0, and
     * Euler to 0 lands on it. */
    T ov(RAD_F32, {1, 1}); ov.f({std::sqrt(2.0f)});
    T s1(RAD_F32, {1}); s1.f({1});
    T z1(RAD_F32, {1}); z1.f({0});
    T c1(RAD_F32, {1}); c1.f({1});
    T p1(RAD_I32, {1}); p1.n({-1});
    T l1(RAD_I32, {1}); l1.n({0});
    T f1(RAD_I32, {1}); f1.n({0});
    T d1(RAD_U32, {1, 2}); d1.n({0, 0});
    T m1(RAD_I32, {1}); m1.n({0});
    T cu1(RAD_I32, {2}); cu1.n({0, 1});
    T x1(RAD_F32, {1, 1}); x1.f({2});
    T h1(RAD_F32, {1, 1});
    A b; b.t(ov).t(s1).t(z1).t(z1).t(c1).t(p1).t(l1).t(f1).t(d1).t(m1).t(m1).t(cu1).t(x1).t(h1)
          .i("M", 1).i("n", 1).i("stride", 1).s("pred", "v").s("dtype", "f32");
    CHECK_OK(run("kd_step", b));
    CHECK_NEAR(x1.at(0), 0.0f, 1e-6);

    /* An unknown sampler is refused, and so is a preview without its operand. */
    T bad(RAD_I32, {1}); bad.n({3});
    A c; c.t(ov).t(s1).t(z1).t(z1).t(c1).t(p1).t(l1).t(f1).t(d1).t(bad).t(m1).t(cu1).t(x1).t(h1)
          .i("M", 1).i("n", 1).i("stride", 1).s("pred", "v").s("dtype", "f32");
    CHECK_EQ(run("kd_step", c), RAD_E_UNSUPPORTED);
    T f4(RAD_I32, {1}); f4.n({4});
    A d; d.t(ov).t(s1).t(z1).t(z1).t(c1).t(p1).t(l1).t(f4).t(d1).t(m1).t(m1).t(cu1).t(x1).t(h1)
          .i("M", 1).i("n", 1).i("stride", 1).s("pred", "v").s("dtype", "f32");
    CHECK_EQ(run("kd_step", d), RAD_E_INVAL);
}

/* gelu_quick rounds the scaled input and the sigmoid to y's dtype before the product. */
TEST(gelu_quick_rounds_like_torch_in_bf16) {
    T x(RAD_BF16, {1, 3}); x.f({1.0f, -2.0f, 0.375f});
    T y(RAD_BF16, {1, 3});
    A a; a.t(x).t(y).i("M", 1).i("n", 3).s("dtype", "bf16");
    CHECK_OK(run("gelu_quick", a));
    for (int i = 0; i < 3; ++i) {
        const float v = x.at(i);
        const float t = round_dt(RAD_BF16, 1.702f * v);
        const float s = round_dt(RAD_BF16, 1.0f / (1.0f + std::exp(-t)));
        CHECK_EQ(y.at(i), round_dt(RAD_BF16, v * s));
    }
    T xf(RAD_F32, {1, 1}); xf.f({0.5f});
    T yf(RAD_F32, {1, 1});
    A b; b.t(xf).t(yf).i("M", 1).i("n", 1).s("dtype", "f32");
    CHECK_OK(run("gelu_quick", b));
    CHECK_EQ(yf.at(0), 0.5f * (1.0f / (1.0f + std::exp(-0.851f))));
}

/* chan_affine: lists are bf16 values; fma is bf16(bf16(x * mul) + add) and inv its inverse. */
TEST(chan_affine_reads_bf16_lists_and_inverts) {
    T x(RAD_BF16, {1, 3}); x.f({1, 1, 1});
    T y(RAD_BF16, {1, 3});
    A a; a.t(x).t(y).i("M", 1).i("n", 3).s("mul", "2 4 0.5").s("add", "1 -1 0.25")
          .s("mode", "fma").s("dtype", "bf16");
    CHECK_OK(run("chan_affine", a));
    CHECK_EQ(y.at(0), 3.0f); CHECK_EQ(y.at(1), 3.0f); CHECK_EQ(y.at(2), 0.75f);
    T z(RAD_BF16, {1, 3});
    A b; b.t(y).t(z).i("M", 1).i("n", 3).s("mul", "2 4 0.5").s("add", "1 -1 0.25")
          .s("mode", "inv").s("dtype", "bf16");
    CHECK_OK(run("chan_affine", b));
    CHECK_EQ(z.at(0), 1.0f); CHECK_EQ(z.at(1), 1.0f); CHECK_EQ(z.at(2), 1.0f);
    /* 0.1 is read as bf16(0.1), not 0.1. */
    T w(RAD_F32, {1, 1});
    T x1(RAD_F32, {1, 1}); x1.f({1});
    A c; c.t(x1).t(w).i("M", 1).i("n", 1).s("mul", "0.1").s("add", "0").s("mode", "fma")
          .s("dtype", "f32");
    CHECK_OK(run("chan_affine", c));
    CHECK_EQ(w.at(0), round_dt(RAD_BF16, 0.1f));
}

/* chan_affine with `round` 0: f32 lists and f32 arithmetic, rounded once into y. fma of 10 with
 * mul 0.3 and add 0.1 is f32(f32(10 * 0.3f) + 0.1f) = 3.1f (the bf16 lists would give
 * 3.10009765625); inv is f32(f32(10 - 0.1f) / 0.3f) = 32.999996f (bf16: 32.75); a bf16 y rounds
 * only the result, 3.09375 and 33. */
TEST(chan_affine_round_0_keeps_f32_lists_and_rounds_once) {
    T x(RAD_F32, {1, 2}); x.f({10, 10});
    T y(RAD_F32, {1, 2});
    A a; a.t(x).t(y).i("M", 1).i("n", 2).s("mul", "0.3 2").s("add", "0.1 -1").s("mode", "fma")
          .s("dtype", "f32").i("round", 0);
    CHECK_OK(run("chan_affine", a));
    CHECK_EQ(y.at(0), 3.1f); CHECK_EQ(y.at(1), 19.0f);
    A b; b.t(x).t(y).i("M", 1).i("n", 2).s("mul", "0.3 2").s("add", "0.1 -1").s("mode", "inv")
          .s("dtype", "f32").i("round", 0);
    CHECK_OK(run("chan_affine", b));
    CHECK_EQ(y.at(0), 32.999996f); CHECK_EQ(y.at(1), 5.5f);

    T yb(RAD_BF16, {1, 2});
    A c; c.t(x).t(yb).i("M", 1).i("n", 2).s("mul", "0.3 2").s("add", "0.1 -1").s("mode", "fma")
          .s("dtype", "bf16").i("round", 0);
    CHECK_OK(run("chan_affine", c));
    CHECK_EQ(yb.at(0), 3.09375f);
    A d; d.t(x).t(yb).i("M", 1).i("n", 2).s("mul", "0.3 2").s("add", "0.1 -1").s("mode", "inv")
          .s("dtype", "bf16").i("round", 0);
    CHECK_OK(run("chan_affine", d));
    CHECK_EQ(yb.at(0), 33.0f);
}

/* conv2d_nhwc: `same` padding sums a zero-padded window; `up` 2 reads the input upsampled
 * nearest-neighbour. */
TEST(conv2d_nhwc_pads_and_upsamples) {
    T x(RAD_F32, {3, 3}); x.f({1, 2, 3, 4, 5, 6, 7, 8, 9});
    T w(RAD_F32, {1, 1, 3, 3}); w.f({1, 1, 1, 1, 1, 1, 1, 1, 1});
    T b(RAD_F32, {1}); b.f({0.5f});
    T y(RAD_F32, {3, 3});
    A a; a.t(x).t(w).t(b).t(y).i("M", 9).i("cin", 1).i("cout", 1).i("k", 3).i("stride", 1)
          .s("pad", "same").i("up", 1).s("dtype", "f32");
    CHECK_OK(run("conv2d_nhwc", a));
    CHECK_EQ(y.at(0), 12.5f);    /* 1 + 2 + 4 + 5 */
    CHECK_EQ(y.at(4), 45.5f);    /* all nine */
    CHECK_EQ(y.at(8), 28.5f);    /* 5 + 6 + 8 + 9 */

    T x2(RAD_F32, {2, 2}); x2.f({1, 2, 3, 4});
    T w1(RAD_F32, {1, 1, 1, 1}); w1.f({1});
    T b0(RAD_F32, {1}); b0.f({0});
    T y2(RAD_F32, {4, 4});
    A c; c.t(x2).t(w1).t(b0).t(y2).i("M", 16).i("cin", 1).i("cout", 1).i("k", 1).i("stride", 1)
          .s("pad", "same").i("up", 2).s("dtype", "f32");
    CHECK_OK(run("conv2d_nhwc", c));
    const float want[16] = { 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 };
    for (int i = 0; i < 16; ++i) CHECK_EQ(y2.at(i), want[i]);
}

/* im2col_nhwc: its rows times the weight read as [cout, cin k k] are conv2d_nhwc, at both paddings
 * of a stride-2 conv, the upsample and a 1x1, with x at a row pitch. Small integers keep every sum
 * exact, so the two orders of accumulation agree to the bit. */
TEST(im2col_nhwc_times_the_weight_is_the_convolution) {
    struct Case { const char* pad; int64_t cin, cout, k, stride, up, H, W; };
    for (const Case& c : { Case{ "same", 3, 4, 3, 1, 1, 5, 6 }, Case{ "same", 2, 3, 3, 2, 1, 7, 5 },
                           Case{ "down", 3, 2, 3, 2, 1, 6, 8 }, Case{ "same", 2, 3, 3, 1, 2, 3, 4 },
                           Case{ "same", 5, 2, 1, 1, 1, 4, 3 } }) {
        const bool down = !std::strcmp(c.pad, "down");
        const int64_t Hv = c.H * c.up, Wv = c.W * c.up, pads = down ? 1 : 2 * (c.k / 2);
        const int64_t Ho = (Hv + pads - c.k) / c.stride + 1, Wo = (Wv + pads - c.k) / c.stride + 1;
        const int64_t KK = c.cin * c.k * c.k, ld = c.W * c.cin + 3;
        std::vector<float> xv((size_t)(c.H * ld)), wv((size_t)(c.cout * KK)), bv((size_t)c.cout);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = (float)((int)(i * 7 % 11) - 5);
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = (float)((int)(i * 5 % 7) - 3);
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = (float)i - 1.0f;
        T xbuf(RAD_F32, {c.H, ld}); xbuf.f(xv);
        T x = xbuf;
        x.t.shape[1] = c.W * c.cin;                  /* [H, W cin] at a row pitch of ld */
        T w(RAD_F32, {c.cout, c.cin, c.k, c.k}); w.f(wv);
        T w2(RAD_F32, {c.cout, KK}); w2.f(wv);
        T b(RAD_F32, {c.cout}); b.f(bv);
        T y(RAD_F32, {Ho, Wo * c.cout});
        A a; a.t(x).t(w).t(b).t(y).i("M", Ho * Wo).i("cin", c.cin).i("cout", c.cout).i("k", c.k)
              .i("stride", c.stride).s("pad", c.pad).i("up", c.up).s("dtype", "f32");
        CHECK_OK(run("conv2d_nhwc", a));
        T u(RAD_F32, {Ho * Wo, KK});
        A d; d.t(x).t(u).i("M", Ho * Wo).i("cin", c.cin).i("k", c.k).i("stride", c.stride)
              .s("pad", c.pad).i("up", c.up).s("dtype", "f32");
        CHECK_OK(run("im2col_nhwc", d));
        T g(RAD_F32, {Ho * Wo, c.cout});
        A e; e.t(u).t(w2).t(b).tnull().t(g).i("M", Ho * Wo).i("N", c.cout).i("K", KK)
              .s("dtype", "f32");
        CHECK_OK(run("gemm_nt_bias", e));
        for (int64_t i = 0; i < Ho * Wo * c.cout; ++i) CHECK_EQ(g.at(i), y.at(i));
        /* Spot the column order: (ci, i, j), the weight's own. */
        const int64_t oy = Ho - 1, ox = 0, ci = c.cin - 1, i = c.k - 1, j = 0;
        const int64_t iy = oy * c.stride + i - (down ? 0 : c.k / 2);
        const int64_t ix = ox * c.stride + j - (down ? 0 : c.k / 2);
        const float want = iy < 0 || iy >= Hv || ix < 0 || ix >= Wv
                               ? 0.0f : xv[(size_t)((iy / c.up) * ld + (ix / c.up) * c.cin + ci)];
        CHECK_EQ(u.at((oy * Wo + ox) * KK + (ci * c.k + i) * c.k + j), want);
    }
    /* y must be exactly the output pixels. */
    T x(RAD_F32, {3, 3}), y(RAD_F32, {8, 9});
    A a; a.t(x).t(y).i("M", 8).i("cin", 1).i("k", 3).i("stride", 1).s("pad", "same").i("up", 1)
          .s("dtype", "f32");
    CHECK_EQ(run("im2col_nhwc", a), RAD_E_SHAPE);
}

/* l2norm_chan: each pixel to unit L2, times sqrt(n), times gamma, then the optional silu; an
 * all-zero pixel stays zero (the 1e-12 floor). */
TEST(l2norm_chan_normalises_each_pixel) {
    T x(RAD_F32, {2, 2}); x.f({3, 4, 0, 0});
    T g(RAD_F32, {2}); g.f({1, 2});
    T y(RAD_F32, {2, 2});
    A a; a.t(x).t(g).t(y).i("M", 2).i("n", 2).s("act", "none").s("dtype", "f32");
    CHECK_OK(run("l2norm_chan", a));
    CHECK_NEAR(y.at(0), 0.6 * std::sqrt(2.0), 1e-6);
    CHECK_NEAR(y.at(1), 1.6 * std::sqrt(2.0), 1e-6);
    CHECK_EQ(y.at(2), 0.0f); CHECK_EQ(y.at(3), 0.0f);
    T ys(RAD_F32, {2, 2});
    A b; b.t(x).t(g).t(ys).i("M", 2).i("n", 2).s("act", "silu").s("dtype", "f32");
    CHECK_OK(run("l2norm_chan", b));
    const double v = 1.6 * std::sqrt(2.0);
    CHECK_NEAR(ys.at(1), v / (1.0 + std::exp(-v)), 1e-6);
}

/* d2s_add deals each input channel group over a 2x2 block; s2d_avg_add is its inverse on that
 * layout. With ft 2 the zero frame the reference pads in counts in the mean. */
TEST(d2s_add_and_s2d_avg_add_are_inverse_shortcuts) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T y(RAD_F32, {2, 2}); y.f({10, 10, 10, 10});
    A a; a.t(x).t(y).i("M", 4).i("cin", 4).i("cout", 1).i("ft", 1).s("dtype", "f32");
    CHECK_OK(run("d2s_add", a));
    CHECK_EQ(y.at(0), 11.0f); CHECK_EQ(y.at(1), 12.0f);
    CHECK_EQ(y.at(2), 13.0f); CHECK_EQ(y.at(3), 14.0f);

    T x8(RAD_F32, {1, 8}); x8.f({0, 0, 0, 0, 5, 6, 7, 8});
    T y8(RAD_F32, {2, 2});
    A a8; a8.t(x8).t(y8).i("M", 4).i("cin", 8).i("cout", 1).i("ft", 2).s("dtype", "f32");
    CHECK_OK(run("d2s_add", a8));
    CHECK_EQ(y8.at(0), 5.0f); CHECK_EQ(y8.at(3), 8.0f);    /* channels 4 + 2i + j */

    T img(RAD_F32, {2, 2}); img.f({1, 2, 3, 4});
    T back(RAD_F32, {1, 4});
    A b; b.t(img).t(back).i("M", 1).i("cin", 1).i("cout", 4).i("ft", 1).s("dtype", "f32");
    CHECK_OK(run("s2d_avg_add", b));
    for (int i = 0; i < 4; ++i) CHECK_EQ(back.at(i), (float)(i + 1));

    T one(RAD_F32, {1, 1});
    A c; c.t(img).t(one).i("M", 1).i("cin", 1).i("cout", 1).i("ft", 2).s("dtype", "f32");
    CHECK_OK(run("s2d_avg_add", c));
    CHECK_EQ(one.at(0), 10.0f / 8.0f);     /* four values and four padded zeros */
}

/* to_rgba8: round(clamp(bf16(bf16(clamp(x, -1, 1) / 2) + 0.5), 0, 1) * 255), half to even. */
TEST(to_rgba8_clamps_halves_and_rounds_to_even) {
    T x(RAD_F32, {1, 5}); x.f({-2, 2, 0, 0.5f, -0.5f});
    T y(RAD_U8, {1, 5});
    A a; a.t(x).t(y).i("M", 1).i("n", 5).s("dtype", "f32");
    CHECK_OK(run("to_rgba8", a));
    CHECK_EQ(y.at(0), 0.0f);
    CHECK_EQ(y.at(1), 255.0f);
    CHECK_EQ(y.at(2), 128.0f);   /* 127.5 to even */
    CHECK_EQ(y.at(3), 191.0f);   /* 191.25 */
    CHECK_EQ(y.at(4), 64.0f);    /* 63.75 */
}

TEST(gelu_tanh_is_the_tanh_approximation) {
    const std::vector<float> xv = { -3, -0.5f, 0, 0.7f, 2.5f };
    T x(RAD_F32, {1, 5}); x.f(xv);
    T y(RAD_F32, {1, 5});
    A a; a.t(x).t(y).i("M", 1).i("n", 5).s("dtype", "f32");
    CHECK_OK(run("gelu_tanh", a));
    for (int i = 0; i < 5; ++i) {
        const double v = xv[(size_t)i];
        CHECK_NEAR(y.at(i), 0.5 * v * (1 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))),
                   1e-6);
    }
}

/* scatter_add_rows adds into the indexed rows, skips a negative index, and refuses an index past
 * the destination before writing anything. */
TEST(scatter_add_rows_accumulates_skips_and_refuses) {
    T v(RAD_F32, {2, 2}); v.f({10, 20, 30, 40});
    T idx(RAD_I32, {2}); idx.n({2, -1});
    T x(RAD_F32, {3, 2}); x.f({1, 1, 2, 2, 3, 3});
    A a; a.t(v).t(idx).t(x).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("scatter_add_rows", a));
    const float want[6] = { 1, 1, 2, 2, 13, 23 };
    for (int i = 0; i < 6; ++i) CHECK_EQ(x.at(i), want[i]);
    T bad(RAD_I32, {2}); bad.n({0, 3});
    A b; b.t(v).t(bad).t(x).i("M", 2).i("n", 2).s("dtype", "f32");
    CHECK_EQ(run("scatter_add_rows", b), RAD_E_INVAL);
    for (int i = 0; i < 6; ++i) CHECK_EQ(x.at(i), want[i]);
}

/* attn_prefix_self: every prefix key is visible, own row i sees own rows j <= i under `causal`.
 * head_dim 1 with zero keys makes every score equal, so each output is the plain mean of the
 * values it may see. */
TEST(attn_prefix_self_sees_the_whole_prefix_and_its_own_causal_rows) {
    const int64_t hd = 1, bs = 2;
    T q(RAD_F32, {2, 1}); q.f({1, 1});
    T k(RAD_F32, {2, 1}); k.f({0, 0});
    T v(RAD_F32, {2, 1}); v.f({20, 40});
    /* kv_cache [blocks 1, kv_heads 1, block 2, 2 * head_dim]: one prefix key, k 0 and v 10. */
    T kv(RAD_F32, {1, 1, bs, 2 * hd}); kv.f({0, 10, 0, 0});
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1}); su.n({1});
    T cu(RAD_I32, {2}); cu.n({0, 2});
    T out(RAD_F32, {2, 1});
    A a; a.t(q).t(k).t(v).t(kv).t(bt).t(su).t(cu).tnull().tnull().tnull().tnull().t(out)
          .i("q_len", 2).i("head_dim", hd).i("gqa", 1).i("block_size", bs).i("causal", 1)
          .s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_prefix_self", a));
    CHECK_EQ(out.at(0), 15.0f);                 /* (10 + 20) / 2 */
    CHECK_NEAR(out.at(1), 70.0 / 3.0, 1e-5);    /* (10 + 20 + 40) / 3 */
    T out2(RAD_F32, {2, 1});
    A b; b.t(q).t(k).t(v).t(kv).t(bt).t(su).t(cu).tnull().tnull().tnull().tnull().t(out2)
          .i("q_len", 2).i("head_dim", hd).i("gqa", 1).i("block_size", bs).i("causal", 0)
          .s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_prefix_self", b));
    CHECK_NEAR(out2.at(0), 70.0 / 3.0, 1e-5);
}

/* quant_cols_fp8: one scale per column over all rows, amax / 448, 1 for an all-zero column.
 * Column 0 [1, -2, 0.5]: scale 2/448, codes 224, -448, 112 (all E4M3 values). Column 1 zero. */
TEST(quant_cols_fp8_scales_each_column_by_its_own_max) {
    T x(RAD_F32, {3, 2}); x.f({1, 0, -2, 0, 0.5f, 0});
    T y(RAD_F8E4M3, {3, 2});
    T s(RAD_F32, {2});
    A a; a.t(x).t(y).t(s).i("M", 3).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("quant_cols_fp8", a));
    CHECK_EQ(s.at(0), 2.0f * (1.0f / 448.0f));
    CHECK_EQ(s.at(1), 1.0f);
    CHECK_EQ(y.at(0), 224.0f); CHECK_EQ(y.at(2), -448.0f); CHECK_EQ(y.at(4), 112.0f);
    CHECK_EQ(y.at(1), 0.0f); CHECK_EQ(y.at(3), 0.0f); CHECK_EQ(y.at(5), 0.0f);
}

/* The fp8 P.V form, prefix-free: w = e4m3(448 p), o = sum(w v) v_scale / (448 sum p). One query
 * against two keys whose scores differ by ln 2: p = 1 and 1/2, w = 448 and 224 exactly, so with
 * value codes 3 and 5 and a column scale of 2, o = 2 (448 3 + 224 5) / (448 1.5) = 2 5.5 / 1.5. */
TEST(attn_prefix_self_fp8_pv_rounds_each_weight_to_e4m3) {
    const float l2 = std::log(2.0f);
    T q(RAD_F32, {3, 1}); q.f({1, 0, 0});
    T k(RAD_F32, {3, 1}); k.f({l2, 0, -9.0f * l2});
    T v(RAD_F8E4M3, {3, 1}); v.f({3, 5, 7});
    T vs(RAD_F32, {1}); vs.f({2});
    T cu(RAD_I32, {2}); cu.n({0, 3});
    T o(RAD_F32, {3, 1});
    A a; a.t(q).t(k).t(v).tnull().tnull().tnull().t(cu).tnull().tnull().tnull().tnull().t(o).t(vs)
          .i("q_len", 3).i("head_dim", 1).i("gqa", 1).i("block_size", 16).i("causal", 1)
          .i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32").d("scale", 1).s("pv", "fp8");
    CHECK_OK(run("attn_prefix_self", a));
    /* Row 0 sees key 0 only: o = 3 2 = 6. Row 1 (q 0) sees keys 0 and 1 at score 0: w 448 each,
     * o = 2 (3 + 5) / 2 = 8. */
    CHECK_NEAR(o.at(0), 6.0f, 1e-5);
    CHECK_NEAR(o.at(1), 8.0f, 1e-5);
    /* Row 2 (q 0) sees all three at score 0. */
    CHECK_NEAR(o.at(2), 2.0f * (3 + 5 + 7) / 3.0f, 1e-5);
    /* Row 0 against keys 0 and 1 directly: scores ln 2 and 0. */
    T q2(RAD_F32, {1, 1}); q2.f({1});
    T k2(RAD_F32, {2, 1}); k2.f({l2, 0});
    T v2(RAD_F8E4M3, {2, 1}); v2.f({3, 5});
    T cu2(RAD_I32, {2}); cu2.n({0, 2});
    T o2(RAD_F32, {2, 1});
    T q3(RAD_F32, {2, 1}); q3.f({1, 1});
    A b; b.t(q3).t(k2).t(v2).tnull().tnull().tnull().t(cu2).tnull().tnull().tnull().tnull().t(o2)
          .t(vs).i("q_len", 2).i("head_dim", 1).i("gqa", 1).i("block_size", 16).i("causal", 0)
          .i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32").d("scale", 1).s("pv", "fp8");
    CHECK_OK(run("attn_prefix_self", b));
    CHECK_NEAR(o2.at(0), 2.0f * 5.5f / 1.5f, 1e-5);
    /* pv fp8 without v_scale, or with a prefix, is refused. */
    A c; c.t(q3).t(k2).t(v2).tnull().tnull().tnull().t(cu2).tnull().tnull().tnull().tnull().t(o2)
          .tnull().i("q_len", 2).i("head_dim", 1).i("gqa", 1).i("block_size", 16)
          .i("causal", 0).i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32")
          .s("pv", "fp8");
    CHECK_EQ(run("attn_prefix_self", c), (int)RAD_E_INVAL);
}

/* With no prefix the op is dense attention over the pass's own rows. */
TEST(attn_prefix_self_with_an_empty_prefix_is_dense_attention) {
    const int64_t M = 3, hd = 4, heads = 2;
    std::vector<float> qv, kvv, vv;
    uint32_t s = 3;
    auto u = [&]() { s = s * 1664525u + 1013904223u; return (float)((int)(s >> 20) % 33 - 16) / 8.0f; };
    for (int64_t i = 0; i < M * heads * hd; ++i) { qv.push_back(u()); kvv.push_back(u()); vv.push_back(u()); }
    T q(RAD_F32, {M, heads * hd}); q.f(qv);
    T k(RAD_F32, {M, heads * hd}); k.f(kvv);
    T v(RAD_F32, {M, heads * hd}); v.f(vv);
    T kv(RAD_F32, {1, heads, 2, 2 * hd});
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T su(RAD_I32, {1}); su.n({0});
    T cu(RAD_I32, {2}); cu.n({0, M});
    T o1(RAD_F32, {M, heads * hd}), o2(RAD_F32, {M, heads * hd});
    A a; a.t(q).t(k).t(v).t(kv).t(bt).t(su).t(cu).tnull().tnull().tnull().tnull().t(o1)
          .i("q_len", M).i("head_dim", hd).i("gqa", 1).i("block_size", 2).i("causal", 1)
          .s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_OK(run("attn_prefix_self", a));
    A b; b.t(q).t(k).t(v).t(cu).t(o2).i("M", M).i("head_dim", hd).i("gqa", 1).i("causal", 1)
          .s("dtype", "f32");
    CHECK_OK(run("attn_dense", b));
    for (int64_t i = 0; i < M * heads * hd; ++i) CHECK_NEAR(o1.at(i), o2.at(i), 1e-5);

    /* The prefix-free form (no cache, block table or prefix lengths) is the same attention, causal
     * or not; one or two of the three prefix operands alone are refused. */
    for (int causal = 0; causal < 2; ++causal) {
        T o3(RAD_F32, {M, heads * hd}), o4(RAD_F32, {M, heads * hd});
        A c; c.t(q).t(k).t(v).tnull().tnull().tnull().t(cu).tnull().tnull().tnull().tnull().t(o3)
              .i("q_len", M).i("head_dim", hd).i("gqa", 1).i("block_size", 2)
              .i("causal", causal).i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32");
        CHECK_OK(run("attn_prefix_self", c));
        A d; d.t(q).t(k).t(v).t(cu).t(o4).i("M", M).i("head_dim", hd).i("gqa", 1)
              .i("causal", causal).s("dtype", "f32");
        CHECK_OK(run("attn_dense", d));
        for (int64_t i = 0; i < M * heads * hd; ++i) CHECK_NEAR(o3.at(i), o4.at(i), 1e-5);
    }
    T o5(RAD_F32, {M, heads * hd});
    A e; e.t(q).t(k).t(v).t(kv).tnull().tnull().t(cu).tnull().tnull().tnull().tnull().t(o5)
          .i("q_len", M).i("head_dim", hd).i("gqa", 1).i("block_size", 2).i("causal", 0)
          .s("q_dtype", "f32").s("kv_dtype", "f32");
    CHECK_EQ(run("attn_prefix_self", e), (int)RAD_E_INVAL);
}

/* ================================================================== video generation
 * docs/OPS.md "Video generation": MiniMax-H3's DiT norm and sampler step, and its decoder's patch
 * unfold, tiles, stitch and positions. */

/* rms_mod: with `gated` the branch first, x = bf16(x + bf16(g * y)) (no tanh); then
 * n1 = bf16(x rsqrt(mean(x^2) + eps)), n2 = bf16(n1 w), m = bf16(n2 bf16(1 + scale)),
 * z = bf16(m + shift), each row of `s` [shift | scale].
 *   Row 0, sequence 0: x [1, -1] gains [0.5 * 2, 0.25 * -4], so it is [2, -2], rms 2; n1 [1, -1],
 *   n2 [1.5, -0.75], times 1 + [0.5, 0.25], plus [0.125, 1]: [2.375, 0.0625]. The halves read the
 *   other way round would give [2.1875, -1.25].
 *   Row 1, sequence 1, gate 0: x [1, 2], rms sqrt(2.5); n1 = bf16(0.63245553) = 0.6328125 and
 *   bf16(1.2649111) = 1.265625; n2 0.94921875 twice; 1 + 2^-8 rounds to 1 (a tie, to even); plus
 *   [0, 0.5]: [0.94921875, 1.453125]. One f32 chain rounded at the end gives
 *   [0.953125, 1.4453125]. */
TEST(rms_mod_rounds_each_step_and_reads_shift_then_scale) {
    T y(RAD_BF16, {2, 2}); y.f({2, -4, 7, 7});
    T g(RAD_BF16, {2, 2}); g.f({0.5f, 0.25f, 0, 0});
    T s(RAD_BF16, {2, 4}); s.f({0.125f, 1, 0.5f, 0.25f,  0, 0.5f, 0.00390625f, 0});
    T w(RAD_BF16, {2}); w.f({1.5f, 0.75f});
    T cu(RAD_I32, {3}); cu.n({0, 1, 2});
    T x(RAD_BF16, {2, 2}); x.f({1, -1, 1, 2});
    T z(RAD_BF16, {2, 2});
    A a; a.t(y).t(g).t(s).t(w).t(cu).t(x).t(z).i("M", 2).i("n", 2).d("eps", 0).i("gated", 1)
          .s("dtype", "bf16");
    CHECK_OK(run("rms_mod", a));
    CHECK_EQ(x.at(0), 2.0f); CHECK_EQ(x.at(1), -2.0f);
    CHECK_EQ(x.at(2), 1.0f); CHECK_EQ(x.at(3), 2.0f);
    CHECK_EQ(z.at(0), 2.375f); CHECK_EQ(z.at(1), 0.0625f);
    CHECK_EQ(z.at(2), 0.94921875f); CHECK_EQ(z.at(3), 1.453125f);

    /* Ungated, without the branch's operands: the residual is normed as it is, and kept. */
    T x2(RAD_BF16, {2, 2}); x2.f({1, -1, 1, 2});
    T z2(RAD_BF16, {2, 2});
    A b; b.tnull().tnull().t(s).t(w).t(cu).t(x2).t(z2).i("M", 2).i("n", 2).d("eps", 0)
          .s("dtype", "bf16");
    CHECK_OK(run("rms_mod", b));
    CHECK_EQ(x2.at(0), 1.0f); CHECK_EQ(x2.at(1), -1.0f);
    CHECK_EQ(z2.at(0), 2.375f); CHECK_EQ(z2.at(1), 0.0625f);
    CHECK_EQ(z2.at(2), 0.94921875f); CHECK_EQ(z2.at(3), 1.453125f);

    /* Gated without a branch is refused. */
    A c; c.tnull().tnull().t(s).t(w).t(cu).t(x2).t(z2).i("M", 2).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_EQ(run("rms_mod", c), RAD_E_SHAPE);
}

/* The optional column map: sequences that share a modulation cell read one table row, and the
 * gate and table may hold fewer rows than there are sequences. A mapped run must equal the same
 * computation with the rows laid out in sequence order and no map. */
TEST(rms_mod_column_map_redirects_sequences_to_table_columns) {
    const std::vector<float> yv = { 2, -4, 2, -4, 2, -4 };
    T w(RAD_BF16, {2}); w.f({1.5f, 0.75f});
    T cu(RAD_I32, {3}); cu.n({0, 1, 3});         /* row 0 is sequence 0, rows 1-2 sequence 1 */
    /* Mapped: columns {B, A} read by sequences {1, 0} -- A's shift/scale/gate land on sequence 0,
     * B's on the rest, exactly the identity run's pairing with the table rows swapped. */
    T sA(RAD_BF16, {2, 4}); sA.f({0, 0.5f, 0.00390625f, 0,   0.125f, 1, 0.5f, 0.25f});
    T gA(RAD_BF16, {2, 2}); gA.f({0, 0,  0.5f, 0.25f});
    T cm(RAD_I32, {2}); cm.n({1, 0});
    T y(RAD_BF16, {3, 2}); y.f(yv);
    T x1(RAD_BF16, {3, 2}); x1.f({1, -1, 1, -1, 1, -1});
    T z1(RAD_BF16, {3, 2});
    A a; a.t(y).t(gA).t(sA).t(w).t(cu).t(x1).t(z1).t(cm).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_OK(run("rms_mod", a));
    CHECK_EQ(x1.at(0), 2.0f); CHECK_EQ(x1.at(1), -2.0f);          /* sequence 0: A's gate */
    CHECK_EQ(z1.at(0), 2.375f); CHECK_EQ(z1.at(1), 0.0625f);      /* sequence 0: A's row */
    CHECK_EQ(z1.at(2), 1.5f);   CHECK_EQ(z1.at(3), -0.25f);       /* rows 1-2: B, ungated */
    CHECK_EQ(z1.at(4), 1.5f);   CHECK_EQ(z1.at(5), -0.25f);

    /* The same output with no map, columns in sequence order. */
    T sB(RAD_BF16, {2, 4}); sB.f({0.125f, 1, 0.5f, 0.25f,  0, 0.5f, 0.00390625f, 0});
    T gB(RAD_BF16, {2, 2}); gB.f({0.5f, 0.25f,  0, 0});
    T x2(RAD_BF16, {3, 2}); x2.f({1, -1, 1, -1, 1, -1});
    T z2(RAD_BF16, {3, 2});
    A b; b.t(y).t(gB).t(sB).t(w).t(cu).t(x2).t(z2).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_OK(run("rms_mod", b));
    CHECK(x1.mem == x2.mem);
    CHECK(z1.mem == z2.mem);

    /* One column for two sequences: the map makes a smaller-than-sequences gate and table legal,
     * and without the map the same call is refused. */
    T s1(RAD_BF16, {1, 4}); s1.f({0.125f, 1, 0.5f, 0.25f});
    T g1(RAD_BF16, {1, 2}); g1.f({0.5f, 0.25f});
    T cm1(RAD_I32, {2}); cm1.n({0, 0});
    T x3(RAD_BF16, {3, 2}); x3.f({1, -1, 1, -1, 1, -1});
    T z3(RAD_BF16, {3, 2});
    A d; d.t(y).t(g1).t(s1).t(w).t(cu).t(x3).t(z3).t(cm1).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_OK(run("rms_mod", d));
    for (int64_t r = 0; r < 3; ++r) {
        CHECK_EQ(z3.at(2 * r), 2.375f);
        CHECK_EQ(z3.at(2 * r + 1), 0.0625f);
    }
    A e; e.t(y).t(g1).t(s1).t(w).t(cu).t(x3).t(z3).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_EQ(run("rms_mod", e), RAD_E_SHAPE);

    /* A map that is not i32, or shorter than the sequence count, is refused. */
    T cmf(RAD_F32, {2}); cmf.f({1, 0});
    T x4(RAD_BF16, {3, 2}); x4.f({1, -1, 1, -1, 1, -1});
    T z4(RAD_BF16, {3, 2});
    A f; f.t(y).t(gA).t(sA).t(w).t(cu).t(x4).t(z4).t(cmf).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_EQ(run("rms_mod", f), RAD_E_SHAPE);
    T cms(RAD_I32, {1}); cms.n({1});
    A g; g.t(y).t(gA).t(sA).t(w).t(cu).t(x4).t(z4).t(cms).i("M", 3).i("n", 2).d("eps", 0)
          .i("gated", 1).s("dtype", "bf16");
    CHECK_EQ(run("rms_mod", g), RAD_E_SHAPE);
}

/* flow_x0_step in f32 steps: st = 1 - 0.3f, r = 0.4f / 0.7f, d = x + st v, x' = r x + (1 - r) d.
 * From x 1, v 3 that is 1.8999999f (0x3ff33333), where the algebraically equal x + (1 - r) st v
 * gives 1.9f; from x -2.5, v 0.75 it is -2.275f. */
TEST(flow_x0_step_blends_toward_the_clean_estimate_in_f32_steps) {
    T v(RAD_F32, {1, 2}); v.f({3, 0.75f});
    T t(RAD_F32, {1}); t.f({0.3f});
    T sg(RAD_F32, {1}); sg.f({0.7f});
    T sn(RAD_F32, {1}); sn.f({0.4f});
    T x(RAD_F32, {1, 2}); x.f({1, -2.5f});
    A a; a.t(v).t(t).t(sg).t(sn).t(x).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("flow_x0_step", a));
    CHECK_EQ(x.at(0), 1.8999999f);
    CHECK(x.at(0) != 1.9f);
    CHECK_EQ(x.at(1), -2.275f);
}

/* patch_unfold: patch rows (f, i, j) of columns (c, a, b, d) dealt to voxel rows (t, yy, xx) and
 * denormalised, f32(f32(x mul[c]) + add[c]) with the lists in f32. Two 1 x 2 x 2 patches side by
 * side (geom (1, 1, 2)), x[p][col] = 100 p + col: voxel row (yy, xx) reads patch xx / 2 at column
 * c 4 + yy 2 + xx % 2. mul 0.3 is f32(0.3), not bf16's 0.30078125 (which would put the first
 * value at -1.796875). Row 8 is past the voxels and keeps its value. */
TEST(patch_unfold_deals_patches_to_voxels_and_denormalises) {
    T x(RAD_F32, {2, 8});
    std::vector<float> xv(16);
    for (int p = 0; p < 2; ++p)
        for (int c = 0; c < 8; ++c) xv[(size_t)(p * 8 + c)] = (float)(100 * p + c);
    x.f(xv);
    T geom(RAD_I32, {3}); geom.n({1, 1, 2});
    T y(RAD_F32, {9, 2}); y.f(std::vector<float>(18, -9.0f));
    A a; a.t(x).t(geom).t(y).i("M", 9).i("C", 2).i("pt", 1).i("ph", 2).i("pw", 2)
          .s("mul", "2 0.3").s("add", "1 -3").s("dtype", "f32");
    CHECK_OK(run("patch_unfold", a));
    const float c0[8] = { 1, 3, 201, 203, 5, 7, 205, 207 };
    const float c1[8] = { -1.8f, -1.5f, 28.2f, 28.5f, -1.2f, -0.9f, 28.8f, 29.1f };
    for (int r = 0; r < 8; ++r) {
        CHECK_EQ(y.at(2 * r), c0[r]);
        CHECK_NEAR(y.at(2 * r + 1), c1[r], 1e-5);
    }
    CHECK_EQ(y.at(16), -9.0f); CHECK_EQ(y.at(17), -9.0f);
}

/* vtile_gather's tiles are diffusers' _split_tiles in latent units. MiniMax-H3's 768 x 1344 frame
 * is 48 x 84 latents; tiles of 16 at least 4 apart: 48 takes 4 tiles (16 4 - 4 3 - 48 = 4 to
 * spare, overlaps [6, 5, 5], starts 0, 10, 21, 32) and 84 takes 7 (overlaps [5, 5, 5, 5, 4, 4],
 * starts 0, 11, 22, 33, 44, 56, 68). x[yy, xx] = 1000 yy + xx; each tile is 256 voxels, a register
 * row and a zero row. */
TEST(vtile_gather_splits_as_diffusers_tiles) {
    const int64_t H = 48, W = 84, period = 258;
    T x(RAD_F32, {H * W, 1});
    std::vector<float> xv((size_t)(H * W));
    for (int64_t i = 0; i < H * W; ++i) xv[(size_t)i] = (float)(1000 * (i / W) + i % W);
    x.f(xv);
    T reg(RAD_F32, {1, 1}); reg.f({0.5f});
    T geom(RAD_I32, {3}); geom.n({1, H, W});
    T y(RAD_F32, {28 * period, 1});
    A a; a.t(x).t(reg).t(geom).t(y).i("M", 28 * period).i("n", 1).i("tile", 16).i("overlap", 4)
          .i("frames", 1).i("period", period).i("n_reg", 1).s("dtype", "f32");
    CHECK_OK(run("vtile_gather", a));
    const int64_t sy[4] = { 0, 10, 21, 32 }, sx[7] = { 0, 11, 22, 33, 44, 56, 68 };
    for (int64_t i = 0; i < 4; ++i)
        for (int64_t j = 0; j < 7; ++j) {
            const int64_t k = i * 7 + j;
            CHECK_EQ(y.at(k * period), (float)(1000 * sy[i] + sx[j]));
            CHECK_EQ(y.at(k * period + 255), (float)(1000 * (sy[i] + 15) + sx[j] + 15));
            CHECK_EQ(y.at(k * period + 256), 0.5f);
            CHECK_EQ(y.at(k * period + 257), 0.0f);
        }
}

/* A tile's voxels are (t, ty, tx) row-major across its frames. Two frames of 3 x 2 in tiles of
 * 2 over 1: rows split at 0 and 1, the columns are one tile; x holds its row index. */
TEST(vtile_gather_lays_each_tile_frame_major_and_refuses_a_short_axis) {
    T x(RAD_F32, {12, 1}); x.f({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    T reg(RAD_F32, {1, 1}); reg.f({-1});
    T geom(RAD_I32, {3}); geom.n({2, 3, 2});
    T y(RAD_F32, {18, 1});
    A a; a.t(x).t(reg).t(geom).t(y).i("M", 18).i("n", 1).i("tile", 2).i("overlap", 1)
          .i("frames", 2).i("period", 9).i("n_reg", 1).s("dtype", "f32");
    CHECK_OK(run("vtile_gather", a));
    const float want[18] = { 0, 1, 2, 3, 6, 7, 8, 9, -1,   2, 3, 4, 5, 8, 9, 10, 11, -1 };
    for (int i = 0; i < 18; ++i) CHECK_EQ(y.at(i), want[i]);

    /* An axis shorter than a tile, frames other than the parameter's, and y short of the tiles. */
    T g1(RAD_I32, {3}); g1.n({2, 1, 2});
    A b; b.t(x).t(reg).t(g1).t(y).i("M", 18).i("n", 1).i("tile", 2).i("overlap", 1)
          .i("frames", 2).i("period", 9).i("n_reg", 1).s("dtype", "f32");
    CHECK_EQ(run("vtile_gather", b), RAD_E_SHAPE);
    T g2(RAD_I32, {3}); g2.n({3, 3, 2});
    A c; c.t(x).t(reg).t(g2).t(y).i("M", 18).i("n", 1).i("tile", 2).i("overlap", 1)
          .i("frames", 2).i("period", 9).i("n_reg", 1).s("dtype", "f32");
    CHECK_EQ(run("vtile_gather", c), RAD_E_SHAPE);
    T y2(RAD_F32, {17, 1});
    A d; d.t(x).t(reg).t(geom).t(y2).i("M", 17).i("n", 1).i("tile", 2).i("overlap", 1)
          .i("frames", 2).i("period", 9).i("n_reg", 1).s("dtype", "f32");
    CHECK_EQ(run("vtile_gather", d), RAD_E_SHAPE);
}

namespace {

/* A 3 x 3 latent of four frames, tiles of 2 over 1 at scale 2 (6 x 6 pixels, 4 x 4 a tile, kept
 * ranges [0, 2) and [2, 6) on each axis, 2-pixel blends), each tile constant: colour c of clip
 * frame fc in tile (i, j) is base[i][j] + 2 fc + c, base [[4, 8], [16, 32]]. */
struct Stitch {
    T x{RAD_BF16, {4 * 16, 12}};
    T geom{RAD_I32, {5}};
    T prev{RAD_BF16, {2 * 36, 3}};
    Stitch(int first, int last) {
        static const float kBase[4] = { 4, 8, 16, 32 };
        std::vector<float> xv((size_t)(4 * 16 * 12));
        for (int64_t k = 0; k < 4; ++k)
            for (int64_t r = 0; r < 16; ++r)
                for (int64_t col = 0; col < 12; ++col)
                    xv[(size_t)((k * 16 + r) * 12 + col)] = kBase[k] + 2.0f * (float)(r / 4) +
                                                            (float)(col / 4);
        x.f(xv);
        geom.n({4, 3, 3, first, last});
        std::vector<float> pv((size_t)(2 * 36 * 3));
        for (size_t i = 0; i < pv.size(); ++i) pv[i] = i < 36 * 3 ? 1.0f : 2.0f;
        prev.f(pv);
    }
    int run_on(T& y, int64_t frames_out) {
        A a; a.t(x).t(geom).t(prev).t(y).i("M", frames_out * 36).i("tile", 2).i("overlap", 1)
              .i("scale", 2).i("pt", 1).i("skip", 0).i("keep", 2).i("blend", 2).i("period", 16)
              .s("mean", "0 0 0").s("std", "0.015625 0.015625 0.015625");
        return run("vtile_stitch", a);
    }
};

}  /* namespace */

/* vtile_stitch: each pixel belongs to the tile whose kept range holds it; its top `ev` rows are
 * blended with the tile above (as decoded), then its left `eh` columns with the tile to its left
 * (as decoded, not blended), wa = bf16(1 - p/e), wb = bf16(p/e). At clip frame 2 the tiles are
 * [[8, 12], [20, 36]], and pixel (3, 3) is row 1 and column 1 of tile (1, 1): V = 0.5 12 + 0.5 36
 * = 24, then S = 0.5 20 + 0.5 V = 22. Blending across first would give 20, a vertically blended
 * left tile 19, a horizontally blended tile above 21.5.
 *
 * In time (skip 0, keep 2, blend 2): output frame k < blend of a chunk that is not first is
 * bf16(bf16(prev[k] wa(k, 2)) + bf16(S wb(k, 2))): frame 0 is prev[0] = 1, frame 1 at (3, 3) is
 * 0.5 2 + 0.5 S(1) = 1 + 10 = 11 (colour 2: 12). prev becomes S(2), S(3), which a last chunk also
 * writes as frames 2 and 3. A byte is round(255 clamp(v / 64, 0, 1)): 1 -> 4, 11 -> 44, 12 -> 48,
 * 18 -> 72, 20 -> 80, 22 -> 88, 24 -> 96. */
TEST(vtile_stitch_blends_down_then_across_and_carries_the_overlap) {
    Stitch st(0, 1);
    T y(RAD_U8, {4 * 36, 3});
    CHECK_OK(st.run_on(y, 4));
    auto px = [](int64_t f, int64_t Y, int64_t X, int64_t c) {
        return (f * 36 + Y * 6 + X) * 3 + c;
    };
    /* prev now holds clip frames 2 and 3, stitched. */
    const float row2[6] = { 8, 8, 20, 16, 12, 12 }, row3[6] = { 14, 14, 20, 22, 24, 24 };
    for (int64_t X = 0; X < 6; ++X) {
        CHECK_EQ(st.prev.at(px(0, 2, X, 0)), row2[X]);
        CHECK_EQ(st.prev.at(px(0, 3, X, 0)), row3[X]);
    }
    CHECK_EQ(st.prev.at(px(1, 3, 3, 0)), 24.0f);
    CHECK_EQ(y.at(px(0, 3, 3, 0)), 4.0f);
    CHECK_EQ(y.at(px(0, 0, 0, 2)), 4.0f);
    CHECK_EQ(y.at(px(1, 3, 3, 0)), 44.0f);
    CHECK_EQ(y.at(px(1, 3, 3, 2)), 48.0f);
    CHECK_EQ(y.at(px(1, 5, 5, 0)), 72.0f);
    CHECK_EQ(y.at(px(2, 3, 3, 0)), 88.0f);
    CHECK_EQ(y.at(px(3, 3, 3, 0)), 96.0f);

    /* A first chunk does not blend in time, and one that is not last writes only its kept
     * frames: frames 0 and 1 at (3, 3) are S(0) = 18 and S(1) = 20. */
    Stitch s1(1, 0);
    T y1(RAD_U8, {2 * 36, 3});
    CHECK_OK(s1.run_on(y1, 2));
    CHECK_EQ(y1.at(px(0, 3, 3, 0)), 72.0f);
    CHECK_EQ(y1.at(px(1, 3, 3, 0)), 80.0f);
    CHECK_EQ(s1.prev.at(px(0, 3, 3, 0)), 22.0f);

    /* Refused, nothing written: bytes short of the last chunk's four frames. */
    Stitch s2(0, 1);
    T y2(RAD_U8, {2 * 36, 3});
    const std::vector<uint8_t> before = s2.prev.mem;
    CHECK_EQ(s2.run_on(y2, 2), RAD_E_SHAPE);
    CHECK(s2.prev.mem == before);
}

/* grid_pos3: token (t, i, j) of each `period`-row tile at scale (2 (c + 0.5) / n - 1) per axis, in
 * f32 steps, the period's last rows (registers) at 0. Tiles of 2 x 3 x 4 every 26 rows, scale 2,
 * into a [3, 30] view of a [3, 32] plane: rows 30 and 31 of each component are not written.
 * 2 g(0, 3) = 2 f32(f32(2 f32(0.5 / 3)) - 1) = -1.3333333f. */
TEST(grid_pos3_places_each_tile_token_and_zeroes_the_rest) {
    T y(RAD_F32, {3, 32}); y.f(std::vector<float>(96, -9.0f));
    y.t.shape[1] = 30;
    A a; a.t(y).i("M", 30).i("T", 2).i("H", 3).i("W", 4).i("period", 26).d("scale", 2.0);
    CHECK_OK(run("grid_pos3", a));
    auto at3 = [&](int c, int m) { return y.at(c * 32 + m); };
    CHECK_EQ(at3(0, 0), -1.0f);  CHECK_EQ(at3(1, 0), -1.3333333f);  CHECK_EQ(at3(2, 0), -1.5f);
    CHECK_EQ(at3(0, 5), -1.0f);  CHECK_EQ(at3(1, 5), 0.0f);          CHECK_EQ(at3(2, 5), -0.5f);
    CHECK_EQ(at3(0, 23), 1.0f);  CHECK_EQ(at3(1, 23), 1.3333333f);   CHECK_EQ(at3(2, 23), 1.5f);
    for (int c = 0; c < 3; ++c) {
        CHECK_EQ(at3(c, 24), 0.0f); CHECK_EQ(at3(c, 25), 0.0f);
        CHECK_EQ(at3(c, 26), at3(c, 0));
        CHECK_EQ(at3(c, 30), -9.0f); CHECK_EQ(at3(c, 31), -9.0f);
    }
    CHECK_EQ(at3(2, 29), 1.5f);
}

/* ================================================================== the tiled encoder
 * docs/OPS.md "Video generation", the tiled encoder: vae.py's split over a one-frame plane, the
 * plane cut into tiles, and the per-tile moments blended back with the RAW neighbours. */

/* 640 x 320 px in latent units is 40 x 20: the rows split at 0, 12, 24 and the columns at 0, 4
 * -- six tiles, every one 16 latents a side (256 px). Level L's volume is the tile over
 * 2^min(L,4) pixels a side, and a plane smaller than one tile is one short tile (vae.py's
 * ([0], [len], [])), 128 x 96 px here with levels down to 8 x 6. */
TEST(vtile_plan_plans_one_frame_the_way_vae_py_splits_it) {
    T geom(RAD_I32, {3}); geom.n({1, 640, 320});
    T y(RAD_I32, {6, 22});
    A a; a.t(geom).t(y).i("M", 6).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_OK(run("vtile_plan", a));
    const int64_t y0[6] = { 0, 0, 192, 192, 384, 384 }, x0[6] = { 0, 64, 0, 64, 0, 64 };
    for (int64_t t = 0; t < 6; ++t) {
        CHECK_EQ(y.iat(t * 22 + 0), y0[t]);
        CHECK_EQ(y.iat(t * 22 + 1), x0[t]);
        CHECK_EQ(y.iat(t * 22 + 2), 256);
        CHECK_EQ(y.iat(t * 22 + 3), 256);
        const int64_t d[6] = { 1, 2, 4, 8, 16, 16 };
        for (int L = 0; L < 6; ++L) {
            CHECK_EQ(y.iat(t * 22 + 4 + 3 * L + 0), 1);
            CHECK_EQ(y.iat(t * 22 + 4 + 3 * L + 1), 256 / d[L]);
            CHECK_EQ(y.iat(t * 22 + 4 + 3 * L + 2), 256 / d[L]);
        }
    }
    T g1(RAD_I32, {3}); g1.n({1, 128, 96});
    T y1(RAD_I32, {1, 22});
    A b; b.t(g1).t(y1).i("M", 1).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_OK(run("vtile_plan", b));
    const int64_t lone[22] = { 0, 0, 128, 96,   1, 128, 96,   1, 64, 48,   1, 32, 24,
                               1, 16, 12,   1, 8, 6,   1, 8, 6 };
    for (int i = 0; i < 22; ++i) CHECK_EQ(y1.iat(i), lone[i]);

    /* A 96 x 768 px plane: one short row band (6 latents) and 48 latents a row, split at 0, 10,
     * 21, 32 -- the slack of 4 going to the overlaps one at a time round-robin (6, 5, 5),
     * so the last column still runs the full 16 latents to the plane's edge. A short band only
     * ever stands alone on its axis, but it still has the other axis' neighbours. */
    T g3(RAD_I32, {3}); g3.n({1, 96, 768});
    T y3(RAD_I32, {4, 22});
    A e; e.t(g3).t(y3).i("M", 4).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_OK(run("vtile_plan", e));
    const int64_t x3[4] = { 0, 160, 336, 512 };
    for (int64_t t = 0; t < 4; ++t) {
        CHECK_EQ(y3.iat(t * 22 + 0), 0);
        CHECK_EQ(y3.iat(t * 22 + 1), x3[t]);
        CHECK_EQ(y3.iat(t * 22 + 2), 96);
        CHECK_EQ(y3.iat(t * 22 + 3), 256);
        const int64_t d[6] = { 1, 2, 4, 8, 16, 16 };
        for (int L = 0; L < 6; ++L) {
            CHECK_EQ(y3.iat(t * 22 + 4 + 3 * L + 0), 1);
            CHECK_EQ(y3.iat(t * 22 + 4 + 3 * L + 1), 96 / d[L]);
            CHECK_EQ(y3.iat(t * 22 + 4 + 3 * L + 2), 256 / d[L]);
        }
    }

    /* A row count that is not the tile count, and a plane the ratio does not divide. */
    T y2(RAD_I32, {5, 22});
    A c; c.t(geom).t(y2).i("M", 5).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_EQ(run("vtile_plan", c), RAD_E_SHAPE);
    T g2(RAD_I32, {3}); g2.n({1, 100, 96});
    A d; d.t(g2).t(y1).i("M", 1).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_EQ(run("vtile_plan", d), RAD_E_SHAPE);

    /* The engine binds M to the band's bound (m.tiles), never to this call: a ceiling above the
     * tiles plans the same rows, and a y that cannot hold them is still refused. */
    T yb(RAD_I32, {6, 22});
    A f; f.t(geom).t(yb).i("M", 33).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_OK(run("vtile_plan", f));
    for (int64_t t = 0; t < 6; ++t) CHECK_EQ(yb.iat(t * 22 + 0), y0[t]);
    T ys(RAD_I32, {5, 22});
    A g; g.t(geom).t(ys).i("M", 33).i("tile", 16).i("overlap", 4).i("ratio", 16);
    CHECK_EQ(run("vtile_plan", g), RAD_E_SHAPE);
}

/* A 20 x 20 px plane in ratio 4, tiles of 3 latents (12 px) with overlap 1: each axis splits at
 * 0 and 2 -- four tiles of 12 x 12 px, 576 rows, tile-major (row band first). x holds
 * (Y, X, Y W + X) per row, so every tile row says which plane pixel it cut. */
TEST(vtile_crop_cuts_the_plane_tile_major) {
    T x(RAD_F32, {400, 3});
    std::vector<float> xv(1200);
    for (int64_t i = 0; i < 400; ++i)
        xv[(size_t)i * 3] = (float)(i / 20), xv[(size_t)i * 3 + 1] = (float)(i % 20),
        xv[(size_t)i * 3 + 2] = (float)i;
    x.f(xv);
    T geom(RAD_I32, {3}); geom.n({1, 20, 20});
    T y(RAD_F32, {576, 3});
    A a; a.t(x).t(geom).t(y).i("M", 576).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_OK(run("vtile_crop", a));
    const int64_t sy[2] = { 0, 8 }, sx[2] = { 0, 8 };   /* latent starts 0 and 2, in px */
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 2; ++j) {
            const int64_t off = (i * 2 + j) * 144;
            for (int64_t k = 0; k < 144; ++k) {
                const int64_t py = sy[i] + k / 12, pxx = sx[j] + k % 12;
                CHECK_EQ(y.at((off + k) * 3 + 0), (float)py);
                CHECK_EQ(y.at((off + k) * 3 + 1), (float)pxx);
                CHECK_EQ(y.at((off + k) * 3 + 2), (float)(py * 20 + pxx));
            }
        }

    /* A plane whose row axis is below one tile and whose column axis is not: one short row band
     * with two column neighbours -- 8 x 20 px in ratio 4 is 2 x 5 latents, columns at 0 and 2
     * (8 px), so two tiles of 8 x 12 px, 192 rows, the overlap column range cut twice. */
    T g4(RAD_I32, {3}); g4.n({1, 8, 20});
    T x4(RAD_F32, {160, 3});
    std::vector<float> x4v(480);
    for (int64_t i = 0; i < 160; ++i)
        x4v[(size_t)i * 3] = (float)(i / 20), x4v[(size_t)i * 3 + 1] = (float)(i % 20),
        x4v[(size_t)i * 3 + 2] = (float)i;
    x4.f(x4v);
    T y4(RAD_F32, {192, 3});
    A e4; e4.t(x4).t(g4).t(y4).i("M", 192).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_OK(run("vtile_crop", e4));
    for (int64_t j = 0; j < 2; ++j)
        for (int64_t k = 0; k < 96; ++k) {
            const int64_t py = k / 12, pxx = 8 * j + k % 12;
            CHECK_EQ(y4.at((j * 96 + k) * 3 + 0), (float)py);
            CHECK_EQ(y4.at((j * 96 + k) * 3 + 1), (float)pxx);
            CHECK_EQ(y4.at((j * 96 + k) * 3 + 2), (float)(py * 20 + pxx));
        }

    /* A plane below one tile is a lone tile -- the crop is the plane itself. */
    T g1(RAD_I32, {3}); g1.n({1, 8, 8});
    T x1(RAD_F32, {64, 3});
    std::vector<float> x1v(192);
    for (int64_t i = 0; i < 64; ++i) x1v[(size_t)i * 3] = (float)i;
    x1.f(x1v);
    T y1(RAD_F32, {64, 3});
    A b; b.t(x1).t(g1).t(y1).i("M", 64).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_OK(run("vtile_crop", b));
    for (int64_t i = 0; i < 64; ++i) CHECK_EQ(y1.at(i * 3), (float)i);

    /* A row count that is not the tiles' pixel sum, and a plane short of geom. */
    T y2(RAD_F32, {575, 3});
    A c; c.t(x).t(geom).t(y2).i("M", 575).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_EQ(run("vtile_crop", c), RAD_E_SHAPE);
    T x2(RAD_F32, {399, 3});
    A d; d.t(x2).t(geom).t(y).i("M", 576).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_EQ(run("vtile_crop", d), RAD_E_SHAPE);

    /* The engine binds M to the band's bound, never to this call: a ceiling above the cut copies
     * the same rows, and a y that cannot hold the tiles is still refused. */
    T yb(RAD_F32, {576, 3});
    A f; f.t(x).t(geom).t(yb).i("M", 33 * 256 * 256).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_OK(run("vtile_crop", f));
    CHECK_EQ(yb.at(575 * 3 + 2), (float)399);
    T ys(RAD_F32, {575, 3});
    A g; g.t(x).t(geom).t(ys).i("M", 33 * 256 * 256).i("tile", 3).i("overlap", 1).i("ratio", 4);
    CHECK_EQ(run("vtile_crop", g), RAD_E_SHAPE);
}

/* A 6 x 6 latent canvas, tiles of 4 with overlap 2: each axis splits at 0 and 2, the kept ranges
 * are [0, 2) and [2, 6), and the blends are two rows and two columns wide. Tile (i, j) is the
 * constant 4 << (2 i + j) -- [[4, 8], [16, 32]] -- so every blend shows its two sources: (2, 0)
 * is the top row of the bottom-left tile fully from the tile above (wb(0, 2) = 0), (3, 0) is its
 * second row, half and half; (2, 2) is the bottom-right tile's corner blended up (8) and then
 * left (16 -- the RAW tile to the left, never a blended one), and (3, 3) takes half of 8 up and
 * then half of 16 left of that: rd(rd(8 0.5) + rd(32 0.5)) = 20, then rd(rd(16 0.5) + rd(20 0.5))
 * = 18. Blending across first, or reading a blended neighbour, gives other numbers. */
TEST(vtile_join_blends_raw_neighbours_down_then_across) {
    T x(RAD_BF16, {64, 1});
    std::vector<float> xv(64);
    const float base[4] = { 4, 8, 16, 32 };
    for (int64_t k = 0; k < 4; ++k)
        for (int64_t r = 0; r < 16; ++r) xv[(size_t)(k * 16 + r)] = base[k];
    x.f(xv);
    T geom(RAD_I32, {3}); geom.n({1, 6, 6});
    T y(RAD_BF16, {36, 1});
    A a; a.t(x).t(geom).t(y).i("M", 36).i("tile", 4).i("overlap", 2).i("n", 1).s("dtype", "bf16");
    CHECK_OK(run("vtile_join", a));
    auto px = [&](int64_t Y, int64_t X) { return y.at(Y * 6 + X); };
    CHECK_EQ(px(0, 0), 4.0f);    CHECK_EQ(px(0, 2), 4.0f);   /* left blend at wb(0, 2) = 0 */
    CHECK_EQ(px(0, 3), 6.0f);    CHECK_EQ(px(2, 0), 4.0f);   /* top blend at wa(0, 2) = 1 */
    CHECK_EQ(px(2, 2), 16.0f);   CHECK_EQ(px(3, 0), 10.0f);
    CHECK_EQ(px(4, 3), 24.0f);   CHECK_EQ(px(3, 3), 18.0f);
    CHECK_EQ(px(5, 5), 32.0f);   CHECK_EQ(px(4, 0), 16.0f);

    /* A 6 x 9 latent canvas, tiles of 4 with overlap 1: rows split at 0, 2 (slack 1 -> overlap
     * 2) and columns at 0, 2, 5 (slack 1 to the FIRST boundary: overlaps 2 then 1 -- distributing
     * the slack to every boundary instead would start the columns at 0, 2, 4). Six tiles of 4 x 4,
     * tile
     * (i, j) the constant 1 << (2 i + j) -- [[1, 2, 4], [8, 16, 32]]. Top before left, RAW
     * neighbours, in bf16 halves: (3, 0) is half of 1 up and half of 8 = 4.5; (3, 5) blends up to
     * half of 4 and half of 8 = 6 and then fully to the RAW left 16; (1, 2) has no top (i = 0)
     * and takes 1 fully left; (5, 3) takes half of 8 left of 16 = 12; (2, 5) takes 4 fully up,
     * then 16 fully left; (3, 6) takes half of 4 up of its 32 = 18 and no left at lx = 1.
     */
    T g5(RAD_I32, {3}); g5.n({1, 6, 9});
    T x5(RAD_BF16, {96, 1});
    std::vector<float> x5v(96);
    for (int64_t t = 0; t < 6; ++t)
        for (int64_t r = 0; r < 16; ++r) x5v[(size_t)t * 16 + r] = (float)(1 << t);
    x5.f(x5v);
    T y5(RAD_BF16, {54, 1});
    A e5; e5.t(x5).t(g5).t(y5).i("M", 54).i("tile", 4).i("overlap", 1).i("n", 1).s("dtype", "bf16");
    CHECK_OK(run("vtile_join", e5));
    auto qx = [&](int64_t Y, int64_t X) { return y5.at(Y * 9 + X); };
    CHECK_EQ(qx(0, 0), 1.0f);    CHECK_EQ(qx(1, 2), 1.0f);    /* no top at i = 0; 1 fully left */
    CHECK_EQ(qx(0, 3), 1.5f);    CHECK_EQ(qx(0, 5), 2.0f);    /* half of 1 left of 2; 2 fully */
    CHECK_EQ(qx(3, 0), 4.5f);    CHECK_EQ(qx(2, 5), 16.0f);   /* half up, then RAW left overrides */
    CHECK_EQ(qx(3, 5), 16.0f);   CHECK_EQ(qx(3, 6), 18.0f);   /* left fully at lx = 0; half up at 1 */
    CHECK_EQ(qx(5, 2), 8.0f);    CHECK_EQ(qx(5, 3), 12.0f);   /* fully left at lx = 0, half at 1 */
    CHECK_EQ(qx(0, 8), 4.0f);    CHECK_EQ(qx(4, 2), 8.0f);    /* past every blend: own or fully left */

    /* A canvas below one tile is a lone tile: the join is the copy. */
    T g1(RAD_I32, {3}); g1.n({1, 3, 3});
    T x1(RAD_BF16, {9, 1});
    std::vector<float> x1v(9);
    for (int64_t i = 0; i < 9; ++i) x1v[(size_t)i] = (float)(i + 1);
    x1.f(x1v);
    T y1(RAD_BF16, {9, 1});
    A b; b.t(x1).t(g1).t(y1).i("M", 9).i("tile", 4).i("overlap", 2).i("n", 1).s("dtype", "bf16");
    CHECK_OK(run("vtile_join", b));
    for (int64_t i = 0; i < 9; ++i) CHECK_EQ(y1.at(i), (float)(i + 1));

    /* M is the canvas, x holds every tile. */
    T y2(RAD_BF16, {35, 1});
    A c; c.t(x).t(geom).t(y2).i("M", 35).i("tile", 4).i("overlap", 2).i("n", 1).s("dtype", "bf16");
    CHECK_EQ(run("vtile_join", c), RAD_E_SHAPE);
    T x2(RAD_BF16, {63, 1});
    A d; d.t(x2).t(geom).t(y).i("M", 36).i("tile", 4).i("overlap", 2).i("n", 1).s("dtype", "bf16");
    CHECK_EQ(run("vtile_join", d), RAD_E_SHAPE);

    /* The engine binds M to the band's bound, never to this call: a ceiling above the canvas
     * blends the same rows, and a y short of the canvas is still refused. */
    T yb(RAD_BF16, {36, 1});
    A f; f.t(x).t(geom).t(yb).i("M", 320 * 320).i("tile", 4).i("overlap", 2).i("n", 1)
             .s("dtype", "bf16");
    CHECK_OK(run("vtile_join", f));
    CHECK_EQ(yb.at(5 * 6 + 5), 32.0f);
    T ys(RAD_BF16, {35, 1});
    A g; g.t(x).t(geom).t(ys).i("M", 320 * 320).i("tile", 4).i("overlap", 2).i("n", 1)
             .s("dtype", "bf16");
    CHECK_EQ(run("vtile_join", g), RAD_E_SHAPE);
}

/* ================================================================== the latent upscaler
 * docs/OPS.md "The latent upscaler": voxel rows (t, i, j) of a T x H x W volume, channels
 * innermost, the volume in `geom`. */

/* conv3d_ndhwc's `same` padding: an all-ones 3 x 3 x 3 kernel over a 3 x 3 x 3 volume of ones
 * counts each voxel's in-volume neighbours, 2 on an axis at an edge and 3 inside, plus the bias:
 * a corner 8.5, the centre 27.5, a face's middle 18.5, an edge's middle 12.5. Refused: an even
 * kernel, and y short of the volume's rows. */
TEST(conv3d_ndhwc_pads_each_axis_with_zeros) {
    T x(RAD_F32, {27, 1}); x.f(std::vector<float>(27, 1.0f));
    T w(RAD_F32, {1, 1, 3, 3, 3}); w.f(std::vector<float>(27, 1.0f));
    T b(RAD_F32, {1}); b.f({0.5f});
    T geom(RAD_I32, {3}); geom.n({3, 3, 3});
    T y(RAD_F32, {27, 1});
    A a; a.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 27).i("cin", 1).i("cout", 1).i("k", 3)
          .i("kt", 3).s("dtype", "f32");
    CHECK_OK(run("conv3d_ndhwc", a));
    CHECK_EQ(y.at(0), 8.5f);      /* (0, 0, 0) */
    CHECK_EQ(y.at(13), 27.5f);    /* (1, 1, 1) */
    CHECK_EQ(y.at(4), 18.5f);     /* (0, 1, 1): the first frame's centre */
    CHECK_EQ(y.at(10), 18.5f);    /* (1, 0, 1): the middle frame's top edge */
    CHECK_EQ(y.at(1), 12.5f);     /* (0, 0, 1) */
    CHECK_EQ(y.at(26), 8.5f);     /* (2, 2, 2) */

    A e; e.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 27).i("cin", 1).i("cout", 1).i("k", 2)
          .i("kt", 3).s("dtype", "f32");
    CHECK_EQ(run("conv3d_ndhwc", e), RAD_E_SHAPE);
    T y2(RAD_F32, {26, 1});
    A s; s.t(x).t(w).t(b).t(geom).tnull().t(y2).i("M", 26).i("cin", 1).i("cout", 1).i("k", 3)
          .i("kt", 3).s("dtype", "f32");
    CHECK_EQ(run("conv3d_ndhwc", s), RAD_E_SHAPE);
}

/* The weight is PyTorch's [cout, cin, kt, k, k] and tap (dt, kh, kw) reads voxel
 * (t + dt - 1, i + kh - 1, j + kw - 1). Two frames of 1 x 3, two channels holding r + 1 at row r;
 * only w[1, 0, 2, 1, 0] = 1 (frame t + 1, column j - 1) and w[1, 1, 0, 1, 2] = 10 (frame t - 1,
 * column j + 1) are set, so output channel 1 at (t, 0, j) is x[(t + 1, j - 1), 0] +
 * 10 x[(t - 1, j + 1), 1] + b[1] and channel 0 is its bias. */
TEST(conv3d_ndhwc_reads_the_weight_as_pytorch_lays_it_out) {
    T x(RAD_F32, {6, 2}); x.f({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    std::vector<float> wv(2 * 2 * 27, 0.0f);
    wv[75] = 1.0f;     /* ((1 * 2 + 0) * 3 + 2) * 9 + 1 * 3 + 0 */
    wv[86] = 10.0f;    /* ((1 * 2 + 1) * 3 + 0) * 9 + 1 * 3 + 2 */
    T w(RAD_F32, {2, 2, 3, 3, 3}); w.f(wv);
    T b(RAD_F32, {2}); b.f({0.5f, -1.0f});
    T geom(RAD_I32, {3}); geom.n({2, 1, 3});
    T y(RAD_F32, {6, 2});
    A a; a.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 6).i("cin", 2).i("cout", 2).i("k", 3)
          .i("kt", 3).s("dtype", "f32");
    CHECK_OK(run("conv3d_ndhwc", a));
    const float want[6] = { -1, 3, 4, 19, 29, -1 };
    for (int r = 0; r < 6; ++r) {
        CHECK_EQ(y.at(2 * r), 0.5f);
        CHECK_EQ(y.at(2 * r + 1), want[r]);
    }
}

/* With `res`, y = bf16(bf16(acc + b) + res): 1 + 0.0087890625 rounds to 1.0078125 first, and
 * 1.0078125 + 2 = 3.0078125 is a tie that goes to 3, where the unrounded sum 3.0087890625 would
 * give 3.015625. -1 + 0.0087890625 rounds to -0.9921875, plus 0.5 is -0.4921875. The residual may
 * be y itself, and adds in place. */
TEST(conv3d_ndhwc_rounds_the_conv_before_the_residual_in_place) {
    T x(RAD_BF16, {2, 1}); x.f({1.0f, -1.0f});
    T w(RAD_BF16, {1, 1, 1, 1, 1}); w.f({1.0f});
    T b(RAD_F32, {1}); b.f({0.0087890625f});
    T geom(RAD_I32, {3}); geom.n({1, 1, 2});
    T y(RAD_BF16, {2, 1}); y.f({2.0f, 0.5f});
    A a; a.t(x).t(w).t(b).t(geom).t(y).t(y).i("M", 2).i("cin", 1).i("cout", 1).i("k", 1)
          .i("kt", 1).s("dtype", "bf16");
    CHECK_OK(run("conv3d_ndhwc", a));
    CHECK_EQ(y.at(0), 3.0f);
    CHECK_EQ(y.at(1), -0.4921875f);

    T res(RAD_BF16, {2, 1}); res.f({2.0f, 0.5f});
    T y2(RAD_BF16, {2, 1});
    A c; c.t(x).t(w).t(b).t(geom).t(res).t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("k", 1)
          .i("kt", 1).s("dtype", "bf16");
    CHECK_OK(run("conv3d_ndhwc", c));
    CHECK(y2.mem == y.mem);
    /* Without a residual, the one rounding: 1.0078125. */
    A d; d.t(x).t(w).t(b).t(geom).tnull().t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("k", 1)
          .i("kt", 1).s("dtype", "bf16");
    CHECK_OK(run("conv3d_ndhwc", d));
    CHECK_EQ(y2.at(0), 1.0078125f);
}

/* The causal video VAE's convolution forms: `stride` and `stride_t` stride after the padding,
 * `causal` 1 pads time in front only, `pad` `down` pads one row and column below and right only,
 * and `reflect` 1 mirrors the spatial padding. Three frames of 2 x 2, causal, spatial stride 2
 * with `down` and `reflect`: one voxel a frame, frame t reading frames t - 2, t - 1 and t. The
 * taps (dt, kh, kw) are w(1,0,0) = 1, w(2,1,1) = 3, w(0,2,2) = 100, w(0,0,2) = 10 and
 * w(0,2,0) = 20; row and column 2 reflect to 0, so the three dt = 0 taps all read (0, 0) and
 * output t is 3 x(t,1,1) + x(t-1,0,0) + 130 x(t-2,0,0) + b, a frame before the first reading
 * zero: 120.5, 250.5 and 3 * 120 + 50 + 130 * 10 + 0.5 = 1710.5. Refused: a stride with a 5-wide
 * kernel, a `pad` neither `same` nor `down`, a reflect pad on an axis of 1, a zero stride, a
 * `causal` neither 0 nor 1, and y short of the strided output. */
TEST(conv3d_ndhwc_causal_forms_stride_mirror_and_pad_front_only) {
    T x(RAD_F32, {12, 1}); x.f({10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120});
    std::vector<float> wv(27, 0.0f);
    wv[9] = 1.0f;     /* (dt 1, kh 0, kw 0): the previous frame's (0, 0) */
    wv[22] = 3.0f;    /* (dt 2, kh 1, kw 1): this frame's (1, 1) */
    wv[8] = 100.0f;   /* (dt 0, kh 2, kw 2): both axes reflect to (0, 0) */
    wv[2] = 10.0f;    /* (dt 0, kh 0, kw 2): the column reflects to 0 */
    wv[6] = 20.0f;    /* (dt 0, kh 2, kw 0): the row reflects to 0 */
    T w(RAD_F32, {1, 1, 3, 3, 3}); w.f(wv);
    T b(RAD_F32, {1}); b.f({0.5f});
    T geom(RAD_I32, {3}); geom.n({3, 2, 2});
    T y(RAD_F32, {3, 1});
    A a; a.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 3).i("cin", 1).i("cout", 1).i("k", 3)
          .i("kt", 3).s("dtype", "f32").i("stride", 2).i("stride_t", 1).i("causal", 1)
          .s("pad", "down").i("reflect", 1);
    CHECK_OK(run("conv3d_ndhwc", a));
    CHECK_EQ(y.at(0), 120.5f);
    CHECK_EQ(y.at(1), 250.5f);
    CHECK_EQ(y.at(2), 1710.5f);

    A e5; e5.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 3).i("cin", 1).i("cout", 1).i("k", 5)
           .i("kt", 3).s("dtype", "f32").i("stride", 2);
    CHECK_EQ(run("conv3d_ndhwc", e5), RAD_E_SHAPE);
    A ep; ep.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 3).i("cin", 1).i("cout", 1).i("k", 3)
            .i("kt", 3).s("dtype", "f32").s("pad", "mirror");
    CHECK_EQ(run("conv3d_ndhwc", ep), RAD_E_SHAPE);
    T g1(RAD_I32, {3}); g1.n({2, 1, 2});
    A eh; eh.t(x).t(w).t(b).t(g1).tnull().t(y).i("M", 4).i("cin", 1).i("cout", 1).i("k", 3)
            .i("kt", 3).s("dtype", "f32").i("causal", 1).i("reflect", 1);
    CHECK_EQ(run("conv3d_ndhwc", eh), RAD_E_SHAPE);
    A e0; e0.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 3).i("cin", 1).i("cout", 1).i("k", 3)
             .i("kt", 3).s("dtype", "f32").i("stride", 0);
    CHECK_EQ(run("conv3d_ndhwc", e0), RAD_E_SHAPE);
    A e2; e2.t(x).t(w).t(b).t(geom).tnull().t(y).i("M", 3).i("cin", 1).i("cout", 1).i("k", 3)
             .i("kt", 3).s("dtype", "f32").i("causal", 2);
    CHECK_EQ(run("conv3d_ndhwc", e2), RAD_E_SHAPE);
    T y2(RAD_F32, {2, 1});
    A es; es.t(x).t(w).t(b).t(geom).tnull().t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("k", 3)
             .i("kt", 3).s("dtype", "f32").i("stride", 2).i("stride_t", 1).i("causal", 1)
             .s("pad", "down").i("reflect", 1);
    CHECK_EQ(run("conv3d_ndhwc", es), RAD_E_SHAPE);
}

/* group_norm's statistics are over every row of a group's channels. Two rows of four channels in
 * two groups: group 0 (channels 0 and 1) holds 1, 1, 3, 3, so mean 2 and var 1 (a per-row mean
 * would make row 0 constant); group 1 holds 8, 12, 12, 8, mean 10 and var 4, rstd 0.5 (eps 0).
 * Normalised, row 0 is (-1, -1, -1, 1) and row 1 (1, 1, 1, -1); times w (1, 2, 3, 4) plus
 * b (0, 0.5, -1, 0). stats gets (2, 1, 10, 0.5) and the rest of it is not written. */
TEST(group_norm_takes_each_group_over_all_rows) {
    T x(RAD_F32, {2, 4}); x.f({1, 1, 8, 12, 3, 3, 12, 8});
    T w(RAD_F32, {4}); w.f({1, 2, 3, 4});
    T b(RAD_F32, {4}); b.f({0, 0.5f, -1, 0});
    T st(RAD_F32, {2 * (2 + 3 * 1024)}); st.f(std::vector<float>(2 * (2 + 3 * 1024), -7.0f));
    T y(RAD_F32, {2, 4});
    A a; a.t(x).t(w).t(b).tnull().t(st).t(y).i("M", 2).i("n", 4).i("groups", 2).d("eps", 0.0)
          .s("act", "none").s("dtype", "f32");
    CHECK_OK(run("group_norm", a));
    const float want[8] = { -1, -1.5f, -4, 4, 1, 2.5f, 2, -4 };
    for (int i = 0; i < 8; ++i) CHECK_EQ(y.at(i), want[i]);
    CHECK_EQ(st.at(0), 2.0f); CHECK_EQ(st.at(1), 1.0f);
    CHECK_EQ(st.at(2), 10.0f); CHECK_EQ(st.at(3), 0.5f);
    CHECK_EQ(st.at(4), -7.0f); CHECK_EQ(st.at(2 * (2 + 3 * 1024) - 1), -7.0f);

    /* Modulated by one row [scale | shift], then silu, in place: v (1 + scale) + shift is
     * (-2, -0.5, -2, 4.25) and (2, 3.5, 1, -3.75), and silu(v) = v / (1 + e^-v). */
    T mod(RAD_F32, {8}); mod.f({1, 0, -0.5f, 0, 0, 1, 0, 0.25f});
    A m; m.t(x).t(w).t(b).t(mod).t(st).t(x).i("M", 2).i("n", 4).i("groups", 2).d("eps", 0.0)
          .s("act", "silu").s("dtype", "f32");
    CHECK_OK(run("group_norm", m));
    const double v[8] = { -2, -0.5, -2, 4.25, 2, 3.5, 1, -3.75 };
    for (int i = 0; i < 8; ++i) CHECK_NEAR(x.at(i), v[i] / (1.0 + std::exp(-v[i])), 1e-5);

    /* Refused: a stats short of groups (2 + 3 1024), groups not dividing n, an unknown act. */
    T sm(RAD_F32, {2 * (2 + 3 * 1024) - 1});
    A r1; r1.t(x).t(w).t(b).tnull().t(sm).t(y).i("M", 2).i("n", 4).i("groups", 2).d("eps", 0.0)
          .s("act", "none").s("dtype", "f32");
    CHECK_EQ(run("group_norm", r1), RAD_E_SHAPE);
    A r2; r2.t(x).t(w).t(b).tnull().t(st).t(y).i("M", 2).i("n", 4).i("groups", 3).d("eps", 0.0)
          .s("act", "none").s("dtype", "f32");
    CHECK_EQ(run("group_norm", r2), RAD_E_SHAPE);
    A r3; r3.t(x).t(w).t(b).tnull().t(st).t(y).i("M", 2).i("n", 4).i("groups", 2).d("eps", 0.0)
          .s("act", "gelu").s("dtype", "f32");
    CHECK_EQ(run("group_norm", r3), RAD_E_INVAL);
}

/* The variance is taken about the mean, not as E[x^2] - mean^2: 4095 and 4097 have mean 4096 and
 * var 1, where an f32 sum of squares (4 4096^2 + 4 = 67108868, 8 apart from its neighbours in
 * f32) cancels to 0. */
TEST(group_norm_keeps_a_small_variance_under_a_large_mean) {
    T x(RAD_F32, {2, 2}); x.f({4095, 4097, 4097, 4095});
    T w(RAD_F32, {2}); w.f({1, 1});
    T b(RAD_F32, {2}); b.f({0, 0});
    T st(RAD_F32, {2 + 3 * 1024});
    T y(RAD_F32, {2, 2});
    A a; a.t(x).t(w).t(b).tnull().t(st).t(y).i("M", 2).i("n", 2).i("groups", 1).d("eps", 0.0)
          .s("act", "none").s("dtype", "f32");
    CHECK_OK(run("group_norm", a));
    const float want[4] = { -1, 1, 1, -1 };
    for (int i = 0; i < 4; ++i) CHECK_EQ(y.at(i), want[i]);
    CHECK_EQ(st.at(0), 4096.0f); CHECK_EQ(st.at(1), 1.0f);
}

/* tconv_dw along the frames: three frames of two rows, channel 0's taps (1, 10, 100) on frames
 * t - 1, t, t + 1 (w[c k + j]) over x = 1..6, channel 1's (0, 1, 0) over -(1..6); frames outside
 * the volume are zero, and each row reads its own position in the other frames. */
TEST(tconv_dw_convolves_each_position_across_frames) {
    T x(RAD_F32, {6, 2}); x.f({1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6});
    T w(RAD_F32, {2, 1, 3, 1, 1}); w.f({1, 10, 100, 0, 1, 0});
    T b(RAD_F32, {2}); b.f({0.5f, 2});
    T geom(RAD_I32, {3}); geom.n({3, 1, 2});
    T y(RAD_F32, {6, 2});
    A a; a.t(x).t(w).t(b).t(geom).t(y).i("M", 6).i("n", 2).i("k", 3).s("dtype", "f32");
    CHECK_OK(run("tconv_dw", a));
    const float c0[6] = { 310.5f, 420.5f, 531.5f, 642.5f, 53.5f, 64.5f };
    for (int r = 0; r < 6; ++r) {
        CHECK_EQ(y.at(2 * r), c0[r]);
        CHECK_EQ(y.at(2 * r + 1), (float)(1 - r));
    }
}

/* resize_bilinear as torch's align_corners=False at a size: 2 x 2 to 4 x 4 gives the familiar
 * [[1, 1.25, 1.75, 2], [1.5, 1.75, 2.25, 2.5], [2.5, 2.75, 3.25, 3.5], [3, 3.25, 3.75, 4]]. The
 * first row's source clamps to 0; the last's is 1.25, whose neighbour is clamped to row 1, so it
 * is row 1. Channel 1 holds the negatives. */
TEST(resize_bilinear_matches_torch_and_clamps_at_the_border) {
    T x(RAD_F32, {4, 2}); x.f({1, -1, 2, -2, 3, -3, 4, -4});
    T geom(RAD_I32, {5}); geom.n({1, 2, 2, 4, 4});
    T y(RAD_F32, {16, 2});
    A a; a.t(x).t(geom).t(y).i("M", 16).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("resize_bilinear", a));
    const float want[16] = { 1, 1.25f, 1.75f, 2, 1.5f, 1.75f, 2.25f, 2.5f,
                             2.5f, 2.75f, 3.25f, 3.5f, 3, 3.25f, 3.75f, 4 };
    for (int r = 0; r < 16; ++r) {
        CHECK_EQ(y.at(2 * r), want[r]);
        CHECK_EQ(y.at(2 * r + 1), -want[r]);
    }
}

/* Frames resize on their own, and shrinking averages: two frames of 4 x 1 to 2 x 1 read source
 * rows 0.5 and 2.5. 3 to 5 at a ratio f32 cannot hold exactly: row 0's source clamps to 0, row
 * 4's (2.2) has its neighbour clamped onto row 2, and rows 1 and 3 read 0.4 and 1.6. */
TEST(resize_bilinear_resizes_each_frame_and_shrinks) {
    T x(RAD_F32, {8, 1}); x.f({1, 2, 4, 8, 10, 20, 40, 80});
    T geom(RAD_I32, {5}); geom.n({2, 4, 1, 2, 1});
    T y(RAD_F32, {4, 1});
    A a; a.t(x).t(geom).t(y).i("M", 4).i("n", 1).s("dtype", "f32");
    CHECK_OK(run("resize_bilinear", a));
    CHECK_EQ(y.at(0), 1.5f); CHECK_EQ(y.at(1), 6.0f);
    CHECK_EQ(y.at(2), 15.0f); CHECK_EQ(y.at(3), 60.0f);

    T x3(RAD_F32, {3, 1}); x3.f({0, 10, 20});
    T g3(RAD_I32, {5}); g3.n({1, 3, 1, 5, 1});
    T y5(RAD_F32, {5, 1});
    A c; c.t(x3).t(g3).t(y5).i("M", 5).i("n", 1).s("dtype", "f32");
    CHECK_OK(run("resize_bilinear", c));
    CHECK_EQ(y5.at(0), 0.0f);
    CHECK_NEAR(y5.at(1), 4.0, 1e-5);
    CHECK_NEAR(y5.at(3), 16.0, 1e-5);
    CHECK_NEAR(y5.at(4), 20.0, 1e-5);
}

/* patch_fold deals voxel rows (yy, xx) of x [8, 2] (x[r, c] = 10 r + c) into the 1 x 2 x 2 patches
 * side by side (geom (1, 1, 2)): patch j's column c 4 + b 2 + d is voxel (b, 2 j + d), times mul
 * plus add, f32. Row 2 is past the patches and keeps its value. */
TEST(patch_fold_gathers_voxels_into_patches_and_normalises) {
    std::vector<float> xv(16);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 2; ++c) xv[(size_t)(r * 2 + c)] = (float)(10 * r + c);
    T x(RAD_F32, {8, 2}); x.f(xv);
    T geom(RAD_I32, {3}); geom.n({1, 1, 2});
    T y(RAD_F32, {3, 8}); y.f(std::vector<float>(24, -9.0f));
    A a; a.t(x).t(geom).t(y).i("M", 3).i("C", 2).i("pt", 1).i("ph", 2).i("pw", 2)
          .s("mul", "1 2").s("add", "0 0.5").s("dtype", "f32");
    CHECK_OK(run("patch_fold", a));
    const float want[16] = { 0, 10, 40, 50, 2.5f, 22.5f, 82.5f, 102.5f,
                             20, 30, 60, 70, 42.5f, 62.5f, 122.5f, 142.5f };
    for (int i = 0; i < 16; ++i) CHECK_EQ(y.at(i), want[i]);
    for (int i = 16; i < 24; ++i) CHECK_EQ(y.at(i), -9.0f);
}

/* patch_fold undoes patch_unfold: two frames of 1 x 2 patches of 2 x 1 x 2 with three channels,
 * unfolded through (x 2 + 1, x 4 - 2, x 0.5 + 3) and folded back through their inverses, exact
 * in f32 at these integers. */
TEST(patch_fold_inverts_patch_unfold) {
    std::vector<float> pv(4 * 12);
    for (size_t i = 0; i < pv.size(); ++i) pv[i] = (float)((int)i * 7 % 23 - 11);
    T p(RAD_F32, {4, 12}); p.f(pv);
    T geom(RAD_I32, {3}); geom.n({2, 1, 2});
    T vox(RAD_F32, {16, 3});
    A u; u.t(p).t(geom).t(vox).i("M", 16).i("C", 3).i("pt", 2).i("ph", 1).i("pw", 2)
          .s("mul", "2 4 0.5").s("add", "1 -2 3").s("dtype", "f32");
    CHECK_OK(run("patch_unfold", u));
    T back(RAD_F32, {4, 12});
    A f; f.t(vox).t(geom).t(back).i("M", 4).i("C", 3).i("pt", 2).i("ph", 1).i("pw", 2)
          .s("mul", "0.5 0.25 2").s("add", "-0.5 0.5 -6").s("dtype", "f32");
    CHECK_OK(run("patch_fold", f));
    for (int i = 0; i < 48; ++i) CHECK_EQ(back.at(i), pv[(size_t)i]);
}

/* flow_renoise: x = f32(f32(0.75 x) + f32(0.25 z(k n + c))) with z the (seed, 1) normal stream,
 * indexed by the row's position k n + c, not its pitch: x is [2, 3] at a pitch of 5, whose last
 * two columns are not written. Both products are exact here, so the expectation has one
 * rounding. Stream 0 (rad_philox_normal) is a different draw. */
TEST(flow_renoise_draws_its_own_stream_by_element) {
    const uint64_t seed = 0x0123456789abcdefull;
    T sg(RAD_F32, {1}); sg.f({0.25f});
    T sd(RAD_U32, {2}); sd.n({(int64_t)(seed & 0xffffffffu), (int64_t)(seed >> 32)});
    const float xv[6] = { 2, 4, -8, 6, 0, 10 };
    T x(RAD_F32, {2, 5});
    x.f({ xv[0], xv[1], xv[2], -9, -9, xv[3], xv[4], xv[5], -9, -9 });
    x.t.shape[1] = 3;
    A a; a.t(sg).t(sd).t(x).i("M", 2).i("n", 3).i("stream", 1).s("dtype", "f32");
    CHECK_OK(run("flow_renoise", a));
    bool differs = false;
    for (int k = 0; k < 2; ++k) {
        for (int c = 0; c < 3; ++c) {
            const uint64_t e = (uint64_t)(k * 3 + c);
            const float z = rad_philox_normal_s(seed, 1, e);
            const float want = 0.75f * xv[k * 3 + c] + 0.25f * z;
            CHECK_EQ(x.at(k * 5 + c), want);
            differs = differs || want != 0.75f * xv[k * 3 + c] + 0.25f * rad_philox_normal(seed, e);
        }
        CHECK_EQ(x.at(k * 5 + 3), -9.0f);
        CHECK_EQ(x.at(k * 5 + 4), -9.0f);
    }
    CHECK(differs);
}

/* cond_aug: x = f32(f32(a x) + f32(oma z(k n + c))) with a = 0.999 the f64 parameter and oma
 * the f32 of the f64 1 - a -- NOT the f32 difference 1.0f - 0.999f, whose low bits differ; the
 * reference subtracts that scalar in python double -- and z the (seed, 2) normal stream. x is
 * [2, 3] at a pitch of 5, last two columns not written. The second call is the caller's
 * per-item restart: item B issued alone with the same (seed, stream) draws from e = 0 again,
 * which a single joined call would not (its row 1 reads k n + c). aug 1 -- the audio twin's
 * disabled spelling -- and a missing aug are refused, not approximated. */
TEST(cond_aug_blends_conditions_with_a_restarted_stream) {
    const uint64_t seed = 0x0000f00dbeef000dull;
    T sd(RAD_U32, {2}); sd.n({(int64_t)(seed & 0xffffffffu), (int64_t)(seed >> 32)});
    const float av = 0.999f, omv = (float)(1.0 - 0.999);
    const float xa[3] = { 1, -2, 3 }, xb[3] = { 0.5f, 0, -4 };
    T x(RAD_F32, {2, 5});
    x.f({ xa[0], xa[1], xa[2], -9, -9, xb[0], xb[1], xb[2], -9, -9 });
    x.t.shape[1] = 3;
    A a1; a1.t(sd).t(x).i("M", 2).i("n", 3).i("stream", 2).d("aug", 0.999).s("dtype", "f32");
    CHECK_OK(run("cond_aug", a1));
    for (int k = 0; k < 2; ++k) {
        for (int c = 0; c < 3; ++c) {
            const float z = rad_philox_normal_s(seed, 2, (uint64_t)(k * 3 + c));
            CHECK_EQ(x.at(k * 5 + c), av * (k ? xb[c] : xa[c]) + omv * z);
        }
        CHECK_EQ(x.at(k * 5 + 3), -9.0f);
        CHECK_EQ(x.at(k * 5 + 4), -9.0f);
    }
    T y(RAD_F32, {1, 3});
    y.f({ xb[0], xb[1], xb[2] });
    A a2; a2.t(sd).t(y).i("M", 1).i("n", 3).i("stream", 2).d("aug", 0.999).s("dtype", "f32");
    CHECK_OK(run("cond_aug", a2));
    for (int c = 0; c < 3; ++c) {
        const float restarted = av * xb[c] + omv * rad_philox_normal_s(seed, 2, (uint64_t)c);
        CHECK_EQ(y.at(c), restarted);
        CHECK(restarted != x.at(5 + c));   // joined, row 1 read element 3 + c instead
    }
    A a3; a3.t(sd).t(y).i("M", 1).i("n", 3).i("stream", 2).d("aug", 1.0).s("dtype", "f32");
    CHECK_EQ(run("cond_aug", a3), RAD_E_SHAPE);
    A a4; a4.t(sd).t(y).i("M", 1).i("n", 3).i("stream", 2).s("dtype", "f32");
    CHECK_EQ(run("cond_aug", a4), RAD_E_SHAPE);
    A a5; a5.t(y).i("M", 1).i("n", 3).i("stream", 2).d("aug", 0.999).s("dtype", "f32");
    CHECK_EQ(run("cond_aug", a5), RAD_E_INVAL);
}

/* ================================================================== video sparse attention
 * docs/OPS.md "Video sparse attention": the tiles over a packed sequence of dense segments and a
 * video grid, vsa_select's top-k of pooled scores and its coarse branch, attn_vsa's block-sparse
 * attention and its gated epilogue. */
namespace {

/* `geom` as the run phase writes it: (frames, latent height, latent width) at RAD_VGEOM_*, and
 * values in the other entries that no tile reads. */
void vsa_geom(T& g, int64_t frames, int64_t lat_h, int64_t lat_w) {
    std::vector<int64_t> v(RAD_VGEOM_N);
    for (int64_t i = 0; i < RAD_VGEOM_N; ++i) v[(size_t)i] = 1000 + i;
    v[RAD_VGEOM_LAT_FRAMES] = frames;
    v[RAD_VGEOM_LAT_H] = lat_h;
    v[RAD_VGEOM_LAT_W] = lat_w;
    g.n(v);
}

A& vsa_params(A& a, int64_t M, int64_t hd, int64_t tt, int64_t th, int64_t tw, int64_t patch,
              double sparsity) {
    return a.i("M", M).i("head_dim", hd).i("tile_t", tt).i("tile_h", th).i("tile_w", tw)
            .i("patch", patch).d("sparsity", sparsity);
}

/* A small geometry with every edge case: dense segments of 13 and 14 rows (a text and a stereo
 * soundtrack of 7 rows a channel), then a 6 x 5 x 7 grid of patches (a 10 x 14 latent at patch
 * 2), 237 rows. In boxes of 4 x 4 x 4 the segments are a tile each and the grid's edge boxes are
 * short on every axis: tiles 2 .. 9 are boxes (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0),
 * (1,0,1), (1,1,0), (1,1,1) of 64, 48, 16, 12, 32, 24, 8 and 6 rows. */
constexpr int64_t kVsaRows = 237;

int64_t vsa_row(int64_t t, int64_t h, int64_t w) { return 27 + (t * 5 + h) * 7 + w; }

}  /* namespace */

/* vsa_select over the small geometry, one column a head (head_dim 1), sparsity 0.8: k =
 * ceil(0.2 8) = 2. Head 0's keys are 1 over box (0,1,1) and 2 over box (1,0,0), 0 over the rest
 * of the grid and 9 over the prefix (no candidate): every video query tile takes tiles 5 and 6,
 * ascending, but box (1,1,1)'s queries are -1, so its scores are -kp and it takes the ties at -0,
 * the two lowest, 2 and 3. A box numbering other than (t, h, w) raster with w fastest names other
 * tiles (w-major makes 5 and 6 into 8 and 3). Head 1's keys are -1 over the whole grid: [2, 3]
 * everywhere. Prefix query tiles' rows, the slot past k and the spare tile row are not written. */
TEST(vsa_select_numbers_the_boxes_t_h_w_and_breaks_ties_low) {
    const int64_t R = kVsaRows;
    std::vector<float> qv((size_t)(R * 2), 1.0f), kv((size_t)(R * 2), 0.0f);
    for (int64_t r = 0; r < 27; ++r) { kv[(size_t)(r * 2)] = 9.0f; kv[(size_t)(r * 2 + 1)] = 5.0f; }
    for (int64_t r = 27; r < R; ++r) kv[(size_t)(r * 2 + 1)] = -1.0f;
    for (int64_t t = 0; t < 4; ++t)
        for (int64_t w = 4; w < 7; ++w) kv[(size_t)(vsa_row(t, 4, w) * 2)] = 1.0f;
    for (int64_t t = 4; t < 6; ++t)
        for (int64_t h = 0; h < 4; ++h)
            for (int64_t w = 0; w < 4; ++w) kv[(size_t)(vsa_row(t, h, w) * 2)] = 2.0f;
    for (int64_t t = 4; t < 6; ++t)
        for (int64_t w = 4; w < 7; ++w) qv[(size_t)(vsa_row(t, 4, w) * 2)] = -1.0f;
    T q(RAD_F32, {R, 2}); q.f(qv);
    T k(RAD_F32, {R, 2}); k.f(kv);
    T segs(RAD_I32, {4}); segs.n({0, 13, 27, R});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 6, 10, 14);
    T sel(RAD_I32, {2, 11, 3}); sel.n(std::vector<int64_t>(66, -7));
    A a; a.t(q).t(k).tnull().t(segs).t(geom).t(sel).tnull();
    vsa_params(a, R, 1, 4, 4, 4, 2, 0.8);
    CHECK_OK(run("vsa_select", a));
    auto at = [&](int64_t h, int64_t i, int64_t s) { return sel.iat((h * 11 + i) * 3 + s); };
    for (int64_t i = 2; i < 10; ++i) {
        CHECK_EQ(at(0, i, 0), (int64_t)(i == 9 ? 2 : 5));
        CHECK_EQ(at(0, i, 1), (int64_t)(i == 9 ? 3 : 6));
        CHECK_EQ(at(1, i, 0), (int64_t)2);
        CHECK_EQ(at(1, i, 1), (int64_t)3);
    }
    for (int64_t h = 0; h < 2; ++h)
        for (int64_t s = 0; s < 3; ++s) {
            CHECK_EQ(at(h, 0, s), (int64_t)-7);
            CHECK_EQ(at(h, 1, s), (int64_t)-7);
            CHECK_EQ(at(h, 10, s), (int64_t)-7);
            for (int64_t i = 2; i < 10; ++i) if (s == 2) CHECK_EQ(at(h, i, s), (int64_t)-7);
        }
}

/* Each dense segment is cut on its own: segments of 13, 0 and 14 rows in boxes of 2 x 2 x 2
 * (8 rows) are tiles [0, 8), [8, 13), [13, 21), [21, 27), four of them, the empty one none; a
 * 2 x 2 x 3 grid follows as tiles 4 (8 rows) and 5 (4 rows). With q 0 every score is 0 and the
 * coarse branch is the plain mean of the six tile means of v = row index: (3.5 + 10 + 16.5 +
 * 23.5 + 32 + 33.5) / 6 = 19.8333, for every tile, prefix ones too. Chunks running across the
 * segments ([8, 16), [16, 24), [24, 27)) would give 20.8333. At sparsity 0 every video tile is
 * selected: [4, 5]. */
TEST(vsa_select_chunks_each_dense_segment_and_averages_every_tile) {
    const int64_t R = 39;
    std::vector<float> rows((size_t)R);
    for (int64_t r = 0; r < R; ++r) rows[(size_t)r] = (float)r;
    T q(RAD_F32, {R, 1}); q.f(std::vector<float>((size_t)R, 0.0f));
    T k(RAD_F32, {R, 1}); k.f(std::vector<float>((size_t)R, 0.0f));
    T v(RAD_F32, {R, 1}); v.f(rows);
    T segs(RAD_I32, {5}); segs.n({0, 13, 13, 27, R});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 2, 2, 3);
    T sel(RAD_I32, {1, 6, 2}); sel.n(std::vector<int64_t>(12, -7));
    T co(RAD_F32, {1, 6, 1});
    A a; a.t(q).t(k).t(v).t(segs).t(geom).t(sel).t(co);
    vsa_params(a, R, 1, 2, 2, 2, 1, 0.0);
    CHECK_OK(run("vsa_select", a));
    CHECK_EQ(sel.iat(8), (int64_t)4);  CHECK_EQ(sel.iat(9), (int64_t)5);
    CHECK_EQ(sel.iat(10), (int64_t)4); CHECK_EQ(sel.iat(11), (int64_t)5);
    for (int64_t i = 0; i < 8; ++i) CHECK_EQ(sel.iat(i), (int64_t)-7);
    for (int64_t i = 0; i < 6; ++i) CHECK_NEAR(co.at(i), 119.0f / 6.0f, 1e-5);
}

/* k = max(1, min(ceil((1 - sparsity) V), V)) in f64, as the trained schedule computes it. One-row
 * tiles (boxes of 1) over a grid with no prefix, q 1 and k = row index, so the selection is the k
 * last tiles. (1 - 0.8) 660 is 131.99999999999997: 132 at MiniMax-H3's 1344 x 768 (37 x 24 x 42
 * in boxes of 4: 10 x 6 x 11). (1 - 0.7) 10 is 3.0000000000000004: 4, where exact arithmetic
 * gives 3. (1 - 0.8) 5 is 0.9999999999999998: 1, and at least one tile is kept. */
TEST(vsa_select_computes_k_in_f64) {
    struct Case { int64_t F, H, W; double sparsity; int64_t k; };
    static const Case kCases[] = { { 10, 6, 11, 0.8, 132 }, { 1, 1, 10, 0.7, 4 },
                                   { 1, 1, 5, 0.8, 1 },     { 1, 1, 7, 0.5, 4 },
                                   { 1, 1, 7, 0.0, 7 },     { 1, 1, 8, 0.8, 2 },
                                   { 2, 2, 2, 0.99, 1 } };
    for (const Case& c : kCases) {
        const int64_t V = c.F * c.H * c.W;
        std::vector<float> kv((size_t)V);
        for (int64_t r = 0; r < V; ++r) kv[(size_t)r] = (float)r;
        T q(RAD_F32, {V, 1}); q.f(std::vector<float>((size_t)V, 1.0f));
        T k(RAD_F32, {V, 1}); k.f(kv);
        T segs(RAD_I32, {2}); segs.n({0, V});
        T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, c.F, c.H, c.W);
        T sel(RAD_I32, {1, V, V + 1}); sel.n(std::vector<int64_t>((size_t)(V * (V + 1)), -1));
        A a; a.t(q).t(k).tnull().t(segs).t(geom).t(sel).tnull();
        vsa_params(a, V, 1, 1, 1, 1, 1, c.sparsity);
        CHECK_OK(run("vsa_select", a));
        for (int64_t i : { (int64_t)0, V - 1 }) {
            int64_t n = 0;
            while (n <= V && sel.iat(i * (V + 1) + n) != -1) ++n;
            CHECK_EQ(n, c.k);
            CHECK_EQ(sel.iat(i * (V + 1)), V - c.k);
            CHECK_EQ(sel.iat(i * (V + 1) + c.k - 1), V - 1);
        }
    }
}

/* Equal scores go to the lower tile, and the selection is written in ascending tile order
 * whatever the scores' order. Five one-row video tiles, q 1, so tile j scores k[j]: of
 * [1, 3, 2, 3, 3], k = 2 (sparsity 0.6) takes 1 and 3, k = 3 (0.4) takes 1, 3 and 4; of
 * [2, 1, 1, 1, 5], k = 2 takes 4 and 0, written [0, 4]. A NaN ranks above every number. */
TEST(vsa_select_breaks_score_ties_towards_the_lower_tile) {
    struct Case { float k[5]; double sparsity; int64_t n; int64_t want[3]; };
    static const Case kCases[] = { { { 1, 3, 2, 3, 3 }, 0.6, 2, { 1, 3, -1 } },
                                   { { 1, 3, 2, 3, 3 }, 0.4, 3, { 1, 3, 4 } },
                                   { { 2, 1, 1, 1, 5 }, 0.6, 2, { 0, 4, -1 } },
                                   { { 2, NAN, 1, 7, 5 }, 0.6, 2, { 1, 3, -1 } } };
    for (const Case& c : kCases) {
        T q(RAD_F32, {5, 1}); q.f({1, 1, 1, 1, 1});
        T k(RAD_F32, {5, 1}); k.f({c.k[0], c.k[1], c.k[2], c.k[3], c.k[4]});
        T segs(RAD_I32, {2}); segs.n({0, 5});
        T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 1, 1, 5);
        T sel(RAD_I32, {1, 5, 3}); sel.n(std::vector<int64_t>(15, -1));
        A a; a.t(q).t(k).tnull().t(segs).t(geom).t(sel).tnull();
        vsa_params(a, 5, 1, 1, 1, 1, 1, c.sparsity);
        CHECK_OK(run("vsa_select", a));
        for (int64_t i = 0; i < 5; ++i)
            for (int64_t s = 0; s < 3; ++s) CHECK_EQ(sel.iat(i * 3 + s), c.want[s]);
    }
}

/* attn_vsa over the small geometry with q 0, so every visible key weighs the same and a row's
 * output is the mean of v = row index over what it sees. Sparsity 0.9: k = 1, and video tile i
 * selects tile 11 - i. Prefix query rows see all 237 rows: 118. Row 27 (box (0,0,0), tile 2)
 * sees the prefix's 27 rows (sum 351) and tile 9's 6 rows (sum 1305): 1656 / 33. Row 236 (tile 9)
 * sees tile 2's 64 rows (sum 5856): 6207 / 91. Row 171, patch (4, 0, 4) of box (1,0,1), tile 7,
 * sees tile 4's 16 rows (sum 1744): 2095 / 43. The prefix query tiles' rows of `sel` are neither
 * read nor checked. */
TEST(attn_vsa_sees_the_prefix_and_the_selected_boxes) {
    const int64_t R = kVsaRows;
    std::vector<float> rows((size_t)R);
    for (int64_t r = 0; r < R; ++r) rows[(size_t)r] = (float)r;
    T q(RAD_F32, {R, 1}); q.f(std::vector<float>((size_t)R, 0.0f));
    T k(RAD_F32, {R, 1}); k.f(std::vector<float>((size_t)R, 0.0f));
    T v(RAD_F32, {R, 1}); v.f(rows);
    T segs(RAD_I32, {4}); segs.n({0, 13, 27, R});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 6, 10, 14);
    T sel(RAD_I32, {1, 10, 1}); sel.n({-5, -5, 9, 8, 7, 6, 5, 4, 3, 2});
    T o(RAD_F32, {R, 1});
    A a; a.t(q).t(k).t(v).t(segs).t(geom).t(sel).tnull().tnull().t(o);
    vsa_params(a, R, 1, 4, 4, 4, 2, 0.9);
    CHECK_OK(run("attn_vsa", a));
    CHECK_NEAR(o.at(0), 118.0f, 1e-4);
    CHECK_NEAR(o.at(26), 118.0f, 1e-4);
    CHECK_NEAR(o.at(27), 1656.0f / 33.0f, 1e-4);
    CHECK_NEAR(o.at(236), 6207.0f / 91.0f, 1e-4);
    CHECK_EQ(vsa_row(4, 0, 4), (int64_t)171);
    CHECK_NEAR(o.at(171), 2095.0f / 43.0f, 1e-4);
}

/* At sparsity 0, k is V and vsa_select selects every video tile in order, and attn_vsa over that
 * selection is attn_dense over the whole sequence: the same keys, taken tile by tile. */
TEST(attn_vsa_at_sparsity_zero_is_dense_attention) {
    const int64_t R = kVsaRows, nh = 2, hd = 8;
    uint32_t s = 7;
    auto draw = [&s](int64_t n) {
        std::vector<float> x((size_t)n);
        for (auto& e : x) {
            s = s * 1664525u + 1013904223u;
            e = (float)((int)((s >> 8) % 2001) - 1000) / 1000.0f;
        }
        return x;
    };
    T q(RAD_F32, {R, nh, hd}); q.f(draw(R * nh * hd));
    T k(RAD_F32, {R, nh, hd}); k.f(draw(R * nh * hd));
    T v(RAD_F32, {R, nh, hd}); v.f(draw(R * nh * hd));
    T segs(RAD_I32, {4}); segs.n({0, 13, 27, R});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 6, 10, 14);
    T sel(RAD_I32, {nh, 10, 8});
    A a; a.t(q).t(k).tnull().t(segs).t(geom).t(sel).tnull();
    vsa_params(a, R, hd, 4, 4, 4, 2, 0.0);
    CHECK_OK(run("vsa_select", a));
    for (int64_t h = 0; h < nh; ++h)
        for (int64_t i = 2; i < 10; ++i)
            for (int64_t j = 0; j < 8; ++j) CHECK_EQ(sel.iat((h * 10 + i) * 8 + j), j + 2);
    T o1(RAD_F32, {R, nh, hd}), o2(RAD_F32, {R, nh, hd});
    A b; b.t(q).t(k).t(v).t(segs).t(geom).t(sel).tnull().tnull().t(o1);
    vsa_params(b, R, hd, 4, 4, 4, 2, 0.0);
    CHECK_OK(run("attn_vsa", b));
    T cu(RAD_I32, {2}); cu.n({0, R});
    A c; c.t(q).t(k).t(v).t(cu).t(o2).i("M", R).i("head_dim", hd).i("gqa", 1).i("causal", 0)
          .s("dtype", "f32");
    CHECK_OK(run("attn_dense", c));
    for (int64_t i = 0; i < R * nh * hd; ++i) CHECK_NEAR(o1.at(i), o2.at(i), 1e-5);

    /* The int8 score form is attn_prefix_self's, prefix-free, over the same selection: k's codes
     * and scales per (row, head) as quant_act_i8g writes them, the query quantised inside. */
    T k8(RAD_I8, {R, nh * hd}), ks(RAD_F32, {R, nh});
    A d; d.t(k).t(k8).t(ks).i("M", R).i("n", nh * hd).i("group", hd).s("dtype", "f32");
    CHECK_OK(run("quant_act_i8g", d));
    T o3(RAD_F32, {R, nh, hd}), o4(RAD_F32, {R, nh * hd});
    A e; e.t(q).t(k).t(v).t(segs).t(geom).t(sel).tnull().tnull().t(o3).t(k8).t(ks).s("qk", "i8");
    vsa_params(e, R, hd, 4, 4, 4, 2, 0.0);
    CHECK_OK(run("attn_vsa", e));
    A f; f.t(q).t(k).t(v).tnull().tnull().tnull().t(cu).tnull().tnull().t(k8).t(ks).t(o4)
          .i("q_len", R).i("head_dim", hd).i("gqa", 1).i("block_size", 16).i("causal", 0)
          .i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32").s("qk", "i8");
    CHECK_OK(run("attn_prefix_self", f));
    for (int64_t i = 0; i < R * nh * hd; ++i) CHECK_NEAR(o3.at(i), o4.at(i), 1e-5);

    /* The fp8 P.V form, likewise: the values as quant_cols_fp8's codes and column scales. */
    T v8(RAD_F8E4M3, {R, nh * hd}), vs(RAD_F32, {nh * hd});
    A g; g.t(v).t(v8).t(vs).i("M", R).i("n", nh * hd).s("dtype", "f32");
    CHECK_OK(run("quant_cols_fp8", g));
    T o5(RAD_F32, {R, nh, hd}), o6(RAD_F32, {R, nh * hd});
    A h; h.t(q).t(k).t(v8).t(segs).t(geom).t(sel).tnull().tnull().t(o5).t(k8).t(ks).t(vs)
          .s("qk", "i8").s("pv", "fp8");
    vsa_params(h, R, hd, 4, 4, 4, 2, 0.0);
    CHECK_OK(run("attn_vsa", h));
    A u; u.t(q).t(k).t(v8).tnull().tnull().tnull().t(cu).tnull().tnull().t(k8).t(ks).t(o6).t(vs)
          .i("q_len", R).i("head_dim", hd).i("gqa", 1).i("block_size", 16).i("causal", 0)
          .i("max_ctx", 0).s("q_dtype", "f32").s("kv_dtype", "f32").s("qk", "i8").s("pv", "fp8");
    CHECK_OK(run("attn_prefix_self", u));
    for (int64_t i = 0; i < R * nh * hd; ++i) CHECK_NEAR(o5.at(i), o6.at(i), 1e-5);
}

/* The coarse branch rounds once. Two video tiles of two rows (a 1 x 1 x 4 grid in boxes of
 * 1 x 1 x 2), q 0, so p is 1/2 each. v rows 1, 1 + 2^-7, 1 + 2^-7, 1 + 2^-7 make the tile means
 * 1 + 2^-8 and 1 + 2^-7 and their mix 1 + 3 2^-9, which bf16 rounds to 1 + 2^-7 = 1.0078125.
 * Rounding the means to bf16 first, or each product, gives 1. */
TEST(vsa_select_rounds_the_coarse_mix_once) {
    const float a7 = 1.0078125f;
    T q(RAD_BF16, {4, 1}); q.f({0, 0, 0, 0});
    T k(RAD_BF16, {4, 1}); k.f({0, 0, 0, 0});
    T v(RAD_BF16, {4, 1}); v.f({1, a7, a7, a7});
    T segs(RAD_I32, {2}); segs.n({0, 4});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 1, 1, 4);
    T sel(RAD_I32, {1, 2, 2});
    T co(RAD_BF16, {1, 2, 1});
    A a; a.t(q).t(k).t(v).t(segs).t(geom).t(sel).t(co);
    vsa_params(a, 4, 1, 1, 1, 2, 1, 0.0);
    CHECK_OK(run("vsa_select", a));
    CHECK_EQ(co.at(0), a7);
    CHECK_EQ(co.at(1), a7);
}

/* The gated epilogue rounds the attention to the output's dtype before the add:
 * out = bf16(bf16(o) + coarse gate). The same two tiles, both visible (sparsity 0), q 0, v rows
 * 1, 1 + 2^-7, 1, 1 + 2^-7 in both columns: o = 1 + 2^-8 everywhere, which bf16 rounds to 1 (a
 * tie, to even). coarse gate = 2^-8 then gives 1 + 2^-8, a tie, 1; -2^-8 gives 1 - 2^-8 =
 * 0.99609375 exactly. Added to the unrounded o they would give 1.0078125 and 1. Tile 1's coarse
 * 3 2^-4 in column 0 gives 1 + 3 2^-8, a tie, 1.015625: the row reads its own tile's coarse.
 * Without coarse and gate the output is bf16(o) = 1. */
TEST(attn_vsa_rounds_the_attention_before_the_gated_coarse_term) {
    const float a7 = 1.0078125f, e4 = 0.0625f;
    T q(RAD_BF16, {4, 2}); q.f({0, 0, 0, 0, 0, 0, 0, 0});
    T k(RAD_BF16, {4, 2}); k.f({0, 0, 0, 0, 0, 0, 0, 0});
    T v(RAD_BF16, {4, 2}); v.f({1, 1, a7, a7, 1, 1, a7, a7});
    T segs(RAD_I32, {2}); segs.n({0, 4});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 1, 1, 4);
    T sel(RAD_I32, {1, 2, 2}); sel.n({0, 1, 0, 1});
    T co(RAD_BF16, {1, 2, 2}); co.f({e4, e4, 3 * e4, e4});
    T gate(RAD_BF16, {4, 2}); gate.f({e4, -e4, e4, -e4, e4, -e4, e4, -e4});
    T o(RAD_BF16, {4, 2});
    A a; a.t(q).t(k).t(v).t(segs).t(geom).t(sel).t(co).t(gate).t(o);
    vsa_params(a, 4, 2, 1, 1, 2, 1, 0.0);
    CHECK_OK(run("attn_vsa", a));
    const float want[8] = { 1.0f, 0.99609375f, 1.0f, 0.99609375f,
                            1.015625f, 0.99609375f, 1.015625f, 0.99609375f };
    for (int i = 0; i < 8; ++i) CHECK_EQ(o.at(i), want[i]);

    T o2(RAD_BF16, {4, 2});
    A b; b.t(q).t(k).t(v).t(segs).t(geom).t(sel).tnull().tnull().t(o2);
    vsa_params(b, 4, 2, 1, 1, 2, 1, 0.0);
    CHECK_OK(run("attn_vsa", b));
    for (int i = 0; i < 8; ++i) CHECK_EQ(o2.at(i), 1.0f);
}

/* What both ops refuse, writing nothing: a last segment that is not the grid's F Hp Wp rows,
 * segments past the operands' rows, a sparsity of 1, and the optional pairs half given (coarse
 * without v, coarse without gate); attn_vsa also a `sel` row that is not k ascending video tiles
 * (RAD_E_INVAL) and either op a `sel` other than i32. */
TEST(vsa_ops_refuse_a_wrong_grid_a_bad_selection_and_half_a_pair) {
    T q(RAD_BF16, {4, 2}); q.f({1, 2, 3, 4, 5, 6, 7, 8});
    T k(RAD_BF16, {4, 2}); k.f({8, 7, 6, 5, 4, 3, 2, 1});
    T v(RAD_BF16, {4, 2}); v.f({1, 1, 2, 2, 3, 3, 4, 4});
    T geom(RAD_I32, {RAD_VGEOM_N}); vsa_geom(geom, 1, 1, 4);
    T co(RAD_BF16, {1, 2, 2}); co.f({1, 1, 1, 1});
    T gate(RAD_BF16, {4, 2}); gate.f({1, 1, 1, 1, 1, 1, 1, 1});
    auto attn = [&](const T& segs, const T& g, const T& sel, bool with_co, bool with_gate,
                    double sparsity) {
        T o(RAD_BF16, {4, 2}); o.f({-3, -3, -3, -3, -3, -3, -3, -3});
        const std::vector<uint8_t> before = o.mem;
        A a; a.t(q).t(k).t(v).t(segs).t(g).t(sel);
        if (with_co) a.t(co); else a.tnull();
        if (with_gate) a.t(gate); else a.tnull();
        a.t(o);
        vsa_params(a, 4, 2, 1, 1, 2, 1, sparsity);
        const int rc = run("attn_vsa", a);
        if (rc < 0) CHECK(o.mem == before);
        return rc;
    };
    T segs(RAD_I32, {2}); segs.n({0, 4});
    T good(RAD_I32, {1, 2, 2}); good.n({0, 1, 0, 1});
    CHECK_OK(attn(segs, geom, good, true, true, 0.0));

    T g3(RAD_I32, {RAD_VGEOM_N}); vsa_geom(g3, 1, 1, 3);
    CHECK_EQ(attn(segs, g3, good, true, true, 0.0), RAD_E_SHAPE);
    T s5(RAD_I32, {2}); s5.n({0, 5});
    CHECK_EQ(attn(s5, geom, good, true, true, 0.0), RAD_E_SHAPE);
    CHECK_EQ(attn(segs, geom, good, true, true, 1.0), RAD_E_SHAPE);
    CHECK_EQ(attn(segs, geom, good, true, false, 0.0), RAD_E_SHAPE);
    for (const std::vector<int64_t>& bad : { std::vector<int64_t>{ 1, 0, 0, 1 },
                                             std::vector<int64_t>{ 0, 1, 0, 2 },
                                             std::vector<int64_t>{ 0, 1, 1, 1 } }) {
        T sel(RAD_I32, {1, 2, 2}); sel.n(bad);
        CHECK_EQ(attn(segs, geom, sel, true, true, 0.0), RAD_E_INVAL);
    }
    T s64(RAD_I64, {1, 2, 2}); s64.n({0, 1, 0, 1});
    CHECK_EQ(attn(segs, geom, s64, true, true, 0.0), RAD_E_DTYPE);

    T sel(RAD_I32, {1, 2, 2}); sel.n({-7, -7, -7, -7});
    const std::vector<uint8_t> before = sel.mem;
    A a; a.t(q).t(k).tnull().t(segs).t(g3).t(sel).tnull();
    vsa_params(a, 4, 2, 1, 1, 2, 1, 0.5);
    CHECK_EQ(run("vsa_select", a), RAD_E_SHAPE);
    A b; b.t(q).t(k).tnull().t(segs).t(geom).t(sel).t(co);
    vsa_params(b, 4, 2, 1, 1, 2, 1, 0.5);
    CHECK_EQ(run("vsa_select", b), RAD_E_SHAPE);
    CHECK(sel.mem == before);
}

/* ================================================================== the audio vocoder
 * BigVGAN's forms (MiniMax-H3's audio decoder). */

/* conv_t1d at kernel 9, stride 5, pad 2: output row o sums x[i] w[o + 2 - 5 i] over the rows with
 * 0 <= o + 2 - 5 i < 9, the bias first. PyTorch's ConvTranspose1d(1, 1, 9, 5, padding 2) over
 * [1, 10] is the full [1, 2, 3, 4, 5, 16, 27, 38, 49, 50, 60, 70, 80, 90] cropped by 2 a side.
 * The second sequence ([100]) reads only itself. Kernel 12 at stride 2 pads 5, past a stride:
 * [1, 10] gives the full product's rows 5..8, [6 + 40, 7 + 50, 8 + 60, 9 + 70]. */
TEST(conv_t1d_takes_any_kernel_of_at_least_a_stride) {
    T x(RAD_F32, {3, 1}); x.f({1, 10, 100});
    T w(RAD_F32, {1, 1, 9}); w.f({1, 2, 3, 4, 5, 6, 7, 8, 9});
    T b(RAD_F32, {1}); b.f({0.5f});
    T cu(RAD_I32, {3}); cu.n({0, 2, 3});
    T y(RAD_F32, {15, 1});
    A a; a.t(x).t(w).t(b).t(cu).t(y).i("M", 3).i("cin", 1).i("cout", 1).i("stride", 5)
          .i("k", 9).s("dtype", "f32");
    CHECK_OK(run("conv_t1d", a));
    const float want[15] = { 3.5f, 4.5f, 5.5f, 16.5f, 27.5f, 38.5f, 49.5f, 50.5f, 60.5f, 70.5f,
                             300.5f, 400.5f, 500.5f, 600.5f, 700.5f };
    for (int i = 0; i < 15; ++i) CHECK_EQ(y.at(i), want[i]);

    T x2(RAD_F32, {2, 1}); x2.f({1, 10});
    T w2(RAD_F32, {1, 1, 12}); w2.f({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    T b2(RAD_F32, {1}); b2.f({0});
    T cu2(RAD_I32, {2}); cu2.n({0, 2});
    T y2(RAD_F32, {4, 1});
    A c; c.t(x2).t(w2).t(b2).t(cu2).t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("stride", 2)
          .i("k", 12).s("dtype", "f32");
    CHECK_OK(run("conv_t1d", c));
    CHECK_EQ(y2.at(0), 46.0f); CHECK_EQ(y2.at(1), 57.0f);
    CHECK_EQ(y2.at(2), 68.0f); CHECK_EQ(y2.at(3), 79.0f);

    /* A kernel short of a stride, and k - stride odd with no pad to break the tie, are refused. */
    A d; d.t(x2).t(w2).t(b2).t(cu2).t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("stride", 4)
          .i("k", 3).s("dtype", "f32");
    CHECK_EQ(run("conv_t1d", d), RAD_E_SHAPE);
    A e; e.t(x2).t(w2).t(b2).t(cu2).t(y2).i("M", 2).i("cin", 1).i("cout", 1).i("stride", 2)
          .i("k", 5).s("dtype", "f32");
    CHECK_EQ(run("conv_t1d", e), RAD_E_SHAPE);
}

/* conv1d's strided form (a DAC encoder's downsample): kernel 2 stride, pad stride / 2, one
 * sequence. x rows [1 .. 8], w [1, 2, 3, 4], stride 2, pad 1: output row m reads
 * x[2m - 1 + t] (zeros outside [0, 8)): 20, 40, 60, 44 — 8 rows halve to exactly 4. */
TEST(conv1d_strided_downsamples_one_sequence) {
    T x(RAD_BF16, {8, 1}); x.f({1, 2, 3, 4, 5, 6, 7, 8});
    T w(RAD_BF16, {1, 1, 4}); w.f({1, 2, 3, 4});
    T cu(RAD_I32, {2}); cu.n({0, 8});
    T y(RAD_BF16, {4, 1});
    A a; a.t(x).t(w).t(cu).tnull().t(y).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
          .i("dil", 1).i("stride", 2).i("pad", 1).s("dtype", "bf16");
    CHECK_OK(run("conv1d", a));
    CHECK_EQ(y.at(0), 20.0f); CHECK_EQ(y.at(1), 40.0f);
    CHECK_EQ(y.at(2), 60.0f); CHECK_EQ(y.at(3), 44.0f);

    /* The refusals: `pad` without a stride, a stride without `pad`, a stride over more than
     * one sequence, a y that is not the formula's rows, and a stride on the f32 rows. */
    A p; p.t(x).t(w).t(cu).tnull().t(y).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
           .i("dil", 1).i("pad", 1).s("dtype", "bf16");
    CHECK_EQ(run("conv1d", p), RAD_E_SHAPE);
    A q; q.t(x).t(w).t(cu).tnull().t(y).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
           .i("dil", 1).i("stride", 2).s("dtype", "bf16");
    CHECK_EQ(run("conv1d", q), RAD_E_SHAPE);
    T cu2(RAD_I32, {3}); cu2.n({0, 4, 8});
    A m; m.t(x).t(w).t(cu2).tnull().t(y).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
           .i("dil", 1).i("stride", 2).i("pad", 1).s("dtype", "bf16");
    CHECK_EQ(run("conv1d", m), RAD_E_SHAPE);
    T y5(RAD_BF16, {5, 1});
    A r; r.t(x).t(w).t(cu).tnull().t(y5).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
           .i("dil", 1).i("stride", 2).i("pad", 1).s("dtype", "bf16");
    CHECK_EQ(run("conv1d", r), RAD_E_SHAPE);
    T xf(RAD_F32, {8, 1}); xf.f({1, 2, 3, 4, 5, 6, 7, 8});
    T wf(RAD_F32, {1, 1, 4}); wf.f({1, 2, 3, 4});
    T yf(RAD_F32, {4, 1});
    A f; f.t(xf).t(wf).t(cu).tnull().t(yf).tnull().i("M", 8).i("cin", 1).i("cout", 1).i("k", 4)
           .i("dil", 1).i("stride", 2).i("pad", 1).s("dtype", "f32");
    CHECK_EQ(run("conv1d", f), RAD_E_DTYPE);
}

/* conv1d `act` clamp: min(max(y, -1), 1) after the bias. */
TEST(conv1d_act_clamp_bounds_the_output) {
    T x(RAD_F32, {4, 1}); x.f({0.5f, 2, -3, -1});
    T w(RAD_F32, {1, 1, 1}); w.f({1});
    T cu(RAD_I32, {2}); cu.n({0, 4});
    T y(RAD_F32, {4, 1});
    T b(RAD_F32, {1}); b.f({0.25f});
    A a; a.t(x).t(w).t(cu).tnull().t(y).t(b).i("M", 4).i("cin", 1).i("cout", 1).i("k", 1)
          .i("dil", 1).s("act", "clamp").s("dtype", "f32");
    CHECK_OK(run("conv1d", a));
    CHECK_EQ(y.at(0), 0.75f); CHECK_EQ(y.at(1), 1.0f);
    CHECK_EQ(y.at(2), -1.0f); CHECK_EQ(y.at(3), -0.75f);
}

/* conv1d `stride`: Fish Audio's causal downsampling conv (modded_dac.py, CausalConvNet), the reach
 * k less the stride padded on the left. k 4, stride 2 over [1..8] pads 2: windows [0 0 1 2],
 * [1 2 3 4], [3 4 5 6], [5 6 7 8] against taps 1, 10, 100, 1000 are 2100, 4321, 6543, 8765. A
 * second sequence [1 2 3 4] starts its own padding; a bound that is not a whole stride, and a
 * stride on the symmetric form, are refused. `cu_div` reads the same bounds in samples. */
TEST(conv1d_stride_downsamples_each_sequence_causally) {
    T x(RAD_F32, {12, 1}); x.f({1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4});
    T w(RAD_F32, {1, 1, 4}); w.f({1, 10, 100, 1000});
    T cu(RAD_I32, {3}); cu.n({0, 8, 12});
    T y(RAD_F32, {6, 1});
    A a; a.t(x).t(w).t(cu).tnull().t(y).i("M", 12).i("cin", 1).i("cout", 1).i("k", 4)
          .i("dil", 1).i("causal", 1).i("stride", 2).s("dtype", "f32");
    CHECK_OK(run("conv1d", a));
    CHECK_EQ(y.at(0), 2100.0f); CHECK_EQ(y.at(1), 4321.0f);
    CHECK_EQ(y.at(2), 6543.0f); CHECK_EQ(y.at(3), 8765.0f);
    CHECK_EQ(y.at(4), 2100.0f); CHECK_EQ(y.at(5), 4321.0f);
    T cu2(RAD_I32, {3}); cu2.n({0, 7, 12});
    A b; b.t(x).t(w).t(cu2).tnull().t(y).i("M", 12).i("cin", 1).i("cout", 1).i("k", 4)
          .i("dil", 1).i("causal", 1).i("stride", 2).s("dtype", "f32");
    CHECK_EQ(run("conv1d", b), RAD_E_SHAPE);
    T w3(RAD_F32, {1, 1, 3}); w3.f({1, 1, 1});
    A c; c.t(x).t(w3).t(cu).tnull().t(y).i("M", 12).i("cin", 1).i("cout", 1).i("k", 3)
          .i("dil", 1).i("stride", 2).s("dtype", "f32");
    CHECK_EQ(run("conv1d", c), RAD_E_SHAPE);
    /* The same bounds in units of 4 rows (samples, 4 a row): the same rows; a bound that is not
     * a whole number of strides in rows is refused. */
    T y2(RAD_F32, {6, 1});
    T cu4(RAD_I32, {3}); cu4.n({0, 32, 48});
    A d; d.t(x).t(w).t(cu4).tnull().t(y2).i("M", 12).i("cin", 1).i("cout", 1).i("k", 4)
          .i("dil", 1).i("causal", 1).i("stride", 2).i("cu_div", 4).s("dtype", "f32");
    CHECK_OK(run("conv1d", d));
    for (int64_t i = 0; i < 6; ++i) CHECK_EQ(y2.at(i), y.at(i));
    T cu6(RAD_I32, {3}); cu6.n({0, 28, 48});
    A e; e.t(x).t(w).t(cu6).tnull().t(y2).i("M", 12).i("cin", 1).i("cout", 1).i("k", 4)
          .i("dil", 1).i("causal", 1).i("stride", 2).i("cu_div", 4).s("dtype", "f32");
    CHECK_EQ(run("conv1d", e), RAD_E_SHAPE);
}

/* vq_encode by hand. in_w scales the first channel by 2; codebook rows 0 and 2 point the same way
 * (a cosine tie, so the lower index), row 1 the other; out_w the identity and out_b
 * [0.5, -0.5]. Row [4, 1]: e [8, 1], code 0, x - [1.5, -0.5] = [2.5, 1.5]. Row [-1, 3]: e [-2, 3],
 * code 1, x - [0.5, 1.5] = [-1.5, 1.5]. Row [0, 0]: e 0, every cosine 0, code 0. */
TEST(vq_encode_takes_the_nearest_code_by_cosine_and_leaves_the_residual) {
    T x(RAD_F32, {3, 2}); x.f({4, 1, -1, 3, 0, 0});
    T iw(RAD_F32, {2, 2}); iw.f({2, 0, 0, 1});
    T ib(RAD_F32, {2}); ib.f({0, 0});
    T cb(RAD_F32, {3, 2}); cb.f({1, 0, 0, 2, 3, 0});
    T ow(RAD_F32, {2, 2}); ow.f({1, 0, 0, 1});
    T ob(RAD_F32, {2}); ob.f({0.5f, -0.5f});
    T codes(RAD_I32, {3}); codes.n({-1, -1, -1});
    A a; a.t(x).t(iw).t(ib).t(cb).t(ow).t(ob).t(codes).i("M", 3).i("n", 2).i("dim", 2)
          .i("bins", 3).s("dtype", "f32");
    CHECK_OK(run("vq_encode", a));
    CHECK_EQ(codes.iat(0), 0); CHECK_EQ(codes.iat(1), 1); CHECK_EQ(codes.iat(2), 0);
    CHECK_EQ(x.at(0), 2.5f); CHECK_EQ(x.at(1), 1.5f);
    CHECK_EQ(x.at(2), -1.5f); CHECK_EQ(x.at(3), 1.5f);
    CHECK_EQ(x.at(4), -1.5f); CHECK_EQ(x.at(5), 0.5f);
    A b; b.t(x).t(iw).t(ib).t(cb).t(ow).t(ob).t(codes).i("M", 3).i("n", 2).i("dim", 2)
          .i("bins", 4).s("dtype", "f32");
    CHECK_EQ(run("vq_encode", b), RAD_E_SHAPE);
}

/* snake_aa's paddings, with the activation the identity (alpha 0: sin(0 u) = 0). Ratio 2, filters
 * of 4: P = 1. Sequence [10, 20, 40] pads to [10, 10, 20, 40, 40]; the transposed f_up [1, 2, 3, 4]
 * gives [10, 20, 40, 60, 50, 80, 100, 160, 160, 240, 120, 160], doubled and cropped by 3 a side:
 * u [120, 100, 160, 200, 320, 320]. u pads 1 left and 2 right, [120, 120, 100, 160, 200, 320,
 * 320, 320, 320], and f_down [1, 2, 3, 4] at stride 2 gives 1300, 2300, 3080. The one-row
 * sequence [5] pads to [5, 5, 5], u [60, 40], [60, 60, 40, 40, 40]: 460. */
TEST(snake_aa_pads_each_sequence_by_replication) {
    T x(RAD_F32, {4, 1}); x.f({10, 20, 40, 5});
    T al(RAD_F32, {1}); al.f({0});
    T be(RAD_F32, {1}); be.f({1});
    T fu(RAD_F32, {1, 1, 4}); fu.f({1, 2, 3, 4});
    T fd(RAD_F32, {1, 1, 4}); fd.f({1, 2, 3, 4});
    T cu(RAD_I32, {3}); cu.n({0, 3, 4});
    T y(RAD_F32, {4, 1});
    A a; a.t(x).t(al).t(be).t(fu).t(fd).t(cu).t(y).i("M", 4).i("n", 1).i("ratio", 2).i("k", 4)
          .s("dtype", "f32");
    CHECK_OK(run("snake_aa", a));
    CHECK_EQ(y.at(0), 1300.0f); CHECK_EQ(y.at(1), 2300.0f);
    CHECK_EQ(y.at(2), 3080.0f); CHECK_EQ(y.at(3), 460.0f);
}

/* snake_aa's SnakeBeta with `log` 1: alpha = beta = 0 read as e^0 = 1, so a = u + sin(u)^2. A
 * constant 0.125 upsamples by f_up [1, 2, 3, 4] to u alternating 2 (2 + 4) 0.125 = 1.5 and
 * 2 (1 + 3) 0.125 = 1.0; f_down 0.25 each is the mean of four. In f32 steps: 2.2982655,
 * 2.1015348, 1.9048042. */
TEST(snake_aa_applies_snakebeta_between_the_filters) {
    T x(RAD_F32, {3, 1}); x.f({0.125f, 0.125f, 0.125f});
    T al(RAD_F32, {1}); al.f({0});
    T be(RAD_F32, {1}); be.f({0});
    T fu(RAD_F32, {1, 1, 4}); fu.f({1, 2, 3, 4});
    T fd(RAD_F32, {1, 1, 4}); fd.f({0.25f, 0.25f, 0.25f, 0.25f});
    T cu(RAD_I32, {2}); cu.n({0, 3});
    T y(RAD_F32, {3, 1});
    A a; a.t(x).t(al).t(be).t(fu).t(fd).t(cu).t(y).i("M", 3).i("n", 1).i("ratio", 2).i("k", 4)
          .i("log", 1).s("dtype", "f32");
    CHECK_OK(run("snake_aa", a));
    CHECK_NEAR(y.at(0), 2.2982655f, 1e-6);
    CHECK_NEAR(y.at(1), 2.1015348f, 1e-6);
    CHECK_NEAR(y.at(2), 1.9048042f, 1e-6);
}

/* ================================================================== the QSA indexer and dflash_conv
 * docs/OPS.md "Sparse attention (QSA)" and "Draft heads". */

/* qsa_score: per block, the relus of each head's dot summed over heads, over sqrt(n); a negative
 * page id scores -inf; blocks at or past nc and rows outside every sequence are not written. Block
 * 1's heads dot to -1 and 2: relu-then-sum gives 2, sum-then-relu would give 1. */
TEST(qsa_score_sums_per_head_relus_and_marks_absent_pages) {
    const int64_t n = 2, heads = 2;
    T q(RAD_BF16, {2, heads * n}); q.f({1, 0, 0, 1, 9, 9, 9, 9});
    T bk(RAD_BF16, {3, n}); bk.f({1, 1, -1, 2, 3, -5});
    T bt(RAD_I32, {1, 4}); bt.n({0, 1, -1, 2});
    T cu(RAD_I32, {2}); cu.n({0, 1});
    T nc(RAD_I32, {1}); nc.n({3});
    T sc(RAD_F32, {2, 4}); sc.f({99, 99, 99, 99, 99, 99, 99, 99});
    A a; a.t(q).t(bk).t(bt).t(cu).t(nc).t(sc).i("M", 2).i("n", n).i("heads", heads)
          .i("blocks", 4).i("seqs", 1).s("dtype", "bf16");
    CHECK_OK(run("qsa_score", a));
    const float r2 = 2.0f / std::sqrt(2.0f);
    CHECK_EQ(sc.at(0), r2);
    CHECK_EQ(sc.at(1), r2);
    CHECK(std::isinf(sc.at(2)) && sc.at(2) < 0);
    CHECK_EQ(sc.at(3), 99.0f);                       /* past nc */
    for (int i = 4; i < 8; ++i) CHECK_EQ(sc.at(i), 99.0f);   /* row 1 is in no sequence */
}

namespace {
/* One qsa_select call over a single sequence: returns sel (topk + 1), nsel and seqused per row. */
struct Sel { std::vector<int64_t> sel, nsel, su; };
Sel qsa_select_run(const std::vector<float>& score, int64_t nb, const std::vector<int64_t>& bt,
                   int64_t nc, const std::vector<int64_t>& pos, int64_t topk, int64_t ratio) {
    const int64_t M = (int64_t)pos.size();
    T sc(RAD_F32, {M, nb}); sc.f(score);
    T tb(RAD_I32, {1, (int64_t)bt.size()}); tb.n(bt);
    T ncs(RAD_I32, {1}); ncs.n({nc});
    T po(RAD_I32, {M}); po.n(pos);
    T cu(RAD_I32, {2}); cu.n({0, M});
    T se(RAD_I32, {M, topk + 1}), ns(RAD_I32, {M}), su(RAD_I32, {M});
    A a; a.t(sc).t(tb).t(ncs).t(po).t(cu).t(se).t(ns).t(su).i("M", M).i("blocks", nb)
          .i("topk", topk).i("ratio", ratio).i("seqs", 1);
    Sel r;
    if (run("qsa_select", a) != RAD_OK) return r;
    for (int64_t i = 0; i < M * (topk + 1); ++i) r.sel.push_back(se.iat(i));
    for (int64_t i = 0; i < M; ++i) { r.nsel.push_back(ns.iat(i)); r.su.push_back(su.iat(i)); }
    return r;
}
}  /* namespace */

/* qsa_select takes the blocks above the topk-th score in block order, then ties at it in block
 * order (lower index first), never a dead page, then the partial page when the query is not on a
 * block's last token; -1 pads to topk + 1, and seqused is the attention length over the table. */
TEST(qsa_select_orders_above_then_ties_and_appends_the_partial_page) {
    const float ninf = -INFINITY;
    /* Scores 3 6 5 1 5 and a dead block; topk 2: block 2 (6), then the first tie at 5 (block 1). */
    const std::vector<float> s1 = { 3, 5, 6, 1, 5, 100, ninf };
    const std::vector<int64_t> bt = { 10, 11, 12, 13, 14, -1, 16 };
    Sel r = qsa_select_run(s1, 7, bt, 6, { 25 }, 2, 4);
    REQUIRE_EQ(r.sel.size(), (size_t)3);
    CHECK_EQ(r.sel[0], 12LL); CHECK_EQ(r.sel[1], 11LL);
    CHECK_EQ(r.sel[2], 16LL);                        /* partial page: (25 + 1) % 4 != 0 */
    CHECK_EQ(r.nsel[0], 3LL);
    CHECK_EQ(r.su[0], 2LL * 4 + 25 % 4 + 1);

    /* All ties: the lower blocks win. */
    const std::vector<float> s2 = { 3, 5, 5, 1, 5, 0, 0 };
    r = qsa_select_run(s2, 7, bt, 6, { 23 }, 2, 4);
    REQUIRE_EQ(r.sel.size(), (size_t)3);
    CHECK_EQ(r.sel[0], 11LL); CHECK_EQ(r.sel[1], 12LL);
    CHECK_EQ(r.sel[2], -1LL);                        /* 23 is a block's last token: no partial */
    CHECK_EQ(r.su[0], 2LL * 4);

    /* The query's own bound: at position 7 only blocks 0 and 1 are in its past. */
    r = qsa_select_run(s1, 7, bt, 6, { 7 }, 4, 4);
    REQUIRE_EQ(r.sel.size(), (size_t)5);
    CHECK_EQ(r.sel[0], 10LL); CHECK_EQ(r.sel[1], 11LL);
    CHECK_EQ(r.sel[2], -1LL); CHECK_EQ(r.sel[4], -1LL);
    CHECK_EQ(r.nsel[0], 2LL);
    CHECK_EQ(r.su[0], 1LL * 4 + 3 + 1);
}

/* ================================================================== kv_resolve
 * The residency map of a streamed group: selected block ids in, ring slots out, a block copied in
 * from the host mirror when it is not resident (docs/OPS.md "kv_resolve",
 * docs/KV-STREAMING.md §3.4). */
namespace {
/* One pool's worth of state. A block's bytes in the mirror are `10 * b + i` for i < block_bytes, so
 * a copied block is recognisable byte by byte; a slot nothing wrote holds NOB (a marker, not a
 * block's bytes). */
enum { kKvNob = 0xEE };
struct KVF {
    const int64_t M, blocks, slots, pitch, bbytes;
    T sel, nsel, res, sblk, sst, ctl, host, pool;

    KVF(int64_t M_, int64_t blocks_, int64_t slots_, int64_t pitch_, int64_t bbytes_)
        : M(M_), blocks(blocks_), slots(slots_), pitch(pitch_), bbytes(bbytes_),
          sel(RAD_I32, {M, pitch}), nsel(RAD_I32, {M}), res(RAD_I32, {blocks_}),
          sblk(RAD_I32, {slots_}), sst(RAD_I32, {slots_, 2}), ctl(RAD_I32, {16}),
          host(RAD_U8, {blocks_, bbytes_}), pool(RAD_U8, {slots_, bbytes_}) {}

    void fill_host() {
        for (int64_t b = 0; b < blocks; ++b)
            for (int64_t i = 0; i < bbytes; ++i)
                host.mem[(size_t)(b * bbytes + i)] = (uint8_t)(10 * b + i);
    }
    /* The pool as a resident block leaves it, and a slot no block occupies. */
    void put(int64_t s, int64_t b) {
        for (int64_t i = 0; i < bbytes; ++i)
            pool.mem[(size_t)(s * bbytes + i)] = (uint8_t)(10 * b + i);
    }
    void clear_slot(int64_t s) {
        for (int64_t i = 0; i < bbytes; ++i) pool.mem[(size_t)(s * bbytes + i)] = (uint8_t)kKvNob;
    }
    int call() {
        A a; a.t(sel).t(nsel).t(res).t(sblk).t(sst).t(ctl).t(host).t(pool)
            .i("M", M).i("blocks", blocks).i("slots", slots).i("sel_pitch", pitch)
            .i("block_bytes", bbytes);
        return run("kv_resolve", a);
    }
    int64_t slot(int64_t s, int64_t i) const { return pool.mem[(size_t)(s * bbytes + i)]; }
};

/* A slot's bytes are block b's, from the mirror. */
#define CHECK_SLOT_HOLDS(F, S, B)                                                                \
    for (int64_t i_ = 0; i_ < (F).bbytes; ++i_)                                                  \
        CHECK_EQ((F).slot((S), i_), (int64_t)(uint8_t)(10 * (B) + i_))

/* A whole int operand, element by element, against the values on paper. */
#define CHECK_ENTRIES(T, ...)                                                                    \
    do {                                                                                         \
        const std::vector<int64_t> want_ = { __VA_ARGS__ };                                      \
        REQUIRE_EQ((int64_t)want_.size(), numel(&(T).t));                                        \
        for (size_t i_ = 0; i_ < want_.size(); ++i_) CHECK_EQ((T).iat((int64_t)i_), want_[i_]);   \
    } while (0)
}  /* namespace */

/* Blocks 2 and 5 are resident; the selection names them (hits), then 6 and 4 (misses). The epoch
 * reaches the hit slots, the two misses are claimed once each and placed in ascending block order
 * (4 then 6) - not the order the rows name them - and the copies land in the mirror's bytes. */
TEST(kv_resolve_hits_take_the_epoch_and_misses_are_placed_in_block_order) {
    KVF f(2, 8, 4, 4, 8);
    f.fill_host();
    f.put(0, 2); f.put(3, 5); f.clear_slot(1); f.clear_slot(2);
    f.res.n({ -1, -1, 0, -1, -1, 3, -1, -1 });
    f.sblk.n({ 2, -1, -1, 5 });
    f.sst.n({ 0, 1, 0, 0, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 2, 5, 6, -1, 2, 4, -1, -1 });
    f.nsel.n({ 3, 2 });
    REQUIRE_EQ(f.call(), RAD_OK);

    /* Block 6 came after block 4 in the selection, and still lands after it: ascending block id. */
    CHECK_ENTRIES(f.res, -1, -1, 0, -1, 1, 3, 2, -1);
    CHECK_ENTRIES(f.sblk, 2, 4, 6, 5);
    /* Slot 0 was referenced already (counter 1), so its use takes it to 2; slot 3's block was
     * resident but cold, as the two placements are, so they land at 1. That is the counter doing
     * its job on a normal decode call. */
    CHECK_ENTRIES(f.sst, 1, 2, 1, 1, 1, 1, 1, 1);
    CHECK_ENTRIES(f.sel, 0, 3, 2, -1, 0, 1, -1, -1);
    CHECK_SLOT_HOLDS(f, 0, 2);
    CHECK_SLOT_HOLDS(f, 1, 4);
    CHECK_SLOT_HOLDS(f, 2, 6);
    CHECK_SLOT_HOLDS(f, 3, 5);
    CHECK_EQ(f.ctl.iat(0), 1);            /* epoch */
    CHECK_EQ(f.ctl.iat(1), 3);            /* hand, just past the last slot taken */
    CHECK_EQ(f.ctl.iat(2), 2);            /* placed */
    CHECK_EQ(f.ctl.iat(3), 0);            /* no overflow */
    CHECK_EQ(f.ctl.iat(4), 2);            /* u64 misses */
    CHECK_EQ(f.ctl.iat(6), 5);            /* u64 lookups: 3 + 2 entries, one each */
    CHECK_EQ(f.ctl.iat(8), 1);            /* u64 calls */
}

/* An unreferenced resident slot is the victim, and the block it held leaves the map: the sweep
 * re-points both maps, so a block that is no longer anywhere resident reads as -1. */
TEST(kv_resolve_evicts_an_unreferenced_slot_and_unmaps_its_block) {
    KVF f(1, 8, 4, 1, 4);
    f.fill_host();
    for (int64_t s = 0; s < 4; ++s) f.put(s, s + 1);
    f.res.n({ -1, 0, 1, 2, 3, -1, -1, -1 });
    f.sblk.n({ 1, 2, 3, 4 });
    f.sst.n({ 0, 1, 0, 0, 0, 1, 0, 0 });   /* slot 1's block is unreferenced */
    f.ctl.n({ 0, 1, 0, 0 });               /* the hand starts on slot 1 */
    f.sel.n({ 0 });
    f.nsel.n({ 1 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, 1, 0, -1, 2, 3, -1, -1, -1);
    CHECK_ENTRIES(f.sblk, 1, 0, 3, 4);
    CHECK_ENTRIES(f.sst, 0, 1, 1, 1, 0, 1, 0, 0);
    CHECK_ENTRIES(f.sel, 1);
    CHECK_SLOT_HOLDS(f, 1, 0);
    CHECK_EQ(f.ctl.iat(1), 2);
    CHECK_EQ(f.ctl.iat(2), 1);
}

/* A free slot is claimed before any live block is evicted, even when the hand starts on an
 * unreferenced resident slot: the ring's empty region is not a victim. The block in that slot
 * stays resident, its map entry and its heat counter untouched. */
TEST(kv_resolve_takes_a_free_slot_before_an_unreferenced_one) {
    KVF f(1, 8, 4, 1, 4);
    f.fill_host();
    f.put(0, 3);
    f.res.n({ -1, -1, -1, 0, -1, -1, -1, -1 });
    f.sblk.n({ 3, -1, -1, -1 });
    f.sst.n({ 0, 0, 0, 0, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });               /* the hand starts on the resident slot */
    f.sel.n({ 0 });
    f.nsel.n({ 1 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, 1, -1, -1, 0, -1, -1, -1, -1);
    CHECK_ENTRIES(f.sblk, 3, 0, -1, -1);
    CHECK_ENTRIES(f.sst, 0, 0, 1, 1, 0, 0, 0, 0);   /* slot 0 never went through CLOCK */
    CHECK_ENTRIES(f.sel, 1);
    CHECK_SLOT_HOLDS(f, 1, 0);
    CHECK_SLOT_HOLDS(f, 0, 3);             /* block 3 was not evicted */
    CHECK_EQ(f.ctl.iat(1), 2);             /* hand just past the slot taken */
    CHECK_EQ(f.ctl.iat(2), 1);
}

/* A block whose id fits the ring and whose own slot is free takes that slot, not the hand's: the
 * map stays res[b] = b while the context is shorter than the ring, which is the layout that makes
 * the attention's gather of adjacent ids sequential inside one layer's stripe. Identity does not
 * move the hand, so ctl[1] is still where the call found it. */
TEST(kv_resolve_places_a_block_in_its_own_slot_when_that_slot_is_free) {
    KVF f(1, 12, 8, 4, 4);
    f.fill_host();
    f.res.n({ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 });
    f.sblk.n({ -1, -1, -1, -1, -1, -1, -1, -1 });
    f.sst.n({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 6, 3, 7, 3 });
    f.nsel.n({ 4 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, -1, -1, -1, 3, -1, -1, 6, 7, -1, -1, -1, -1);
    CHECK_ENTRIES(f.sblk, -1, -1, -1, 3, -1, -1, 6, 7);
    CHECK_ENTRIES(f.sel, 6, 3, 7, 3);
    CHECK_SLOT_HOLDS(f, 3, 3);
    CHECK_SLOT_HOLDS(f, 6, 6);
    CHECK_SLOT_HOLDS(f, 7, 7);
    CHECK_EQ(f.ctl.iat(1), 0);            /* identity never moves the hand */
    CHECK_EQ(f.ctl.iat(2), 3);
}

/* A clash is not an eviction. Block 3's own slot holds block 5, so the placement is the ordinary
 * free-slot walk from the hand -- slot 0 here -- and block 5 stays resident and referenced. Block
 * 4's slot is free and takes it even though the hand has moved past it. */
TEST(kv_resolve_own_slot_occupied_falls_back_to_a_free_slot) {
    KVF f(1, 12, 8, 3, 4);
    f.fill_host();
    f.put(3, 5);
    f.res.n({ -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1 });
    f.sblk.n({ -1, -1, -1, 5, -1, -1, -1, -1 });
    f.sst.n({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 3, 4, -1 });
    f.nsel.n({ 2 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, -1, -1, -1, 0, 4, 3, -1, -1, -1, -1, -1, -1);
    CHECK_ENTRIES(f.sblk, 3, -1, -1, 5, 4, -1, -1, -1);
    CHECK_ENTRIES(f.sel, 0, 4, -1);
    CHECK_SLOT_HOLDS(f, 0, 3);
    CHECK_SLOT_HOLDS(f, 4, 4);
    CHECK_SLOT_HOLDS(f, 3, 5);
    CHECK_EQ(f.ctl.iat(1), 1);            /* only the fallback moved it */
    CHECK_EQ(f.ctl.iat(2), 2);
}

/* Second chance: every slot is referenced, so the hand clears all four bits and the miss takes the
 * slot it started on, one full turn later. The bit it passed stays cleared. */
TEST(kv_resolve_gives_a_referenced_slot_a_second_chance) {
    KVF f(1, 8, 4, 2, 4);
    f.fill_host();
    for (int64_t s = 0; s < 4; ++s) f.put(s, s + 4);
    f.res.n({ -1, -1, -1, -1, 0, 1, 2, 3 });
    f.sblk.n({ 4, 5, 6, 7 });
    f.sst.n({ 0, 1, 0, 1, 0, 1, 0, 1 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 0, 1 });
    f.nsel.n({ 2 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, 0, 1, -1, -1, -1, -1, 2, 3);
    CHECK_ENTRIES(f.sblk, 0, 1, 6, 7);
    CHECK_ENTRIES(f.sst, 1, 1, 1, 1, 0, 0, 0, 0);   /* 2 and 3 lost their second chance */
    CHECK_ENTRIES(f.sel, 0, 1);
    CHECK_SLOT_HOLDS(f, 0, 0);
    CHECK_SLOT_HOLDS(f, 1, 1);
    CHECK_SLOT_HOLDS(f, 2, 6);            /* untouched: still block 6 */
    CHECK_EQ(f.ctl.iat(1), 2);
}

/* The heat counter is what makes this different from a one-bit CLOCK: slot 0 was hit on two
 * earlier calls (counter 2), slots 1 and 2 once each. A CLOCK sweep would clear all three bits and
 * evict slot 0 on the second pass; the counter halves 2 -> 1 -> 0, so slot 0 survives two passes and
 * slot 1, whose single step of heat the first pass spends, is the victim. Same inputs, a different
 * slot: the fixture is here to pin the policy and not only agreement between two implementations. */
TEST(kv_resolve_a_slot_hit_twice_survives_two_sweep_passes) {
    KVF f(1, 32, 3, 1, 8);
    f.fill_host();
    for (int64_t s = 0; s < 3; ++s) f.put(s, 10 + s);
    f.res.n({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
    f.sblk.n({ 10, 11, 12 });
    f.sst.n({ 0, 2, 0, 1, 0, 1 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 20 });
    f.nsel.n({ 1 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.sblk, 10, 20, 12);     /* slot 0, the hot one, was not the victim */
    /* The sweep writes only the counter; the epoch column is what a hit or a placement stamps, so
     * slots 0 and 2 keep the 0 they entered with. */
    CHECK_ENTRIES(f.sst, 0, 0, 1, 1, 0, 0);
    CHECK_ENTRIES(f.sel, 1);
    CHECK_EQ(f.ctl.iat(1), 2);
    CHECK_EQ(f.ctl.iat(2), 1);
}

/* The halving sequence, across three calls, with the hand put back on slot 0 so each call passes
 * the hot slot once -- the same visits a hand that wraps once gives it. A saturated 3 is 1 after one
 * pass (a one-bit CLOCK would have cleared it to 0, and the next call would evict it), 1 is 0 after
 * the second, and only the third pass takes the slot. Each step is checked, because the whole of the
 * policy is the difference between 1 and 0 here. */
TEST(kv_resolve_halves_a_saturated_counter_three_to_one_to_zero) {
    KVF f(1, 32, 3, 1, 8);
    f.fill_host();
    for (int64_t s = 0; s < 3; ++s) f.put(s, 10 + s);
    f.res.n({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
    f.sblk.n({ 10, 11, 12 });
    f.sst.n({ 0, 3, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 20 });
    f.nsel.n({ 1 });
    REQUIRE_EQ(f.call(), RAD_OK);
    /* 3 -> 1: the pass cost it one halving, not the whole counter. Slot 1 (cold) was the victim. */
    CHECK_ENTRIES(f.sblk, 10, 20, 12);
    CHECK_ENTRIES(f.sst, 0, 1, 1, 1, 0, 0);

    f.sel.n({ 21 });
    ((int32_t*)f.ctl.t.data)[1] = 0;       /* the hand back on slot 0, the epoch untouched */
    REQUIRE_EQ(f.call(), RAD_OK);
    /* 1 -> 0: slot 0 passed again and is still resident; slot 2 (cold) is the victim. */
    CHECK_ENTRIES(f.sblk, 10, 20, 21);
    CHECK_ENTRIES(f.sst, 0, 0, 1, 0, 2, 1);

    f.sel.n({ 22 });
    ((int32_t*)f.ctl.t.data)[1] = 0;
    REQUIRE_EQ(f.call(), RAD_OK);
    /* 0: the third visit takes it. */
    CHECK_ENTRIES(f.sblk, 22, 20, 21);
    CHECK_ENTRIES(f.sst, 3, 1, 1, 0, 2, 1);
    CHECK_EQ(f.res.iat(10), -1);           /* evicted on the third pass */
    CHECK_EQ(f.res.iat(20), 1);
    CHECK_EQ(f.res.iat(21), 2);
    CHECK_EQ(f.res.iat(22), 0);
}

/* A saturated counter and a slot placed on the previous call differ by two steps of heat, and the
 * sweep spends the fresh one first: {3,1,1,1} walks two full laps, leaving the 3 at 1 then 0 and
 * taking slot 1. A one-bit CLOCK, for which all four are "referenced", evicts slot 0 -- the hottest
 * block in the ring. */
TEST(kv_resolve_a_saturated_counter_outlasts_a_fresh_placement) {
    KVF f(1, 32, 4, 1, 8);
    f.fill_host();
    for (int64_t s = 0; s < 4; ++s) f.put(s, 10 + s);
    f.res.n({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
    f.sblk.n({ 10, 11, 12, 13 });
    f.sst.n({ 0, 3, 0, 1, 0, 1, 0, 1 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 20 });
    f.nsel.n({ 1 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.sblk, 10, 20, 12, 13);
    CHECK_ENTRIES(f.sst, 0, 0, 1, 1, 0, 0, 0, 0);
    CHECK_EQ(f.ctl.iat(1), 2);
}

/* One bump a slot a call, not one an entry: block 10 is resident and named by both entries, so its
 * counter goes 1 -> 2 rather than 1 -> 3. The device kernel counts the first thread to stamp a slot
 * (`atomicExch`), so a per-entry bump in the reference would be a real disagreement, not a detail. */
TEST(kv_resolve_counts_one_bump_a_slot_a_call) {
    KVF f(1, 32, 3, 2, 8);
    f.fill_host();
    for (int64_t s = 0; s < 3; ++s) f.put(s, 10 + s);
    f.res.n({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
    f.sblk.n({ 10, 11, 12 });
    f.sst.n({ 0, 1, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 10, 10 });
    f.nsel.n({ 2 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.sst, 1, 2, 0, 0, 0, 0);
    CHECK_ENTRIES(f.sel, 0, 0);
    CHECK_EQ(f.ctl.iat(2), 0);
}

/* Overflow: three hits hold slots 0..2 and two misses want the one slot left. The first is placed,
 * the second has nowhere and must NOT steal the slot the first took (every slot of the ring is
 * marked with this call's epoch, so the sweep passes it), nor any hit's. ctl[3] latches, the
 * unplaced block reverts to -1, and its selection entry is rewritten to slot 0: leaving the block id
 * would make a device attention dereference an address past the ring, and ctl[3] is what says the
 * table is unusable. */
TEST(kv_resolve_overflow_latches_instead_of_taking_a_slot_it_just_filled) {
    KVF f(1, 8, 4, 5, 4);
    f.fill_host();
    for (int64_t s = 0; s < 3; ++s) f.put(s, s);
    f.clear_slot(3);
    f.res.n({ 0, 1, 2, -1, -1, -1, -1, -1 });
    f.sblk.n({ 0, 1, 2, -1 });
    f.sst.n({ 0, 1, 0, 1, 0, 1, 0, 0 });
    f.ctl.n({ 0, 3, 0, 0 });               /* the hand starts on the one free slot */
    f.sel.n({ 6, 7, 0, 1, 2 });
    f.nsel.n({ 5 });
    REQUIRE_EQ(f.call(), RAD_OK);

    CHECK_ENTRIES(f.res, 0, 1, 2, -1, -1, -1, 3, -1);
    CHECK_ENTRIES(f.sblk, 0, 1, 2, 6);
    CHECK_ENTRIES(f.sst, 1, 2, 1, 2, 1, 2, 1, 1);   /* three hits on referenced slots, one placement */
    /* Entry 1 is block 7, unplaced: it names slot 0, a defined address inside the ring, and ctl[3]
     * below is what marks the table unusable. */
    CHECK_ENTRIES(f.sel, 3, 0, 0, 1, 2);
    CHECK_SLOT_HOLDS(f, 3, 6);             /* the second miss did not overwrite the first */
    CHECK_SLOT_HOLDS(f, 0, 0);
    CHECK_EQ(f.ctl.iat(2), 1);
    CHECK_EQ(f.ctl.iat(3), 1);             /* overflow, latched */
    CHECK_EQ(f.ctl.iat(4), 1);
    CHECK_EQ(f.ctl.iat(6), 5);
}

/* A launch's claim count is derived from its own operands, never a constant: one claim is one
 * distinct absent block a selection entry names, so a launch makes at most `M * sel_pitch` of them
 * -- 2052 for four rows of 513, which is what `--max-num-seqs 4` selects -- and the miss list has to
 * hold every one of them. This runs the same fixture at 2048 and at 2052 claims and holds the wider
 * launch to the narrower one's placement: every block the two share lands in the same slot, so the
 * launch above the old fixed list is not a different placement, and ctl[12] reports the depth the
 * list had to hold. */
TEST(kv_resolve_places_2052_claims_exactly_as_it_places_2048) {
    const int64_t bbytes = 8, blocks = 4096, pitch = 513, M = 4, slots = 2100, first = 100;
    /* Every slot free: ids below the ring take their own slot, and the ids past it walk the free
     * slots from the hand. */
    auto launch = [&](int64_t per_row) {
        KVF f(M, blocks, slots, pitch, bbytes);
        f.fill_host();
        std::vector<int64_t> sel((size_t)(M * pitch), -1), nsel((size_t)M, per_row);
        for (int64_t q = 0; q < M; ++q)
            for (int64_t i = 0; i < per_row; ++i)
                sel[(size_t)(q * pitch + i)] = first + q * per_row + i;
        f.sel.n(sel);
        f.nsel.n(nsel);
        f.res.n(std::vector<int64_t>((size_t)blocks, -1));
        f.sblk.n(std::vector<int64_t>((size_t)slots, -1));
        f.sst.n(std::vector<int64_t>((size_t)(slots * 2), 0));
        f.ctl.n(std::vector<int64_t>(16, 0));
        CHECK_EQ(f.call(), RAD_OK);
        return f;
    };
    KVF small = launch(512);   /* 4 x 512 = 2048 claims */
    KVF large = launch(513);   /* 4 x 513 = 2052 claims */

    /* 2052 claims are all placed, ascending from the first, and the counters say so. Identity
     * covers every id below the ring: block `first + k` takes slot `first + k`. The 52 ids past
     * `slots` (2100..2151) fall back to the first free slots after the hand, which is still 0. */
    for (int64_t k = 0; k < 2000; ++k) {              /* ids first .. slots - 1 */
        CHECK_EQ(large.res.iat(first + k), first + k);
        CHECK_EQ(large.sblk.iat(first + k), first + k);
    }
    for (int64_t j = 0; j < 52; ++j) {                /* ids slots .. first + 2051 */
        CHECK_EQ(large.res.iat(slots + j), j);
        CHECK_EQ(large.sblk.iat(j), slots + j);
    }
    CHECK_EQ(large.ctl.iat(12), 2052);    /* the claim list's depth */
    CHECK_EQ(large.ctl.iat(10), 2052);    /* every claim is a miss: no block was resident */
    CHECK_EQ(large.ctl.iat(2), 2052);
    CHECK_EQ(large.ctl.iat(1), 52);       /* identity never moves the hand: 52 fallbacks did */
    CHECK_EQ(large.ctl.iat(3), 0);
    CHECK_SLOT_HOLDS(large, 0, slots);
    CHECK_SLOT_HOLDS(large, 51, first + 2051);
    CHECK_SLOT_HOLDS(large, 2051, first + 1951);

    /* The 2048-claim launch places its own blocks in exactly those slots. The four blocks only the
     * wider launch names (2148..2151) are the sole difference, so they are skipped. */
    for (int64_t s = 0; s < slots; ++s)
        if (large.sblk.iat(s) < first + 2048)
            CHECK_EQ(small.sblk.iat(s), large.sblk.iat(s));
    for (int64_t k = 0; k < 2048; ++k)
        CHECK_EQ(small.res.iat(first + k), large.res.iat(first + k));
    CHECK_EQ(small.ctl.iat(12), 2048);
    CHECK_EQ(small.ctl.iat(2), 2048);
    /* The rewrite: every valid entry names the slot of its block, and the padding past nsel[q] is
     * -1 (the 2048-claim launch has one such entry a row). */
    for (int64_t q = 0; q < M; ++q)
        for (int64_t i = 0; i < pitch; ++i) {
            const int64_t e = q * pitch + i;
            /* The block's own id while it fits the ring, then the slot it fell back to. The wider
             * launch packs 513 a row, the narrower one 512. */
            const int64_t bl = first + q * pitch + i;
            const int64_t bs = first + q * 512 + i;
            CHECK_EQ(large.sel.iat(e), bl < slots ? bl : bl - slots);
            if (i < 512) CHECK_EQ(small.sel.iat(e), bs < slots ? bs : bs - slots);
            else         CHECK_EQ(small.sel.iat(e), -1);
        }
}


TEST(kv_resolve_refuses_a_selection_that_does_not_fit_the_maps) {
    KVF f(1, 8, 4, 2, 4);
    f.fill_host();
    f.res.n({ -1, -1, -1, -1, -1, -1, -1, -1 });
    f.sblk.n({ -1, -1, -1, -1 });
    f.sst.n({ 0, 0, 0, 0, 0, 0, 0, 0 });
    f.ctl.n({ 0, 0, 0, 0 });
    f.sel.n({ 8, -1 });                    /* block 8 is outside [0, blocks) */
    f.nsel.n({ 1 });
    CHECK_EQ(f.call(), RAD_E_INVAL);
    CHECK_EQ(f.sel.iat(0), 8);             /* nothing rewritten */
    CHECK_ENTRIES(f.res, -1, -1, -1, -1, -1, -1, -1, -1);
    CHECK_ENTRIES(f.sel, 8, -1);
    CHECK_EQ(f.ctl.iat(0), 0);             /* no epoch bump either */
    f.sel.n({ 1, 1 });
    f.nsel.n({ 3 });                       /* more entries than the row holds */
    CHECK_EQ(f.call(), RAD_E_SHAPE);
    CHECK_EQ(f.ctl.iat(0), 0);
}

/* qsa_tail_store: only the last `ring` tokens of a sequence write, to row pos % ring of the
 * sequence's slot; a sequence with no slot and every untargeted row are left alone. */
TEST(qsa_tail_store_writes_the_last_ring_tokens_into_the_sequences_slot) {
    const int64_t n = 2, ratio = 4, ring = 6, slots = 4;
    /* Sequence 0: rows 0..7 at positions 10..17. Sequence 1: rows 8..9, no slot. Sequence 2:
     * rows 10..11 at positions 5, 6 (wrapping the ring). */
    std::vector<float> kv;
    for (int t = 0; t < 12; ++t) { kv.push_back((float)t); kv.push_back((float)-t); }
    T k(RAD_BF16, {12, n}); k.f(kv);
    T pos(RAD_I32, {12}); pos.n({10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 5, 6});
    T cu(RAD_I32, {4}); cu.n({0, 8, 10, 12});
    std::vector<float> init((size_t)(slots * ring * n), -100.0f);
    T st(RAD_BF16, {slots, 1, ring, n}); st.f(init);
    T sx(RAD_I32, {3, 2}); sx.n({3, 0, -1, 0, 1, 0});
    A a; a.t(k).t(pos).t(cu).t(st).t(sx).i("M", 12).i("n", n).i("ratio", ratio).i("ring", ring)
          .i("seqs", 3).s("dtype", "bf16");
    CHECK_OK(run("qsa_tail_store", a));
    auto at = [&](int64_t slot, int64_t row, int64_t d) { return st.at((slot * ring + row) * n + d); };
    for (int t = 2; t < 8; ++t) {                        /* positions 12..17 -> rows 0..5 */
        CHECK_EQ(at(3, (10 + t) % ring, 0), (float)t);
        CHECK_EQ(at(3, (10 + t) % ring, 1), (float)-t);
    }
    CHECK_EQ(at(1, 5, 0), 10.0f); CHECK_EQ(at(1, 0, 0), 11.0f);
    CHECK_EQ(at(1, 1, 0), -100.0f);
    for (int64_t row = 0; row < ring; ++row) {
        CHECK_EQ(at(0, row, 0), -100.0f);
        CHECK_EQ(at(2, row, 0), -100.0f);
    }
}

/* qsa_work over a batch whose first sequence starts mid-block: block 2 takes positions 8..10 from
 * the tail (row q % ring of the sequence's slot) and 11 from the batch, block 3 is all batch; the
 * second sequence (positions 0..2) completes nothing; slots past the count are -1 and position 0. */
TEST(qsa_work_gathers_completing_blocks_from_the_tail_and_the_batch) {
    const int64_t n = 2, ratio = 4, ring = 7, slots = 2, work = 4;
    std::vector<float> kv;
    for (int t = 0; t < 9; ++t) { kv.push_back((float)(100 + t)); kv.push_back((float)(200 + t)); }
    T k(RAD_BF16, {9, n}); k.f(kv);
    T pos(RAD_I32, {9}); pos.n({11, 12, 13, 14, 15, 16, 0, 1, 2});
    T cu(RAD_I32, {3}); cu.n({0, 6, 9});
    std::vector<float> tv;
    for (int s = 0; s < slots; ++s)
        for (int r = 0; r < ring; ++r) { tv.push_back((float)(-10 * s - r)); tv.push_back(0.5f); }
    T tail(RAD_BF16, {slots, 1, ring, n}); tail.f(tv);
    T sx(RAD_I32, {2, 1}); sx.n({1, 0});
    T bt(RAD_I32, {2, 8}); bt.n({50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67});
    std::vector<float> nines((size_t)(work * ratio * n), 9.0f);
    T stage(RAD_BF16, {work * ratio, n}); stage.f(nines);
    T page(RAD_I32, {work}), bpos(RAD_I32, {work}), nc(RAD_I32, {2});
    A a; a.t(k).t(pos).t(cu).t(tail).t(sx).t(bt).tnull().t(stage).t(page).t(bpos).t(nc)
          .i("M", 9).i("n", n).i("ratio", ratio).i("ring", ring).i("seqs", 2).i("work", work)
          .s("dtype", "bf16");
    CHECK_OK(run("qsa_work", a));
    CHECK_EQ(nc.iat(0), 4LL); CHECK_EQ(nc.iat(1), 0LL);
    CHECK_EQ(page.iat(0), 52LL); CHECK_EQ(page.iat(1), 53LL);
    CHECK_EQ(page.iat(2), -1LL); CHECK_EQ(page.iat(3), -1LL);
    CHECK_EQ(bpos.iat(0), 8LL); CHECK_EQ(bpos.iat(1), 12LL);
    CHECK_EQ(bpos.iat(2), 0LL); CHECK_EQ(bpos.iat(3), 0LL);
    /* Block 2: tail rows 8 % 7, 9 % 7, 10 % 7 of slot 1, then batch row 0 (position 11). */
    CHECK_EQ(stage.at(0 * n), -11.0f); CHECK_EQ(stage.at(1 * n), -12.0f);
    CHECK_EQ(stage.at(2 * n), -13.0f); CHECK_EQ(stage.at(3 * n), 100.0f);
    /* Block 3: batch rows 1..4 (positions 12..15). */
    for (int i = 0; i < 4; ++i) CHECK_EQ(stage.at((4 + i) * n), (float)(101 + i));
    CHECK_EQ(stage.at(8 * n), 9.0f);                 /* an empty slot's rows are not written */

    /* With the step's rotary components, a first token in the batch reads its own and an earlier
     * one its position plus the first row's shift. */
    std::vector<int64_t> rv;
    for (int c = 0; c < 3; ++c)
        for (int t = 0; t < 9; ++t) rv.push_back(1000 * (c + 1) + t);
    T rope(RAD_I32, {3, 9}); rope.n(rv);
    T bp3(RAD_I32, {work, 3});
    A b; b.t(k).t(pos).t(cu).t(tail).t(sx).t(bt).t(rope).t(stage).t(page).t(bp3).t(nc)
          .i("M", 9).i("n", n).i("ratio", ratio).i("ring", ring).i("seqs", 2).i("work", work)
          .s("dtype", "bf16");
    CHECK_OK(run("qsa_work", b));
    for (int c = 0; c < 3; ++c) CHECK_EQ(bp3.iat(0 * 3 + c), 8LL + (1000 - 11));   /* 8 + rope[0][0] - 11 */
    for (int c = 0; c < 3; ++c) CHECK_EQ(bp3.iat(1 * 3 + c), 1000LL * (c + 1) + 1);
}

/* ================================================================== QSA tree verify
 *
 * A tree verify's rows must select what a sequential decode of their own root-to-node path selects
 * (docs/QSA.md, "Tree verify"), and the committed state it leaves must be the sequential decode's.
 * These tests build explicit trees and run the real op chain twice: once as a tree (qsa_tail_compact,
 * qsa_work, the main and the scratch qsa_block_key, qsa_tail_store, the committed and the region
 * qsa_score, qsa_select with the tree operands) and, per node, once as the ordinary chain over the
 * node's path from the same starting state. Visible key sets are compared, not table order: a tree
 * row names its committed pages in `sel` and its region keys in `vmask`, a chain row names all of
 * them as pages. The block keys the two ways produce and the scores they give are compared bit for
 * bit. The tree op parameters (topk, ratio) are small so contexts of a few dozen tokens are far
 * above the exactness bound, and a key palette makes equal scores common so the tie rule decides. */
namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    int below(int n) { return (int)(next() % (uint32_t)n); }
};

struct QWorld {
    int64_t n = 8, heads = 2, ratio = 4, ring = 0, topk = 4, nbmax = 40, rot = 0, RW = 8;
    int64_t work = 24;                       /* a launch's work-list capacity */
    int64_t nseq = 1;                        /* sequences: slot s, table row s, pages s * nbmax + b */
    T tail, bkp, tab, wk;
    QWorld(int64_t ring_, int64_t topk_, int64_t rot_, int64_t nseq_ = 1)
        : ring(ring_), topk(topk_), rot(rot_), nseq(nseq_), tail(RAD_BF16, { nseq_, 1, ring_, 8 }),
          bkp(RAD_BF16, { nseq_ * 40, 8 }), tab(RAD_I32, { nseq_, 40 }), wk(RAD_F32, { 8 }) {
        std::vector<int64_t> id;
        for (int64_t s = 0; s < nseq; ++s)
            for (int64_t b = 0; b < nbmax; ++b) id.push_back(s * nbmax + b);
        tab.n(id);                           /* page id == block index (plus the sequence's base) */
        wk.f({ 1, 1, 1, 1, 1, 1, 1, 1 });
        /* Unwritten pages and ring rows are recognisable. */
        tail.f(std::vector<float>((size_t)(nseq * ring * n), -77.0f));
        bkp.f(std::vector<float>((size_t)(nseq * nbmax * n), -55.0f));
    }

    A bkey(const T& k, const T& page, const T& bpos, T& out, int64_t M) const {
        A a; a.t(k).t(wk).t(page).t(bpos).t(out).i("M", M).i("n", n).i("ratio", ratio)
              .i("rotary_dim", rot).d("eps", 1e-6).d("theta", 1e4).d("scale", 1).d("wadd", 0)
              .s("dtype", "bf16");
        return a;
    }
};

/* A key row from a palette of five vectors, or a fresh one. */
std::vector<float> qkey(Rng& r, int64_t n, bool palette) {
    std::vector<float> v((size_t)n);
    if (palette) {
        const int p = r.below(3);
        for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)(((p + 1) * (int)(i + 1)) % 5 - 2);
    } else {
        for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)(r.below(9) - 4) * 0.5f;
    }
    return v;
}

/* The ordinary chain over `K` (rows at consecutive positions from pos0): work, block keys, tail
 * store, then score and select for every row. `Q` may be empty (no scoring). Returns the last
 * row's selection. */
struct ChainOut { std::vector<int64_t> sel; int64_t nsel = 0, su = 0; std::vector<float> score; int64_t nc = 0; };

void qsa_chain(QWorld& w, const std::vector<std::vector<float>>& K,
               const std::vector<std::vector<float>>& Q, int64_t pos0, bool score, ChainOut* outp) {
    const int64_t R = (int64_t)K.size(), n = w.n, ratio = w.ratio;
    const int64_t work = (R + ratio - 1) / ratio + 2;
    T k(RAD_BF16, { R, n }), pos(RAD_I32, { R }), cu(RAD_I32, { 2 }), sx(RAD_I32, { 1, 1 });
    std::vector<float> kv;
    std::vector<int64_t> pv;
    for (int64_t t = 0; t < R; ++t) { kv.insert(kv.end(), K[(size_t)t].begin(), K[(size_t)t].end()); pv.push_back(pos0 + t); }
    k.f(kv); pos.n(pv); cu.n({ 0, R }); sx.n({ 0 });
    T stage(RAD_BF16, { work * ratio, n }), page(RAD_I32, { work }), bpos(RAD_I32, { work }), nc(RAD_I32, { 1 });
    A a; a.t(k).t(pos).t(cu).t(w.tail).t(sx).t(w.tab).tnull().t(stage).t(page).t(bpos).t(nc)
          .i("M", R).i("n", n).i("ratio", ratio).i("ring", w.ring).i("seqs", 1).i("work", work)
          .s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_work", a), RAD_OK);
    A b = w.bkey(stage, page, bpos, w.bkp, work);
    REQUIRE_EQ(run("qsa_block_key", b), RAD_OK);
    A c; c.t(k).t(pos).t(cu).t(w.tail).t(sx).i("M", R).i("n", n).i("ratio", ratio).i("ring", w.ring)
          .i("seqs", 1).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_tail_store", c), RAD_OK);
    if (!score) return;
    ChainOut& o = *outp;
    o.nc = nc.iat(0);
    T q(RAD_BF16, { R, w.heads * n }), sc(RAD_F32, { R, w.nbmax });
    std::vector<float> qv;
    for (int64_t t = 0; t < R; ++t) qv.insert(qv.end(), Q[(size_t)t].begin(), Q[(size_t)t].end());
    q.f(qv);
    A d; d.t(q).t(w.bkp).t(w.tab).t(cu).t(nc).t(sc).i("M", R).i("n", n).i("heads", w.heads)
          .i("blocks", w.nbmax).i("seqs", 1).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_score", d), RAD_OK);
    T se(RAD_I32, { R, w.topk + 1 }), ns(RAD_I32, { R }), su(RAD_I32, { R });
    A e; e.t(sc).t(w.tab).t(nc).t(pos).t(cu).t(se).t(ns).t(su).i("M", R).i("blocks", w.nbmax)
          .i("topk", w.topk).i("ratio", ratio).i("seqs", 1);
    REQUIRE_EQ(run("qsa_select", e), RAD_OK);
    const int64_t r = R - 1;
    for (int64_t i = 0; i <= w.topk; ++i) o.sel.push_back(se.iat(r * (w.topk + 1) + i));
    o.nsel = ns.iat(r); o.su = su.iat(r);
    for (int64_t bk = 0; bk < w.nbmax; ++bk) o.score.push_back(sc.at(r * w.nbmax + bk));
}

/* Everything a tree step leaves for the attention and the checks. */
struct TreeOut {
    std::vector<std::vector<int64_t>> sel;      /* per row, topk + 1 */
    std::vector<int64_t> nsel, su, vmask, nc;
    std::vector<std::vector<float>> rscore;     /* per row, RW */
    std::vector<int64_t> rnc;
    std::vector<std::vector<float>> skey;       /* per row, n */
    std::vector<int64_t> spage;
};

/* One tree verify step of one sequence: rows are nodes (row 0 the anchor), par[j] the parent,
 * positions base + depth. `na`/`path` (null when absent) run the compaction first. */
void qsa_tree_step(QWorld& w, const std::vector<std::vector<float>>& K,
                   const std::vector<std::vector<float>>& Q, const std::vector<int64_t>& par,
                   int64_t base, int64_t na, const std::vector<int64_t>* path, bool score,
                   TreeOut* outp, int64_t work_cap = 0) {
    const int64_t R = (int64_t)K.size(), n = w.n, ratio = w.ratio, RW = w.RW;
    const int64_t work = work_cap ? work_cap : w.work;
    std::vector<int64_t> depth((size_t)R, 0), pv, sv;
    for (int64_t j = 0; j < R; ++j) {
        depth[(size_t)j] = j ? depth[(size_t)par[(size_t)j]] + 1 : 0;
        pv.push_back(base + depth[(size_t)j]);
        sv.push_back(j ? par[(size_t)j] : -1);
    }
    T k(RAD_BF16, { R, n }), pos(RAD_I32, { R }), cu(RAD_I32, { 2 }), sx(RAD_I32, { 1, 1 }), sp(RAD_I32, { R });
    std::vector<float> kv;
    for (int64_t t = 0; t < R; ++t) kv.insert(kv.end(), K[(size_t)t].begin(), K[(size_t)t].end());
    k.f(kv); pos.n(pv); cu.n({ 0, R }); sx.n({ 0 }); sp.n(sv);
    T nav(RAD_I32, { 1 }); nav.n({ na });
    T pth(RAD_I32, { 1, 16 });
    if (path) { std::vector<int64_t> pp = *path; pp.resize(16, 0); pth.n(pp); }
    if (path) {
        A cmp; cmp.t(pos).t(cu).t(nav).t(pth).t(w.tail).t(sx).i("M", 1).i("n", n).i("ring", w.ring)
                  .i("seqs", 1).s("dtype", "bf16");
        REQUIRE_EQ(run("qsa_tail_compact", cmp), RAD_OK);
    }
    T stage(RAD_BF16, { work * ratio, n }), page(RAD_I32, { work }), bpos(RAD_I32, { work }), nc(RAD_I32, { 1 });
    T sst(RAD_BF16, { R * ratio, n }), spg(RAD_I32, { R }), sbp(RAD_I32, { R });
    T rtb(RAD_I32, { R, RW }), rnc(RAD_I32, { R }), rcu(RAD_I32, { R + 1 });
    A a; a.t(k).t(pos).t(cu).t(w.tail).t(sx).t(w.tab).tnull().t(stage).t(page).t(bpos).t(nc)
          .t(sp).t(nav).t(sst).t(spg).t(sbp).t(rtb).t(rnc).t(rcu)
          .i("M", R).i("n", n).i("ratio", ratio).i("ring", w.ring).i("seqs", 1).i("work", work)
          .s("dtype", "bf16");
    if (!path) a.ts[12] = RadTensor{};            /* num_accepted absent: no heal */
    REQUIRE_EQ(run("qsa_work", a), RAD_OK);
    A b = w.bkey(stage, page, bpos, w.bkp, work);
    REQUIRE_EQ(run("qsa_block_key", b), RAD_OK);
    T skey(RAD_BF16, { R, n });
    skey.f(std::vector<float>((size_t)(R * n), -33.0f));
    A b2 = w.bkey(sst, spg, sbp, skey, R);
    REQUIRE_EQ(run("qsa_block_key", b2), RAD_OK);
    A c; c.t(k).t(pos).t(cu).t(w.tail).t(sx).t(sp).i("M", R).i("n", n).i("ratio", ratio)
          .i("ring", w.ring).i("seqs", 1).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_tail_store", c), RAD_OK);
    TreeOut& o = *outp;
    o.nc.push_back(nc.iat(0));
    for (int64_t j = 0; j < R; ++j) {
        o.spage.push_back(spg.iat(j));
        o.rnc.push_back(rnc.iat(j));
        std::vector<float> sk;
        for (int64_t d = 0; d < n; ++d) sk.push_back(skey.at(j * n + d));
        o.skey.push_back(sk);
    }
    if (!score) return;
    T q(RAD_BF16, { R, w.heads * n }), sc(RAD_F32, { R, w.nbmax }), rsc(RAD_F32, { R, RW });
    std::vector<float> qv;
    for (int64_t t = 0; t < R; ++t) qv.insert(qv.end(), Q[(size_t)t].begin(), Q[(size_t)t].end());
    q.f(qv);
    A d; d.t(q).t(w.bkp).t(w.tab).t(cu).t(nc).t(sc).i("M", R).i("n", n).i("heads", w.heads)
          .i("blocks", w.nbmax).i("seqs", 1).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_score", d), RAD_OK);
    A d2; d2.t(q).t(skey).t(rtb).t(rcu).t(rnc).t(rsc).i("M", R).i("n", n).i("heads", w.heads)
            .i("blocks", RW).i("seqs", R).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_score", d2), RAD_OK);
    T se(RAD_I32, { R, w.topk + 1 }), ns(RAD_I32, { R }), su(RAD_I32, { R }), vm(RAD_I64, { R });
    A e; e.t(sc).t(w.tab).t(nc).t(pos).t(cu).t(se).t(ns).t(su).t(sp).t(rsc).t(vm)
          .i("M", R).i("blocks", w.nbmax).i("topk", w.topk).i("ratio", ratio).i("seqs", 1);
    REQUIRE_EQ(run("qsa_select", e), RAD_OK);
    for (int64_t j = 0; j < R; ++j) {
        std::vector<int64_t> s;
        for (int64_t i = 0; i <= w.topk; ++i) s.push_back(se.iat(j * (w.topk + 1) + i));
        o.sel.push_back(s);
        o.nsel.push_back(ns.iat(j)); o.su.push_back(su.iat(j)); o.vmask.push_back(vm.iat(j));
        std::vector<float> rs;
        for (int64_t i = 0; i < RW; ++i) rs.push_back(rsc.at(j * RW + i));
        o.rscore.push_back(rs);
    }
}

/* A random tree of `R` rows. */
std::vector<int64_t> qsa_rand_tree(Rng& r, int64_t R, int deep) {
    std::vector<int64_t> par((size_t)R, -1);
    std::vector<int64_t> depth((size_t)R, 0);
    for (int64_t j = 1; j < R; ++j) {
        for (int tries = 0;; ++tries) {
            int64_t p = deep ? (j - 1 - r.below((int)std::min<int64_t>(j, 3))) : r.below((int)j);
            if (tries > 6) p = r.below((int)j);          /* a deep chain ran out of shallow parents */
            if (depth[(size_t)p] < 6) { par[(size_t)j] = p; depth[(size_t)j] = depth[(size_t)p] + 1; break; }
        }
    }
    return par;
}

/* The tokens a chain row at `P` reads: its table's pages in full and the partial one up to P % ratio. */
std::set<int64_t> qsa_chain_visible(const ChainOut& c, int64_t P, int64_t ratio) {
    std::set<int64_t> v;
    const bool partial = (P + 1) % ratio != 0;
    for (int64_t e = 0; e < c.nsel; ++e) {
        const int64_t pg = c.sel[(size_t)e];
        const int64_t cnt = (partial && e == c.nsel - 1) ? P % ratio + 1 : ratio;
        for (int64_t i = 0; i < cnt; ++i) v.insert(pg * ratio + i);
    }
    return v;
}

/* The tokens a tree row reads: its committed pages in full, then the region keys its mask names. */
std::set<int64_t> qsa_tree_visible(const TreeOut& o, int64_t row, const std::vector<int64_t>& pos,
                                   int64_t base, int64_t ratio) {
    std::set<int64_t> v;
    for (int64_t e = 0; e < o.nsel[(size_t)row]; ++e)
        for (int64_t i = 0; i < ratio; ++i) v.insert(o.sel[(size_t)row][(size_t)e] * ratio + i);
    const int64_t rb = base / ratio * ratio;
    for (int k = 0; k < 64; ++k) {
        if (!((o.vmask[(size_t)row] >> k) & 1)) continue;
        if (rb + k < base) v.insert(rb + k);
        else v.insert(pos[(size_t)(rb + k - base)]);   /* node j's position */
    }
    return v;
}

/* One scenario: a prefix of `base` tokens, a random tree, every node against its sequential path. */
void qsa_tree_scenario(uint64_t seed, int64_t base, int64_t R, int64_t topk, int64_t rot, bool palette,
                       int64_t ratio_unused, const char* tag, int* ties_seen) {
    (void)ratio_unused;
    Rng r(seed);
    const int64_t ring = 4 + 12;
    QWorld w(ring, topk, rot);
    const int64_t n = w.n;
    std::vector<std::vector<float>> pk;
    for (int64_t t = 0; t < base; ++t) pk.push_back(qkey(r, n, palette));
    qsa_chain(w, pk, {}, 0, false, nullptr);            /* the prefix, as a prefill */
    const std::vector<uint8_t> tail0 = w.tail.mem, bk0 = w.bkp.mem;

    std::vector<std::vector<float>> K, Q;
    for (int64_t j = 0; j < R; ++j) {
        K.push_back(qkey(r, n, palette));
        std::vector<float> q;
        for (int64_t h = 0; h < w.heads; ++h) { auto x = qkey(r, n, palette); q.insert(q.end(), x.begin(), x.end()); }
        Q.push_back(q);
    }
    const std::vector<int64_t> par = qsa_rand_tree(r, R, seed & 1);
    TreeOut o;
    qsa_tree_step(w, K, Q, par, base, 0, nullptr, true, &o);

    /* The verify wrote no committed page but the anchor's block. */
    const int64_t first_open = base / w.ratio;
    for (int64_t b = first_open + 1; b < w.nbmax; ++b)
        CHECK_EQ(w.bkp.at(b * n), -55.0f);
    CHECK_EQ(o.nc[0], first_open);
    /* Node j's raw key is at ring row (base + j) % ring, distinct per node. */
    for (int64_t j = 0; j < R; ++j)
        for (int64_t d = 0; d < n; ++d)
            CHECK_EQ(w.tail.at(((base + j) % w.ring) * n + d), K[(size_t)j][(size_t)d]);

    std::vector<int64_t> pos, depth((size_t)R, 0);
    for (int64_t j = 0; j < R; ++j) {
        depth[(size_t)j] = j ? depth[(size_t)par[(size_t)j]] + 1 : 0;
        pos.push_back(base + depth[(size_t)j]);
    }
    for (int64_t x = 0; x < R; ++x) {
        std::vector<int64_t> rows;
        for (int64_t y = x; y >= 0; y = par[(size_t)y]) { rows.push_back(y); if (y == 0) break; }
        std::reverse(rows.begin(), rows.end());
        std::vector<std::vector<float>> pK, pQ;
        for (int64_t y : rows) { pK.push_back(K[(size_t)y]); pQ.push_back(Q[(size_t)y]); }
        std::memcpy(w.tail.mem.data(), tail0.data(), tail0.size());
        std::memcpy(w.bkp.mem.data(), bk0.data(), bk0.size());
        ChainOut c;
        qsa_chain(w, pK, pQ, base, true, &c);
        const int64_t P = pos[(size_t)x];
        const int64_t ntot = (P + 1) / w.ratio;
        const std::string what = std::string(tag) + " seed " + std::to_string(seed) + " node " + std::to_string(x);
        /* The scratch block keys are the chain's committed ones. */
        for (int64_t i = 0; i < o.rnc[(size_t)x]; ++i) {
            const int64_t blk = first_open + i;
            CHECK(o.rscore[(size_t)x][(size_t)i] == c.score[(size_t)blk]);
            if (o.rscore[(size_t)x][(size_t)i] != c.score[(size_t)blk])
                fprintf(stderr, "    %s: region score %lld differs\n", what.c_str(), (long long)blk);
        }
        CHECK_EQ(o.rnc[(size_t)x], ntot - first_open);
        const auto cv = qsa_chain_visible(c, P, w.ratio);
        const auto tv = qsa_tree_visible(o, x, pos, base, w.ratio);
        CHECK(cv == tv);
        if (cv != tv) fprintf(stderr, "    %s: visible sets differ (chain %zu tree %zu)\n", what.c_str(), cv.size(), tv.size());
        /* The mask itself, by flat slot identity: bit (base - rb) + y for the ancestor-or-self y
         * (a sibling at the same depth is another bit), bit q - rb for a committed q, each set iff
         * the chain read the token at that position; nothing else is set. */
        {
            const int64_t rb = base / w.ratio * w.ratio;
            uint64_t want = 0;
            for (int64_t q = rb; q < base; ++q)
                if (cv.count(q)) want |= 1ull << (q - rb);
            for (int64_t y : rows)
                if (cv.count(pos[(size_t)y])) want |= 1ull << (base - rb + y);
            CHECK_EQ((uint64_t)o.vmask[(size_t)x], want);
            if ((uint64_t)o.vmask[(size_t)x] != want)
                fprintf(stderr, "    %s: vmask %llx want %llx\n", what.c_str(),
                        (unsigned long long)o.vmask[(size_t)x], (unsigned long long)want);
        }
        /* The keys the row reads, by content (keys are distinct unless the palette is on): the
         * tree's from its pages and the flat slots, the chain's from the positions it read. */
        if (!palette) {
            const int64_t rb = base / w.ratio * w.ratio;
            auto keyat = [&](int64_t q) -> const std::vector<float>& {
                return q < base ? pk[(size_t)q] : K[(size_t)rows[(size_t)(q - base)]];
            };
            std::vector<std::vector<float>> tk, ck;
            for (int64_t e = 0; e < o.nsel[(size_t)x]; ++e)
                for (int64_t i = 0; i < w.ratio; ++i) tk.push_back(pk[(size_t)(o.sel[(size_t)x][(size_t)e] * w.ratio + i)]);
            for (int k2 = 0; k2 < 64; ++k2) {
                if (!((o.vmask[(size_t)x] >> k2) & 1)) continue;
                if (rb + k2 < base) tk.push_back(pk[(size_t)(rb + k2)]);
                else tk.push_back(K[(size_t)(rb + k2 - base)]);     /* flat slot -> node row */
            }
            for (int64_t q : cv) ck.push_back(keyat(q));
            std::sort(tk.begin(), tk.end()); std::sort(ck.begin(), ck.end());
            CHECK(tk == ck);
        }
        /* Below the bound everything is selected. */
        if (ntot <= topk) CHECK_EQ(o.nsel[(size_t)x] + (ntot - first_open), (int64_t)ntot);
        else if (ties_seen) ++*ties_seen;
        CHECK_EQ(o.su[(size_t)x], o.nsel[(size_t)x] * w.ratio);
    }
    /* A node whose block completes has a scratch key equal to the chain's committed page. */
    for (int64_t x = 0; x < R; ++x) {
        if ((pos[(size_t)x] + 1) % w.ratio != 0) { CHECK_EQ(o.spage[(size_t)x], -1LL); continue; }
        CHECK_EQ(o.spage[(size_t)x], x);
    }
}

}  /* namespace */

/* Below the bound (every complete block fits in topk) a tree row's committed pages and region mask
 * together are exactly what its sequential path reads, for every base alignment and tree shape. */
TEST(qsa_tree_rows_equal_a_sequential_decode_below_the_bound) {
    int tag = 0;
    for (int64_t base = 17; base <= 22; ++base)        /* every alignment, anchor on a block end too */
        for (uint64_t seed = 1; seed <= 3; ++seed)
            qsa_tree_scenario(seed * 10 + (uint64_t)base, base, 11, 40, tag++ % 2 ? 4 : 0, false, 4,
                              "below", nullptr);
}

/* Above it (topk 3 of ~10 blocks) the sets differ per row; ties are common with the palette (no
 * rotation), and the tie rule, the union ranking and the region mask must still agree with the
 * sequential decode's choice for every node. */
TEST(qsa_tree_rows_equal_a_sequential_decode_above_the_bound) {
    int above = 0;
    for (int64_t base = 31; base <= 38; ++base)
        for (uint64_t seed = 1; seed <= 4; ++seed)
            for (int64_t topk : { 2, 3, 5 })
                qsa_tree_scenario(seed * 100 + (uint64_t)base, base, 12, topk, seed % 2 ? 0 : 4,
                                  seed % 2 == 1, 4, "above", &above);
    CHECK(above > 100);
}

/* A chain through qsa_tree is the ordinary decode: parents j - 1, every row's vmask is its prefix. */
TEST(qsa_tree_chain_rows_see_their_prefix_of_the_region) {
    Rng r(5);
    QWorld w(16, 64, 0);
    std::vector<std::vector<float>> pk, K, Q;
    for (int t = 0; t < 22; ++t) pk.push_back(qkey(r, w.n, false));
    qsa_chain(w, pk, {}, 0, false, nullptr);
    std::vector<int64_t> par = { -1, 0, 1, 2, 3, 4 };
    for (int j = 0; j < 6; ++j) {
        K.push_back(qkey(r, w.n, false));
        std::vector<float> q;
        for (int64_t h = 0; h < w.heads; ++h) { auto x = qkey(r, w.n, false); q.insert(q.end(), x.begin(), x.end()); }
        Q.push_back(q);
    }
    TreeOut o;
    qsa_tree_step(w, K, Q, par, 22, 0, nullptr, true, &o);
    /* base 22: b0 = 5, region keys start at 20; row j reads slots 20, 21 and 22 .. 22 + j; every
     * block is selected below the bound. */
    for (int j = 0; j < 6; ++j) CHECK_EQ(o.vmask[(size_t)j], (int64_t)((1ull << (3 + j)) - 1));
    CHECK_EQ(o.nsel[0], 5LL);
}

/* The heal: after a verify that accepted a path, the next step rebuilds the blocks the path's
 * non-anchor tokens completed from the compacted tail, equal to what a sequential decode of the
 * accepted tokens wrote (block keys and ring rows), and a verify itself commits nothing but its
 * anchor's block. */
TEST(qsa_tree_heal_rebuilds_the_accepted_paths_blocks_from_the_compacted_tail) {
    for (int64_t base = 18; base <= 25; ++base)
        for (uint64_t seed = 1; seed <= 4; ++seed) {
            Rng r(seed * 7 + (uint64_t)base);
            QWorld w(16, 4, 4);
            const int64_t n = w.n;
            std::vector<std::vector<float>> pk;
            for (int64_t t = 0; t < base; ++t) pk.push_back(qkey(r, n, false));
            qsa_chain(w, pk, {}, 0, false, nullptr);
            const std::vector<uint8_t> tail0 = w.tail.mem, bk0 = w.bkp.mem;

            const int64_t R = 12;
            std::vector<std::vector<float>> K, Q;
            for (int64_t j = 0; j < R; ++j) { K.push_back(qkey(r, n, false)); Q.push_back(std::vector<float>((size_t)(w.heads * n), 1.0f)); }
            const std::vector<int64_t> par = qsa_rand_tree(r, R, 1);
            /* The deepest node's path is accepted. */
            std::vector<int64_t> depth((size_t)R, 0);
            int64_t best = 0;
            for (int64_t j = 1; j < R; ++j) { depth[(size_t)j] = depth[(size_t)par[(size_t)j]] + 1; if (depth[(size_t)j] > depth[(size_t)best]) best = j; }
            std::vector<int64_t> path;
            for (int64_t y = best; y >= 0; y = par[(size_t)y]) { path.push_back(y); if (y == 0) break; }
            std::reverse(path.begin(), path.end());
            const int64_t na = (int64_t)path.size();
            TreeOut v1;
            qsa_tree_step(w, K, Q, par, base, 0, nullptr, false, &v1);   /* the verify */

            /* Sequential decode of the accepted tokens from the same starting state. */
            const std::vector<uint8_t> tail1 = w.tail.mem, bk1 = w.bkp.mem;
            std::memcpy(w.tail.mem.data(), tail0.data(), tail0.size());
            std::memcpy(w.bkp.mem.data(), bk0.data(), bk0.size());
            std::vector<std::vector<float>> PK;
            for (int64_t y : path) PK.push_back(K[(size_t)y]);
            qsa_chain(w, PK, {}, base, false, nullptr);
            const std::vector<uint8_t> tailS = w.tail.mem, bkS = w.bkp.mem;

            /* The next step: a tree of a bonus anchor and two children at base + na. */
            std::memcpy(w.tail.mem.data(), tail1.data(), tail1.size());
            std::memcpy(w.bkp.mem.data(), bk1.data(), bk1.size());
            std::vector<std::vector<float>> K2, Q2;
            for (int j = 0; j < 3; ++j) { K2.push_back(qkey(r, n, false)); Q2.push_back(std::vector<float>((size_t)(w.heads * n), 1.0f)); }
            std::vector<int64_t> pth = path;
            TreeOut v2;
            qsa_tree_step(w, K2, Q2, { -1, 0, 0 }, base + na, na, &pth, false, &v2);

            const std::string what = "heal base " + std::to_string(base) + " seed " + std::to_string(seed);
            /* Every block complete below the new anchor equals the sequential decode's page. */
            const int64_t last_blk = (base + na - 1 + 1) / w.ratio;       /* blocks [0, last_blk) */
            for (int64_t b = 0; b < last_blk; ++b)
                for (int64_t d = 0; d < n; ++d) {
                    const float want = bf16_to_f32(((const uint16_t*)bkS.data())[b * n + d]);
                    CHECK(w.bkp.at(b * n + d) == want);
                    if (w.bkp.at(b * n + d) != want) {
                        fprintf(stderr, "    %s: block %lld lane %lld\n", what.c_str(), (long long)b, (long long)d);
                        break;
                    }
                }
            /* The committed positions' ring rows hold the accepted tokens' raw keys. */
            for (int64_t t = 0; t < na; ++t)
                for (int64_t d = 0; d < n; ++d) {
                    const int64_t row = (base + t) % w.ring;
                    const float want = bf16_to_f32(((const uint16_t*)tailS.data())[row * n + d]);
                    CHECK(w.tail.at(row * n + d) == want);
                }
        }
}

/* qsa_tail_compact by hand: node rows stored flat at (base_prev + j) % ring move to depth order,
 * ascending in place; an identity path, a one-token commit and a sequence with no slot change
 * nothing. */
TEST(qsa_tail_compact_moves_the_accepted_path_to_depth_order) {
    const int64_t n = 2, ring = 7, slots = 3;
    std::vector<float> init;
    for (int64_t s = 0; s < slots; ++s)
        for (int64_t r = 0; r < ring; ++r) { init.push_back((float)(100 * s + r)); init.push_back((float)(-r)); }
    T st(RAD_BF16, { slots, 1, ring, n }); st.f(init);
    /* Sequence 0: slot 2, previous anchor at 9 (ring row 2), na 3, path 0, 2, 4: tail rows
     * (9 + 2) % 7 = 4 -> (9 + 1) % 7 = 3, then (9 + 4) % 7 = 6 -> (9 + 2) % 7 = 4. Sequence 1: slot 0,
     * na 1. Sequence 2: no slot. Sequence 3: identity path. */
    T pos(RAD_I32, { 4 }); pos.n({ 12, 5, 20, 31 });
    T cu(RAD_I32, { 5 }); cu.n({ 0, 1, 2, 3, 4 });
    T na(RAD_I32, { 4 }); na.n({ 3, 1, 4, 3 });
    T path(RAD_I32, { 4, 5 }); path.n({ 0, 2, 4, 0, 0,   0, 0, 0, 0, 0,   0, 1, 2, 3, 0,   0, 1, 2, 0, 0 });
    T sx(RAD_I32, { 4, 1 }); sx.n({ 2, 0, -1, 1 });
    A a; a.t(pos).t(cu).t(na).t(path).t(st).t(sx).i("M", 4).i("n", n).i("ring", ring).i("seqs", 4)
          .s("dtype", "bf16");
    CHECK_OK(run("qsa_tail_compact", a));
    auto at = [&](int64_t s, int64_t r, int64_t d) { return st.at((s * ring + r) * n + d); };
    CHECK_EQ(at(2, 3, 0), 200.0f + 4); CHECK_EQ(at(2, 4, 0), 200.0f + 6);
    CHECK_EQ(at(2, 6, 0), 200.0f + 6); CHECK_EQ(at(2, 2, 0), 200.0f + 2);
    CHECK_EQ(at(0, 0, 0), 0.0f); CHECK_EQ(at(1, 3, 0), 100.0f + 3);
    /* A path entry past the ring is refused before any write. */
    path.n({ 0, 2, 9, 0, 0 });
    CHECK_EQ(run("qsa_tail_compact", a), RAD_E_INVAL);
}

/* qsa_work on a tree step with M-RoPE planes (RadBatch::rope_pos). Three facts, on a chain with a
 * sibling (positions 24..31, the anchor on a block boundary) over a tail of earlier keys:
 *  1. Text planes (every component = position + one shift): every output equals the plane-less
 *     tree step's, the positions moved by the shift in all three components. With shift 0 this is
 *     the equivalence the engine's old "drop rope_pos on a tree step" workaround relied on, which
 *     holds only for a request with no media shift.
 *  2. Planes that differ per row (a media request's rows, or anything else that is not
 *     position + shift): a scratch block whose first token is in the step takes the components of
 *     its ancestor row at that depth, not position + shift of the anchor.
 *  3. A block whose first token is before the step is text: its index plus the anchor row's shift. */
TEST(qsa_work_tree_with_mrope_planes_rotates_scratch_blocks_at_their_ancestors_components) {
    const int64_t n = 2, ratio = 4, ring = 16, work = 6, R = 9;
    for (int64_t base : { 24LL, 22LL }) {
    const std::vector<int64_t> par = { -1, 0, 1, 2, 3, 4, 5, 6, 3 };   /* a chain 0..7, row 8 under 3 */
    std::vector<int64_t> dep(R, 0), pv, sv;
    for (int64_t j = 0; j < R; ++j) {
        dep[(size_t)j] = j ? dep[(size_t)par[(size_t)j]] + 1 : 0;
        pv.push_back(base + dep[(size_t)j]);
        sv.push_back(par[(size_t)j]);
    }
    std::vector<float> kv, tv;
    for (int64_t t = 0; t < R; ++t) { kv.push_back((float)(100 + t)); kv.push_back((float)(200 + t)); }
    for (int64_t r = 0; r < ring; ++r) { tv.push_back((float)(-r)); tv.push_back(0.25f); }
    /* The outputs, sized for the plane-less (rank 1) or the plane (three columns) form. */
    struct Outs {
        T stage, page, bpos, sst, spg, sbp, rtb, rnc;
        Outs(bool mc, int64_t work_, int64_t R_)
            : stage(RAD_BF16, { work_ * 4, 2 }), page(RAD_I32, { work_ }),
              bpos(RAD_I32, mc ? std::vector<int64_t>{ work_, 3 } : std::vector<int64_t>{ work_ }),
              sst(RAD_BF16, { R_ * 4, 2 }), spg(RAD_I32, { R_ }),
              sbp(RAD_I32, mc ? std::vector<int64_t>{ R_, 3 } : std::vector<int64_t>{ R_ }),
              rtb(RAD_I32, { R_, 8 }), rnc(RAD_I32, { R_ }) {}
    };
    auto step = [&](const std::vector<int64_t>* rope_v, Outs& o) {
        T &page = o.page, &bpos = o.bpos, &stage = o.stage, &spg = o.spg, &sbp = o.sbp, &sst = o.sst,
          &rtb = o.rtb, &rnc = o.rnc;
        T k(RAD_BF16, { R, n }); k.f(kv);
        T pos(RAD_I32, { R }); pos.n(pv);
        T cu(RAD_I32, { 2 }); cu.n({ 0, R });
        T tail(RAD_BF16, { 1, 1, ring, n }); tail.f(tv);
        T sx(RAD_I32, { 1, 1 }); sx.n({ 0 });
        std::vector<int64_t> id;
        for (int64_t b = 0; b < 40; ++b) id.push_back(b + 50);
        T bt(RAD_I32, { 1, 40 }); bt.n(id);
        T sp(RAD_I32, { R }); sp.n(sv);
        T nc(RAD_I32, { 1 }), rcu(RAD_I32, { R + 1 });
        A a; a.t(k).t(pos).t(cu).t(tail).t(sx).t(bt);
        T rope(RAD_I32, { 3, R });
        if (rope_v) { rope.n(*rope_v); a.t(rope); } else { a.tnull(); }
        a.t(stage).t(page).t(bpos).t(nc).t(sp).tnull().t(sst).t(spg).t(sbp).t(rtb).t(rnc).t(rcu)
         .i("M", R).i("n", n).i("ratio", ratio).i("ring", ring).i("seqs", 1).i("work", work)
         .s("dtype", "bf16");
        REQUIRE_EQ(run("qsa_work", a), RAD_OK);
    };
    Outs O0(false, work, R);
    step(nullptr, O0);
    T &pg0 = O0.page, &bp0 = O0.bpos, &st0 = O0.stage, &sg0 = O0.spg, &sb0 = O0.sbp, &ss0 = O0.sst,
      &rt0 = O0.rtb, &rn0 = O0.rnc;

    /* 1. Text planes with shift 0 and with shift 1000. */
    for (int64_t shift : { 0LL, 1000LL }) {
        std::vector<int64_t> rv;
        for (int c = 0; c < 3; ++c)
            for (int64_t j = 0; j < R; ++j) rv.push_back(pv[(size_t)j] + shift);
        Outs O1(true, work, R);
        step(&rv, O1);
        T &pg = O1.page, &bp = O1.bpos, &st = O1.stage, &sg = O1.spg, &sb = O1.sbp, &ss = O1.sst,
          &rt = O1.rtb, &rn = O1.rnc;
        for (int64_t i = 0; i < work; ++i) {
            CHECK_EQ(pg.iat(i), pg0.iat(i));
            for (int c = 0; c < 3; ++c) CHECK_EQ(bp.iat(i * 3 + c), bp0.iat(i) + (pg0.iat(i) >= 0 ? shift : 0));
        }
        for (int64_t x = 0; x < R; ++x) {
            CHECK_EQ(sg.iat(x), sg0.iat(x));
            CHECK_EQ(rn.iat(x), rn0.iat(x));
            for (int c = 0; c < 3; ++c) CHECK_EQ(sb.iat(x * 3 + c), sg0.iat(x) >= 0 ? sb0.iat(x) + shift : 0);
        }
        CHECK(st.mem == st0.mem);
        CHECK(ss.mem == ss0.mem);
        CHECK(rt.mem == rt0.mem);
    }

    /* 2 and 3. Planes that are not position + shift. A row completing a block (position + 1 a
     * multiple of ratio) takes, for a first token q0 in the step, its ancestor row's components,
     * else (q0 before the anchor: text from the tail) q0 + the shift of row 0. */
    std::vector<int64_t> rv;
    for (int c = 0; c < 3; ++c)
        for (int64_t j = 0; j < R; ++j) rv.push_back(1000 * (c + 1) + 7 * j);
    Outs O2(true, work, R);
    step(&rv, O2);
    T &sg = O2.spg, &sb = O2.sbp;
    int64_t completing = 0, in_step = 0;
    for (int64_t x = 0; x < R; ++x) {
        const bool done = (pv[(size_t)x] + 1) % ratio == 0;
        CHECK_EQ(sg.iat(x), done ? x : -1LL);
        const int64_t q0 = pv[(size_t)x] - ratio + 1;
        int64_t y = x;
        while (y > 0 && pv[(size_t)y] > q0) y = par[(size_t)y];
        for (int c = 0; c < 3; ++c) {
            const int64_t want = !done ? 0 : q0 >= base ? 1000 * (c + 1) + 7 * y
                                                        : q0 + (1000 - base);
            CHECK_EQ(sb.iat(x * 3 + c), want);
        }
        completing += done;
        in_step += done && q0 >= base;
    }
    CHECK(completing >= 1);
    CHECK(in_step >= (base == 24 ? 2 : 1));    /* both branches are exercised */
    }
}

/* qsa_tail_store stores a tree row at its flat slot, siblings at distinct rows. */
TEST(qsa_tail_store_stores_tree_rows_at_their_flat_slots) {
    const int64_t n = 2, ring = 8;
    T k(RAD_BF16, { 4, n }); k.f({ 1, 1, 2, 2, 3, 3, 4, 4 });
    T pos(RAD_I32, { 4 }); pos.n({ 10, 11, 11, 12 });
    T cu(RAD_I32, { 2 }); cu.n({ 0, 4 });
    T st(RAD_BF16, { 1, 1, ring, n }); st.f(std::vector<float>((size_t)(ring * n), 0.0f));
    T sx(RAD_I32, { 1, 1 }); sx.n({ 0 });
    T sp(RAD_I32, { 4 }); sp.n({ -1, 0, 0, 1 });
    A a; a.t(k).t(pos).t(cu).t(st).t(sx).t(sp).i("M", 4).i("n", n).i("ratio", 4).i("ring", ring)
          .i("seqs", 1).s("dtype", "bf16");
    CHECK_OK(run("qsa_tail_store", a));
    for (int j = 0; j < 4; ++j) CHECK_EQ(st.at(((10 + j) % ring) * n), (float)(j + 1));
}

/* ------------------------------------------------------------------ batches of several sequences
 *
 * The per-row tables are keyed by absolute batch row, the work list and the sstage rows by absolute
 * position in the launch, so a tree at t0 > 0, two trees and a tree beside an ordinary sequence are
 * where an offset error would hide. Each case runs the two-step flow (a verify, then the next step
 * with the accepted path: compaction, heal, tree) for several sequences in ONE batch and compares
 * every output, per sequence, with the same sequence run alone (rows from 0, slot 0, table row 0,
 * the ordinary operand list for a non-tree sequence). The alone run is what the scenario tests
 * above pin against the sequential decode. */
namespace {

struct QSeq {
    bool tree = true;
    int64_t base = 0, na = 1;
    std::vector<int64_t> par, path;                 /* tree: parents, accepted path (na entries) */
    std::vector<std::vector<float>> K, Q;           /* per row */
};

struct BOut {
    std::vector<int64_t> nc, spage, sbp, rnc, nsel, su, vmask;
    std::vector<std::vector<int64_t>> rtab, sel;
    std::vector<std::vector<float>> rscore, score, sstage, skey;
};

/* One step over `S` in one batch; the world has one slot, table row and page range per sequence. */
void qsa_batch_step(QWorld& w, const std::vector<QSeq>& S, bool treeops, BOut* o) {
    const int64_t n = w.n, ratio = w.ratio, RW = w.RW, NS = (int64_t)S.size();
    std::vector<int64_t> cuv{ 0 }, posv, spv, nav, slotv;
    int64_t T_ = 0;
    for (int64_t s = 0; s < NS; ++s) {
        const QSeq& q = S[(size_t)s];
        const int64_t R = (int64_t)q.K.size();
        std::vector<int64_t> depth((size_t)R, 0);
        for (int64_t j = 0; j < R; ++j) {
            depth[(size_t)j] = q.tree ? (j ? depth[(size_t)q.par[(size_t)j]] + 1 : 0) : j;
            posv.push_back(q.base + depth[(size_t)j]);
            spv.push_back(!q.tree ? -2 : j ? q.par[(size_t)j] : -1);
        }
        T_ += R; cuv.push_back(T_);
        nav.push_back(q.na);
        slotv.push_back(NS - 1 - s);                    /* slots reversed: the slot is the sequence's */
    }
    const int64_t Tn = T_;
    T k(RAD_BF16, { Tn, n }), pos(RAD_I32, { Tn }), cu(RAD_I32, { NS + 1 }), sx(RAD_I32, { NS, 1 });
    T sp(RAD_I32, { Tn }), nat(RAD_I32, { NS }), pth(RAD_I32, { NS, 16 });
    std::vector<float> kv, qv;
    std::vector<int64_t> pp((size_t)(NS * 16), 0);
    bool any_path = false;
    for (int64_t s = 0; s < NS; ++s) {
        for (size_t j = 0; j < S[(size_t)s].K.size(); ++j) {
            kv.insert(kv.end(), S[(size_t)s].K[j].begin(), S[(size_t)s].K[j].end());
            qv.insert(qv.end(), S[(size_t)s].Q[j].begin(), S[(size_t)s].Q[j].end());
        }
        for (size_t i = 0; i < S[(size_t)s].path.size(); ++i) pp[(size_t)(s * 16) + i] = S[(size_t)s].path[i];
        any_path |= S[(size_t)s].na >= 2;
    }
    k.f(kv); pos.n(posv); cu.n(cuv); sx.n(slotv); sp.n(spv); nat.n(nav); pth.n(pp);
    if (any_path) {
        A cmp; cmp.t(pos).t(cu).t(nat).t(pth).t(w.tail).t(sx).i("M", NS).i("n", n).i("ring", w.ring)
                  .i("seqs", NS).s("dtype", "bf16");
        REQUIRE_EQ(run("qsa_tail_compact", cmp), RAD_OK);
    }
    const int64_t work = 12 * NS + 8;
    T stage(RAD_BF16, { work * ratio, n }), page(RAD_I32, { work }), bpos(RAD_I32, { work }), nc(RAD_I32, { NS });
    T sst(RAD_BF16, { Tn * ratio, n }), spg(RAD_I32, { Tn }), sbp(RAD_I32, { Tn });
    T rtb(RAD_I32, { Tn, RW }), rnc(RAD_I32, { Tn }), rcu(RAD_I32, { Tn + 1 });
    sst.f(std::vector<float>((size_t)(Tn * ratio * n), -9.0f));
    A a; a.t(k).t(pos).t(cu).t(w.tail).t(sx).t(w.tab).tnull().t(stage).t(page).t(bpos).t(nc);
    if (treeops) a.t(sp).t(nat).t(sst).t(spg).t(sbp).t(rtb).t(rnc).t(rcu);
    else a.tnull().t(nat);
    a.i("M", Tn).i("n", n).i("ratio", ratio).i("ring", w.ring).i("seqs", NS).i("work", work).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_work", a), RAD_OK);
    A b = w.bkey(stage, page, bpos, w.bkp, work);
    REQUIRE_EQ(run("qsa_block_key", b), RAD_OK);
    T skey(RAD_BF16, { Tn, n });
    skey.f(std::vector<float>((size_t)(Tn * n), -33.0f));
    if (treeops) {
        A b2 = w.bkey(sst, spg, sbp, skey, Tn);
        REQUIRE_EQ(run("qsa_block_key", b2), RAD_OK);
    }
    A c; c.t(k).t(pos).t(cu).t(w.tail).t(sx);
    if (treeops) c.t(sp);
    c.i("M", Tn).i("n", n).i("ratio", ratio).i("ring", w.ring).i("seqs", NS).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_tail_store", c), RAD_OK);

    T q(RAD_BF16, { Tn, w.heads * n }), sc(RAD_F32, { Tn, w.nbmax }), rsc(RAD_F32, { Tn, RW });
    q.f(qv);
    A d; d.t(q).t(w.bkp).t(w.tab).t(cu).t(nc).t(sc).i("M", Tn).i("n", n).i("heads", w.heads)
          .i("blocks", w.nbmax).i("seqs", NS).s("dtype", "bf16");
    REQUIRE_EQ(run("qsa_score", d), RAD_OK);
    if (treeops) {
        A d2; d2.t(q).t(skey).t(rtb).t(rcu).t(rnc).t(rsc).i("M", Tn).i("n", n).i("heads", w.heads)
                .i("blocks", RW).i("seqs", Tn).s("dtype", "bf16");
        REQUIRE_EQ(run("qsa_score", d2), RAD_OK);
    }
    T se(RAD_I32, { Tn, w.topk + 1 }), ns(RAD_I32, { Tn }), su(RAD_I32, { Tn }), vm(RAD_I64, { Tn });
    vm.n(std::vector<int64_t>((size_t)Tn, 0x5555));
    A e; e.t(sc).t(w.tab).t(nc).t(pos).t(cu).t(se).t(ns).t(su);
    if (treeops) e.t(sp).t(rsc).t(vm);
    e.i("M", Tn).i("blocks", w.nbmax).i("topk", w.topk).i("ratio", ratio).i("seqs", NS);
    REQUIRE_EQ(run("qsa_select", e), RAD_OK);

    for (int64_t s = 0; s < NS; ++s) o->nc.push_back(nc.iat(s));
    for (int64_t j = 0; j < Tn; ++j) {
        o->nsel.push_back(ns.iat(j)); o->su.push_back(su.iat(j));
        o->vmask.push_back(treeops ? vm.iat(j) : 0);
        std::vector<int64_t> sl;
        for (int64_t i = 0; i <= w.topk; ++i) sl.push_back(se.iat(j * (w.topk + 1) + i));
        o->sel.push_back(sl);
        std::vector<float> scv, rs, ssv, skv;
        for (int64_t i = 0; i < w.nbmax; ++i) scv.push_back(sc.at(j * w.nbmax + i));
        o->score.push_back(scv);
        if (!treeops) {
            o->spage.push_back(-1); o->sbp.push_back(0); o->rnc.push_back(0);
            o->rtab.push_back({}); o->rscore.push_back({}); o->sstage.push_back({}); o->skey.push_back({});
            continue;
        }
        o->spage.push_back(spg.iat(j)); o->sbp.push_back(sbp.iat(j)); o->rnc.push_back(rnc.iat(j));
        std::vector<int64_t> rt;
        for (int64_t i = 0; i < RW; ++i) { rt.push_back(rtb.iat(j * RW + i)); rs.push_back(rsc.at(j * RW + i)); }
        o->rtab.push_back(rt); o->rscore.push_back(rs);
        if (spg.iat(j) >= 0) {
            for (int64_t i = 0; i < ratio * n; ++i) ssv.push_back(sst.at(j * ratio * n + i));
            for (int64_t i = 0; i < n; ++i) skv.push_back(skey.at(j * n + i));
        }
        o->sstage.push_back(ssv); o->skey.push_back(skv);
    }
}

/* A random sequence mid-flight: a prefill, then a verify (tree) or a decode (chain), whose state is
 * left in `w1` (a one-sequence world). Returns the next step's sequence. */
QSeq qsa_next_step(Rng& r, QWorld& w1, bool tree, int64_t base0, bool palette) {
    const int64_t n = w1.n;
    std::vector<std::vector<float>> pk;
    for (int64_t t = 0; t < base0; ++t) pk.push_back(qkey(r, n, palette));
    qsa_chain(w1, pk, {}, 0, false, nullptr);
    auto rows = [&](int64_t R, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q) {
        for (int64_t j = 0; j < R; ++j) {
            K.push_back(qkey(r, n, palette));
            std::vector<float> q;
            for (int64_t h = 0; h < w1.heads; ++h) { auto x = qkey(r, n, palette); q.insert(q.end(), x.begin(), x.end()); }
            Q.push_back(q);
        }
    };
    QSeq out;
    out.tree = tree;
    std::vector<std::vector<float>> K1, Q1;
    if (tree) {
        const int64_t R1 = 6 + r.below(7);
        rows(R1, K1, Q1);
        const std::vector<int64_t> par = qsa_rand_tree(r, R1, r.below(2));
        TreeOut v1;
        qsa_tree_step(w1, K1, Q1, par, base0, 0, nullptr, false, &v1);
        std::vector<int64_t> depth((size_t)R1, 0);
        int64_t best = 0;
        for (int64_t j = 1; j < R1; ++j) { depth[(size_t)j] = depth[(size_t)par[(size_t)j]] + 1; if (depth[(size_t)j] > depth[(size_t)best]) best = j; }
        for (int64_t y = best;; y = par[(size_t)y]) { out.path.push_back(y); if (y == 0) break; }
        std::reverse(out.path.begin(), out.path.end());
        out.na = (int64_t)out.path.size();
        out.base = base0 + out.na;
        const int64_t R2 = 3 + r.below(8);
        rows(R2, out.K, out.Q);
        out.par = qsa_rand_tree(r, R2, r.below(2));
    } else {
        const int64_t R1 = 2 + r.below(5);
        rows(R1, K1, Q1);
        qsa_chain(w1, K1, {}, base0, false, nullptr);
        out.na = 1;
        out.base = base0 + R1;
        rows(1 + r.below(4), out.K, out.Q);
    }
    return out;
}

/* Copies one sequence's world (slot 0, table row 0) into slot/page range `s` of the batch world. */
void qsa_place(QWorld& bw, int64_t s, const QWorld& w1) {
    const int64_t slot = bw.nseq - 1 - s;
    std::memcpy(bw.tail.mem.data() + (size_t)(slot * bw.ring * bw.n) * 2, w1.tail.mem.data(), (size_t)(w1.ring * w1.n) * 2);
    std::memcpy(bw.bkp.mem.data() + (size_t)(s * bw.nbmax * bw.n) * 2, w1.bkp.mem.data(), (size_t)(w1.nbmax * w1.n) * 2);
}

void qsa_batch_case(uint64_t seed, const std::vector<bool>& kinds, int64_t topk, bool palette, const char* tag) {
    Rng r(seed);
    const int64_t NS = (int64_t)kinds.size();
    QWorld bw(16, topk, 0, NS);
    std::vector<QSeq> S;
    std::vector<std::unique_ptr<QWorld>> alone;     /* a T keeps a pointer into its own bytes: no copies */
    for (int64_t s = 0; s < NS; ++s) {
        alone.push_back(std::make_unique<QWorld>(16, topk, 0));
        QWorld& w1 = *alone.back();
        S.push_back(qsa_next_step(r, w1, kinds[(size_t)s], 17 + r.below(16), palette));
        qsa_place(bw, s, w1);
    }
    bool anytree = false;
    for (const QSeq& q : S) anytree |= q.tree;
    BOut B;
    qsa_batch_step(bw, S, anytree, &B);
    int64_t t0 = 0;
    for (int64_t s = 0; s < NS; ++s) {
        const QSeq& q = S[(size_t)s];
        const int64_t R = (int64_t)q.K.size();
        BOut A1;
        qsa_batch_step(*alone[(size_t)s], { q }, q.tree, &A1);
        const std::string what = std::string(tag) + " seed " + std::to_string(seed) + " seq " + std::to_string(s);
        CHECK_EQ(B.nc[(size_t)s], A1.nc[0]);
        for (int64_t x = 0; x < R; ++x) {
            const size_t bx = (size_t)(t0 + x), ax = (size_t)x;
            bool same = B.nsel[bx] == A1.nsel[ax] && B.su[bx] == A1.su[ax] && B.vmask[bx] == A1.vmask[ax] &&
                        B.rnc[bx] == A1.rnc[ax] && B.sbp[bx] == A1.sbp[ax] &&
                        B.sstage[bx] == A1.sstage[ax] && B.skey[bx] == A1.skey[ax];
            if (q.tree) same = same && B.rscore[bx] == A1.rscore[ax];
            /* The scratch row is the absolute batch row; table entries are absolute rows too. */
            same = same && (B.spage[bx] == (A1.spage[ax] < 0 ? -1 : t0 + A1.spage[ax]));
            for (size_t i = 0; q.tree && i < A1.rtab[ax].size(); ++i)
                same = same && B.rtab[bx][i] == (A1.rtab[ax][i] < 0 ? -1 : t0 + A1.rtab[ax][i]);
            /* sel names pages: the sequence's range is offset by s * nbmax. */
            for (size_t i = 0; i < A1.sel[ax].size(); ++i)
                same = same && B.sel[bx][i] == (A1.sel[ax][i] < 0 ? -1 : A1.sel[ax][i] + s * bw.nbmax);
            for (int64_t b = 0; b < A1.nc[0]; ++b) same = same && B.score[bx][(size_t)b] == A1.score[ax][(size_t)b];
            CHECK(same);
            if (!same) fprintf(stderr, "    %s: row %lld differs from the sequence alone\n", what.c_str(), (long long)x);
        }
        /* The state the step leaves: the sequence's ring slot and block-key pages. */
        const int64_t slot = bw.nseq - 1 - s;
        CHECK(std::memcmp(bw.tail.mem.data() + (size_t)(slot * bw.ring * bw.n) * 2, alone[(size_t)s]->tail.mem.data(), (size_t)(bw.ring * bw.n) * 2) == 0);
        CHECK(std::memcmp(bw.bkp.mem.data() + (size_t)(s * bw.nbmax * bw.n) * 2, alone[(size_t)s]->bkp.mem.data(), (size_t)(bw.nbmax * bw.n) * 2) == 0);
        t0 += R;
    }
}

}  /* namespace */

TEST(qsa_batch_tree_at_nonzero_offset_matches_the_sequence_alone) {
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        qsa_batch_case(seed, { false, true }, 40, seed % 2 == 0, "chain+tree below");
        qsa_batch_case(seed + 50, { false, true }, 3, seed % 2 == 0, "chain+tree above");
    }
}

TEST(qsa_batch_two_trees_match_each_sequence_alone) {
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        qsa_batch_case(seed + 100, { true, true }, 40, seed % 2 == 0, "tree+tree below");
        qsa_batch_case(seed + 150, { true, true }, 3, seed % 2 == 0, "tree+tree above");
        qsa_batch_case(seed + 170, { true, true }, 2, true, "tree+tree ties");
    }
}

TEST(qsa_batch_tree_beside_an_ordinary_sequence_matches_each_alone) {
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        qsa_batch_case(seed + 200, { true, false, true }, 40, false, "tree+chain+tree below");
        qsa_batch_case(seed + 250, { true, false, true }, 3, seed % 2 == 0, "tree+chain+tree above");
        qsa_batch_case(seed + 270, { false, true, false, true }, 4, false, "chain,tree,chain,tree");
    }
}

/* qsa_tail_compact and qsa_tail_store on a batch by hand: slots reversed against the sequences, a
 * tree at t0 > 0, an ordinary sequence between two trees. */
TEST(qsa_tail_ops_by_hand_on_a_mixed_batch) {
    const int64_t n = 2, ring = 8, slots = 3;
    std::vector<float> init;
    for (int64_t s = 0; s < slots; ++s)
        for (int64_t r = 0; r < ring; ++r) { init.push_back((float)(100 * s + r)); init.push_back((float)(-r)); }
    T st(RAD_BF16, { slots, 1, ring, n }); st.f(init);
    /* seq 0 (rows 0..1): tree, slot 2, anchor 12, na 2 (path 0, 3) -> base_prev 10; seq 1 (rows 2..4):
     * ordinary, slot 1, na 1; seq 2 (row 5): tree, slot 0, anchor 21, na 3 (path 0, 2, 3) -> base_prev 18. */
    T pos(RAD_I32, { 6 }); pos.n({ 12, 13, 30, 31, 32, 21 });
    T cu(RAD_I32, { 4 }); cu.n({ 0, 2, 5, 6 });
    T na(RAD_I32, { 3 }); na.n({ 2, 1, 3 });
    T path(RAD_I32, { 3, 4 }); path.n({ 0, 3, 0, 0,   0, 0, 0, 0,   0, 2, 3, 0 });
    T sx(RAD_I32, { 3, 1 }); sx.n({ 2, 1, 0 });
    A a; a.t(pos).t(cu).t(na).t(path).t(st).t(sx).i("M", 3).i("n", n).i("ring", ring).i("seqs", 3)
          .s("dtype", "bf16");
    CHECK_OK(run("qsa_tail_compact", a));
    auto at = [&](int64_t s, int64_t r) { return st.at((s * ring + r) * n); };
    /* seq 0: row (10 + 1) % 8 = 3 <- (10 + 3) % 8 = 5. */
    CHECK_EQ(at(2, 3), 200.0f + 5); CHECK_EQ(at(2, 5), 200.0f + 5); CHECK_EQ(at(2, 2), 200.0f + 2);
    /* seq 2: rows (18 + 1) % 8 = 3 <- (18 + 2) % 8 = 4, then (18 + 2) % 8 = 4 <- (18 + 3) % 8 = 5. */
    CHECK_EQ(at(0, 3), 4.0f); CHECK_EQ(at(0, 4), 5.0f); CHECK_EQ(at(0, 5), 5.0f);
    /* The ordinary sequence's slot is untouched. */
    for (int64_t r = 0; r < ring; ++r) CHECK_EQ(at(1, r), 100.0f + r);

    /* qsa_tail_store: tree rows at flat slots from THEIR anchor, the ordinary sequence at positions. */
    T k(RAD_BF16, { 6, n }); k.f({ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 });
    T sp(RAD_I32, { 6 }); sp.n({ -1, 0, -2, -2, -2, -1 });
    A c; c.t(k).t(pos).t(cu).t(st).t(sx).t(sp).i("M", 6).i("n", n).i("ratio", 4).i("ring", ring)
          .i("seqs", 3).s("dtype", "bf16");
    CHECK_OK(run("qsa_tail_store", c));
    CHECK_EQ(at(2, 12 % 8), 1.0f); CHECK_EQ(at(2, 13 % 8), 2.0f);
    CHECK_EQ(at(1, 30 % 8), 3.0f); CHECK_EQ(at(1, 31 % 8), 4.0f); CHECK_EQ(at(1, 32 % 8), 5.0f);
    CHECK_EQ(at(0, 21 % 8), 6.0f);
}

/* dflash_conv: tap 0 always, tap 1 only past a block's first row; each tap's coefficient is base
 * plus its group's delta, read from the first taps * NG columns of a twice-as-wide row. */
TEST(dflash_conv_resets_its_taps_at_each_block) {
    const int64_t T_ = 4, H = 4, NG = 2, taps = 2;
    std::vector<float> xv;
    for (int i = 0; i < T_ * H; ++i) xv.push_back((float)(i + 1));
    T x(RAD_F32, {T_, H}); x.f(xv);
    std::vector<float> dv;
    for (int t = 0; t < T_; ++t) {
        dv.insert(dv.end(), { 0, 0, 0, 1 });           /* delta[t][0][g] = 0, delta[t][1][g] = g */
        dv.insert(dv.end(), { 1000, 1000, 1000, 1000 });  /* the other half of the projection */
    }
    T dl(RAD_F32, {T_, 2 * taps * NG}); dl.f(dv);
    T base(RAD_F32, {taps, H}); base.f({1, 1, 1, 1, 0.5f, 0.5f, 0.5f, 0.5f});
    T out(RAD_F32, {T_, H});
    A a; a.t(x).t(dl).t(base).t(out).i("T", T_).i("hidden_size", H).i("taps", taps)
          .i("group_size", 2).i("NG", NG).i("block_size", 2).s("dtype", "f32");
    CHECK_OK(run("dflash_conv", a));
    const float want[16] = { 1, 2, 3, 4, 5.5f, 7, 11.5f, 14, 9, 10, 11, 12, 17.5f, 19, 31.5f, 34 };
    for (int i = 0; i < 16; ++i) CHECK_EQ(out.at(i), want[i]);
    A b; b.t(x).t(dl).t(base).t(x).i("T", T_).i("hidden_size", H).i("taps", taps)
          .i("group_size", 2).i("NG", NG).i("block_size", 2).s("dtype", "f32");
    CHECK_EQ(run("dflash_conv", b), RAD_E_INVAL);      /* out aliasing x */
}

/* ================================================================== production and drafter ops
 * The ops with no case above, each on a hand-checkable input. */

/* rmsnorm_quant_fp8 with a residual stores the sum first and takes the variance from the narrowed
 * sum: 1 + 2^-9 is 1.0 in bf16, so the normed row is exactly 1 (from the f32 sum it would be
 * 0.996 after rounding). An all-zero group gets scale 1 and zero codes. */
TEST(rmsnorm_quant_fp8_norms_the_narrowed_sum) {
    T x(RAD_BF16, {2, 2}); x.f({1, 1, 0, 0});
    T res(RAD_BF16, {2, 2}); res.f({0x1p-9f, 0x1p-9f, 0, 0});
    T w(RAD_BF16, {2}); w.f({1, 1});
    T q(RAD_F8E4M3, {2, 2}), s(RAD_F32, {2, 1}), ob(RAD_BF16, {2, 2});
    A a; a.t(x).t(res).t(w).t(q).t(s).t(ob).i("M", 2).i("n", 2).d("eps", 1e-6).i("group", 2)
          .s("dtype", "bf16");
    CHECK_OK(run("rmsnorm_quant_fp8", a));
    CHECK_EQ(res.at(0), 1.0f); CHECK_EQ(res.at(1), 1.0f);
    CHECK_EQ(ob.at(0), 1.0f); CHECK_EQ(ob.at(1), 1.0f);
    CHECK_EQ(s.at(0), 1.0f / 448.0f);
    CHECK_EQ(q.at(0), 448.0f); CHECK_EQ(q.at(1), 448.0f);
    CHECK_EQ(s.at(1), 1.0f);
    CHECK_EQ(q.at(2), 0.0f); CHECK_EQ(q.at(3), 0.0f);
}

/* scale_rows: one scalar per row, through `act`, then the optional add of the narrowed product. */
TEST(scale_rows_gates_each_row_by_its_scalar) {
    T x(RAD_F32, {2, 2}); x.f({1, 2, 3, 4});
    T s(RAD_F32, {2}); s.f({0, 2});
    T y(RAD_F32, {2, 2});
    A a; a.t(x).t(s).tnull().t(y).i("M", 2).i("n", 2).s("act", "sigmoid").s("dtype", "f32");
    CHECK_OK(run("scale_rows", a));
    CHECK_EQ(y.at(0), 0.5f); CHECK_EQ(y.at(1), 1.0f);
    CHECK_NEAR(y.at(2), 3.0 / (1.0 + std::exp(-2.0)), 1e-6);
    T add(RAD_F32, {2, 2}); add.f({10, 20, 30, 40});
    T y2(RAD_F32, {2, 2});
    A b; b.t(x).t(s).t(add).t(y2).i("M", 2).i("n", 2).s("act", "none").s("dtype", "f32");
    CHECK_OK(run("scale_rows", b));
    CHECK_EQ(y2.at(0), 10.0f); CHECK_EQ(y2.at(1), 20.0f);
    CHECK_EQ(y2.at(2), 36.0f); CHECK_EQ(y2.at(3), 48.0f);
}

/* tile_grid: diffusers' tiled_decode boxes -- a tile every `stride` latents while it starts inside
 * the axis, `tile` long or to the edge -- one row (t, h, w, t0, y0, x0, k, count) a tile, rows past
 * the count zero. 29 rows at stride 14 are bands of 16, 15 and 1; 16 columns are bands of 16 and
 * 2. Under `intervals` (the diffusion decoder's _tile_intervals) an axis of `tile` cells or fewer
 * is one band and a last band shorter than `min` joins the one before: 61 frames at 40 / 28 are
 * [0, 40) and [28, 61); 68 rows at 96 / 88 one band; 100 columns at 96 / 88, min 13, one band
 * of 96 and then [88, 100) is 12 < 13, merged: [0, 100). The last time band carries `tail` more. */
TEST(tile_grid_cuts_where_tiled_decode_does) {
    T g(RAD_I32, {3}); g.n({3, 29, 16});
    T y(RAD_I32, {7, 8});
    A a; a.t(g).t(y).i("M", 7).i("tile", 16).i("stride", 14);
    CHECK_OK(run("tile_grid", a));
    const int32_t want[7][8] = { { 3, 16, 16, 0, 0, 0, 0, 6 },   { 3, 16, 2, 0, 0, 14, 1, 6 },
                                 { 3, 15, 16, 0, 14, 0, 2, 6 },  { 3, 15, 2, 0, 14, 14, 3, 6 },
                                 { 3, 1, 16, 0, 28, 0, 4, 6 },   { 3, 1, 2, 0, 28, 14, 5, 6 },
                                 { 0, 0, 0, 0, 0, 0, 0, 0 } };
    for (int k = 0; k < 7; ++k)
        for (int e = 0; e < 8; ++e) CHECK_EQ(y.iat(k * 8 + e), want[k][e]);
    T g2(RAD_I32, {3}); g2.n({61, 68, 100});
    T y2(RAD_I32, {3, 8});
    A b; b.t(g2).t(y2).i("M", 3).i("tile", 96).i("stride", 88).s("rule", "intervals").i("min", 13)
        .i("tile_t", 40).i("stride_t", 28).i("min_t", 6).i("tail", 8);
    CHECK_OK(run("tile_grid", b));
    const int32_t want2[3][8] = { { 40, 68, 100, 0, 0, 0, 0, 2 }, { 41, 68, 100, 28, 0, 0, 1, 2 },
                                  { 0, 0, 0, 0, 0, 0, 0, 0 } };
    for (int k = 0; k < 3; ++k)
        for (int e = 0; e < 8; ++e) CHECK_EQ(y2.iat(k * 8 + e), want2[k][e]);
}

/* attn_na3d is attention over each query's window: on a 2 x 4 x 5 volume with a 3 x 3 x 3 window
 * (2 in time, the axis's length), the window of the corner voxel (0, 0, 0) is shifted inward to
 * start at (0, 0, 0), and of (1, 3, 4) to start at (0, 1, 2); against attention over exactly those
 * keys. */
TEST(attn_na3d_attends_the_inward_shifted_window) {
    const int64_t Tn = 2, Hn = 4, Wn = 5, V = Tn * Hn * Wn, hd = 4;
    auto val = [](int64_t i) { return (float)((i * 29) % 23 - 11) / 8.0f; };
    std::vector<float> qv((size_t)(V * hd)), kv((size_t)(V * hd)), vv((size_t)(V * hd));
    for (int64_t i = 0; i < V * hd; ++i) { qv[(size_t)i] = val(i); kv[(size_t)i] = val(i + 7); vv[(size_t)i] = val(i + 13); }
    T q(RAD_F32, {V, hd}); q.f(qv);
    T k(RAD_F32, {V, hd}); k.f(kv);
    T v(RAD_F32, {V, hd}); v.f(vv);
    T g(RAD_I32, {3}); g.n({Tn, Hn, Wn});
    T o(RAD_F32, {V, hd});
    A a; a.t(q).t(k).t(v).t(g).t(o).i("M", V).i("head_dim", hd).i("kt", 3).i("kh", 3).i("kw", 3)
        .s("dtype", "f32");
    CHECK_OK(run("attn_na3d", a));
    struct Q { int64_t t, i, j, t0, i0, j0; };
    for (const Q c : { Q{ 0, 0, 0, 0, 0, 0 }, Q{ 1, 3, 4, 0, 1, 2 }, Q{ 0, 2, 2, 0, 1, 1 } }) {
        const int64_t r = (c.t * Hn + c.i) * Wn + c.j;
        std::vector<double> sc;
        std::vector<int64_t> keys;
        for (int64_t a1 = c.t0; a1 < c.t0 + 2; ++a1)
            for (int64_t b1 = c.i0; b1 < c.i0 + 3; ++b1)
                for (int64_t c1 = c.j0; c1 < c.j0 + 3; ++c1) {
                    const int64_t key = (a1 * Hn + b1) * Wn + c1;
                    double d = 0;
                    for (int64_t e = 0; e < hd; ++e) d += (double)qv[(size_t)(r * hd + e)] * kv[(size_t)(key * hd + e)];
                    sc.push_back(d / 2.0);
                    keys.push_back(key);
                }
        double mx = -1e30, sum = 0;
        for (double x : sc) mx = std::max(mx, x);
        for (double& x : sc) { x = std::exp(x - mx); sum += x; }
        for (int64_t e = 0; e < hd; ++e) {
            double want = 0;
            for (size_t n = 0; n < keys.size(); ++n) want += sc[n] / sum * vv[(size_t)(keys[n] * hd + e)];
            CHECK_NEAR(o.at(r * hd + e), want, 1e-5);
        }
    }
}

/* vol_index: a 2 x 2 x 3 volume's coordinates, component-major; voxel 7 is (1, 0, 1). */
TEST(vol_index_writes_each_voxels_coordinates) {
    T g(RAD_I32, {3}); g.n({2, 2, 3});
    T y(RAD_I32, {3, 12});
    A a; a.t(g).t(y).i("M", 12);
    CHECK_OK(run("vol_index", a));
    CHECK_EQ(y.iat(7), 1); CHECK_EQ(y.iat(12 + 7), 0); CHECK_EQ(y.iat(24 + 7), 1);
    CHECK_EQ(y.iat(11), 1); CHECK_EQ(y.iat(12 + 11), 1); CHECK_EQ(y.iat(24 + 11), 2);
}

/* tile_crop copies a box out of the volume: a 2 x 3 x 4 volume of its row indices, the 1 x 2 x 2
 * box at (1, 1, 1). */
TEST(tile_crop_copies_the_box) {
    T x(RAD_F32, {24, 1});
    std::vector<float> v(24);
    for (int i = 0; i < 24; ++i) v[(size_t)i] = (float)i;
    x.f(v);
    T g(RAD_I32, {3}); g.n({2, 3, 4});
    T b(RAD_I32, {6}); b.n({1, 2, 2, 1, 1, 1});
    T y(RAD_F32, {4, 1});
    A a; a.t(x).t(g).t(b).t(y).i("M", 4).i("C", 1).s("dtype", "f32");
    CHECK_OK(run("tile_crop", a));
    CHECK_EQ(y.at(0), 17.0f); CHECK_EQ(y.at(1), 18.0f);
    CHECK_EQ(y.at(2), 21.0f); CHECK_EQ(y.at(3), 22.0f);
}

/* tile_stitch: tiles cut from one picture stitch back to that picture's bytes (every blend mixes
 * equal values; with an extent of 2 the weights are 0, 0.5 and 1, so exactly); and across a seam
 * between a tile of zeros and one of ones, the first column of the right tile is the left tile's
 * margin and the second their mean. A 5 x 5 latent grid at tile 4, stride 3, a latent 1 x 1 patch
 * of 2 x 2 pixels: bands of 4 and 2 latents, 8 and 4 pixels, kept to 6. */
TEST(tile_stitch_blends_as_tiled_decode_and_keeps_a_cut_picture) {
    const int64_t H = 5, W = 5, tile = 4, stride = 3, p = 2, Hp = H * p, Wp = W * p;
    auto band = [&](int64_t L, int64_t k) { return std::min(tile, L - k * stride); };
    auto band_of = [](int64_t L, int64_t k, int64_t t, int64_t s) { return std::min(t, L - k * s); };
    std::vector<float> img((size_t)(Hp * Wp));
    for (int64_t i = 0; i < Hp * Wp; ++i) img[(size_t)i] = (float)((i * 37) % 63 - 31) / 32.0f;
    auto store = [&](auto value) {
        std::vector<float> v;
        for (int64_t i = 0; i < 2; ++i)
            for (int64_t j = 0; j < 2; ++j)
                for (int64_t py = 0; py < band(H, i); ++py)      /* patch rows of the tile */
                    for (int64_t px = 0; px < band(W, j); ++px)
                        for (int64_t dx = 0; dx < p; ++dx)        /* order wh: column offset outer */
                            for (int64_t dy = 0; dy < p; ++dy)
                                v.push_back(value(i, j, py * p + dy, px * p + dx));
        return v;
    };
    auto run_stitch = [&](const std::vector<float>& tiles, std::vector<uint8_t>* out) {
        T x(RAD_F32, {(int64_t)tiles.size() / (p * p), p * p}); x.f(tiles);
        T g(RAD_I32, {3}); g.n({1, H, W});
        T f(RAD_I32, {1}); f.n({0});
        T y(RAD_U8, {Hp, Wp});
        A a; a.t(x).t(g).t(f).t(y).i("M", Hp).i("tile", tile).i("stride", stride).i("scale", 1)
            .i("p", p).i("C", 1).s("dtype", "f32");
        CHECK_OK(run("tile_stitch", a));
        out->resize((size_t)(Hp * Wp));
        for (int64_t i = 0; i < Hp * Wp; ++i) (*out)[(size_t)i] = y.mem[(size_t)i];
    };
    std::vector<uint8_t> got;
    run_stitch(store([&](int64_t i, int64_t j, int64_t yy, int64_t xx) {
                   return img[(size_t)((i * stride * p + yy) * Wp + j * stride * p + xx)];
               }), &got);
    for (int64_t i = 0; i < Hp * Wp; ++i) {
        const float u = std::min(std::max(img[(size_t)i] / 2.0f + 0.5f, 0.0f), 1.0f);
        CHECK_EQ((int)got[(size_t)i], (int)std::nearbyint(u * 255.0f));
    }
    /* Left tiles zeros, right tiles ones: row 0 is 0 to column 6, the margin (0) at 6, the mean at
     * 7, then 1 (bytes 128, 128, 191, 255). */
    run_stitch(store([](int64_t, int64_t j, int64_t, int64_t) { return j ? 1.0f : 0.0f; }), &got);
    CHECK_EQ((int)got[5], 128); CHECK_EQ((int)got[6], 128);
    CHECK_EQ((int)got[7], 191); CHECK_EQ((int)got[8], 255);

    /* In time too: 4 cells of 2 frames, the first one frame (7 frames), cut at tile_t 3, stride_t 2
     * into cells [0, 3) (frames 0..4) and [2, 4) (frames 3..6); each tile is its box of a 7-frame
     * clip, and every frame stitches back to the clip's bytes. */
    const int64_t nfr = 7;
    std::vector<float> clip((size_t)(nfr * Hp * Wp));
    for (int64_t i = 0; i < (int64_t)clip.size(); ++i) clip[(size_t)i] = (float)((i * 53) % 61 - 30) / 32.0f;
    std::vector<float> tiles3;
    for (int64_t kt = 0; kt < 2; ++kt) {
        const int64_t P = kt == 0 ? 0 : 3, F = kt == 0 ? 5 : 4;
        for (int64_t i = 0; i < 2; ++i)
            for (int64_t j = 0; j < 2; ++j)
                for (int64_t f = 0; f < F; ++f)
                    for (int64_t py = 0; py < band_of(H, i, tile, stride); ++py)
                        for (int64_t px = 0; px < band_of(W, j, tile, stride); ++px)
                            for (int64_t dx = 0; dx < p; ++dx)
                                for (int64_t dy = 0; dy < p; ++dy)
                                    tiles3.push_back(clip[(size_t)(((P + f) * Hp + i * stride * p +
                                                                    py * p + dy) * Wp +
                                                                   j * stride * p + px * p + dx)]);
    }
    {
        T x(RAD_F32, {(int64_t)tiles3.size() / (p * p), p * p}); x.f(tiles3);
        T g(RAD_I32, {3}); g.n({4, H, W});
        T f(RAD_I32, {1}); f.n({0});
        T y(RAD_U8, {nfr * Hp, Wp});
        A a; a.t(x).t(g).t(f).t(y).i("M", nfr * Hp).i("tile", tile).i("stride", stride)
            .i("tile_t", 3).i("stride_t", 2).i("scale", 1).i("scale_t", 2).i("p", p).i("C", 1)
            .s("dtype", "f32");
        CHECK_OK(run("tile_stitch", a));
        int64_t bad = 0;
        for (int64_t i = 0; i < nfr * Hp * Wp; ++i) {
            const float u = std::min(std::max(clip[(size_t)i] / 2.0f + 0.5f, 0.0f), 1.0f);
            bad += (int)y.mem[(size_t)i] != (int)std::nearbyint(u * 255.0f);
        }
        CHECK_EQ(bad, (int64_t)0);
    }
}

TEST(sigmoid_is_the_logistic_function) {
    T x(RAD_F32, {1, 4}); x.f({0, 100, -100, 1});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(y).i("M", 1).i("n", 4).s("dtype", "f32");
    CHECK_OK(run("sigmoid", a));
    CHECK_EQ(y.at(0), 0.5f); CHECK_EQ(y.at(1), 1.0f);
    CHECK_NEAR(y.at(2), 0.0, 1e-30);
    CHECK_NEAR(y.at(3), 1.0 / (1.0 + std::exp(-1.0)), 1e-7);
}

/* fill writes `value` (0 when absent) over its extent at the operand's row pitch, and nothing
 * between rows. */
TEST(fill_writes_its_value_at_the_row_pitch_only) {
    T buf(RAD_F32, {2, 5}); buf.f({9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
    T view = buf;
    view.t = buf.t;
    view.t.shape[1] = 3;                  /* [2, 3] at a row pitch of 5 */
    A a; a.t(view).i("M", 2).i("n", 3).s("dtype", "f32").d("value", 1.5);
    CHECK_OK(run("fill", a));
    const float want[10] = { 1.5f, 1.5f, 1.5f, 9, 9, 1.5f, 1.5f, 1.5f, 9, 9 };
    for (int i = 0; i < 10; ++i) CHECK_EQ(buf.at(i), want[i]);
    A b; b.t(view).i("M", 2).i("n", 3).s("dtype", "f32");
    CHECK_OK(run("fill", b));
    CHECK_EQ(buf.at(0), 0.0f); CHECK_EQ(buf.at(3), 9.0f);
}

/* had_quant_act_i8g is the same at int8 codes: amax / 127, nothing folded either. */
TEST(had_quant_act_i8g_rotates_before_the_absmax) {
    T x(RAD_F32, {1, 4}); x.f({1, 1, 1, -1});
    T q(RAD_I8, {1, 4}), s(RAD_F32, {1, 2});
    A a; a.t(x).t(q).t(s).i("M", 1).i("n", 4).i("group", 2).s("dtype", "f32");
    CHECK_OK(run("had_quant_act_i8g", a));
    CHECK_EQ(s.at(0), 2.0f / 127.0f); CHECK_EQ(s.at(1), 2.0f / 127.0f);
    CHECK_EQ(q.at(0), 127.0f); CHECK_EQ(q.at(1), 0.0f);
    CHECK_EQ(q.at(2), 0.0f); CHECK_EQ(q.at(3), 127.0f);
}

namespace {
void put_pair(T& t, int64_t i, int32_t id, float v) {
    std::memcpy((float*)t.t.data + 2 * i, &id, 4);
    ((float*)t.t.data)[2 * i + 1] = v;
}
}  /* namespace */

/* row_topk_merge over two ranks' candidate sets equals row_topk over the unsharded row: values
 * descending, ties to the lower id, a -1 hole dropped. */
TEST(row_topk_merge_equals_a_top_r_over_the_unsharded_row) {
    /* Twelve columns, ranks own [0, 6) and [6, 12). */
    const std::vector<float> row = { 0.1f, 0.2f, 0.5f, 0.5f, 0.0f, 1.0f,
                                     -1.0f, 1.0f, 0.3f, 0.75f, 0.5f, -2.0f };
    T x(RAD_F32, {1, 12}); x.f(row);
    T ix(RAD_I32, {1, 4}), vx(RAD_F32, {1, 4});
    A a; a.t(x).t(ix).t(vx).tnull().i("M", 1).i("N", 12).i("R", 4).s("dtype", "f32");
    CHECK_OK(run("row_topk", a));
    CHECK_EQ(ix.iat(0), 5LL); CHECK_EQ(ix.iat(1), 7LL);
    CHECK_EQ(ix.iat(2), 9LL); CHECK_EQ(ix.iat(3), 2LL);

    T g(RAD_F32, {1, 2 * 4 * 2});
    put_pair(g, 0, 5, 1.0f); put_pair(g, 1, 2, 0.5f); put_pair(g, 2, 3, 0.5f); put_pair(g, 3, -1, -INFINITY);
    put_pair(g, 4, 7, 1.0f); put_pair(g, 5, 9, 0.75f); put_pair(g, 6, 10, 0.5f); put_pair(g, 7, 8, 0.3f);
    T im(RAD_I32, {1, 4}), vm(RAD_F32, {1, 4});
    A b; b.t(g).t(im).t(vm).i("M", 1).i("R", 4).i("world_size", 2).s("dtype", "f32");
    CHECK_OK(run("row_topk_merge", b));
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(im.iat(i), ix.iat(i));
        CHECK_EQ(vm.at(i), vx.at(i));
    }
    /* A row of holes merges to -1 and -inf. */
    T h(RAD_F32, {1, 4});
    put_pair(h, 0, -1, 3.0f); put_pair(h, 1, -1, 2.0f);
    T ih(RAD_I32, {1, 2}), vh(RAD_F32, {1, 2});
    A c; c.t(h).t(ih).t(vh).i("M", 1).i("R", 1).i("world_size", 2).s("dtype", "f32");
    CHECK_OK(run("row_topk_merge", c));
    CHECK_EQ(ih.iat(0), -1LL);
    CHECK(std::isinf(vh.at(0)) && vh.at(0) < 0);
}

/* logit_rerank rescores each candidate with the exact head row, sorts descending with ties to the
 * lower id, and turns an id outside this rank's slice into a pad that sorts last. */
TEST(logit_rerank_rescores_sorts_and_pads) {
    T x(RAD_F32, {1, 2}); x.f({1, 2});
    T head(RAD_F32, {4, 2}); head.f({1, 0, 0, 1, 1, 1, 2, 0.5f});   /* scores 1, 2, 3, 3 */
    T in(RAD_I32, {1, 5}); in.n({3, 0, 2, 9, -1});
    T ix(RAD_I32, {1, 5}), vx(RAD_F32, {1, 5});
    A a; a.t(x).t(head).t(in).t(ix).t(vx).tnull().i("M", 1).i("R", 5).i("n_vocab", 4)
          .i("n_embd", 2).s("dtype", "f32");
    CHECK_OK(run("logit_rerank", a));
    const int64_t wid[5] = { 2, 3, 0, -1, -1 };
    const float wv[3] = { 3, 3, 1 };
    for (int i = 0; i < 5; ++i) CHECK_EQ(ix.iat(i), wid[i]);
    for (int i = 0; i < 3; ++i) CHECK_EQ(vx.at(i), wv[i]);
    CHECK(std::isinf(vx.at(4)) && vx.at(4) < 0);

    /* A slice holding ids 2 and 3: id 1 is another rank's. */
    T hs(RAD_F32, {2, 2}); hs.f({1, 1, 2, 0.5f});
    T in2(RAD_I32, {1, 2}); in2.n({1, 3});
    T ix2(RAD_I32, {1, 2}), vx2(RAD_F32, {1, 2});
    A b; b.t(x).t(hs).t(in2).t(ix2).t(vx2).tnull().i("M", 1).i("R", 2).i("n_vocab", 2)
          .i("n_embd", 2).i("vocab_off", 2).s("dtype", "f32");
    CHECK_OK(run("logit_rerank", b));
    CHECK_EQ(ix2.iat(0), 3LL); CHECK_EQ(vx2.at(0), 3.0f);
    CHECK_EQ(ix2.iat(1), -1LL);
}

/* dflash_select's walk scores each step's candidates against the previous step's pick, not the
 * anchor: here step 1 picks token 0 after step 0 picked 1, where the anchor would pick 3. */
TEST(dflash_select_walks_from_its_own_previous_pick) {
    T cand(RAD_I32, {2, 2}); cand.n({1, 2, 3, 0});
    T un(RAD_F32, {2, 2}); un.f({0, 0, 0, 0.5f});
    T anc(RAD_I32, {1}); anc.n({0});
    T pred(RAD_F32, {4, 1}); pred.f({1, -1, 2, 0});
    T succ(RAD_F32, {4, 1}); succ.f({0, 3, 1, 2});
    T tok(RAD_I32, {1, 2});
    A a; a.t(cand).t(un).tnull().t(anc).t(pred).t(succ).t(tok).i("M", 1).i("steps", 2)
          .i("top_k", 2).i("rank", 1).i("n_vocab", 4).s("dtype", "f32");
    CHECK_OK(run("dflash_select", a));
    CHECK_EQ(tok.iat(0), 1LL);
    CHECK_EQ(tok.iat(1), 0LL);
}

/* The DSpark reference scores all vocabulary entries after applying the Markov bias. Both
 * sequences choose tokens outside their base-logit top 16; step 1 conditions on the prior greedy
 * choice, and the step-0 tie across tile boundaries resolves to the lower token id. */
TEST(dflash_markov_greedy_scans_full_vocab_and_keeps_sequences_independent) {
    constexpr int V = 130, R = 32, M = 2, steps = 2, tiles = 3;
    std::vector<float> logits_values(M * steps * V, 0.0f);
    for (int r = 0; r < M * steps; r += 2)
        for (int v = 0; v < 16; ++v) logits_values[r * V + v] = 100 - v;
    T logits(RAD_BF16, {M * steps, V}); logits.f(logits_values);
    T anchor(RAD_I32, {M, 2}); anchor.n({1, -1, 2, -1});
    std::vector<float> pred_values(V * R, 0.0f), succ_values(V * R, 0.0f);
    pred_values[1 * R + 0] = 1;       /* sequence 0, step 0 predecessor */
    succ_values[127 * R + 0] = 200;   /* outside the base top 16 */
    succ_values[129 * R + 0] = 200;   /* equal biased score in a later tile */
    pred_values[127 * R + 1] = 1;     /* sequence 0 step 1 depends on token 127 */
    succ_values[100 * R + 1] = 100;
    pred_values[2 * R + 2] = 1;       /* sequence 1 has a separate anchor/codebook walk */
    succ_values[80 * R + 2] = 200;
    pred_values[80 * R + 3] = 1;
    succ_values[101 * R + 3] = 90;
    T pred(RAD_BF16, {V, R}); pred.f(pred_values);
    T succ(RAD_BF16, {V, R}); succ.f(succ_values);
    T tile_scores(RAD_F32, {M, tiles});
    T tile_tokens(RAD_I32, {M, tiles});
    T out(RAD_I32, {M, steps});
    A a; a.t(logits).t(anchor).t(pred).t(succ).t(tile_scores).t(tile_tokens).t(out)
          .i("M", M).i("steps", steps).i("rank", R).i("n_vocab", V)
          .i("anchor_stride", 2).s("dtype", "bf16");
    CHECK_OK(run("dflash_markov_greedy", a));
    CHECK_EQ(out.iat(0), 127LL);
    CHECK_EQ(out.iat(1), 100LL);
    CHECK_EQ(out.iat(2), 80LL);
    CHECK_EQ(out.iat(3), 101LL);
    CHECK_EQ(tile_tokens.iat(0), 0LL);
    CHECK_EQ(tile_tokens.iat(1), 100LL);
    CHECK_EQ(tile_tokens.iat(2), 128LL);
    CHECK_EQ(tile_scores.at(0), 0.0f);
    CHECK_EQ(tile_scores.at(1), 100.0f);
    CHECK_EQ(tile_scores.at(2), 0.0f);
}

TEST(dflash_markov_f32_preserves_a_near_tie_and_refuses_bf16) {
    T logits(RAD_F32, {1, 2}); logits.f({1.001f, 1.002f});
    T anchor(RAD_I32, {1}); anchor.n({0});
    T pred(RAD_BF16, {2, 32}), succ(RAD_BF16, {2, 32});
    T score(RAD_F32, {1, 1}), token(RAD_I32, {1, 1}), out(RAD_I32, {1, 1});
    A a; a.t(logits).t(anchor).t(pred).t(succ).t(score).t(token).t(out)
        .i("M", 1).i("steps", 1).i("rank", 32).i("n_vocab", 2).s("dtype", "bf16");
    CHECK_OK(run("dflash_markov_greedy_f32", a));
    CHECK_EQ(out.iat(0), 1);
    T bad_anchor(RAD_F32, {1}); bad_anchor.f({0});
    a.ts[1] = bad_anchor.t;
    CHECK_EQ(run("dflash_markov_greedy_f32", a), RAD_E_SHAPE);
    a.ts[1] = anchor.t;
    CHECK(run("dflash_markov_greedy", a) != RAD_OK);
    T bf_logits(RAD_BF16, {1, 2}); bf_logits.f({1.001f, 1.002f});
    a.ts[0] = bf_logits.t;
    CHECK(run("dflash_markov_greedy_f32", a) != RAD_OK);
    CHECK_OK(run("dflash_markov_greedy", a));
    CHECK_EQ(out.iat(0), 0);
}

/* Independent pitches in the two scratch arrays are part of the tensor contract. Guard cells
 * distinguish correct row addressing from treating both outputs as tightly packed. */
TEST(dflash_markov_greedy_padded_scratch_rows) {
    constexpr int V=130, R=32, M=2, tiles=3;
    T logits(RAD_BF16, {M, V}), anchor(RAD_I32, {M}); anchor.n({1, 2});
    T pred(RAD_BF16, {V, R}), succ(RAD_BF16, {V, R});
    st_dt(RAD_BF16, pred.t.data, R, 1); st_dt(RAD_BF16, pred.t.data, 2*R+1, 1);
    st_dt(RAD_BF16, succ.t.data, 129*R, 4); st_dt(RAD_BF16, succ.t.data, 80*R+1, 8);
    T scores(RAD_F32, {M, 5}); scores.f(std::vector<float>(10, -777));
    T tokens(RAD_I32, {M, 7}); tokens.n(std::vector<int64_t>(14, -777));
    scores.t.shape[1]=tiles; tokens.t.shape[1]=tiles;
    T out(RAD_I32, {M, 1});
    A a; a.t(logits).t(anchor).t(pred).t(succ).t(scores).t(tokens).t(out)
        .i("M", M).i("steps", 1).i("rank", R).i("n_vocab", V).s("dtype", "bf16");
    CHECK_OK(run("dflash_markov_greedy", a));
    CHECK_EQ(out.iat(0), 129); CHECK_EQ(out.iat(1), 80);
    CHECK_EQ(tokens.iat(2), 129); CHECK_EQ(tokens.iat(7+1), 80);
    CHECK_EQ(scores.at(2), 4); CHECK_EQ(scores.at(5+1), 8);
    for (int i : {3,4,8,9}) CHECK_EQ(scores.at(i), -777);
    for (int i : {3,4,5,6,10,11,12,13}) CHECK_EQ(tokens.iat(i), -777);
}

/* A one-row, one-layer DSpark oracle assembled from libref's ordinary ops. The literals check the
 * architecture-specific chain rather than only each primitive: each tap has its own RMS gain,
 * the normalized taps feed fc/h_norm then context K/V projections, context and draft K/V land in
 * the paged cache, and a gated attention residual plus the exact-GELU MLP gives the layer output. */
TEST(dspark_taps_context_kv_and_layer_output_match_hand_oracle) {
    T tap0_src(RAD_F32, {1, 2}); tap0_src.f({1, 1});
    T tap0_gain(RAD_BF16, {2}); tap0_gain.f({1, 2});
    T tap0(RAD_BF16, {1, 2});
    A tap0a; tap0a.t(tap0_src).t(tap0_gain).t(tap0)
                 .i("M", 1).i("n", 2).d("eps", 0).s("dtype", "bf16");
    CHECK_OK(run("rmsnorm", tap0a));
    CHECK_EQ(tap0.at(0), 1.0f); CHECK_EQ(tap0.at(1), 2.0f);

    T tap1_src(RAD_F32, {1, 2}); tap1_src.f({2, 2});
    T tap1_gain(RAD_BF16, {2}); tap1_gain.f({2, 1});
    T tap1(RAD_BF16, {1, 2});
    A tap1a; tap1a.t(tap1_src).t(tap1_gain).t(tap1)
                 .i("M", 1).i("n", 2).d("eps", 0).s("dtype", "bf16");
    CHECK_OK(run("rmsnorm", tap1a));
    CHECK_EQ(tap1.at(0), 2.0f); CHECK_EQ(tap1.at(1), 1.0f);

    T taps(RAD_BF16, {1, 4}); taps.f({tap0.at(0), tap0.at(1), tap1.at(0), tap1.at(1)});
    T fc_w(RAD_BF16, {2, 4}); fc_w.f({1, 0, 0, 0,  0, 0, 0, 1});
    T context(RAD_BF16, {1, 2});
    A fca; fca.t(taps).t(fc_w).t(context).i("M", 1).i("N", 2).i("K", 4).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", fca));
    CHECK_EQ(context.at(0), 1.0f); CHECK_EQ(context.at(1), 1.0f);

    T context_gain(RAD_BF16, {2}); context_gain.f({1, 1});
    T context_norm(RAD_BF16, {1, 2});
    A cna; cna.t(context).t(context_gain).t(context_norm)
            .i("M", 1).i("n", 2).d("eps", 0).s("dtype", "bf16");
    CHECK_OK(run("rmsnorm", cna));
    CHECK_EQ(context_norm.at(0), 1.0f); CHECK_EQ(context_norm.at(1), 1.0f);

    T wq(RAD_BF16, {2, 2}); wq.f({0, 0, 0, 0});
    T wk(RAD_BF16, {2, 2}); wk.f({0, 0, 0, 0});
    T wv(RAD_BF16, {2, 2}); wv.f({0, 0, 1, 1});
    T q_proj(RAD_BF16, {1, 2}), k_ctx(RAD_BF16, {1, 2}), v_ctx(RAD_BF16, {1, 2});
    A qa; qa.t(context_norm).t(wq).t(q_proj).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    A ka; ka.t(context_norm).t(wk).t(k_ctx).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    A va; va.t(context_norm).t(wv).t(v_ctx).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", qa)); CHECK_OK(run("gemm_nt", ka)); CHECK_OK(run("gemm_nt", va));
    CHECK_EQ(k_ctx.at(0), 0.0f); CHECK_EQ(k_ctx.at(1), 0.0f);
    CHECK_EQ(v_ctx.at(0), 0.0f); CHECK_EQ(v_ctx.at(1), 2.0f);

    /* The query projection is from the draft row, but has the same simple normalized input. */
    T query_src(RAD_F32, {1, 2}); query_src.f({1, 1});
    T query_norm(RAD_BF16, {1, 2});
    A qna; qna.t(query_src).t(context_gain).t(query_norm)
            .i("M", 1).i("n", 2).d("eps", 0).s("dtype", "bf16");
    CHECK_OK(run("rmsnorm", qna));
    T k_query(RAD_BF16, {1, 2}), v_query(RAD_BF16, {1, 2});
    A kqa; kqa.t(query_norm).t(wk).t(k_query).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    A vqa; vqa.t(query_norm).t(wv).t(v_query).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", kqa)); CHECK_OK(run("gemm_nt", vqa));

    T keys(RAD_BF16, {2, 1, 2}); keys.f({k_ctx.at(0), k_ctx.at(1), k_query.at(0), k_query.at(1)});
    T values(RAD_BF16, {2, 1, 2}); values.f({v_ctx.at(0), v_ctx.at(1), v_query.at(0), v_query.at(1)});
    T slots(RAD_I32, {2}); slots.n({0, 1});
    T cache(RAD_BF16, {1, 1, 2, 4});
    A store; store.t(keys).t(values).t(slots).t(cache)
              .i("M", 2).i("head_dim", 2).i("n_head_kv", 1).i("block_size", 2).s("kv_dtype", "bf16");
    CHECK_OK(run("kv_store", store));
    for (int slot = 0; slot < 2; ++slot) {
        CHECK_EQ(cache.at(slot * 4 + 0), 0.0f); CHECK_EQ(cache.at(slot * 4 + 1), 0.0f);
        CHECK_EQ(cache.at(slot * 4 + 2), 0.0f); CHECK_EQ(cache.at(slot * 4 + 3), 2.0f);
    }

    T q(RAD_BF16, {1, 1, 2}); q.f({q_proj.at(0), q_proj.at(1)});
    T bt(RAD_I32, {1, 1}); bt.n({0});
    T used(RAD_I32, {1}); used.n({2});
    T attended(RAD_BF16, {1, 1, 2});
    A attn; attn.t(q).t(cache).t(bt).t(used).tnull().tnull().tnull().t(attended)
               .i("q_len", 1).i("head_dim", 2).i("gqa", 1).i("block_size", 2)
               .i("causal", 0).i("window", 513).i("n_head", 1)
               .s("q_dtype", "bf16").s("kv_dtype", "bf16");
    CHECK_OK(run("attn_paged_anchor_window", attn));
    CHECK_EQ(attended.at(0), 0.0f); CHECK_EQ(attended.at(1), 2.0f);

    T gate_w(RAD_BF16, {1, 2}); gate_w.f({0, 0});
    T gate_logit(RAD_BF16, {1, 1}), gate(RAD_BF16, {1, 1});
    A ga; ga.t(query_norm).t(gate_w).t(gate_logit)
           .i("M", 1).i("N", 1).i("K", 2).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", ga));
    A sig; sig.t(gate_logit).t(gate).i("M", 1).i("n", 1).s("dtype", "bf16");
    CHECK_OK(run("sigmoid", sig));
    CHECK_EQ(gate.at(0), 0.5f);
    T gated(RAD_BF16, {1, 2});
    A scale; scale.t(attended).t(gate).tnull().t(gated)
              .i("M", 1).i("n", 2).s("act", "none").s("dtype", "bf16");
    CHECK_OK(run("scale_rows", scale));
    CHECK_EQ(gated.at(0), 0.0f); CHECK_EQ(gated.at(1), 1.0f);

    T wo(RAD_BF16, {2, 2}); wo.f({1, 0, 0, 1});
    T delta(RAD_BF16, {1, 2});
    A oa; oa.t(gated).t(wo).t(delta).i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", oa));
    CHECK_EQ(delta.at(0), 0.0f); CHECK_EQ(delta.at(1), 1.0f);
    T residual(RAD_F32, {1, 2}); residual.f({1, 1});
    T after_attn(RAD_F32, {1, 2});
    A adda; adda.t(residual).t(delta).t(after_attn).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", adda));
    CHECK_EQ(after_attn.at(0), 1.0f); CHECK_EQ(after_attn.at(1), 2.0f);

    T ff_norm(RAD_BF16, {1, 2});
    A fna; fna.t(after_attn).t(context_gain).t(ff_norm)
            .i("M", 1).i("n", 2).d("eps", 0).s("dtype", "bf16");
    CHECK_OK(run("rmsnorm", fna));
    T ff_w(RAD_BF16, {2, 2}); ff_w.f({1, 0, 1, 0});
    T gate_up(RAD_BF16, {1, 2});
    A fua; fua.t(ff_norm).t(ff_w).t(gate_up)
             .i("M", 1).i("N", 2).i("K", 2).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", fua));
    T ff(RAD_BF16, {1, 1});
    A act; act.t(gate_up).t(gate_up).t(ff)
             .i("M", 1).i("n", 1).s("act", "gelu").s("dtype", "bf16");
    CHECK_OK(run("act_mul", act));
    T down_w(RAD_BF16, {2, 1}); down_w.f({1, 0});
    T down(RAD_BF16, {1, 2});
    A da; da.t(ff).t(down_w).t(down).i("M", 1).i("N", 2).i("K", 1).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", da));
    const float normalized = bf16_to_f32((uint16_t)f32_to_bf16(1.0f / std::sqrt(2.5f)));
    const float gelu = 0.5f * normalized * (1.0f + std::erf(normalized / std::sqrt(2.0f)));
    const float rounded_gelu = bf16_to_f32((uint16_t)f32_to_bf16(gelu));
    const float expected_ff = bf16_to_f32((uint16_t)f32_to_bf16(rounded_gelu * normalized));
    CHECK_NEAR(ff.at(0), expected_ff, 1e-6);
    CHECK_EQ(down.at(0), expected_ff); CHECK_EQ(down.at(1), 0.0f);
    T layer_out(RAD_F32, {1, 2});
    A outa; outa.t(after_attn).t(down).t(layer_out).i("M", 1).i("n", 2).s("dtype", "f32");
    CHECK_OK(run("add", outa));
    CHECK_NEAR(layer_out.at(0), 1.0f + expected_ff, 1e-6);
    CHECK_EQ(layer_out.at(1), 2.0f);
}

TEST(gemm_nt_q_bias_adds_a_bias_per_column) {
    T a_(RAD_I8, {1, 2}); a_.n({1, 2});
    T as(RAD_F32, {1}); as.f({0.5f});
    T b_(RAD_I8, {2, 2}); b_.n({1, 1, 2, 0});
    T bs(RAD_F32, {2}); bs.f({2, 1});
    T bias(RAD_F32, {2}); bias.f({10, 20});
    T y(RAD_F32, {1, 2});
    A a; a.t(a_).t(as).t(b_).t(bs).t(bias).t(y).i("M", 1).i("N", 2).i("K", 2).i("group", 2)
          .s("dtype", "i8a8");
    CHECK_OK(run("gemm_nt_q_bias", a));
    CHECK_EQ(y.at(0), 13.0f);
    CHECK_EQ(y.at(1), 21.0f);
}

/* gram_accum adds x^T x into h (twice is twice once); with a scale, x is dequantised per group. */
TEST(gram_accum_accumulates_the_dequantised_gram) {
    T a_(RAD_BF16, {2, 2}); a_.f({1, 2, 3, 4});
    T h(RAD_F32, {2, 2});
    for (int round = 1; round <= 2; ++round) {
        A a; a.t(a_).tnull().t(h).i("M", 2).i("n", 2).i("group", 2).s("dtype", "bf16");
        CHECK_OK(run("gram_accum", a));
        CHECK_EQ(h.at(0), 10.0f * round); CHECK_EQ(h.at(1), 14.0f * round);
        CHECK_EQ(h.at(2), 14.0f * round); CHECK_EQ(h.at(3), 20.0f * round);
    }
    T q(RAD_F8E4M3, {2, 2}); q.f({1, 2, 3, 4});
    T s(RAD_F32, {2, 1}); s.f({0.5f, 2});
    T h2(RAD_F32, {2, 2});
    A b; b.t(q).t(s).t(h2).i("M", 2).i("n", 2).i("group", 2).s("dtype", "fp8a8");
    CHECK_OK(run("gram_accum", b));
    CHECK_EQ(h2.at(0), 36.25f); CHECK_EQ(h2.at(1), 48.5f); CHECK_EQ(h2.at(3), 65.0f);
}

/* ================================================================== unsupported inputs refused
 * libref returns an error for what it cannot compute instead of RAD_OK with another function's
 * answer (docs/OPS.md, Conventions). */
TEST(gemm_nt_q_refuses_unread_codes_and_operands) {
    T a_(RAD_I8, {1, 32}), as(RAD_F32, {1, 1});
    T b4(RAD_U4, {2, 32}), bs(RAD_F16, {2, 1});
    T y(RAD_F32, {1, 2}); y.f({7, 7});
    A a; a.t(a_).t(as).t(b4).t(bs).t(y).tnull().tnull().i("M", 1).i("N", 2).i("K", 32)
          .i("group", 32).s("dtype", "w4a8_asym");
    CHECK_EQ(run("gemm_nt_q", a), RAD_E_DTYPE);
    CHECK_EQ(y.at(0), 7.0f);

    /* A zero-point correction it does not read, and a reference exponent beside a non-mxfp4
     * weight. */
    T b8(RAD_I8, {2, 32});
    T asum(RAD_F32, {2, 16});
    A b; b.t(a_).t(as).t(b8).t(bs).t(y).t(asum).tnull().i("M", 1).i("N", 2).i("K", 32)
          .i("group", 32).s("dtype", "w8a8");
    CHECK_EQ(run("gemm_nt_q", b), RAD_E_UNSUPPORTED);
    T bref(RAD_U8, {2});
    A c; c.t(a_).t(as).t(b8).t(bs).t(y).tnull().t(bref).i("M", 1).i("N", 2).i("K", 32)
          .i("group", 32).s("dtype", "w8a8");
    CHECK_EQ(run("gemm_nt_q", c), RAD_E_UNSUPPORTED);
    CHECK_EQ(y.at(0), 7.0f);
}

TEST(cast_refuses_a_dtype_it_cannot_convert) {
    T x(RAD_U4, {1, 4}); x.raw({0x21, 0x43});
    T y(RAD_F32, {1, 4});
    A a; a.t(x).t(y).i("M", 1).i("n", 4).s("from", "u4").s("to", "f32");
    CHECK_EQ(run("cast", a), RAD_E_DTYPE);
    T z(RAD_U4, {1, 4});
    A b; b.t(y).t(z).i("M", 1).i("n", 4).s("from", "f32").s("to", "u4");
    CHECK_EQ(run("cast", b), RAD_E_DTYPE);
}

/* mxfp4a8 reads an E8M0 scale plane as 2^(e - 127), as it reads a u8 one, and accepts `b_ref`
 * beside the e2m1 weight without changing the product: 32 codes of 1.5 at 2^-1 against ones. */
TEST(gemm_nt_q_mxfp4_reads_e8m0_scales_and_accepts_the_reference) {
    T a_(RAD_F8E4M3, {1, 32}); a_.f(std::vector<float>(32, 1.0f));
    T as(RAD_F32, {1, 1}); as.f({1});
    std::vector<uint8_t> codes(16, 0x33);                /* e2m1 0x3 is 1.5 */
    T b(RAD_FP4E2M1, {1, 32}); b.raw(codes);
    T se(RAD_E8M0, {1, 1}); se.raw({126});
    T su(RAD_U8, {1, 1}); su.raw({126});
    T y1(RAD_F32, {1, 1}), y2(RAD_F32, {1, 1}), y3(RAD_F32, {1, 1});
    A a; a.t(a_).t(as).t(b).t(se).t(y1).tnull().tnull().i("M", 1).i("N", 1).i("K", 32)
          .i("group", 32).s("dtype", "mxfp4a8");
    CHECK_OK(run("gemm_nt_q", a));
    CHECK_EQ(y1.at(0), 24.0f);
    A c; c.t(a_).t(as).t(b).t(su).t(y2).tnull().tnull().i("M", 1).i("N", 1).i("K", 32)
          .i("group", 32).s("dtype", "mxfp4a8");
    CHECK_OK(run("gemm_nt_q", c));
    CHECK_EQ(y2.at(0), 24.0f);
    A d; d.t(a_).t(as).t(b).t(se).t(y3).tnull().t(se).i("M", 1).i("N", 1).i("K", 32)
          .i("group", 32).s("dtype", "mxfp4a8");
    CHECK_OK(run("gemm_nt_q", d));
    CHECK_EQ(y3.at(0), 24.0f);
}

/* An operand of a dtype libref cannot read never reaches an op. */
TEST(an_unknown_operand_dtype_is_refused_at_launch) {
    T x(RAD_F32, {1, 4}); x.f({1, 2, 3, 4});
    T y(RAD_F32, {1, 4}); y.f({5, 5, 5, 5});
    T xb = x;
    xb.t.dtype = RAD_DT_PLUGIN_BASE + 3;
    A a; a.t(xb).tnull().t(y).i("M", 1).i("n", 4).d("eps", 1e-6).s("dtype", "f32");
    CHECK_EQ(run("rmsnorm", a), RAD_E_DTYPE);
    CHECK_EQ(y.at(0), 5.0f);
    T yb = y;
    yb.t.dtype = RAD_E8M0;                /* readable as a scale, but libref cannot store it */
    A b; b.t(x).tnull().t(yb).i("M", 1).i("n", 4).d("eps", 1e-6).s("dtype", "f32");
    CHECK_EQ(run("rmsnorm", b), RAD_E_DTYPE);
}

/* ================================================================== the collective rendezvous
 * A rank that times out withdraws its arrival, so the next collective pairs from a clean start;
 * once no rank is mid-collective, a new world size is accepted. */
TEST(collective_rendezvous_recovers_after_a_timeout) {
    auto set_timeout = (void (*)(int))plugin_sym("ref_collective_timeout_ms");
    REQUIRE(set_timeout != nullptr);
    set_timeout(50);
    {   /* Rank 0 alone: its peer never calls, so it times out. */
        T x(RAD_F32, {2}); x.f({1, 2});
        A a; a.t(x).tnull().i("world_size", 2).i("numel", 2).s("dtype", "f32").i("exact", 1);
        CHECK_EQ(run("all_reduce", a, 0, 2), RAD_E_DEVICE);
    }
    T x0(RAD_F32, {2}); x0.f({1, 2});
    T x1(RAD_F32, {2}); x1.f({10, 20});
    int st[2] = { 1, 1 };
    auto body = [&](int r, T& x) {
        A a; a.t(x).tnull().i("world_size", 2).i("numel", 2).s("dtype", "f32").i("exact", 1);
        st[r] = run("all_reduce", a, r, 2);
    };
    set_timeout(10000);       /* the pair below must not race a short timeout */
    std::thread t1(body, 1, std::ref(x1));
    body(0, x0);
    t1.join();
    CHECK_OK(st[0]); CHECK_OK(st[1]);
    CHECK_EQ(x0.at(0), 11.0f); CHECK_EQ(x1.at(1), 22.0f);

    /* Then three ranks in the same process. */
    T y[3] = { T(RAD_F32, {1}), T(RAD_F32, {1}), T(RAD_F32, {1}) };
    for (int r = 0; r < 3; ++r) y[r].f({(float)(r + 1)});
    int s3[3] = { 1, 1, 1 };
    auto body3 = [&](int r) {
        A a; a.t(y[r]).tnull().i("world_size", 3).i("numel", 1).s("dtype", "f32").i("exact", 1);
        s3[r] = run("all_reduce", a, r, 3);
    };
    std::thread u1(body3, 1), u2(body3, 2);
    body3(0);
    u1.join(); u2.join();
    for (int r = 0; r < 3; ++r) { CHECK_OK(s3[r]); CHECK_EQ(y[r].at(0), 6.0f); }
    set_timeout(0);
}

/* ================================================================== the GEMM summation order
 * Each output of libref's GEMMs is one chain (ref_common.h): acc from zero, acc + a[k] * b[k] over
 * ascending k with two roundings a step, then the bias; quantised, a sub-chain per scale group
 * folded as acc + (sub * sa) * sb. The tiled kernels (ref_gemm.h) run many chains side by side and
 * must still give exactly those bits, here at shapes that are not multiples of the tiles and with
 * strided activation rows. The expectations are the chains written out, each step rounded through
 * a volatile so no contraction can creep in. */
namespace {

uint32_t g_chain_rng = 0x2545F491u;
float chain_val() {
    g_chain_rng = g_chain_rng * 1664525u + 1013904223u;
    return (float)((int)(g_chain_rng >> 16) % 2001 - 1000) / 256.0f;
}

/* The quantised chain over one output: `a` and `b` are the widened codes, `sa(g)` and `sb(g)` the
 * group scales (1 for an absent stream). */
template <class SA, class SB>
float group_chain(const float* a, const float* b, int64_t K, int64_t g, const SA& sa, const SB& sb) {
    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < K; k0 += g) {
        float sub = 0.0f;
        for (int64_t k = k0; k < k0 + g && k < K; ++k) {
            volatile float p = a[k] * b[k];
            sub = sub + p;
        }
        volatile float t = sub * sa(k0 / g);
        volatile float u = t * sb(k0 / g);
        acc = acc + u;
    }
    return acc;
}

bool same_bits(float x, float y) { return std::memcmp(&x, &y, sizeof x) == 0; }

}  /* namespace */

TEST(gemm_nt_is_one_ascending_chain_per_output) {
    const int64_t M = 5, N = 37, K = 300, lda = 311;
    T abuf(RAD_BF16, {M, lda});
    std::vector<float> av((size_t)(M * lda)), bv((size_t)(N * K)), biasv((size_t)N);
    for (auto& v : av) v = round_dt(RAD_BF16, chain_val());
    for (auto& v : bv) v = round_dt(RAD_BF16, chain_val());
    for (auto& v : biasv) v = chain_val();
    abuf.f(av);
    T a_ = abuf;
    a_.t = abuf.t;
    a_.t.shape[1] = K;                         /* [M, K] at a row pitch of lda */
    T b_(RAD_BF16, {N, K}); b_.f(bv);
    T bias(RAD_F32, {N}); bias.f(biasv);
    T y(RAD_F32, {M, N}), yb(RAD_F32, {M, N});
    A a; a.t(a_).t(b_).t(y).i("M", M).i("N", N).i("K", K).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt", a));
    A b; b.t(a_).t(b_).t(bias).tnull().t(yb).i("M", M).i("N", N).i("K", K).s("dtype", "bf16");
    CHECK_OK(run("gemm_nt_bias", b));
    int64_t bad = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                volatile float p = av[(size_t)(m * lda + k)] * bv[(size_t)(n * K + k)];
                acc = acc + p;
            }
            bad += !same_bits(y.at(m * N + n), acc);
            volatile float wb = acc + biasv[(size_t)n];
            bad += !same_bits(yb.at(m * N + n), wb);
        }
    CHECK_EQ(bad, 0LL);
}

/* fp8a8 against a block grid that N = 37 only partly fills, K three groups of 128. */
TEST(gemm_nt_q_folds_each_group_into_one_chain) {
    const int64_t M = 7, N = 37, K = 384, g = 128;
    std::vector<float> av((size_t)(M * K)), bv((size_t)(N * K));
    for (auto& v : av) v = fp8e4m3_to_f32(f32_to_fp8e4m3(chain_val()));
    for (auto& v : bv) v = fp8e4m3_to_f32(f32_to_fp8e4m3(chain_val()));
    std::vector<float> sav((size_t)(M * 3)), sbv(3);
    for (auto& v : sav) v = std::fabs(chain_val()) / 64.0f + 0.01f;
    for (auto& v : sbv) v = round_dt(RAD_BF16, std::fabs(chain_val()) / 64.0f + 0.01f);
    T a_(RAD_F8E4M3, {M, K}); a_.f(av);
    T as(RAD_F32, {M, 3}); as.f(sav);
    T b_(RAD_F8E4M3, {N, K}); b_.f(bv);
    T bs(RAD_BF16, {1, 3}); bs.f(sbv);
    T y(RAD_F32, {M, N});
    A a; a.t(a_).t(as).t(b_).t(bs).t(y).tnull().tnull().i("M", M).i("N", N).i("K", K)
          .i("group", g).s("dtype", "fp8a8");
    CHECK_OK(run("gemm_nt_q", a));
    int64_t bad = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const float want = group_chain(&av[(size_t)(m * K)], &bv[(size_t)(n * K)], K, g,
                                           [&](int64_t gi) { return sav[(size_t)(m * 3 + gi)]; },
                                           [&](int64_t gi) { return sbv[(size_t)gi]; });
            bad += !same_bits(y.at(m * N + n), want);
        }
    CHECK_EQ(bad, 0LL);
}

/* moe_gemm_q: each sorted row is its expert's chain against its token's activation row, and a
 * dropped slot is a zero row. */
TEST(moe_gemm_q_rows_are_their_experts_chains) {
    const int64_t M = 4, top_k = 2, ne = 3, N = 20, K = 256, g = 128, T_ = M * top_k;
    std::vector<float> av((size_t)(M * K)), wv((size_t)(2 * N * K)), w1v((size_t)(N * K));
    for (auto& v : av) v = fp8e4m3_to_f32(f32_to_fp8e4m3(chain_val()));
    for (auto& v : wv) v = fp8e4m3_to_f32(f32_to_fp8e4m3(chain_val()));
    for (auto& v : w1v) v = fp8e4m3_to_f32(f32_to_fp8e4m3(chain_val()));
    std::vector<float> sav((size_t)(M * 2)), wsv((size_t)(2 * 2)), ws1v(2);
    for (auto& v : sav) v = std::fabs(chain_val()) / 64.0f + 0.01f;
    for (auto& v : wsv) v = round_dt(RAD_BF16, std::fabs(chain_val()) / 64.0f + 0.01f);
    for (auto& v : ws1v) v = round_dt(RAD_BF16, std::fabs(chain_val()) / 64.0f + 0.01f);
    T a_(RAD_F8E4M3, {M, K}); a_.f(av);
    T as(RAD_F32, {M, 2}); as.f(sav);
    T w(RAD_F8E4M3, {2, N, K}); w.f(wv);                 /* experts 0 and 2 */
    T ws(RAD_BF16, {2, 1, 2}); ws.f(wsv);
    T w1(RAD_F8E4M3, {1, N, K}); w1.f(w1v);              /* expert 1 */
    T ws1(RAD_BF16, {1, 1, 2}); ws1.f(ws1v);
    /* Experts own sorted rows [0, 3), [3, 5), [5, 7); row 7 is a dropped slot. */
    T st(RAD_I32, {T_}); st.n({0, 5, 6, 1, 2, 3, 4, -1});
    T off(RAD_I32, {ne + 1}); off.n({0, 3, 5, 7});
    T y(RAD_F32, {T_, N});
    A a; a.t(a_).t(as).t(w).t(ws).t(st).t(off).t(w1).t(ws1).t(y)
          .i("M", M).i("N", N).i("K", K).i("n_expert", ne).i("top_k", top_k).i("group", g)
          .s("a_order", "token").s("dtype", "fp8a8");
    CHECK_OK(run("moe_gemm_q", a));
    const int64_t flat[T_] = { 0, 5, 6, 1, 2, 3, 4, -1 };
    const int64_t expert[T_] = { 0, 0, 0, 1, 1, 2, 2, -1 };
    int64_t bad = 0;
    for (int64_t i = 0; i < T_; ++i)
        for (int64_t n = 0; n < N; ++n) {
            float want = 0.0f;
            if (expert[i] >= 0) {
                const int64_t tok = flat[i] / top_k, e = expert[i];
                const float* wr = e == 1 ? &w1v[(size_t)(n * K)] : &wv[(size_t)((e / 2) * N * K + n * K)];
                const float* sr = e == 1 ? &ws1v[0] : &wsv[(size_t)((e / 2) * 2)];
                want = group_chain(&av[(size_t)(tok * K)], wr, K, g,
                                   [&](int64_t gi) { return sav[(size_t)(tok * 2 + gi)]; },
                                   [&](int64_t gi) { return sr[gi]; });
            }
            bad += !same_bits(y.at(i * N + n), want);
        }
    CHECK_EQ(bad, 0LL);
}

/* ================================================================== every op, generically
 * A small geometry per op, the operands sized and filled from libref's own descriptions
 * (RadKernelInfo::opd_shape) as rad-kbench fills them. Two contracts are checked across the whole
 * vocabulary: each op runs on its own description (so a fixture recording can draw it), and each
 * refuses a call over zero rows with RAD_E_SHAPE without writing (docs/OPS.md, Conventions). */
namespace {

struct Geo {
    const char* op;
    const char* key;               /* the ranged parameter */
    int64_t     r;                 /* its value in the running case */
    void (*set)(A&, int64_t);      /* every parameter, the ranged one at the given value */
};

#define G(op, key, r, body) { op, key, r, [](A& a, int64_t R) { body; } }
std::vector<Geo> geometries() {
    return {
        G("position_scale", "M", 3, a.i("M", R).i("n", 8).i("original", 16384)
                                       .d("beta", 0.1).s("dtype", "f32")),
        G("embed_attn", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("gqa", 2)
                                    .i("causal", 0).s("dtype", "f32")),
        G("ministral_rope", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2)
                                       .i("n_head_kv", 1).d("theta", 1e6).s("dtype", "f32")),
        G("pixtral_rope", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2)
                                     .d("theta", 10000.0).s("dtype", "f32")),
        G("pixtral_merge", "M", 4, a.i("M", R).i("n", 8).i("merge", 2).s("dtype", "f32")),
        G("rmsnorm", "M", 3, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "f32")),
        G("rmsnorm_rows", "M", 3, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "f32")),
        G("rmsnorm_p", "M", 3, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "f32")),
        G("rmsnorm_add", "M", 3, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "f32")),
        G("hc_enter", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).s("dtype", "f32")),
        G("hc_read", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).i("lowrank", 2).d("eps", 1e-6)
                             .s("dtype", "f32").i("group", 4).i("rotate", 1)),
        G("hc_write", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).s("dtype", "f32")),
        G("hc_project", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).d("scale", 1.0).s("dtype", "f32")),
        G("mtp_enter", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).d("eps", 1e-6).s("dtype", "f32")),
        G("grid_embed", "M", 3, a.i("M", R).i("n", 8).i("side", 4).s("dtype", "f32")),
        G("grid_embed_aa", "M", 3, a.i("M", R).i("n", 8).i("side", 4).s("dtype", "f32")),
        G("grid_embed_cpe", "M", 3, a.i("M", R).i("n", 8).i("side", 4).s("dtype", "f32")),
        G("lili_cand", "M", 3, a.i("M", R).i("k", 4).i("world", 1).i("steps", 3).i("n", 8)
                                .s("dtype", "f32")),
        G("lili_attn", "M", 8, a.i("M", R).i("steps", 2).i("k", 2).i("n_head", 2).i("head_dim", 4)
                                .s("dtype", "f32")),
        G("lili_factor", "M", 8, a.i("M", R).i("L", 4).i("n", 8).s("dtype", "f32")),
        G("lili_walk", "M", 2, a.i("M", R).i("steps", 3).i("k", 2).i("n", 8).s("dtype", "f32")
                                .d("eps", 1e-4).d("scale", 8.0)),
        G("layernorm", "M", 3, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "f32")),
        G("add", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("mul", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("mul_weight", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("act_mul", "M", 3, a.i("M", R).i("n", 8).s("act", "silu").s("dtype", "f32")),
        G("seg_mean", "M", 6, a.i("M", R).i("n", 8).s("dtype", "f32").i("n_seq", 2)),
        G("seg_softmax_pool", "M", 6, a.i("M", R).i("n", 8).s("dtype", "f32").i("n_seq", 2)),
        G("seg_mean_bf16_sum", "M", 6, a.i("M", R).i("n", 8).s("dtype", "bf16").i("n_seq", 2)),
        G("pool_rows", "M", 6, a.i("M", R).i("n", 8).i("k", 2).s("dtype", "f32")),
        G("pool_cols", "M", 6, a.i("M", R).i("n", 8).i("k", 2).s("dtype", "f32")),
        G("point_fourier", "M", 3, a.i("M", R).i("n", 3).i("freqs", 2).s("dtype", "f32")),
        G("relu", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("clamp", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("conv2d", "M", 8, a.i("M", R).i("width", 4).i("cin", 2).i("cout", 3).i("kh", 3).i("kw", 3)
                            .i("stride", 2).i("pad", 1).s("dtype", "f32")),
        G("conv2d_unfold", "M", 8, a.i("M", R).i("width", 4).i("cin", 2).i("kh", 3).i("kw", 3)
                                   .i("stride", 2).i("pad", 1).s("dtype", "f32")),
        G("conv1d_dw", "M", 8, a.i("M", R).i("n", 8).i("k", 3).s("dtype", "f32")),
        G("rel_sinusoid", "M", 5, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("seg_sinusoid", "M", 5, a.i("M", R).i("n", 8).i("period", 3).s("dtype", "f32")),
        G("attn_local_rel", "M", 8, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("left", 3)
                                    .i("right", 0).d("softcap", 50.0).s("dtype", "f32")),
        G("silu_mul", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("gelu_mul", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("gelu", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("silu", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("sigmoid", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("fill", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32").d("value", 1.5)),
        G("logit_softcap", "M", 3, a.i("M", R).i("n", 8).d("cap", 30.0).s("dtype", "f32")),
        G("scale", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("softmax", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("gather_rows", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("scatter_rows", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("ple_gate", "M", 2, a.i("M", R).i("n", 8).i("hc", 2).d("eps", 1e-6).s("dtype", "f32")),
        G("ple_conv", "M", 6, a.i("M", R).i("n", 8).i("width", 3).i("dilation", 2).s("dtype", "f32")),
        G("ple_conv_bias", "M", 6, a.i("M", R).i("n", 8).i("width", 4).i("dilation", 1)
                                    .s("dtype", "f32")),
        G("causal_conv1d", "M", 6, a.i("M", R).i("n", 8).i("width", 4).s("dtype", "f32")),
        G("ngram_ids", "M", 4, a.i("M", R).i("heads", 2).i("ngram", 3).i("eos", 1)),
        G("cast", "M", 3, a.i("M", R).i("n", 8).s("from", "f32").s("to", "bf16")),
        G("quant_act_i8", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("quant_act_i8g", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("quant_act_fp8", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("quant_cols_fp8", "M", 3, a.i("M", R).i("n", 32).s("dtype", "f32")),
        G("had_quant_act_i8g", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("had_quant_act_i8", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("rmsnorm_had_quant_i8", "M", 3, a.i("M", R).i("n", 32).d("eps", 1e-6).i("group", 16)
                                          .s("dtype", "f32")),
        G("gated_had_quant_i8", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("gram_accum", "M", 3, a.i("M", R).i("n", 16).i("group", 8).s("dtype", "fp8a8")),
        G("dequant", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "f32")),
        G("dequant_w4", "N", 16, a.i("N", R).i("K", 64).i("group", 64).i("twos", 1)
                                   .s("dtype", "w4a8")),
        G("dequant_w4", "N", 16, a.i("N", R).i("K", 128).i("group", 128).i("twos", 0)
                                   .s("dtype", "w4a8")),
        G("rmsnorm_quant_fp8", "M", 3, a.i("M", R).i("n", 32).d("eps", 1e-6).i("group", 16)
                                       .s("dtype", "bf16")),
        G("gated_quant_fp8", "M", 3, a.i("M", R).i("n", 32).i("group", 16).s("dtype", "bf16")
                                     .s("act", "silu")),
        G("gemm_nt_q_gated", "M", 3, a.i("M", R).i("N", 32).i("K", 128).i("group", 128)
                                     .s("dtype", "fp8a8").s("act", "silu")),
        G("dflash_select", "M", 2, a.i("M", R).i("steps", 3).i("top_k", 4).i("rank", 8)
                                   .i("n_vocab", 30).s("dtype", "bf16")),
        G("dflash_markov_greedy", "M", 2, a.i("M", R).i("steps", 3).i("rank", 32)
                                             .i("n_vocab", 130).i("anchor_stride", 1)
                                             .s("dtype", "bf16")),
        G("dflash_markov_greedy_f32", "M", 2, a.i("M", R).i("steps", 3).i("rank", 32)
                                                 .i("n_vocab", 130).i("anchor_stride", 1)
                                                 .s("dtype", "bf16")),
        G("dflash_tree", "M", 2, a.i("M", R).i("steps", 3).i("top_k", 4).i("rank", 8)
                                 .i("n_vocab", 30).i("nodes", 3).s("dtype", "bf16")),
        G("mtp_tree_grow", "M", 2, a.i("M", R).i("round", 1).i("rounds", 3).i("nodes", 6)
                                   .i("width", 3)),
        G("gemm_nt", "M", 3, a.i("M", R).i("N", 20).i("K", 40).s("dtype", "f32")),
        G("gemm_nt_bias", "M", 3, a.i("M", R).i("N", 20).i("K", 40).s("dtype", "f32")),
        G("gemm_nt_q", "M", 3, a.i("M", R).i("N", 20).i("K", 256).i("group", 128).s("dtype", "fp8a8")),
        G("gemm_nt_q_bias", "M", 3, a.i("M", R).i("N", 20).i("K", 256).i("group", 128)
                                    .s("dtype", "fp8a8")),
        G("lora_shrink", "M", 3, a.i("M", R).i("K", 40).i("R", 8).i("S", 2).s("dtype", "f32")),
        G("lora_expand", "M", 3, a.i("M", R).i("N", 20).i("R", 8).i("S", 2).s("dtype", "f32")),
        G("lora_expand_mm", "M", 3, a.i("M", R).i("N", 20).i("R", 8).i("S", 2).s("dtype", "f32")),
        G("attn_paged", "q_len", 3, a.i("q_len", R).i("head_dim", 8).i("gqa", 2).i("block_size", 4)
                                    .i("causal", 1).i("window", 0).s("q_dtype", "f32")
                                    .s("kv_dtype", "f32").i("n_head", 4).i("ctx", 9)),
        G("attn_paged_anchor_window", "q_len", 3,
          a.i("q_len", R).i("head_dim", 8).i("gqa", 2).i("block_size", 4).i("causal", 1)
           .i("window", 4).s("q_dtype", "f32").s("kv_dtype", "f32").i("n_head", 4).i("ctx", 9)),
        G("attn_paged_tree", "q_len", 3, a.i("q_len", R).i("head_dim", 8).i("gqa", 2)
                                         .i("block_size", 4).i("causal", 1).i("window", 0)
                                         .s("q_dtype", "f32").s("kv_dtype", "f32").i("n_head", 4)
                                         .i("ctx", 9)),
        G("attn_paged_tree_sel", "q_len", 3, a.i("q_len", R).i("head_dim", 8).i("gqa", 2)
                                             .i("block_size", 4).i("causal", 1).i("window", 0)
                                             .s("q_dtype", "f32").s("kv_dtype", "f32")
                                             .i("n_head", 4).i("ctx", 9).i("sel_pitch", 3)),
        G("attn_paged_sink", "q_len", 3, a.i("q_len", R).i("head_dim", 8).i("gqa", 2)
                                         .i("block_size", 4).i("causal", 0).i("window", 0)
                                         .s("q_dtype", "f32").s("kv_dtype", "f32").i("n_head", 4)
                                         .i("ctx", 9)),
        G("attn_dense", "M", 6, a.i("M", R).i("head_dim", 8).i("gqa", 2).i("causal", 1)
                                .s("dtype", "f32").i("n_head", 4)),
        /* One-row segments, so M is also the segment count (n_seq, the description's hint for
         * cu_seqlens). The real grids are op_sam_attn_*'s. */
        G("sam_attn", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("grid_h", 1)
                              .i("grid_w", 1).i("window", 1).i("n_seq", R)
                              .d("scale", 0.3535533905932738).s("dtype", "bf16")),
        G("kv_store", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head_kv", 2).i("block_size", 4)
                              .s("kv_dtype", "f32")),
        G("kv_store_streamed", "M", 3,
          a.i("M", R).i("head_dim", 8).i("n_head_kv", 2).i("block_size", 8)
           .s("kv_dtype", "i8")),
        G("rope", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("n_head_kv", 1)
                          .d("theta", 1e4).d("scale", 1.0).s("mode", "neox")),
        G("rope_rows", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("n_head_kv", 1)
                          .d("theta", 1e4).d("scale", 1.0).s("mode", "neox")),
        G("qk_norm_rope", "M", 3, a.i("M", R).i("head_dim", 8).i("n_head", 2).i("n_head_kv", 1)
                                  .d("theta", 1e4).d("eps", 1e-6)),
        G("rope_table", "M", 3, a.i("M", R).i("rot", 8).d("theta", 1e4).d("scale", 1.0)
                                .s("dtype", "f32")),
        G("qsa_block_key", "M", 2, a.i("M", R).i("n", 32).i("ratio", 4).i("rotary_dim", 8)
                                   .d("eps", 1e-6).d("theta", 1e4).s("dtype", "bf16")),
        G("qsa_score", "M", 4, a.i("M", R).i("n", 32).i("heads", 2).i("blocks", 6).i("seqs", 2)
                               .s("dtype", "bf16")),
        G("qsa_select", "M", 4, a.i("M", R).i("blocks", 6).i("topk", 3).i("ratio", 4).i("seqs", 2)),
        G("kv_resolve", "M", 3, a.i("M", R).i("blocks", 8).i("slots", 6).i("sel_pitch", 4)
                                 .i("block_bytes", 8)),
        G("qsa_tail_store", "M", 4, a.i("M", R).i("n", 32).i("ratio", 4).i("ring", 6).i("seqs", 2)
                                    .s("dtype", "bf16")),
        G("qsa_tail_compact", "M", 4, a.i("M", R).i("n", 32).i("ring", 6).i("seqs", 2)
                                      .s("dtype", "bf16")),
        G("qsa_work", "M", 6, a.i("M", R).i("n", 32).i("ratio", 4).i("ring", 6).i("seqs", 2)
                              .i("work", 8).s("dtype", "bf16")),
        G("dflash_conv", "T", 6, a.i("T", R).i("hidden_size", 32).i("taps", 2).i("group_size", 16)
                                 .i("NG", 2).i("block_size", 4).s("dtype", "bf16")),
        G("scale_rows", "M", 3, a.i("M", R).i("n", 8).s("act", "sigmoid").s("dtype", "f32")),
        G("router_topk", "M", 3, a.i("M", R).i("n_expert", 6).i("top_k", 2).i("norm", 1)
                                 .s("dtype", "f32")),
        G("router_topk_scatter", "M", 3, a.i("M", R).i("n_expert", 6).i("n_local", 4)
                                         .i("top_k", 2).i("norm", 1).s("dtype", "f32")),
        G("router_gemm_topk_scatter", "M", 3, a.i("M", R).i("K", 32).i("n_expert", 4)
                                              .i("n_local", 4).i("top_k", 2).i("norm", 0)
                                              .s("dtype", "f32")),
        G("moe_scatter", "M", 3, a.i("M", R).i("n_expert", 6).i("top_k", 2)),
        G("moe_gemm", "M", 3, a.i("M", R).i("N", 16).i("K", 32).i("n_expert", 4).i("top_k", 2)
                              .s("a_order", "token").s("dtype", "f32")),
        G("moe_gemm_q", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 4).i("top_k", 2)
                                .i("group", 128).s("a_order", "token").s("dtype", "fp8a8")),
        G("moe_gemm_q", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 4).i("top_k", 2)
                                .i("group", 128).s("a_order", "token").s("dtype", "w4a8")),
        G("moe_gemm_q", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 4).i("top_k", 2)
                                .i("group", 128).s("a_order", "sorted").s("dtype", "w4a8")),
        G("moe_gemm_q", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 4).i("top_k", 2)
                                .i("group", 128).s("a_order", "token").s("dtype", "w4a4")),
        G("moe_gemm_q", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 4).i("top_k", 2)
                                .i("group", 128).s("a_order", "sorted").s("dtype", "w4a4")),
        G("moe_gemm_q_mod4", "M", 3, a.i("M", R).i("N", 16).i("K", 128).i("n_expert", 5)
                                     .i("top_k", 2).i("group", 128).s("a_order", "sorted")
                                     .s("dtype", "fp8a8")),
        G("moe_gather", "M", 3, a.i("M", R).i("n", 8).i("top_k", 2).s("dtype", "f32")),
        G("row_topk", "M", 3, a.i("M", R).i("N", 20).i("R", 4).s("dtype", "f32")),
        G("row_topk_merge", "M", 3, a.i("M", R).i("R", 4).i("world_size", 2).s("dtype", "f32")),
        G("pair_argmax", "M", 3, a.i("M", R).i("n", 4).s("dtype", "f32")),
        G("logit_rerank", "M", 3, a.i("M", R).i("R", 4).i("n_vocab", 20).i("n_embd", 8)
                                  .s("dtype", "f32")),
        G("gdn_conv_prep", "M", 6, a.i("M", R).i("head_k", 8).i("head_v", 8).i("chunk", 4)
                                   .i("conv_width", 4).i("n_head_k", 1).i("n_head_v", 2)
                                   .i("n_seq", 2)),
        /* One sequence: a fixture's cu_seqlens spreads the ranged value (here q_len, not the
         * token count) over the sequences. */
        G("gdn_conv_update", "q_len", 2, a.i("q_len", R).i("head_k", 8).i("head_v", 8)
                                         .i("conv_width", 4).i("n_head_k", 1).i("n_head_v", 2)),
        G("gdn_kkt_solve", "M", 6, a.i("M", R).i("head_k", 8).i("head_v", 8).i("chunk", 4)
                                   .i("n_head_k", 1).i("n_head_v", 2).i("n_seq", 2)),
        G("gdn_chunk_scan", "M", 6, a.i("M", R).i("head_k", 8).i("head_v", 8).i("chunk", 4)
                                    .i("n_head_k", 1).i("n_head_v", 2).i("n_seq", 2)),
        G("gdn_recurrent_update", "q_len", 2, a.i("q_len", R).i("head_k", 8).i("head_v", 8)
                                              .i("n_head_k", 1).i("n_head_v", 2)),
        G("gdn_conv_recurrent_update", "q_len", 2, a.i("q_len", R).i("head_k", 8).i("head_v", 8)
                                                   .i("conv_width", 4).i("n_head_k", 1)
                                                   .i("n_head_v", 2)),
        G("gdn_gated_rmsnorm", "M", 3, a.i("M", R).i("channels", 8).d("eps", 1e-6).s("act", "silu")),
        /* No row count: `off`, the window's write offset, sets the entries (off + 4). */
        G("kv_window_compact", "off", 2, a.i("off", R).s("dtype", "f32")),
        G("all_reduce", "numel", 16, a.i("world_size", 1).i("numel", R).s("dtype", "f32")
                                     .i("exact", 1)),
        G("all_gather", "numel", 16, a.i("world_size", 1).i("numel", R).s("dtype", "f32")),
        G("embed_lookup", "M", 3, a.i("M", R).i("n_embd", 8).i("n_vocab", 20).s("dtype", "f32")),
        G("embed_lookup_q", "M", 3, a.i("M", R).i("n_embd", 8).i("n_vocab", 20)
                                    .s("dtype", "fp8_e4m3")),
        G("logits_gemm", "M", 3, a.i("M", R).i("n_vocab", 20).i("n_embd", 8).s("dtype", "f32")),
        G("sample_penalties", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_dry", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_ngram_ban", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_temp", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_topk", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_topp", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_minp", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_typical", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_xtc", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_mask", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_pick", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_merge_topk", "M", 2, a.i("M", R).i("n_cand", 16)),
        G("sample_argmax", "M", 2, a.i("M", R).i("n_vocab", 64)),
        G("sample_logprobs", "M", 2, a.i("M", R).i("n_vocab", 64).i("k", 4)),
        G("ln_mod", "M", 4, a.i("M", R).i("n", 8).d("eps", 1e-6).s("dtype", "bf16")),
        G("gate_add", "M", 4, a.i("M", R).i("n", 8).s("dtype", "bf16")),
        G("timestep_embed", "M", 3, a.i("M", R).i("n", 8).d("scale", 1000.0)
                                    .d("max_period", 10000.0).d("clamp_min", 0.3).s("dtype", "f32")),
        /* The curve form: the same op over a precomputed table instead of the sinusoid. */
        G("timestep_embed", "M", 3, a.i("M", R).i("n", 8).i("curve", 5).d("clamp_min", 0.3)
                                    .s("dtype", "f32")),
        G("gather_rows2", "M", 4, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("lane_gather", "M", 4, a.i("M", R).i("n", 8).i("stride", 4).s("dtype", "bf16")),
        G("lane_store", "M", 4, a.i("M", R).i("n", 8).i("stride", 8).s("dtype", "bf16")),
        G("randn_lanes", "M", 4, a.i("M", R).i("n", 8).i("stride", 4).s("dtype", "bf16")),
        G("flow_euler", "M", 4, a.i("M", R).i("n", 8).i("stride", 4).s("dtype", "bf16")),
        G("kd_input", "M", 4, a.i("M", R).i("n", 4).i("stride", 4).s("dtype", "bf16")),
        G("kd_step", "M", 4, a.i("M", R).i("n", 4).i("stride", 4).s("pred", "eps")
                             .s("dtype", "bf16")),
        G("chan_affine", "M", 2, a.i("M", R).i("n", 3).s("mul", "1 2 3").s("add", "0 1 2")
                                 .s("mode", "fma").s("dtype", "bf16")),
        G("conv2d_nhwc", "M", 16, a.i("M", R).i("cin", 2).i("cout", 3).i("k", 3).i("stride", 1)
                                  .s("pad", "same").i("up", 1).s("dtype", "bf16")),
        G("im2col_nhwc", "M", 16, a.i("M", R).i("cin", 2).i("k", 3).i("stride", 1)
                                  .s("pad", "same").i("up", 1).s("dtype", "bf16")),
        G("l2norm_chan", "M", 3, a.i("M", R).i("n", 8).s("act", "silu").s("dtype", "f32")),
        G("d2s_add", "M", 16, a.i("M", R).i("cin", 4).i("cout", 2).i("ft", 1).s("dtype", "bf16")),
        G("s2d_avg_add", "M", 16, a.i("M", R).i("cin", 2).i("cout", 4).i("ft", 1).s("dtype", "bf16")),
        G("to_rgba8", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("gelu_tanh", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("gelu_quick", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("scatter_add_rows", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("attn_prefix_self", "q_len", 3, a.i("q_len", R).i("head_dim", 8).i("gqa", 2)
                                          .i("block_size", 4).i("causal", 1).s("q_dtype", "bf16")
                                          .s("kv_dtype", "bf16").i("n_head", 4).i("ctx", 5)),
        G("mhc_pre", "M", 3, a.i("M", R).i("n", 8).i("hc", 4).i("iterations", 20).d("eps", 1e-5)
                             .d("hc_eps", 1e-6).s("dtype", "f32")),
        G("mhc_pre_partial", "M", 3, a.i("M", R).i("n", 8).i("hc", 4).i("chunk", 8).i("first", 1)
                                     .i("chunks", 2).s("dtype", "f32")),
        G("mhc_pre_finish", "M", 3, a.i("M", R).i("n", 8).i("hc", 4).i("iterations", 20)
                                    .d("eps", 1e-5).d("hc_eps", 1e-6).i("chunk", 8)
                                    .s("dtype", "f32")),
        G("mhc_post", "M", 3, a.i("M", R).i("n", 8).i("hc", 4).s("dtype", "f32")),
        G("mhc_contract", "M", 3, a.i("M", R).i("n", 8).i("hc", 4).s("dtype", "f32")),
        G("kda_update", "M", 3, a.i("M", R).i("heads", 2).i("head_k", 8).i("head_v", 8)
                                .d("lower_bound", -5.0).d("scale", 0.35).d("l2_eps", 1e-6)
                                .s("mode", "prefill").s("dtype", "f32")),
        G("ssd_update", "M", 3, a.i("M", R).i("heads", 4).i("head_dim", 4).i("d_state", 8)
                                .i("groups", 2).s("mode", "prefill").s("dtype", "f32")),
        G("mla_attn", "M", 3, a.i("M", R).i("latent_dim", 8).i("heads", 2).i("blocks", 2)
                              .i("block_size", 4).d("scale", 0.125).s("dtype", "f32")),
        G("mla_project", "M", 3, a.i("M", R).i("heads", 2).i("head_dim", 4).i("latent_dim", 6)
                                 .s("mode", "absorb").s("dtype", "bf16")),
        G("mla_attn_paged", "M", 3, a.i("M", R).i("heads", 2).i("latent_dim", 8).i("rope_dim", 4)
                                    .i("block_size", 4).d("scale", 0.25).i("ctx", 6)
                                    .s("dtype", "f32")),
        G("dsa_score", "M", 4, a.i("M", R).i("n", 8).i("heads", 2).i("blocks", 3).i("seqs", 2)
                               .s("dtype", "fp8_e4m3")),
        G("dsa_pool", "M", 3, a.i("M", R).i("n", 8).i("ratio", 4).s("dtype", "bf16")),
        G("dsa_query", "M", 3, a.i("M", R).i("n", 8).s("dtype", "bf16")),
        G("gemm_nt_f32", "M", 3, a.i("M", R).i("N", 20).i("K", 40).s("dtype", "bf16")),
        G("swiglu_clamped", "M", 3, a.i("M", R).i("n", 8).d("limit", 0.5).s("dtype", "f32")),
        G("swiglu_step", "M", 3, a.i("M", R).i("n", 8).d("limit", 0.5).s("dtype", "f32")),
        G("router_sigmoid_group_topk", "M", 3, a.i("M", R).i("n_expert", 8).i("top_k", 2)
                                               .i("n_group", 4).i("topk_group", 2).i("norm", 1)
                                               .d("route_scale", 2.5).s("dtype", "f32")),
        G("router_sigmoid_topk", "M", 3, a.i("M", R).i("n_expert", 6).i("top_k", 2).i("norm", 1)
                                         .d("route_scale", 2.5).s("dtype", "f32")),
        G("conv1d", "M", 8, a.i("M", R).i("cin", 2).i("cout", 3).i("k", 3).i("dil", 1)
                            .s("dtype", "f32")),
        G("conv_t1d", "M", 8, a.i("M", R).i("cin", 2).i("cout", 3).i("stride", 2).s("dtype", "f32")),
        G("snake", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("cfg_topk_sample", "M", 4, a.i("M", R).i("n_vocab", 64).i("k", 8).i("rep", 2)
                                     .d("cfg", 1.5).i("stream", 0).s("dtype", "f32")),
        G("rvq_embed", "M", 3, a.i("M", R).i("n_embd", 8).i("n_codes", 3).i("vocab", 16)
                               .i("off0", 4).d("scale", 1.0).s("dtype", "bf16")),
        G("frame_mix_store", "M", 2, a.i("M", R).i("n_embd", 8).i("L", 3).i("stride", 4)
                                     .i("rep", 2).s("dtype", "bf16")),
        G("end_status", "M", 2, a.i("M", R).i("end", 5).i("rep", 2)),
        G("ras_pick", "M", 2, a.i("M", R).i("win", 4).i("lo", 1).i("hi", 9).i("rep", 1)),
        G("vq_encode", "M", 3, a.i("M", R).i("n", 16).i("dim", 4).i("bins", 24)
                                .s("dtype", "f32")),
        G("resample_nearest", "M", 6, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("fourier_embed", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("win_input", "M", 8, a.i("M", R).i("n_lat", 4).i("n_zero", 2).i("n_cond", 3)
                               .i("stride", 4).s("dtype", "bf16")),
        G("win_prep", "M", 8, a.i("M", R).i("n", 4).i("stride", 4).i("ov", 2).s("dtype", "bf16")),
        G("win_finish", "M", 8, a.i("M", R).i("n", 4).i("n_cond", 3).i("stride", 4).i("ov", 2)
                                .s("dtype", "bf16")),
        G("chan_split", "M", 3, a.i("M", R).i("n", 4).i("parts", 2).s("dtype", "f32")),
        G("interleave2", "M", 4, a.i("M", R).s("dtype", "f32")),
        G("iota", "M", 4, a.i("M", R).i("start", 0).i("step", 1)),
        G("cu_repeat", "M", 3, a.i("M", R).i("mult", 2)),
        G("int_affine", "M", 3, a.i("M", R).s("src", "0 -1 2").s("mul", "2 0 4").s("add", "-1 5 -3")),
        G("sub_i32", "M", 3, a.i("M", R)),
        G("frame_store", "M", 2, a.i("M", R).i("n_codes", 4).i("stride", 6).i("rep", 2)
                                 .i("skip", 0)),
        G("rms_mod", "M", 4, a.i("M", R).i("n", 8).d("eps", 1e-6).i("gated", 1).s("dtype", "bf16")),
        G("flow_x0_step", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("flow_res_multistep", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("patch_unfold", "M", 32, a.i("M", R).i("C", 3).i("pt", 1).i("ph", 2).i("pw", 2)
                                   .s("mul", "1 2 3").s("add", "0 1 2").s("dtype", "f32")),
        G("vtile_gather", "M", 60, a.i("M", R).i("n", 8).i("tile", 2).i("overlap", 1)
                                   .i("frames", 3).i("period", 15).i("n_reg", 2)
                                   .s("dtype", "f32")),
        G("vtile_stitch", "M", 108, a.i("M", R).i("tile", 2).i("overlap", 1).i("scale", 2)
                                    .i("pt", 2).i("skip", 1).i("keep", 2).i("blend", 1)
                                    .i("period", 13).s("mean", "0.485 0.456 0.406")
                                    .s("std", "0.229 0.224 0.225")),
        G("grid_pos3", "M", 40, a.i("M", R).i("T", 2).i("H", 3).i("W", 4).i("period", 27)
                                .d("scale", 1.0)),
        G("snake_aa", "M", 8, a.i("M", R).i("n", 8).i("ratio", 2).i("k", 12).s("dtype", "f32")),
        /* A 3 x 3 x 3 grid in boxes of 2 and two dense segments of its 27 rows; attn_vsa at
         * k = 1, the one selection its description draws. */
        G("vsa_select", "M", 81, a.i("M", R).i("head_dim", 8).i("tile_t", 2).i("tile_h", 2)
                                 .i("tile_w", 2).i("patch", 1).d("sparsity", 0.8)
                                 .s("dtype", "f32")),
        G("attn_vsa", "M", 81, a.i("M", R).i("head_dim", 8).i("tile_t", 2).i("tile_h", 2)
                               .i("tile_w", 2).i("patch", 1).d("sparsity", 0.9)
                               .s("dtype", "f32")),
        G("conv3d_ndhwc", "M", 27, a.i("M", R).i("cin", 3).i("cout", 5).i("k", 3).i("kt", 3)
                                   .s("dtype", "f32")),
        G("group_norm", "M", 4, a.i("M", R).i("n", 8).i("groups", 2).d("eps", 1e-5)
                                .s("act", "silu").s("dtype", "f32")),
        G("tconv_dw", "M", 8, a.i("M", R).i("n", 8).i("k", 5).s("dtype", "f32")),
        G("resize_bilinear", "M", 8, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("patch_fold", "M", 8, a.i("M", R).i("C", 3).i("pt", 1).i("ph", 2).i("pw", 2)
                                .s("mul", "1 2 3").s("add", "0 1 2").s("dtype", "f32")),
        G("flow_renoise", "M", 3, a.i("M", R).i("n", 8).i("stream", 1).s("dtype", "f32")),
        G("cond_aug", "M", 3, a.i("M", R).i("n", 8).i("stream", 2).d("aug", 0.999)
                             .s("dtype", "f32")),
        /* The drawn frame is 20 latent units a side (ref_shapes.cpp), so a tile of 16 with
         * overlap 4 splits it 2 x 2: the plan has 4 rows, the crop (ratio 16) 4 tiles of 256
         * pixels -- 512 x 512 px of plane into 4 x 256 x 256 = 262144 rows -- and the join
         * blends 4 x 16 x 16 = 1024 moment rows into the 400-row canvas. */
        G("vtile_plan", "M", 4, a.i("M", R).i("tile", 16).i("overlap", 4).i("ratio", 16)),
        G("vtile_crop", "M", 262144, a.i("M", R).i("tile", 16).i("overlap", 4).i("ratio", 16)
                                    .s("dtype", "f32")),
        G("vtile_join", "M", 400, a.i("M", R).i("tile", 16).i("overlap", 4).i("n", 48)
                                 .s("dtype", "bf16")),
        G("attn_cross", "M", 6, a.i("M", R).i("head_dim", 8).i("gqa", 2).i("causal", 0)
                                .s("dtype", "f32").i("n_head", 2)),
        G("table_add", "M", 3, a.i("M", R).i("n", 8).i("L", 3).s("dtype", "f32")),
        G("mix_rows", "M", 3, a.i("M", R).i("n", 8).i("k", 4).s("dtype", "f32")),
        G("chan_stats", "M", 3, a.i("M", R).i("n", 8).s("mode", "norm").s("dtype", "f32")),
        G("rope_split", "M", 3, a.i("M", R).i("n_head", 2).i("head_dim", 8).i("dim", 16)
                                .i("n_dims", 3).d("theta", 10000.0).s("max", "20 2048 2048")
                                .s("dtype", "f32")),
        G("head_gate", "M", 3, a.i("M", R).i("n_head", 2).i("head_dim", 8).s("dtype", "f32")),
        G("ltx_guide", "M", 3, a.i("M", R).i("n", 8).s("dtype", "f32")),
        G("pixel_norm", "M", 4, a.i("M", R).i("n", 8).d("eps", 1e-6).s("form", "rsqrt")
                                .i("ileave", 2).d("mult", 2.5).s("dtype", "f32")),
        /* M 7 is a cube of side 1 and 14 one of side 2, so the zero-row call sees the volume
         * change. */
        G("d2s3d", "M", 7, a.i("M", R).i("C", 3).i("st", 2).i("sh", 2).i("sw", 2).s("dtype", "f32")),
        G("s2d3d", "M", 7, a.i("M", R).i("C", 6).i("st", 1).i("sh", 1).i("sw", 1)
                           .i("group", 2).s("mode", "mean").s("dtype", "f32")),
        G("pix_patch", "M", 4, a.i("M", R).i("p", 2).i("C", 3).s("dtype", "f32")),
        G("stft_logmel", "M", 3, a.i("M", R).i("hop", 4).i("nfft", 8).i("n_mel", 3).i("ch", 2)
                                 .s("dtype", "f32")),
        G("bwe_up", "M", 6, a.i("M", R).i("ratio", 3).i("ch", 2).s("dtype", "f32")),
        G("tile_grid", "M", 4, a.i("M", R).i("tile", 4).i("stride", 3)),
        G("vol_index", "M", 7, a.i("M", R)),
        G("attn_na3d", "M", 7, a.i("M", R).i("head_dim", 8).i("kt", 3).i("kh", 3).i("kw", 3)
                               .s("dtype", "f32")),
        G("tile_crop", "M", 7, a.i("M", R).i("C", 3).s("dtype", "f32")),
        G("tile_stitch", "M", 100, a.i("M", R).i("tile", 4).i("stride", 3).i("scale", 2).i("p", 2)
                                   .i("C", 3).s("dtype", "f32")),
    };
}
#undef G

/* Ops whose description declines every geometry (a rotation, a candidate merge or a signed index
 * cannot be drawn; ref_shapes.cpp says why). The sweeps skip them; their hand cases above cover
 * them. */
bool undrawable(const char* op) {
    // Complete raster grids need the hand-built cases in nemotron35_test, not random indices.
    return !std::strcmp(op, "sample_merge_topk") || !std::strcmp(op, "gather_rows2") ||
           !std::strcmp(op, "pixtral_merge") ||
           /* sample_logprobs rows (sorted pairs, positive sums): avx_check_hand.cpp builds them */
           !std::strcmp(op, "lili_cand");
}

uint32_t lcg(uint32_t* s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }

/* Operands built from a description, filled the way rad-kbench fills them: each fill law's domain,
 * indices in their range, cu_seqlens splitting `rows` evenly. `zero` marks the dimensions to
 * empty, for the zero-row call. */
struct Built {
    std::vector<std::vector<uint8_t>> mem;
    std::vector<RadTensor> t;
};

void fill_operand(const RadOpdDesc& d, int64_t rows, int64_t prev_lead, uint32_t seed,
                  std::vector<uint8_t>& m, RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    uint32_t s = seed;
    if (d.fill == RAD_FILL_INDEX) {
        if (d.flags & RAD_OPD_F_IDX_CU) {
            const int64_t S = n > 1 ? n - 1 : 1;
            for (int64_t i = 0; i < n; ++i) st_int(t.dtype, t.data, i, (i * rows) / S);
        } else if (d.idx_const >= 0) {
            for (int64_t i = 0; i < n; ++i) st_int(t.dtype, t.data, i, d.idx_const);
        } else {
            const int64_t range = d.idx_max > 0 ? d.idx_max : (prev_lead > 0 ? prev_lead : 1);
            std::vector<int64_t> pool;
            if (d.flags & RAD_OPD_F_IDX_UNIQUE)
                for (int64_t i = 0; i < range; ++i) pool.push_back(i);
            for (int64_t i = 0; i < n; ++i) {
                if (d.flags & RAD_OPD_F_IDX_UNIQUE) {
                    if (i >= range) { st_int(t.dtype, t.data, i, -1); continue; }
                    const int64_t j = i + (int64_t)(lcg(&s) % (uint32_t)(range - i));
                    std::swap(pool[(size_t)i], pool[(size_t)j]);
                    st_int(t.dtype, t.data, i, pool[(size_t)i]);
                } else {
                    st_int(t.dtype, t.data, i, (int64_t)(lcg(&s) % (uint32_t)range));
                }
            }
        }
        return;
    }
    const int bits = rad_dtype_bits(t.dtype);
    if (bits > 0 && bits < 8) {                  /* packed codes: any bit pattern is a value */
        for (size_t i = 0; i < m.size(); ++i) m[i] = (uint8_t)lcg(&s);
        return;
    }
    if (t.dtype == RAD_E8M0) {
        for (int64_t i = 0; i < n; ++i) ((uint8_t*)t.data)[i] = (uint8_t)(123 + lcg(&s) % 9);
        return;
    }
    std::vector<float> v((size_t)n);
    for (auto& x : v) x = (float)((int)(lcg(&s) % 2001) - 1000) / 1000.0f;
    const int64_t last = d.rank ? d.shape[d.rank - 1] : 1;
    if (d.fill == RAD_FILL_SIGMOID) {
        for (auto& x : v) x = 1.0f / (1.0f + std::exp(-2.0f * x));
    } else if (d.fill == RAD_FILL_GATE_CUMSUM) {
        const int64_t heads = last, rws = heads ? n / heads : n;
        const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : rws;
        for (int64_t h = 0; h < heads; ++h) {
            float acc = 0;
            for (int64_t r = 0; r < rws; ++r) {
                if (r % ch == 0) acc = 0;
                acc -= std::log1p(std::exp(v[(size_t)(r * heads + h)]));
                v[(size_t)(r * heads + h)] = acc;
            }
        }
    } else if (d.fill == RAD_FILL_TRI_LOWER) {
        /* [rows, heads, chunk]: unit diagonal at row % chunk, small below, zero above. */
        const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : 1;
        const int64_t heads = d.rank >= 3 ? d.shape[1] : 1;
        const int64_t rws = n / (heads * ch);
        for (int64_t r = 0; r < rws; ++r)
            for (int64_t h = 0; h < heads; ++h)
                for (int64_t j = 0; j < ch; ++j) {
                    float& x = v[(size_t)((r * heads + h) * ch + j)];
                    x = j > r % ch ? 0.0f : (j == r % ch ? 1.0f : x * 0.25f);
                }
    } else if (d.fill == RAD_FILL_UNIT_ROW && last > 0) {
        for (int64_t r = 0; r < n / last; ++r) {
            double ss = 0;
            for (int64_t i = 0; i < last; ++i) ss += (double)v[(size_t)(r * last + i)] * v[(size_t)(r * last + i)];
            const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
            for (int64_t i = 0; i < last; ++i) v[(size_t)(r * last + i)] *= inv;
        }
    }
    for (int64_t i = 0; i < n; ++i) st_dt(t.dtype, t.data, i, v[(size_t)i]);
}

/* Describes every operand of `op` with the parameters in `a`; RAD_E_UNSUPPORTED when the
 * description declines. */
int describe(const char* op, A& a, std::vector<RadOpdDesc>* d) {
    const RadKernelInfo* k = kern(op);
    const RadOpSchema* sc = schema(op);
    if (!k || !sc || !k->opd_shape) return RAD_E_NOKERNEL;
    d->assign((size_t)sc->n_operands, RadOpdDesc{});
    for (int i = 0; i < sc->n_operands; ++i) {
        const int rc = k->opd_shape(a.ps.data(), (int)a.ps.size(), i, &(*d)[(size_t)i]);
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

void build(const std::vector<RadOpdDesc>& d, const std::vector<bool>& zero_dims_of,
           const std::vector<RadOpdDesc>& d2, int64_t rows, Built* b) {
    b->mem.assign(d.size(), {});
    b->t.assign(d.size(), RadTensor{});
    int64_t prev_lead = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        RadTensor& t = b->t[i];
        if (d[i].flags & RAD_OPD_F_ABSENT) { prev_lead = 0; continue; }
        t.dtype = d[i].dtype;
        t.rank = d[i].rank;
        RadOpdDesc de = d[i];                       /* the description at the extents built */
        int64_t n = 1;
        for (uint32_t k = 0; k < d[i].rank; ++k) {
            int64_t e = d[i].shape[k];
            if (!zero_dims_of.empty() && zero_dims_of[i] && d2[i].shape[k] != e) e = 0;
            t.shape[k] = de.shape[k] = e;
            n *= e;
        }
        int64_t acc = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = acc; acc *= std::max<int64_t>(1, t.shape[k]); }
        const int64_t bytes = rad_dtype_bytes(t.dtype, n) > 0 ? rad_dtype_bytes(t.dtype, n) : n * 4;
        b->mem[i].assign((size_t)bytes + 64, 0xA5);
        t.data = b->mem[i].data();
        fill_operand(de, rows, prev_lead, 0x9E3779B9u ^ (uint32_t)(i * 7919), b->mem[i], t);
        prev_lead = t.rank ? t.shape[0] : 0;
    }
}

}  /* namespace */

/* Every libref row has a geometry here, so an op added without one fails this test by name. */
TEST(every_libref_op_has_a_sweep_geometry) {
    const std::vector<Geo> g = geometries();
    for (int i = 0; i < api().kernel_count(); ++i) {
        const char* op = api().kernel_at(i)->op;
        bool found = false;
        for (const Geo& e : g) found = found || !std::strcmp(e.op, op);
        if (!found) fprintf(stderr, "    no sweep geometry for '%s'\n", op);
        CHECK(found);
    }
}

/* Each op runs on operands sized and filled from its own description: the description agrees with
 * the kernel, so rad-kbench can draw and record the op. */
TEST(every_op_runs_on_its_own_description) {
    for (const Geo& g : geometries()) {
        if (undrawable(g.op)) continue;
        A a; g.set(a, g.r);
        std::vector<RadOpdDesc> d;
        const int rc = describe(g.op, a, &d);
        if (rc != RAD_OK) fprintf(stderr, "    %s: description declined (%s)\n", g.op, rad_strerror(rc));
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK) continue;
        Built b;
        build(d, {}, {}, g.r, &b);
        a.ts = b.t;
        const int st = run(g.op, a);
        if (st != RAD_OK) fprintf(stderr, "    %s: %s\n", g.op, rad_strerror(st));
        CHECK_EQ(st, RAD_OK);
    }
}

/* Zero rows is not a call: every op refuses it with RAD_E_SHAPE and writes nothing. The row
 * dimensions are found by describing the op at the running value and at twice it; every extent
 * that changes is set to zero. */
TEST(every_op_refuses_zero_rows_without_writing) {
    for (const Geo& g : geometries()) {
        if (undrawable(g.op)) continue;
        A a1; g.set(a1, g.r);
        A a2; g.set(a2, 2 * g.r);
        std::vector<RadOpdDesc> d1, d2;
        if (describe(g.op, a1, &d1) != RAD_OK || describe(g.op, a2, &d2) != RAD_OK) {
            fprintf(stderr, "    %s: no description to empty\n", g.op);
            CHECK(false);
            continue;
        }
        std::vector<bool> zero(d1.size(), true);
        Built b;
        build(d1, zero, d2, 0, &b);
        A a; g.set(a, 0);
        a.ts = b.t;
        const std::vector<std::vector<uint8_t>> before = b.mem;
        const int st = run(g.op, a);
        if (st != RAD_E_SHAPE) fprintf(stderr, "    %s at zero rows: %s\n", g.op, rad_strerror(st));
        CHECK_EQ(st, RAD_E_SHAPE);
        if (b.mem != before) fprintf(stderr, "    %s wrote at zero rows\n", g.op);
        CHECK(b.mem == before);
    }
}

/* ================================================================== the anchor state form
 * gdn_recurrent_update under `anchor` keeps one state a sequence and replays the previous step's
 * accepted prefix from its factor block (docs/OPS.md, `state_form`). With an f32 state the replay
 * repeats the per-candidate arithmetic exactly, so over a schedule of acceptances 1, 2, 3, 1 both
 * forms must give the same `o` bit for bit. The anchor's six columns slide by 0, +1, 0, -1 pages, so
 * the stamp and the column algebra are exercised; and a run that ignores num_accepted (always 1)
 * must differ, so the case is sensitive to the replay. */
namespace {
struct AnchorRun {
    int     status     = RAD_OK;   /* the first refusal of any call, else RAD_OK */
    int64_t differ_pc  = 0;        /* steps whose anchor `o` differs from the per-candidate one */
    int64_t differ_one = 0;        /* the same with num_accepted held at 1 */
};

/* Four steps of two sequences at head [V, K], QL tokens a sequence a step, accepting
 * `nacc_sched[step]` of the previous step's tokens, under per_candidate and under anchor. */
AnchorRun anchor_against_per_candidate(int64_t K, int64_t V, int64_t QL, const int nacc_sched[4]) {
    const int64_t H = 2, N = 2, T_ = N * QL, PAGES = 7;
    const int win_sched[4] = { 0, 1, 1, 0 };
    AnchorRun r;
    uint32_t rs = 77;
    auto rnd = [&rs]() { rs = rs * 1664525u + 1013904223u; return (float)((int)(rs >> 9) % 2001 - 1000) / 1000.0f; };
    std::vector<float> init((size_t)(N * H * V * K));
    for (auto& x : init) x = 0.1f * rnd();
    T alog(RAD_F32, {H}); alog.f({-2.0f, -1.75f});
    T dtb(RAD_F32, {H}); dtb.f({-1.0f, -0.5f});
    std::vector<float> nwv((size_t)V);
    for (auto& x : nwv) x = 1.0f + 0.25f * rnd();
    T nw(RAD_F32, {V}); nw.f(nwv);
    T cu(RAD_I32, {N + 1}); cu.n({0, QL, 2 * QL});

    /* Per-candidate: slot n * QL + t per candidate token, the initial state at column 0. */
    T st_pc(RAD_F32, {N * QL, H, V, K});
    /* Anchor: PAGES slots a sequence, the initial state at the first window's column 2. */
    T st_an(RAD_F32, {N * PAGES, H, V, K}), st_one(RAD_F32, {N * PAGES, H, V, K});
    for (int64_t n = 0; n < N; ++n)
        for (int64_t i = 0; i < H * V * K; ++i) {
            const float x = init[(size_t)(n * H * V * K + i)];
            ((float*)st_pc.t.data)[(n * QL) * H * V * K + i] = x;
            ((float*)st_an.t.data)[(n * PAGES + win_sched[0] + 2) * H * V * K + i] = x;
            ((float*)st_one.t.data)[(n * PAGES + win_sched[0] + 2) * H * V * K + i] = x;
        }
    T idx_pc(RAD_I32, {N, QL});
    for (int64_t n = 0; n < N; ++n)
        for (int64_t t = 0; t < QL; ++t) ((int32_t*)idx_pc.t.data)[n * QL + t] = (int32_t)(n * QL + t);

    for (int step = 0; step < 4; ++step) {
        std::vector<float> qv((size_t)(T_ * K)), kv((size_t)(T_ * K)), vv((size_t)(T_ * H * V));
        std::vector<float> avv((size_t)(T_ * H)), bvv((size_t)(T_ * H)), zv((size_t)(T_ * H * V));
        for (auto& x : qv) x = rnd();
        for (auto& x : kv) x = rnd();
        for (auto& x : vv) x = rnd();
        for (auto& x : avv) x = rnd();
        for (auto& x : bvv) x = rnd();
        for (auto& x : zv) x = rnd();
        T q(RAD_BF16, {T_, 1, K}); q.f(qv);
        T k(RAD_BF16, {T_, 1, K}); k.f(kv);
        T v(RAD_BF16, {T_, H, V}); v.f(vv);
        T av(RAD_BF16, {T_, H}); av.f(avv);
        T bv(RAD_BF16, {T_, H}); bv.f(bvv);
        T z(RAD_BF16, {T_, H, V}); z.f(zv);
        T nacc(RAD_I32, {N}); nacc.n({nacc_sched[step], nacc_sched[step]});
        T ones(RAD_I32, {N}); ones.n({1, 1});
        T idx_an(RAD_I32, {N, 6});
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c = 0; c < 6; ++c)
                ((int32_t*)idx_an.t.data)[n * 6 + c] = (int32_t)(n * PAGES + win_sched[step] + c);
        auto go = [&](T& st, T& idx, T& na, const char* form, T& o) {
            A a; a.t(q).t(k).t(v).t(av).t(bv).t(alog).t(dtb).t(st).t(idx).t(na).t(cu).t(z).t(nw)
                  .t(o).tnull().tnull().i("q_len", QL).i("head_k", K).i("head_v", V)
                  .i("n_head_v", H).d("scale", 0.125).d("eps", 1e-6).s("act", "silu")
                  .s("state_form", form);
            return run("gdn_recurrent_update", a);
        };
        T o_pc(RAD_F32, {T_, H, V}), o_an(RAD_F32, {T_, H, V}), o_one(RAD_F32, {T_, H, V});
        for (int s : { go(st_pc, idx_pc, nacc, "per_candidate", o_pc),
                       go(st_an, idx_an, nacc, "anchor", o_an),
                       go(st_one, idx_an, ones, "anchor", o_one) })
            if (s < 0) { r.status = s; return r; }
        r.differ_pc += o_pc.mem != o_an.mem;
        r.differ_one += o_pc.mem != o_one.mem;
    }
    return r;
}
}  /* namespace */

TEST(gdn_recurrent_anchor_replays_the_accepted_prefix) {
    const int sched[4] = { 1, 2, 3, 1 };
    const AnchorRun r = anchor_against_per_candidate(64, 64, 3, sched);
    CHECK_OK(r.status);
    CHECK_EQ(r.differ_pc, 0LL);
    CHECK(r.differ_one > 0);
}

/* r4d_selftest's anchor case (gdn_recurrent_anchor) at libr4d's kernel shape: 128 x 128 heads,
 * four 32-row groups, two heads, two sequences of three tokens, the window sliding by +1, 0 and -1
 * over acceptances 1, 2, 3, 1. The selftest compares the card's kernel with a host walk of the
 * per-candidate chain; here libref's anchor form must equal its per-candidate form bit for bit. */
TEST(gdn_recurrent_anchor_at_libr4ds_shape_follows_the_selftests_schedule) {
    const int sched[4] = { 1, 2, 3, 1 };
    const AnchorRun r = anchor_against_per_candidate(128, 128, 3, sched);
    CHECK_OK(r.status);
    CHECK_EQ(r.differ_pc, 0LL);
    CHECK(r.differ_one > 0);
}

/* libr4d's kernel replays each 32-row group of V from that group's own copy of k and its own
 * (g, beta) in the factor block (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip, the
 * replay loop's `grp`). A step writes equal copies, but a block drawn at random (as a kernel
 * recording draws the state) holds four different ones, and a reader that took the first copy for
 * every row would answer differently. Here the block is random, three groups (V = 80, the last
 * partial), two tokens replayed: the anchor libref publishes at column 2 must be the per-group
 * replay written out, and a first-copy-only replay must differ from it. */
TEST(gdn_recurrent_anchor_replays_each_row_group_from_its_own_copy) {
    const int64_t K = 32, V = 80, SLOTS = 6, NACC = 2;
    const int64_t groups = (V + 31) / 32, fstr = groups * K + V + 2 * groups;
    const int64_t vb = groups * K, gb = groups * K + V;
    uint32_t rs = 5;
    auto rnd = [&rs]() { rs = rs * 1664525u + 1013904223u; return (float)((int)(rs >> 9) % 2001 - 1000) / 2000.0f; };
    T st(RAD_F32, {SLOTS, 1, V, K});
    float* S = (float*)st.t.data;
    for (int64_t i = 0; i < SLOTS * V * K; ++i) S[i] = rnd();
    const int64_t fb = 4 * V * K;                    /* column 4's block: the factors read */
    const int64_t tmax = std::min<int64_t>(16, (V * K - 1) / fstr);
    S[fb + tmax * fstr] = 2.0f;                      /* stamp 2: no shift, the anchor at column 2 */
    for (int64_t t = 0; t < NACC; ++t)
        for (int64_t gi = 0; gi < groups; ++gi) S[fb + t * fstr + gb + 2 * gi] = -std::fabs(S[fb + t * fstr + gb + 2 * gi]);
    std::vector<float> before(S, S + SLOTS * V * K);

    T q(RAD_F32, {1, 1, K}), k(RAD_F32, {1, 1, K}), v(RAD_F32, {1, 1, V});
    T av(RAD_F32, {1, 1}), bv(RAD_F32, {1, 1}), z(RAD_F32, {1, 1, V}), o(RAD_F32, {1, 1, V});
    std::vector<float> qv((size_t)K), kv((size_t)K), vv((size_t)V), zv((size_t)V), ones((size_t)V, 1.0f);
    for (auto& x : qv) x = rnd();
    for (auto& x : kv) x = rnd();
    for (auto& x : vv) x = rnd();
    for (auto& x : zv) x = rnd();
    q.f(qv); k.f(kv); v.f(vv); z.f(zv); av.f({0.1f}); bv.f({0.2f});
    T alog(RAD_F32, {1}); alog.f({-2.0f});
    T dtb(RAD_F32, {1}); dtb.f({-1.0f});
    T nw(RAD_F32, {V}); nw.f(ones);
    T idx(RAD_I32, {1, 6}); idx.n({0, 1, 2, 3, 4, 5});
    T nacc(RAD_I32, {1}); nacc.n({NACC});
    T cu(RAD_I32, {2}); cu.n({0, 1});
    A a; a.t(q).t(k).t(v).t(av).t(bv).t(alog).t(dtb).t(st).t(idx).t(nacc).t(cu).t(z).t(nw).t(o)
          .tnull().tnull().i("q_len", 1).i("head_k", K).i("head_v", V).i("n_head_v", 1)
          .d("scale", 0.125).d("eps", 1e-6).s("act", "silu").s("state_form", "anchor");
    REQUIRE_EQ(run("gdn_recurrent_update", a), RAD_OK);

    /* The replay written out, from the block as it was: per group, its own k copy and (g, beta),
     * or (first_only) group 0's for every row. */
    auto replay = [&](bool first_only) {
        std::vector<float> h(before.begin() + 2 * V * K, before.begin() + 3 * V * K);
        for (int64_t t = 0; t < NACC; ++t) {
            const float* f = before.data() + fb + t * fstr;
            for (int64_t e = 0; e < V; ++e) {
                const int64_t gi = first_only ? 0 : e / 32;
                const float dec = std::exp(f[gb + 2 * gi]), bt = f[gb + 2 * gi + 1];
                float* r = h.data() + e * K;
                double dot = 0.0;
                for (int64_t d = 0; d < K; ++d) { r[d] *= dec; dot += (double)r[d] * f[gi * K + d]; }
                const float u = bt * (f[vb + e] - (float)dot);
                for (int64_t d = 0; d < K; ++d) r[d] += u * f[gi * K + d];
            }
        }
        return h;
    };
    const std::vector<float> want = replay(false), first = replay(true);
    double err = 0.0, apart = 0.0;
    for (int64_t i = 0; i < V * K; ++i) {
        const double got = S[2 * V * K + i];
        err = std::max(err, std::fabs(got - want[(size_t)i]));
        apart = std::max(apart, std::fabs(got - first[(size_t)i]));
    }
    CHECK(err < 1e-5);
    CHECK(apart > 1e-2);
}

/* A head whose [V, K] block cannot hold libr4d's 16 tokens of factors holds as many as fit before
 * the stamp: at 16 x 16 a token's factors are 34 floats of the block's 256, so 7. A full 7-token
 * step and a 7-token accepted prefix replay exactly; an 8-token step is refused, not written past
 * the block. */
TEST(gdn_recurrent_anchor_fits_a_small_heads_block) {
    const int sched[4] = { 1, 7, 4, 7 };
    const AnchorRun r = anchor_against_per_candidate(16, 16, 7, sched);
    CHECK_OK(r.status);
    CHECK_EQ(r.differ_pc, 0LL);
    CHECK(r.differ_one > 0);
    const int one[4] = { 1, 1, 1, 1 };
    CHECK_EQ(anchor_against_per_candidate(16, 16, 8, one).status, RAD_E_SHAPE);
}

/* ================================================================== integer and packed stores
 * ref_common.h st_dt, through ops that store what they read. */

/* A float store into an integer dtype rounds to nearest even, saturates to the type's range and
 * stores NaN as 0: never wraps, never undefined. */
TEST(integer_stores_round_saturate_and_zero_nan) {
    const std::vector<float> xv = { NAN, 3e9f, -3e9f, 2.5f, -2.5f, 70000.0f };
    T x(RAD_F32, {1, 6}); x.f(xv);
    T y32(RAD_I32, {1, 6}), y16(RAD_U16, {1, 6}), y8(RAD_I8, {1, 6});
    A a; a.t(x).t(y32).i("M", 1).i("n", 6).s("from", "f32").s("to", "i32");
    CHECK_OK(run("cast", a));
    const int64_t w32[6] = { 0, INT32_MAX, INT32_MIN, 2, -2, 70000 };
    for (int i = 0; i < 6; ++i) CHECK_EQ(y32.iat(i), w32[i]);
    A b; b.t(x).t(y16).i("M", 1).i("n", 6).s("from", "f32").s("to", "u16");
    CHECK_OK(run("cast", b));
    const float w16[6] = { 0, 65535, 0, 2, 0, 65535 };
    for (int i = 0; i < 6; ++i) CHECK_EQ(y16.at(i), w16[i]);
    A c; c.t(x).t(y8).i("M", 1).i("n", 6).s("from", "f32").s("to", "i8");
    CHECK_OK(run("cast", c));
    const float w8[6] = { 0, 127, -128, 2, -2, 127 };
    for (int i = 0; i < 6; ++i) CHECK_EQ(y8.at(i), w8[i]);
}

/* Packed codes move exactly: a gather of 3-code int4 rows, whose odd rows start mid-byte, into a
 * destination whose rows share bytes, and a cast to e2m1 that rounds to the nearest code. */
TEST(gather_rows_moves_packed_codes_exactly) {
    /* Codes 0..8 of a [3, 3] int4 plane: 1 2 3 | 4 5 6 | 7 -8 -7. */
    T x(RAD_I4, {3, 3}); x.raw({0x21, 0x43, 0x65, 0x87, 0x09});
    T idx(RAD_I32, {3}); idx.n({2, 0, 1});
    T y(RAD_I4, {3, 3});
    A a; a.t(x).t(idx).t(y).i("M", 3).i("n", 3).s("dtype", "i4");
    CHECK_OK(run("gather_rows", a));
    const float want[9] = { 7, -8, -7, 1, 2, 3, 4, 5, 6 };
    for (int i = 0; i < 9; ++i) CHECK_EQ(y.at(i), want[i]);

    T f(RAD_F32, {1, 6}); f.f({0.74f, 0.76f, -2.5f, 7.0f, -0.25f, 5.0f});
    T e(RAD_FP4E2M1, {1, 6});
    A b; b.t(f).t(e).i("M", 1).i("n", 6).s("from", "f32").s("to", "fp4_e2m1");
    CHECK_OK(run("cast", b));
    /* 0.74 -> 0.5, 0.76 -> 1, -2.5 ties to the even code (2), 7 saturates, -0.25 ties to +-0,
     * 5 ties to the even code (4). */
    const float wf[6] = { 0.5f, 1.0f, -2.0f, 6.0f, 0.0f, 4.0f };
    for (int i = 0; i < 6; ++i) CHECK_EQ(e.at(i), wf[i]);
}

/* ================================================================== sam_attn
 * Unlimited-OCR's SAM tower attention (arch/common/rad_block_sam.h, libref/ref_sam_attn.cpp).
 *
 * The expected values are a direct port of deepencoder.py's Attention.forward /
 * add_decomposed_rel_pos / window_partition in double, not a second call into libref: the padding
 * rule, the per-head rel_pos channel dot and the resampling of a table along its row axis each have
 * a way to be wrong that an oracle sharing libref's code could not see. */

namespace {

struct SamIn {
    int64_t gh = 1, gw = 1, win = 0, nh = 1, hd = 1, nseq = 1;
    int64_t raw_rows = 0;                 /* 0: the geometry's own 2*tile-1; else the staged rows */
    std::vector<float> q, k, v, rh, rw;   /* [M, nh*hd] and [tab(), hd], row-major */
    int64_t W()    const { return nh * hd; }
    int64_t rows() const { return nseq * gh * gw; }
    int64_t tab()  const {
        if (raw_rows > 0) return raw_rows;
        return 2 * (win > 0 ? win : (gh > gw ? gh : gw)) - 1;
    }
};

/* A deterministic draw in [-1, 1]: index-driven, so a wrong pitch or row map reads other numbers. */
static void sam_fill(SamIn& in, uint32_t seed) {
    uint32_t s = seed | 1u;
    auto next = [&s] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffffu) / 32767.5f - 1.0f; };
    in.q.resize((size_t)(in.rows() * in.W()));
    in.k.resize(in.q.size());
    in.v.resize(in.q.size());
    for (auto* x : { &in.q, &in.k, &in.v })
        for (auto& f : *x) f = next();
    in.rh.resize((size_t)(in.tab() * in.hd));
    in.rw.resize(in.rh.size());
    for (auto* x : { &in.rh, &in.rw })
        for (auto& f : *x) f = 0.1f * next();
}

/* F.interpolate(..., size=want, mode="linear"), align_corners False, over dim 0. */
static std::vector<double> sam_table(const std::vector<float>& raw, int64_t hd, int64_t want) {
    const int64_t rows = hd > 0 ? (int64_t)raw.size() / hd : 0;
    std::vector<double> out((size_t)(want * hd), 0.0);
    const double span = (double)rows / (double)want;
    for (int64_t d = 0; d < want; ++d) {
        double src = ((double)d + 0.5) * span - 0.5;
        if (src < 0.0) src = 0.0;
        if (src > (double)(rows - 1)) src = (double)(rows - 1);
        const int64_t i0 = (int64_t)src, i1 = i0 + 1 < rows ? i0 + 1 : i0;
        const double f = src - (double)i0;
        for (int64_t c = 0; c < hd; ++c)
            out[(size_t)(d * hd + c)] = raw[(size_t)(i0 * hd + c)] * (1.0 - f) +
                                        raw[(size_t)(i1 * hd + c)] * f;
    }
    return out;
}

/* `drop_pads` is the WRONG reading of window_partition: pad keys removed from the softmax instead of
 * attending as zeros. Used only to prove a case can tell the two apart. */
static std::vector<float> sam_oracle(const SamIn& in, bool drop_pads = false) {
    const int64_t W = in.W(), R = in.gh * in.gw, M = in.rows();
    const int64_t th = in.win > 0 ? in.win : in.gh, tw = in.win > 0 ? in.win : in.gw;
    const std::vector<double> Rh = sam_table(in.rh, in.hd, 2 * th - 1);
    const std::vector<double> Rw = sam_table(in.rw, in.hd, 2 * tw - 1);
    const double scale = 1.0 / std::sqrt((double)in.hd);
    std::vector<float> out((size_t)(M * W), 0.0f);
    std::vector<double> logits, acc;
    for (int64_t s = 0; s < in.nseq; ++s)
        for (int64_t ih = 0; ih < in.gh; ++ih)
            for (int64_t iw = 0; iw < in.gw; ++iw) {
                const int64_t row = s * R + ih * in.gw + iw;
                const int64_t ty = in.win > 0 ? ih / in.win : 0;
                const int64_t tx = in.win > 0 ? iw / in.win : 0;
                const int64_t lh = ih - ty * th, lw = iw - tx * tw;
                for (int64_t h = 0; h < in.nh; ++h) {
                    const float* qf = &in.q[(size_t)(row * W + h * in.hd)];
                    logits.clear();
                    std::vector<int64_t> kvrow;
                    for (int64_t jh = 0; jh < th; ++jh) {
                        const double* hrp = &Rh[(size_t)((lh - jh + th - 1) * in.hd)];
                        double hr = 0.0;
                        for (int64_t c = 0; c < in.hd; ++c) hr += qf[c] * hrp[c];
                        for (int64_t jw = 0; jw < tw; ++jw) {
                            const int64_t kh = ty * th + jh, kw = tx * tw + jw;
                            const bool real = kh < in.gh && kw < in.gw;
                            if (drop_pads && !real) continue;
                            const double* wrp = &Rw[(size_t)((lw - jw + tw - 1) * in.hd)];
                            double sc = hr;
                            for (int64_t c = 0; c < in.hd; ++c) sc += qf[c] * wrp[c];
                            if (real) {
                                const float* kf = &in.k[(size_t)((s * R + kh * in.gw + kw) * W +
                                                                h * in.hd)];
                                double d = 0.0;
                                for (int64_t c = 0; c < in.hd; ++c) d += qf[c] * kf[c];
                                sc += scale * d;
                                kvrow.push_back(s * R + kh * in.gw + kw);
                            } else {
                                kvrow.push_back(-1);
                            }
                            logits.push_back(sc);
                        }
                    }
                    double mx = -INFINITY;
                    for (double v : logits) mx = std::max(mx, v);
                    double sum = 0.0;
                    for (double& v : logits) { v = std::exp(v - mx); sum += v; }
                    acc.assign((size_t)in.hd, 0.0);
                    for (size_t j = 0; j < logits.size(); ++j) {
                        if (kvrow[j] < 0) continue;
                        const float* vf = &in.v[(size_t)(kvrow[j] * W + h * in.hd)];
                        const double p = logits[j] / sum;
                        for (int64_t c = 0; c < in.hd; ++c) acc[(size_t)c] += p * vf[c];
                    }
                    for (int64_t c = 0; c < in.hd; ++c)
                        out[(size_t)(row * W + h * in.hd + c)] = (float)acc[(size_t)c];
                }
            }
    return out;
}

/* Run the op; `pitch` > 0 stores every operand at that row pitch (a column slice of a wider
 * buffer), 0 uses W. Returns the output read back at the same pitch. */
static std::vector<float> sam_run(const SamIn& in, int64_t pitch, int* rc) {
    const int64_t W = in.W(), R = in.gh * in.gw, M = in.rows();
    const int64_t ld = pitch > 0 ? pitch : W;
    T q(RAD_BF16, { M, ld }), k(RAD_BF16, { M, ld }), v(RAD_BF16, { M, ld }), o(RAD_BF16, { M, ld });
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < W; ++c) {
            st_dt(RAD_BF16, q.t.data, r * ld + c, in.q[(size_t)(r * W + c)]);
            st_dt(RAD_BF16, k.t.data, r * ld + c, in.k[(size_t)(r * W + c)]);
            st_dt(RAD_BF16, v.t.data, r * ld + c, in.v[(size_t)(r * W + c)]);
        }
    T rh(RAD_BF16, { in.tab(), in.hd }), rw(RAD_BF16, { in.tab(), in.hd });
    for (size_t i = 0; i < in.rh.size(); ++i) {
        st_dt(RAD_BF16, rh.t.data, (int64_t)i, in.rh[i]);
        st_dt(RAD_BF16, rw.t.data, (int64_t)i, in.rw[i]);
    }
    T cu(RAD_I32, { in.nseq + 1 });
    for (int64_t s = 0; s <= in.nseq; ++s) st_int(RAD_I32, cu.t.data, s, s * R);

    A a;
    /* A pitch wider than W is a strided VIEW of the storage: shape [M, W], stride[0] = ld, which is
     * what the tower's column slices of one fused projection are. */
    auto view = [&](const T& x) {
        RadTensor v = x.t;
        v.shape[1] = W;
        v.stride[0] = ld;
        v.stride[1] = 1;
        return v;
    };
    a.ts.push_back(view(q));
    a.ts.push_back(view(k));
    a.ts.push_back(view(v));
    a.t(cu);
    a.ts.push_back(view(o));
    a.t(rh).t(rw)
     .i("M", M).i("head_dim", in.hd).i("n_head", in.nh).i("grid_h", in.gh).i("grid_w", in.gw)
     .i("window", in.win).d("scale", 1.0 / std::sqrt((double)in.hd)).s("dtype", "bf16");
    *rc = run("sam_attn", a);
    std::vector<float> out((size_t)(M * W), 0.0f);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < W; ++c) out[(size_t)(r * W + c)] = ld_dt(RAD_BF16, o.t.data, r * ld + c);
    return out;
}

/* Compare two [M, W] rows: max absolute difference and the number outside 2e-3 + 2e-2 * |want|. */
static void sam_cmp(const std::vector<float>& got, const std::vector<float>& want, float* maxd,
                    int64_t* bad) {
    *maxd = 0.0f;
    *bad = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const float d = std::fabs(got[i] - want[i]);
        if (d > *maxd) *maxd = d;
        if (d > 2e-3f + 2e-2f * std::fabs(want[i])) ++*bad;
    }
}

}  /* namespace */

/* One row, one key: the softmax is 1 and the output is v, bit for bit after bf16 rounding. The same
 * with window 1 over a 1x3 grid, where every row is a tile of its own. */
TEST(op_sam_attn_single_key_returns_v) {
    SamIn one;
    one.gh = one.gw = 1; one.win = 1; one.nh = 1; one.hd = 4;
    one.q = { 3.0f, -1.0f, 0.5f, 2.0f };
    one.k = { 1.0f, 2.0f, 3.0f, 4.0f };
    one.v = { 0.5f, -1.0f, 2.0f, 0.25f };
    one.rh.assign(4, 0.0f);
    one.rw.assign(4, 0.0f);
    int rc = 0;
    std::vector<float> got = sam_run(one, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    for (int64_t c = 0; c < 4; ++c) CHECK_EQ(got[(size_t)c], one.v[(size_t)c]);

    SamIn strip = one;
    strip.gh = 1; strip.gw = 3;
    strip.q = { 1, 2, 3, 4, -1, 0, 1, -2, 0.5f, 0.5f, 0.5f, 0.5f };
    strip.k = { 1, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2 };
    strip.v = { 0.5f, 0.25f, 0.125f, 0.0625f, -0.5f, -0.25f, -0.125f, -0.0625f,
                9.0f, 8.0f, 7.0f, 6.0f };
    strip.rh.assign(4, 0.25f);
    strip.rw.assign(4, -0.25f);
    got = sam_run(strip, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    for (int64_t i = 0; i < 12; ++i) CHECK_EQ(got[(size_t)i], strip.v[(size_t)i]);
}

/* Window 14 over 16x16: the grid pads to 28x28, so the last two rows and columns of the last window
 * are ZERO keys that still take part in the softmax. Compared against the double port, then against
 * the port with the pads removed: the two differ, which is what makes this case able to see the
 * rule. */
TEST(op_sam_attn_windowed_matches_the_reference_port) {
    SamIn in;
    in.gh = 16; in.gw = 16; in.win = 14; in.nh = 2; in.hd = 8;
    sam_fill(in, 0x51ed2701u);
    int rc = 0;
    const std::vector<float> got = sam_run(in, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    float maxd = 0.0f; int64_t bad = 0;
    const std::vector<float> want = sam_oracle(in);
    sam_cmp(got, want, &maxd, &bad);
    CHECK_EQ(bad, 0LL);
    CHECK(maxd < 2e-3f);

    const std::vector<float> nopad = sam_oracle(in, true);
    float pad_max = 0.0f;
    for (size_t i = 0; i < nopad.size(); ++i)
        pad_max = std::max(pad_max, std::fabs(nopad[i] - want[i]));
    CHECK(pad_max > 1e-3f);          /* the case can tell the two rules apart */
}

/* Window 0: the whole segment is one tile, over a square grid and over a 6x10 one whose rel_pos
 * tables are 2*10-1 rows long and must be resampled to 2*6-1 for the row axis. */
TEST(op_sam_attn_global_and_non_square) {
    SamIn sq;
    sq.gh = 16; sq.gw = 16; sq.win = 0; sq.nh = 2; sq.hd = 8;
    sam_fill(sq, 0xa341316cu);
    int rc = 0;
    float maxd = 0.0f; int64_t bad = 0;
    sam_cmp(sam_run(sq, 0, &rc), sam_oracle(sq), &maxd, &bad);
    CHECK_EQ(rc, (int)RAD_OK);
    CHECK_EQ(bad, 0LL);
    CHECK(maxd < 2e-3f);

    SamIn ns;
    ns.gh = 6; ns.gw = 10; ns.win = 0; ns.nh = 3; ns.hd = 4;
    sam_fill(ns, 0x7f4a7c15u);
    CHECK_EQ(ns.tab(), 19LL);
    sam_cmp(sam_run(ns, 0, &rc), sam_oracle(ns), &maxd, &bad);
    CHECK_EQ(rc, (int)RAD_OK);
    CHECK_EQ(bad, 0LL);
    CHECK(maxd < 2e-3f);

    /* The production case the tower hits: a 40x40 global block whose wanted rel_pos rows are 79,
     * served from the checkpoint's 127-row tables (both axes resize). */
    SamIn p40;
    p40.gh = p40.gw = 40; p40.win = 0; p40.nh = 2; p40.hd = 8; p40.raw_rows = 127;
    sam_fill(p40, 0x2545f491u);
    CHECK_EQ(p40.tab(), 127LL);
    sam_cmp(sam_run(p40, 0, &rc), sam_oracle(p40), &maxd, &bad);
    CHECK_EQ(rc, (int)RAD_OK);
    CHECK_EQ(bad, 0LL);
    CHECK(maxd < 2e-3f);
}

/* Two segments: each is one tile (window 0). The second segment's rows are computed from its own
 * data only, so changing the first segment's q/k/v must leave them bit for bit unchanged while the
 * first segment's move. */
TEST(op_sam_attn_segments_are_independent) {
    SamIn in;
    in.gh = 8; in.gw = 8; in.win = 0; in.nh = 2; in.hd = 8; in.nseq = 2;
    sam_fill(in, 0x2545f491u);
    int rc = 0;
    const std::vector<float> base = sam_run(in, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    float maxd = 0.0f; int64_t bad = 0;
    sam_cmp(base, sam_oracle(in), &maxd, &bad);
    CHECK_EQ(bad, 0LL);
    CHECK(maxd < 2e-3f);

    SamIn alt = in;
    for (int64_t i = 0; i < 64 * in.W(); ++i) {        /* the first segment's rows */
        alt.q[(size_t)i] = -alt.q[(size_t)i];
        alt.k[(size_t)i] += 0.25f;
    }
    const std::vector<float> moved = sam_run(alt, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    const size_t half = base.size() / 2;
    bool first_changed = false, second_same = true;
    for (size_t i = 0; i < half; ++i)
        first_changed |= base[i] != moved[i];
    for (size_t i = half; i < base.size(); ++i)
        second_same &= base[i] == moved[i];
    CHECK(first_changed);
    CHECK(second_same);
}

/* The tower's q/k/v are the three column slices of one fused projection, so every operand is read
 * at a row pitch wider than W. The result must be bit-identical to the dense-pitch one. */
TEST(op_sam_attn_reads_at_a_row_pitch) {
    SamIn in;
    in.gh = 16; in.gw = 16; in.win = 14; in.nh = 2; in.hd = 8;
    sam_fill(in, 0x1234567u);
    int rc = 0;
    const std::vector<float> dense = sam_run(in, 0, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    const std::vector<float> strided = sam_run(in, in.W() + 3, &rc);
    CHECK_EQ(rc, (int)RAD_OK);
    for (size_t i = 0; i < dense.size(); ++i) CHECK_EQ(strided[i], dense[i]);
}

/* Refusals: a segment that is not one grid, an absent table, a negative window, a zero head. */
TEST(op_sam_attn_refuses_malformed_geometry) {
    SamIn in;
    in.gh = 4; in.gw = 4; in.win = 0; in.nh = 1; in.hd = 4; in.nseq = 2;
    sam_fill(in, 0x9e3779b9u);
    const int64_t W = in.W(), R = in.gh * in.gw, M = in.rows();

    auto build = [&](T& q, T& k, T& v, T& cu, T& o, T& rh, T& rw, A& a, int64_t window) {
        a.t(q).t(k).t(v).t(cu).t(o).t(rh).t(rw)
         .i("M", M).i("head_dim", in.hd).i("n_head", in.nh).i("grid_h", in.gh)
         .i("grid_w", in.gw).i("window", window).d("scale", 0.5).s("dtype", "bf16");
    };
    T q(RAD_BF16, { M, W }), k(RAD_BF16, { M, W }), v(RAD_BF16, { M, W }), o(RAD_BF16, { M, W });
    q.f(in.q); k.f(in.k); v.f(in.v);
    T rh(RAD_BF16, { in.tab(), in.hd }), rw(RAD_BF16, { in.tab(), in.hd });
    rh.f(in.rh); rw.f(in.rw);

    /* cu says the first segment is half a grid. */
    T cu_short(RAD_I32, { 3 }); cu_short.n({ 0, R / 2, M });
    A a1; build(q, k, v, cu_short, o, rh, rw, a1, 0);
    CHECK_EQ(run("sam_attn", a1), (int)RAD_E_SHAPE);

    T cu_ok(RAD_I32, { 3 }); cu_ok.n({ 0, R, M });
    A a2; build(q, k, v, cu_ok, o, rh, rw, a2, 0);
    CHECK_OK(run("sam_attn", a2));

    /* An absent rel_pos table. */
    A a3; a3.t(q).t(k).t(v).t(cu_ok).t(o).t(rh).tnull()
          .i("M", M).i("head_dim", in.hd).i("n_head", in.nh).i("grid_h", in.gh)
          .i("grid_w", in.gw).i("window", 0).d("scale", 0.5).s("dtype", "bf16");
    CHECK_EQ(run("sam_attn", a3), (int)RAD_E_INVAL);

    /* A negative window, and a zero head width. */
    A a4; build(q, k, v, cu_ok, o, rh, rw, a4, -1);
    CHECK_EQ(run("sam_attn", a4), (int)RAD_E_SHAPE);
    A a5; a5.t(q).t(k).t(v).t(cu_ok).t(o).t(rh).t(rw)
          .i("M", M).i("head_dim", 0).i("n_head", in.nh).i("grid_h", in.gh)
          .i("grid_w", in.gw).i("window", 0).d("scale", 0.5).s("dtype", "bf16");
    CHECK_EQ(run("sam_attn", a5), (int)RAD_E_SHAPE);
}

/* ================================================================== tree verification of the delta net
 * gdn_conv_update, gdn_recurrent_update (anchor form) and the folded pair take `spec_parent` and
 * `spec_path` (docs/OPS.md, "Tree verification of the delta net"). The oracle here is the op itself
 * run sequentially: a tree row must equal the last row of a CHAIN decode of its root-to-row path,
 * from the same committed state, bit for bit; and a step that follows a tree and is handed the
 * accepted rows as `spec_path` must leave the caches, and give the outputs, of a chain that decoded
 * exactly those rows. Heads are 16 x 16 (libref's anchor block then holds 7 rows' records), f32
 * throughout, so every comparison is on bytes. */
namespace {
namespace gt {
constexpr int64_t K = 16, V = 16, Hg = 1, Hv = 2, W = 4, CD = 2 * Hg * K + Hv * V;
constexpr int64_t MAXR = 7;                    /* rows a 16 x 16 anchor block holds */
constexpr int64_t DEPTH = W - 2 + MAXR;        /* conv window entries */

struct Rows { std::vector<float> x, a, b, z; };

uint32_t g_rs = 12345;
float rnd() {
    g_rs = g_rs * 1664525u + 1013904223u;
    return (float)((int)(g_rs >> 9) % 2001 - 1000) / 1000.0f;
}

Rows gen(int n) {
    Rows r;
    r.x.resize((size_t)(n * CD)); r.a.resize((size_t)(n * Hv)); r.b.resize((size_t)(n * Hv));
    r.z.resize((size_t)(n * Hv * V));
    for (auto* v : { &r.x, &r.a, &r.b, &r.z }) for (auto& e : *v) e = rnd();
    return r;
}
Rows pick(const Rows& s, const std::vector<int>& rows) {
    Rows r;
    for (int i : rows) {
        r.x.insert(r.x.end(), s.x.begin() + i * CD, s.x.begin() + (i + 1) * CD);
        r.a.insert(r.a.end(), s.a.begin() + i * Hv, s.a.begin() + (i + 1) * Hv);
        r.b.insert(r.b.end(), s.b.begin() + i * Hv, s.b.begin() + (i + 1) * Hv);
        r.z.insert(r.z.end(), s.z.begin() + i * Hv * V, s.z.begin() + (i + 1) * Hv * V);
    }
    return r;
}

T dup(const T& s) { T r = s; r.t.data = r.mem.data(); return r; }

/* The layer's weights, fixed across a case. */
struct Weights {
    T w, alog, dtb, nw;
    Weights() : w(RAD_F32, { CD, W }), alog(RAD_F32, { Hv }), dtb(RAD_F32, { Hv }),
                nw(RAD_F32, { V }) {
        std::vector<float> v((size_t)(CD * W));
        for (auto& e : v) e = 0.5f * rnd();
        w.f(v);
        alog.f({ -2.0f, -1.75f }); dtb.f({ -1.0f, -0.5f });
        std::vector<float> n((size_t)V);
        for (auto& e : n) e = 1.0f + 0.25f * rnd();
        nw.f(n);
    }
};
const Weights& weights() { static Weights w; return w; }

/* One sequence's caches: the conv window and the anchor state (slot 0; slot 1 the factors). */
struct Cache {
    T cs, st;
    Cache() : cs(RAD_F32, { 1, CD, DEPTH }), st(RAD_F32, { 2, Hv, V, K }) {}
};
Cache fresh() {
    Cache c;
    std::vector<float> w((size_t)(CD * DEPTH)), s((size_t)(2 * Hv * V * K), 0.0f);
    for (auto& e : w) e = rnd();
    for (size_t i = 0; i < (size_t)(Hv * V * K); ++i) s[i] = 0.1f * rnd();
    c.cs.f(w); c.st.f(s);
    return c;
}
Cache dup(const Cache& c) { Cache r; r.cs = dup(c.cs); r.st = dup(c.st); return r; }

T ivec(const std::vector<int>& v) {
    T t(RAD_I32, { (int64_t)v.size() });
    t.n(std::vector<int64_t>(v.begin(), v.end()));
    return t;
}
T path_t(const std::vector<int>& v) {
    T t(RAD_I32, { 1, (int64_t)v.size() });
    t.n(std::vector<int64_t>(v.begin(), v.end()));
    return t;
}

/* The folded op over n rows: parents / path null for a chain / no path. */
int fused(Cache& c, const Rows& r, int n, const std::vector<int>* par,
          const std::vector<int>* path, int nacc, T& o, const char* form = "anchor") {
    const Weights& w = weights();
    T x(RAD_F32, { n, CD }), a(RAD_F32, { n, Hv }), b(RAD_F32, { n, Hv }),
      z(RAD_F32, { n, Hv, V });
    x.f(r.x); a.f(r.a); b.f(r.b); z.f(r.z);
    T cidx(RAD_I32, { 1 }); cidx.n({ 0 });
    T sidx(RAD_I32, { 1, 6 }); sidx.n({ 0, 0, 0, 1, 1, 1 });
    T na(RAD_I32, { 1 }); na.n({ nacc });
    T cu(RAD_I32, { 2 }); cu.n({ 0, n });
    T sp = par ? ivec(*par) : T(RAD_I32, { 1 }), pp = path ? path_t(*path) : T(RAD_I32, { 1, 1 });
    A f; f.t(x).t(w.w).tnull().t(c.cs).t(cidx).t(a).t(b).t(w.alog).t(w.dtb).t(c.st).t(sidx)
          .t(na).t(cu).t(z).t(w.nw).t(o).tnull().tnull();
    if (par) f.t(sp); else f.tnull();
    if (path) f.t(pp); else f.tnull();
    f.i("q_len", n).i("head_k", K).i("head_v", V).i("conv_width", W).i("n_head_v", Hv)
     .d("scale", 0.25).d("eps", 1e-6).s("act", "silu").s("state_form", form);
    return run("gdn_conv_recurrent_update", f);
}

/* The same as the pair: gdn_conv_update, then gdn_recurrent_update over its q, k and v. */
int pair(Cache& c, const Rows& r, int n, const std::vector<int>* par,
         const std::vector<int>* path, int nacc, T& o) {
    const Weights& w = weights();
    T x(RAD_F32, { n, CD }), a(RAD_F32, { n, Hv }), b(RAD_F32, { n, Hv }),
      z(RAD_F32, { n, Hv, V });
    x.f(r.x); a.f(r.a); b.f(r.b); z.f(r.z);
    T cidx(RAD_I32, { 1 }); cidx.n({ 0 });
    T sidx(RAD_I32, { 1, 6 }); sidx.n({ 0, 0, 0, 1, 1, 1 });
    T na(RAD_I32, { 1 }); na.n({ nacc });
    T cu(RAD_I32, { 2 }); cu.n({ 0, n });
    T q(RAD_F32, { n, Hg, K }), k(RAD_F32, { n, Hg, K }), v(RAD_F32, { n, Hv, V });
    T sp = par ? ivec(*par) : T(RAD_I32, { 1 }), pp = path ? path_t(*path) : T(RAD_I32, { 1, 1 });
    A cv; cv.t(x).t(w.w).tnull().t(c.cs).t(cidx).t(na).t(cu).t(q).t(k).t(v);
    if (par) cv.t(sp); else cv.tnull();
    if (path) cv.t(pp); else cv.tnull();
    cv.i("q_len", n).i("head_k", K).i("head_v", V).i("conv_width", W);
    int rc = run("gdn_conv_update", cv);
    if (rc != RAD_OK) return rc;
    A rr; rr.t(q).t(k).t(v).t(a).t(b).t(w.alog).t(w.dtb).t(c.st).t(sidx).t(na).t(cu).t(z)
            .t(w.nw).t(o).tnull().tnull();
    if (par) rr.t(sp); else rr.tnull();
    if (path) rr.t(pp); else rr.tnull();
    rr.i("q_len", n).i("head_k", K).i("head_v", V).i("n_head_v", Hv).d("scale", 0.25)
      .d("eps", 1e-6).s("act", "silu").s("state_form", "anchor");
    return run("gdn_recurrent_update", rr);
}

bool same_rows(const T& a, int ra, const T& b, int rb) {
    const size_t n = (size_t)(Hv * V) * 4;
    return std::memcmp(a.mem.data() + (size_t)ra * n, b.mem.data() + (size_t)rb * n, n) == 0;
}
/* Conv window entries [0, e) of every channel, and the anchor state slot, equal. */
bool same_caches(const Cache& a, const Cache& b, int64_t entries) {
    for (int64_t c = 0; c < CD; ++c)
        for (int64_t e = 0; e < entries; ++e)
            if (a.cs.at(c * DEPTH + e) != b.cs.at(c * DEPTH + e)) return false;
    return std::memcmp(a.st.mem.data(), b.st.mem.data(), (size_t)(Hv * V * K) * 4) == 0;
}

/* Rows from the anchor (row 0) down to r. */
std::vector<int> root_path(const std::vector<int>& par, int r) {
    std::vector<int> p;
    for (int x = r; x >= 0; x = par[x] < 0 ? (x > 0 ? x - 1 : -1) : par[x]) p.push_back(x);
    return std::vector<int>(p.rbegin(), p.rend());
}

/* Every row of the tree equals the chain over its path, and every path committed through
 * `spec_path` equals the chain that decoded just that path, followed by an ordinary step and by a
 * tree step. */
void tree_case(const std::vector<int>& par) {
    const int R = (int)par.size();
    const Rows rows = gen(R);
    const Cache init = fresh();

    Cache ct = dup(init);
    T o_tree(RAD_F32, { R, Hv, V });
    REQUIRE_EQ(fused(ct, rows, R, &par, nullptr, 1, o_tree), RAD_OK);

    /* The pair agrees with the fold. */
    {
        Cache cp = dup(init);
        T o_pair(RAD_F32, { R, Hv, V });
        REQUIRE_EQ(pair(cp, rows, R, &par, nullptr, 1, o_pair), RAD_OK);
        CHECK(o_pair.mem == o_tree.mem);
        CHECK(cp.cs.mem == ct.cs.mem);
        CHECK(cp.st.mem == ct.st.mem);
    }

    const Rows next = gen(3);
    const std::vector<int> par2 = { -1, 0, 0, 1 };
    const Rows next2 = gen(4);
    for (int r = 0; r < R; ++r) {
        const std::vector<int> path = root_path(par, r);
        const int len = (int)path.size();
        {   /* Row r is the last row of the chain over its path. */
            Cache cq = dup(init);
            T o_seq(RAD_F32, { len, Hv, V });
            REQUIRE_EQ(fused(cq, pick(rows, path), len, nullptr, nullptr, 1, o_seq), RAD_OK);
            CHECK(same_rows(o_tree, r, o_seq, len - 1));
        }
        /* Commit the path, then an ordinary chain step, then a tree step. */
        for (int variant = 0; variant < 2; ++variant) {
            const Rows& n2 = variant ? next2 : next;
            const int n = variant ? 4 : 3;
            const std::vector<int>* p2 = variant ? &par2 : nullptr;
            Cache c1 = dup(ct);
            T o2(RAD_F32, { n, Hv, V });
            REQUIRE_EQ(fused(c1, n2, n, p2, &path, len, o2), RAD_OK);

            Cache c0 = dup(init);
            T o1(RAD_F32, { len, Hv, V }), o2s(RAD_F32, { n, Hv, V });
            REQUIRE_EQ(fused(c0, pick(rows, path), len, nullptr, nullptr, 1, o1), RAD_OK);
            REQUIRE_EQ(fused(c0, n2, n, p2, nullptr, len, o2s), RAD_OK);
            CHECK(o2.mem == o2s.mem);
            CHECK(same_caches(c1, c0, W - 2 + n));
        }
    }
}
}  /* namespace gt */
}  /* namespace */

/* A chain written as a tree (-2 on every row, or its explicit parents) is the chain. */
TEST(gdn_tree_chain_is_the_plain_chain) {
    const std::vector<int> chain = { -1, 0, 1, 2, 3 }, ord = { -1, -2, -2, -2, -2 };
    const gt::Rows rows = gt::gen(5);
    const gt::Cache init = gt::fresh();
    T o0(RAD_F32, { 5, gt::Hv, gt::V }), o1 = gt::dup(o0), o2 = gt::dup(o0);
    gt::Cache c0 = gt::dup(init), c1 = gt::dup(init), c2 = gt::dup(init);
    CHECK_OK(gt::fused(c0, rows, 5, nullptr, nullptr, 1, o0));
    CHECK_OK(gt::fused(c1, rows, 5, &chain, nullptr, 1, o1));
    CHECK_OK(gt::fused(c2, rows, 5, &ord, nullptr, 1, o2));
    CHECK(o0.mem == o1.mem);
    CHECK(o0.mem == o2.mem);
    CHECK(c0.cs.mem == c1.cs.mem);
    CHECK(c0.st.mem == c1.st.mem);
    CHECK(c0.st.mem == c2.st.mem);
    gt::tree_case(chain);
}

/* Stage 1's shape: a chain of two, then three leaves off the anchor. */
TEST(gdn_tree_root_fork_leaves_equal_their_sequential_paths) {
    gt::tree_case({ -1, 0, 1, 0, 0, 0 });
}

/* A leaf under the anchor and under a chain node. */
TEST(gdn_tree_leaves_at_two_depths) {
    gt::tree_case({ -1, 0, 1, 2, 1, 0, 3 });
}

/* Two non-leaf children of the anchor, each with children of its own, and a non-leaf node whose
 * children are one leaf and one non-leaf: the state a branch starts from is its parent's, not the
 * last row's. */
TEST(gdn_tree_two_nonleaf_children_equal_their_sequential_paths) {
    gt::tree_case({ -1, 0, 1, 1, 0, 4, 4 });
    gt::tree_case({ -1, 0, 0, 1, 2, 3, 4 });
    gt::tree_case({ -1, 0, 1, 0, 3, 2, 4 });
}

/* The window follows the ancestor chain past the block's first row: a depth-1 node's taps are the
 * anchor and the committed window, a depth-2 node's the node, the anchor and the window's newest. */
TEST(gdn_tree_conv_taps_follow_the_ancestor_chain_and_the_window) {
    const std::vector<int> par = { -1, 0, 0, 2, 2 };
    const int R = 5;
    const gt::Rows rows = gt::gen(R);
    const gt::Cache init = gt::fresh();
    gt::Cache c = gt::dup(init);
    const gt::Weights& w = gt::weights();
    T x(RAD_F32, { R, gt::CD }); x.f(rows.x);
    T cidx(RAD_I32, { 1 }); cidx.n({ 0 });
    T na(RAD_I32, { 1 }); na.n({ 1 });
    T cu(RAD_I32, { 2 }); cu.n({ 0, R });
    T q(RAD_F32, { R, gt::Hg, gt::K }), k(RAD_F32, { R, gt::Hg, gt::K }),
      v(RAD_F32, { R, gt::Hv, gt::V });
    T sp = gt::ivec(par);
    A cv; cv.t(x).t(w.w).tnull().t(c.cs).t(cidx).t(na).t(cu).t(q).t(k).t(v).t(sp).tnull()
          .i("q_len", R).i("head_k", gt::K).i("head_v", gt::V).i("conv_width", gt::W);
    REQUIRE_EQ(run("gdn_conv_update", cv), RAD_OK);
    /* Channel 5 (a q channel), by hand: window entries 0..2 are hist = [h0, h1, h2], read at
     * num_accepted - 1 = 0. Row 3's chain is 3, 2, 0, then the window's newest, h2. */
    const int64_t ch = 5;
    auto xv = [&](int row) { return rows.x[(size_t)(row * gt::CD + ch)]; };
    auto hist = [&](int i) { return init.cs.at(ch * gt::DEPTH + i); };
    auto wt = [&](int j) { return w.w.at(ch * gt::W + j); };
    auto silu = [](float a) { return a / (1.0f + std::exp(-a)); };
    const float want3 = silu(((0.0f + wt(0) * hist(2)) + wt(1) * xv(0)) + wt(2) * xv(2) + wt(3) * xv(3));
    CHECK_NEAR(q.at(3 * gt::K + ch), want3, 1e-5);
    /* Row 1's chain is 1, 0, then the window's newest two: h2, h1. */
    const float want1 = silu(((wt(0) * hist(1) + wt(1) * hist(2)) + wt(2) * xv(0)) + wt(3) * xv(1));
    CHECK_NEAR(q.at(1 * gt::K + ch), want1, 1e-5);
}

/* The window left after a tree step is in the chain layout (raw input of row r at entry r + 2),
 * whatever the tree, and the next step reads it through spec_path. */
TEST(gdn_tree_conv_window_gather_equals_compaction) {
    const std::vector<int> par = { -1, 0, 1, 0, 3, 3 };
    const std::vector<int> path = { 0, 3, 5 };
    const int R = 6;
    const gt::Rows rows = gt::gen(R), next = gt::gen(3);
    const gt::Weights& w = gt::weights();
    gt::Cache c = gt::fresh();

    auto conv = [&](gt::Cache& cc, const gt::Rows& r, int n, const std::vector<int>* sp,
                    const std::vector<int>* pp, int nacc, T& q, T& k, T& v) {
        T x(RAD_F32, { n, gt::CD }); x.f(r.x);
        T cidx(RAD_I32, { 1 }); cidx.n({ 0 });
        T na(RAD_I32, { 1 }); na.n({ nacc });
        T cu(RAD_I32, { 2 }); cu.n({ 0, n });
        T spt = sp ? gt::ivec(*sp) : T(RAD_I32, { 1 }), ppt = pp ? gt::path_t(*pp) : T(RAD_I32, { 1, 1 });
        A a; a.t(x).t(w.w).tnull().t(cc.cs).t(cidx).t(na).t(cu).t(q).t(k).t(v);
        if (sp) a.t(spt); else a.tnull();
        if (pp) a.t(ppt); else a.tnull();
        a.i("q_len", n).i("head_k", gt::K).i("head_v", gt::V).i("conv_width", gt::W);
        return run("gdn_conv_update", a);
    };
    T q(RAD_F32, { R, gt::Hg, gt::K }), k(RAD_F32, { R, gt::Hg, gt::K }),
      v(RAD_F32, { R, gt::Hv, gt::V });
    REQUIRE_EQ(conv(c, rows, R, &par, nullptr, 1, q, k, v), RAD_OK);

    /* Gathered through spec_path ... */
    gt::Cache g = gt::dup(c);
    T q1(RAD_F32, { 3, gt::Hg, gt::K }), k1(RAD_F32, { 3, gt::Hg, gt::K }),
      v1(RAD_F32, { 3, gt::Hv, gt::V });
    REQUIRE_EQ(conv(g, next, 3, nullptr, &path, 3, q1, k1, v1), RAD_OK);

    /* ... or compacted first and read as a chain. */
    gt::Cache m = gt::dup(c);
    T sidx(RAD_I32, { 1 }); sidx.n({ 0 });
    T na(RAD_I32, { 1 }); na.n({ 3 });
    T pt = gt::path_t(path);
    A kc; kc.t(m.cs).t(sidx).t(na).t(pt).i("off", gt::W - 2);
    REQUIRE_EQ(run("kv_window_compact", kc), RAD_OK);
    T q2(RAD_F32, { 3, gt::Hg, gt::K }), k2(RAD_F32, { 3, gt::Hg, gt::K }),
      v2(RAD_F32, { 3, gt::Hv, gt::V });
    REQUIRE_EQ(conv(m, next, 3, nullptr, nullptr, 3, q2, k2, v2), RAD_OK);
    CHECK(q1.mem == q2.mem);
    CHECK(k1.mem == k2.mem);
    CHECK(v1.mem == v2.mem);
    for (int64_t ch = 0; ch < gt::CD; ++ch)
        for (int64_t e = 0; e < gt::W - 2 + 3; ++e)
            CHECK_EQ(g.cs.at(ch * gt::DEPTH + e), m.cs.at(ch * gt::DEPTH + e));
}

TEST(gdn_tree_refusals) {
    const gt::Rows rows = gt::gen(4);
    const gt::Cache init = gt::fresh();
    T o(RAD_F32, { 4, gt::Hv, gt::V });
    {   /* a parent at or after its child */
        const std::vector<int> bad = { -1, 0, 2, 1 };
        gt::Cache c = gt::dup(init);
        CHECK_EQ(gt::fused(c, rows, 4, &bad, nullptr, 1, o), (int)RAD_E_INVAL);
    }
    {   /* a tree names this step's row states: the per-candidate form has none */
        const std::vector<int> par = { -1, 0, 0, 1 };
        gt::Cache c = gt::dup(init);
        CHECK_EQ(gt::fused(c, rows, 4, &par, nullptr, 1, o, "per_candidate"), (int)RAD_E_UNSUPPORTED);
    }
    {   /* more rows than the factor block holds (tmax 7) */
        const gt::Rows big = gt::gen(8);
        std::vector<int> par(8, -2);
        par[0] = -1;
        T ob(RAD_F32, { 8, gt::Hv, gt::V });
        gt::Cache c = gt::dup(init);
        CHECK_EQ(gt::fused(c, big, 8, &par, nullptr, 1, ob), (int)RAD_E_SHAPE);
    }
    {   /* a committed row past the records the block holds */
        const std::vector<int> path = { 0, 9 };
        gt::Cache c = gt::dup(init);
        CHECK_EQ(gt::fused(c, rows, 4, nullptr, &path, 2, o), (int)RAD_E_SHAPE);
    }
    {   /* num_accepted beyond spec_path's pitch */
        const std::vector<int> path = { 0, 1 };
        gt::Cache c = gt::dup(init);
        CHECK_EQ(gt::fused(c, rows, 4, nullptr, &path, 3, o), (int)RAD_E_SHAPE);
    }
}

/* kv_window_compact: entry off + t takes the entry of row path[t], t ascending, per channel and
 * only for the sequences whose num_accepted says so; i32 and bf16 windows alike. */
TEST(op_kv_window_compact_moves_committed_rows_down) {
    for (uint32_t dt : { (uint32_t)RAD_I32, (uint32_t)RAD_BF16 }) {
        const int64_t S = 3, C = 3, E = 9, off = 2;
        T cache(dt, { S, C, E });
        for (int64_t i = 0; i < S * C * E; ++i) st_dt(dt, cache.t.data, i, (float)(i % 100));
        const T before = gt::dup(cache);
        T sidx(RAD_I32, { 3, 2 }); sidx.n({ 0, 7, 1, 7, -1, 7 });   /* stride 2: column 0 is the slot; -1 pads */
        T nacc(RAD_I32, { 3 }); nacc.n({ 4, 1, 3 });
        T path(RAD_I32, { 3, 5 });
        path.n({ 0, 2, 3, 5, 0, 0, 1, 2, 3, 4, 0, 1, 4, 0, 0 });
        A a; a.t(cache).t(sidx).t(nacc).t(path).i("off", off);
        CHECK_OK(run("kv_window_compact", a));
        for (int64_t s = 0; s < S; ++s)
            for (int64_t c = 0; c < C; ++c)
                for (int64_t e = 0; e < E; ++e) {
                    const int64_t i = (s * C + c) * E + e;
                    float want = before.at(i);
                    if (s == 0 && e == off + 1) want = before.at((s * C + c) * E + off + 2);
                    if (s == 0 && e == off + 2) want = before.at((s * C + c) * E + off + 3);
                    if (s == 0 && e == off + 3) want = before.at((s * C + c) * E + off + 5);
                    CHECK_EQ(cache.at(i), want);
                }
        /* A path that does not ascend is refused (in place it would read an overwritten entry). */
        T bad(RAD_I32, { 3, 5 });
        bad.n({ 0, 0, 3, 5, 0, 0, 1, 2, 3, 4, 0, 1, 4, 0, 0 });
        A b; b.t(cache).t(sidx).t(nacc).t(bad).i("off", off);
        CHECK_EQ(run("kv_window_compact", b), (int)RAD_E_INVAL);
    }
}

/* ================================================================== mtp_tree_grow
 * A serial draft head's best-first tree (abi/rad_mtp_tree.h), by hand: M = 1, R = 2 rounds, N = 3
 * nodes, W = 2 frontier rows. The log-softmax of the eight logits 3,2,1,0,-1,-2,-3,-4 is -0.4584,
 * -1.4584, -2.4584 for its first three columns (a shift does not change it). */
namespace {
const int64_t kTgW = 2, kTgR = 2, kTgN = 3;
int tg_state_words(int64_t N, int64_t W, int64_t R) { return (int)(4 + W + 6 * (N + 1) + 4 * 8 * W * R); }

int tg_launch(int round, T& didx, T& dval, T& st, T& tok, T& par, T* hpos = nullptr, T* hrow = nullptr,
              T* tab = nullptr, T* sel = nullptr, T* nsel = nullptr, T* vmask = nullptr,
              T* rb = nullptr, int64_t N = kTgN, int64_t R = kTgR) {
    A b;
    b.t(didx).t(dval).t(st);
    if (hpos) b.t(*hpos); else b.tnull();
    if (hrow) b.t(*hrow); else b.tnull();
    if (tab) b.t(*tab); else b.tnull();
    b.t(tok).t(par);
    if (sel) b.t(*sel).t(*nsel).t(*vmask).t(*rb);
    b.i("M", 1).i("round", round).i("rounds", R).i("nodes", N).i("width", kTgW);
    if (sel) b.i("block_size", 4).i("sel_pitch", (int)sel->t.shape[1]);
    return run("mtp_tree_grow", b);
}
}  /* namespace */

TEST(mtp_tree_grow_takes_the_best_of_the_pool_and_hangs_it_on_its_parent) {
    const int64_t sw = tg_state_words(kTgN, kTgW, kTgR);
    T st(RAD_I32, { 1, sw });
    T tok(RAD_I32, { 1, kTgN }), par(RAD_I32, { 1, kTgN });
    const std::vector<float> steep = { 3, 2, 1, 0, -1, -2, -3, -4 };
    const std::vector<float> flat  = { 0, 0, 0, 0, 0, 0, 0, 0 };

    /* Round 1: the root's two best are nodes 1 and 2, in column order. */
    T d1(RAD_I32, { 1, 8 }), v1(RAD_F32, { 1, 8 });
    d1.n({ 10, 11, 12, 13, 14, 15, 16, 17 });
    v1.f(steep);
    CHECK_OK(tg_launch(1, d1, v1, st, tok, par));
    CHECK_EQ(tok.iat(0), 10); CHECK_EQ(par.iat(0), 0);
    CHECK_EQ(tok.iat(1), 11); CHECK_EQ(par.iat(1), 0);

    /* Round 2, the last: node 1's best child (cum -0.9168) beats the root's third (-2.4584). */
    {
        T st2 = clone_t(st);
        T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
        d2.n({ 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37 });
        std::vector<float> both = steep; both.insert(both.end(), steep.begin(), steep.end());
        v2.f(both);
        CHECK_OK(tg_launch(2, d2, v2, st2, tok, par));
        CHECK_EQ(tok.iat(0), 10); CHECK_EQ(par.iat(0), 0);
        CHECK_EQ(tok.iat(1), 11); CHECK_EQ(par.iat(1), 0);
        CHECK_EQ(tok.iat(2), 20); CHECK_EQ(par.iat(2), 1);
    }
    /* Flat children (-2.079 each, so cum -2.537) lose to the root's third candidate (-2.4584): the
     * pool carries the round-1 leftovers, and the new node hangs off the root. */
    {
        T st2 = clone_t(st);
        T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
        d2.n({ 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37 });
        std::vector<float> both = flat; both.insert(both.end(), flat.begin(), flat.end());
        v2.f(both);
        CHECK_OK(tg_launch(2, d2, v2, st2, tok, par));
        CHECK_EQ(tok.iat(2), 12); CHECK_EQ(par.iat(2), 0);
    }
    /* A pool that runs dry (the second pass's ids are all negative) pads with node 1's token under
     * the root: 12 nodes over 3 rounds, of which only the root's eight candidates exist. */
    {
        const int64_t N = 12, R = 3;
        T stn(RAD_I32, { 1, tg_state_words(N, kTgW, R) });
        T tokn(RAD_I32, { 1, N }), parn(RAD_I32, { 1, N });
        T dn(RAD_I32, { 2, 8 }), vn(RAD_F32, { 2, 8 });
        dn.n(std::vector<int64_t>(16, -1));
        vn.f(std::vector<float>(16, 0.0f));
        CHECK_OK(tg_launch(1, d1, v1, stn, tokn, parn, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, N, R));
        CHECK_OK(tg_launch(2, dn, vn, stn, tokn, parn, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, N, R));
        CHECK_OK(tg_launch(3, dn, vn, stn, tokn, parn, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, nullptr, N, R));
        /* the eight real candidates in order of log probability (the column order), then four pads */
        for (int64_t j = 0; j < 8; ++j) { CHECK_EQ(tokn.iat(j), 10 + j); CHECK_EQ(parn.iat(j), 0); }
        for (int64_t j = 8; j < N; ++j) { CHECK_EQ(tokn.iat(j), 10); CHECK_EQ(parn.iat(j), 0); }
    }
}

/* --draft-coupling with a stage-2 tree (abi/rad_mtp_tree_rrs.h): the same op with the draw, card and
 * state-plane operands draws every frontier row's children without replacement and counts the
 * children it creates into the node's state, in draft order. A greedy draw or a draw staged for
 * another card is the uncoupled op, word for word. M = 1, R = 3, N = 5, W = 2. */
namespace {
int tg_coupled(int round, T& didx, T& dval, T& st, T& tok, T& par, T* draw, T* card, T* plane,
               int rheo_m = -1) {
    A b;
    b.t(didx).t(dval).t(st).tnull().tnull().tnull().t(tok).t(par);
    b.tnull().tnull().tnull().tnull().tnull().tnull();
    if (draw) b.t(*draw).t(*card).t(*plane); else b.tnull().tnull().tnull();
    b.i("M", 1).i("round", round).i("rounds", 3).i("nodes", 5).i("width", kTgW).i("n_vocab", 1000);
    if (rheo_m >= 0) b.i("rheo_m", rheo_m);
    return run("mtp_tree_grow", b);
}
}  /* namespace */

TEST(mtp_tree_grow_with_a_coupled_draw_creates_each_nodes_children_in_draft_order) {
    const int64_t N = 5, R = 3, sw = tg_state_words(N, kTgW, R), SW = (int64_t)RAD_MTP_COUPLE_STATE_WORDS;
    const std::vector<float> steep = { 3, 2, 1, 0.5f, -1, -2, -3, -4 };
    auto run_tree = [&](uint32_t mode, int32_t card, bool coupled, T* tok_out, T* par_out, T* plane) {
        T st(RAD_I32, { 1, sw });
        T tok(RAD_I32, { 1, N }), par(RAD_I32, { 1, N });
        T draw(RAD_U32, { 1, (int64_t)RAD_MTP_COUPLE_DRAW_WORDS }), cs(RAD_I32, { 1 });
        RadMtpCoupleDraw d{};
        d.seed = 77; d.base = 5; d.seq = 1; d.card = 0; d.mode = mode;
        std::memcpy(draw.mem.data(), &d, sizeof d);
        cs.n({ card });
        for (int round = 1; round <= (int)R; ++round) {
            const int64_t rows = round == 1 ? 1 : kTgW;
            T di(RAD_I32, { rows, 8 }), dv(RAD_F32, { rows, 8 });
            std::vector<int64_t> ids;
            std::vector<float> vs;
            for (int64_t r = 0; r < rows; ++r)
                for (int c = 0; c < 8; ++c) { ids.push_back(100 * (round + r) + c); vs.push_back(steep[(size_t)c]); }
            di.n(ids); dv.f(vs);
            CHECK_OK(tg_coupled(round, di, dv, st, tok, par, coupled ? &draw : nullptr,
                                coupled ? &cs : nullptr, coupled ? plane : nullptr));
        }
        *tok_out = clone_t(tok);
        *par_out = clone_t(par);
    };
    T tokc(RAD_I32, { 1, N }), parc(RAD_I32, { 1, N }), plane(RAD_U32, { N, 1, SW });
    run_tree(1, 0, true, &tokc, &parc, &plane);
    /* node states: Q over the eight, children as the tree has them, in node order */
    int coupled_nodes = 0;
    for (int64_t f = 0; f < N; ++f) {
        RadMtpCoupleState s;
        std::memcpy(&s, plane.mem.data() + (size_t)f * (size_t)SW * 4, sizeof s);
        std::vector<int64_t> kids;
        for (int64_t j = 0; j < N; ++j)
            if (parc.iat(j) == f) kids.push_back(tokc.iat(j));
        if (kids.empty()) { CHECK(s.nkids == 0); continue; }
        ++coupled_nodes;
        CHECK_EQ(s.status, 1);
        CHECK(rad_mtp_couple_state_valid(s, 1000));
        CHECK_EQ((size_t)s.nkids, kids.size());
        for (size_t q = 0; q < kids.size(); ++q) CHECK_EQ((int64_t)s.ids[q], kids[q]);
    }
    CHECK(coupled_nodes >= 2);
    /* a greedy draw: the uncoupled op, states untouched */
    T tok0(RAD_I32, { 1, N }), par0(RAD_I32, { 1, N }), plane0(RAD_U32, { N, 1, SW });
    T tokg(RAD_I32, { 1, N }), parg(RAD_I32, { 1, N }), planeg(RAD_U32, { N, 1, SW });
    run_tree(0, 0, true, &tokg, &parg, &planeg);
    run_tree(1, 0, false, &tok0, &par0, &plane0);
    for (int64_t j = 0; j < N; ++j) {
        CHECK_EQ(tokg.iat(j), tok0.iat(j));
        CHECK_EQ(parg.iat(j), par0.iat(j));
    }
    for (size_t w = 0; w < (size_t)(N * SW); ++w) {
        uint32_t x;
        std::memcpy(&x, planeg.mem.data() + w * 4, 4);
        CHECK_EQ(x, 0u);
    }
}

/* --draft-siblings rheo: the same op with `rheo_m` = m writes states whose first m children are the m
 * largest q (the slot order) and whose child m is the one tail draw (scheme m + 1); without the
 * param the states are the Gumbel ones (scheme 0). */
TEST(mtp_tree_grow_with_rheo_m_draws_the_top_q_children_and_one_tail_child) {
    const int64_t N = 5, R = 3, sw = tg_state_words(N, kTgW, R), SW = (int64_t)RAD_MTP_COUPLE_STATE_WORDS;
    const std::vector<float> steep = { 3, 2, 1, 0.5f, -1, -2, -3, -4 };
    for (int m = 0; m <= 2; ++m) {
        T st(RAD_I32, { 1, sw });
        T tok(RAD_I32, { 1, N }), par(RAD_I32, { 1, N });
        T draw(RAD_U32, { 1, (int64_t)RAD_MTP_COUPLE_DRAW_WORDS }), cs(RAD_I32, { 1 }), plane(RAD_U32, { N, 1, SW });
        RadMtpCoupleDraw d{};
        d.seed = 77 + (uint64_t)m; d.base = 5; d.seq = 1; d.card = 0; d.mode = 1;
        std::memcpy(draw.mem.data(), &d, sizeof d);
        cs.n({ 0 });
        for (int round = 1; round <= (int)R; ++round) {
            const int64_t rows = round == 1 ? 1 : kTgW;
            T di(RAD_I32, { rows, 8 }), dv(RAD_F32, { rows, 8 });
            std::vector<int64_t> ids;
            std::vector<float> vs;
            for (int64_t r = 0; r < rows; ++r)
                for (int c = 0; c < 8; ++c) { ids.push_back(100 * (round + r) + c); vs.push_back(steep[(size_t)c]); }
            di.n(ids); dv.f(vs);
            CHECK_OK(tg_coupled(round, di, dv, st, tok, par, &draw, &cs, &plane, m));
        }
        int nodes_with_kids = 0, total_kids = 0;
        for (int64_t f = 0; f < N; ++f) {
            RadMtpCoupleState s;
            std::memcpy(&s, plane.mem.data() + (size_t)f * (size_t)SW * 4, sizeof s);
            std::vector<int64_t> kids;   /* in node order, repeats dropped (a short pool pads with node 1's token) */
            for (int64_t j = 0; j < N; ++j)
                if (par.iat(j) == f && std::find(kids.begin(), kids.end(), tok.iat(j)) == kids.end())
                    kids.push_back(tok.iat(j));
            if (kids.empty()) continue;
            ++nodes_with_kids;
            CHECK_EQ(s.status, 1);
            CHECK(rad_mtp_couple_state_valid(s, 1000));
            CHECK_EQ(s.scheme, m + 1);
            CHECK(kids.size() <= (size_t)(m + 1));
            /* a node counts the children it really made, in draft order; a placeholder (a dry pool's copy,
             * score -inf, e.g. the second root child when m = 0 leaves one candidate) counts none */
            CHECK(s.nkids >= 0 && (size_t)s.nkids <= kids.size());
            if (m > 0) CHECK_EQ((size_t)s.nkids, kids.size());
            total_kids += s.nkids;
            for (int q = 0; q < s.nkids; ++q) CHECK_EQ((int64_t)s.ids[q], kids[(size_t)q]);
            /* slots 0 .. m-1: the largest q in descending order (steep is descending in the column) */
            const int nt = std::min<int>(m, s.count);
            for (int q = 1; q < nt; ++q) CHECK(s.q[q - 1] >= s.q[q]);
        }
        CHECK(nodes_with_kids >= 2);
        CHECK(total_kids >= 3);
    }
}

/* The hidden-state routing: every frontier row continues from its PARENT's head output, found in a
 * per-node store (row `s * Q + node`, Q = 1 + (R - 1) * W), because the pool may hang a node off a
 * node of any earlier pass. M = 2 sequences, W = 2, R = 4, N = 6, Q = 7. */
namespace {
int tg_route(int round, int64_t M, T& didx, T& dval, T& st, T& tok, T& par, T& src, T& dst,
             int64_t N = 6, int64_t R = 4) {
    A b;
    b.t(didx).t(dval).t(st).tnull().tnull().tnull().t(tok).t(par);
    b.tnull().tnull().tnull().tnull().t(src).t(dst);
    b.i("M", (int)M).i("round", round).i("rounds", R).i("nodes", N).i("width", kTgW);
    return run("mtp_tree_grow", b);
}
std::vector<int64_t> tg_ids(int64_t base) {
    std::vector<int64_t> v;
    for (int64_t i = 0; i < 8; ++i) v.push_back(base + i);
    return v;
}
}  /* namespace */

TEST(mtp_tree_grow_routes_each_frontier_row_to_its_parents_hidden_state) {
    const int64_t M = 2, N = 6, R = 4, Q = 1 + (R - 1) * kTgW;
    T st(RAD_I32, { M, tg_state_words(N, kTgW, R) });
    T tok(RAD_I32, { M, N }), par(RAD_I32, { M, N });
    T src(RAD_I32, { M * kTgW }), dst(RAD_I32, { M * kTgW });
    const std::vector<float> steep = { 3, 2, 1, 0, -1, -2, -3, -4 };
    const std::vector<float> peak  = { 3, -40, -40, -40, -40, -40, -40, -40 };
    const std::vector<float> root  = { 3, 2, -40, -41, -42, -43, -44, -45 };

    /* Round 1: both sequences' roots (store row s * Q) have nodes 1 and 2. */
    {
        T d(RAD_I32, { M, 8 }), v(RAD_F32, { M, 8 });
        std::vector<int64_t> di = tg_ids(10);
        const std::vector<int64_t> d2 = tg_ids(10);
        di.insert(di.end(), d2.begin(), d2.end());
        std::vector<float> vf = root;
        vf.insert(vf.end(), root.begin(), root.end());
        d.n(di); v.f(vf);
        CHECK_OK(tg_route(1, M, d, v, st, tok, par, src, dst));
        for (int64_t s = 0; s < M; ++s) {
            CHECK_EQ(src.iat(s * 2 + 0), s * Q);     CHECK_EQ(src.iat(s * 2 + 1), s * Q);
            CHECK_EQ(dst.iat(s * 2 + 0), s * Q + 1); CHECK_EQ(dst.iat(s * 2 + 1), s * Q + 2);
        }
    }
    /* Round 2 over rows s * 2 + w (w the slot of node 1 + w). Sequence 0: row 0 offers nothing, so
     * both new nodes (3, 4) hang off node 2, row 1. Sequence 1: row 0 is peaked (its best child is
     * the best of the pool) and row 1 steep: node 3 off node 1, node 4 off node 2. */
    {
        T d(RAD_I32, { 4, 8 }), v(RAD_F32, { 4, 8 });
        std::vector<int64_t> di(8, -1);
        for (const int64_t base : { 20, 30, 40 }) {
            const auto x = tg_ids(base);
            di.insert(di.end(), x.begin(), x.end());
        }
        std::vector<float> vf(8, 0.0f);
        vf.insert(vf.end(), steep.begin(), steep.end());
        vf.insert(vf.end(), peak.begin(), peak.end());
        vf.insert(vf.end(), steep.begin(), steep.end());
        d.n(di); v.f(vf);
        CHECK_OK(tg_route(2, M, d, v, st, tok, par, src, dst));
        CHECK_EQ(par.iat(0 * N + 0), 2); CHECK_EQ(par.iat(0 * N + 1), 2);
        CHECK_EQ(src.iat(0), 0 * Q + 2); CHECK_EQ(src.iat(1), 0 * Q + 2);
        CHECK_EQ(dst.iat(0), 0 * Q + 3); CHECK_EQ(dst.iat(1), 0 * Q + 4);
        CHECK_EQ(par.iat(1 * N + 0), 1); CHECK_EQ(par.iat(1 * N + 1), 2);
        CHECK_EQ(src.iat(2), 1 * Q + 1); CHECK_EQ(src.iat(3), 1 * Q + 2);
        CHECK_EQ(dst.iat(2), 1 * Q + 3); CHECK_EQ(dst.iat(3), 1 * Q + 4);
    }
    /* Round 3: nothing new on the frontier, so the best leftovers are node 2's later children,
     * whose parent sits in an EARLIER pass than the last: the source is still node 2's store row. */
    {
        T d(RAD_I32, { 4, 8 }), v(RAD_F32, { 4, 8 });
        d.n(std::vector<int64_t>(32, -1));
        v.f(std::vector<float>(32, 0.0f));
        CHECK_OK(tg_route(3, M, d, v, st, tok, par, src, dst));
        for (int64_t s = 0; s < M; ++s) {
            CHECK_EQ(par.iat(s * N + 0), 2); CHECK_EQ(par.iat(s * N + 1), 2);
            CHECK_EQ(src.iat(s * 2 + 0), s * Q + 2); CHECK_EQ(src.iat(s * 2 + 1), s * Q + 2);
            CHECK_EQ(dst.iat(s * 2 + 0), s * Q + 5); CHECK_EQ(dst.iat(s * 2 + 1), s * Q + 6);
        }
    }
    /* A pool that runs dry pads with a node under the root, whose row is the root's. */
    {
        T st1(RAD_I32, { 1, tg_state_words(N, kTgW, R) });
        T tk(RAD_I32, { 1, N }), pr(RAD_I32, { 1, N }), s1(RAD_I32, { 2 }), d1(RAD_I32, { 2 });
        T d(RAD_I32, { 1, 8 }), v(RAD_F32, { 1, 8 });
        std::vector<int64_t> di(8, -1);
        di[0] = 7;
        d.n(di); v.f(std::vector<float>(8, 1.0f));
        CHECK_OK(tg_route(1, 1, d, v, st1, tk, pr, s1, d1));
        CHECK_EQ(s1.iat(0), 0); CHECK_EQ(s1.iat(1), 0);
        CHECK_EQ(d1.iat(0), 1); CHECK_EQ(d1.iat(1), 2);
    }
    /* A row past the new nodes (the node budget is spent) reads the root's row and writes nowhere:
     * N = 3, R = 3, W = 2, so round 2 can create one node. Q = 5. */
    {
        const int64_t N3 = 3, R3 = 3;
        T st1(RAD_I32, { 1, tg_state_words(N3, kTgW, R3) });
        T tk(RAD_I32, { 1, N3 }), pr(RAD_I32, { 1, N3 }), s1(RAD_I32, { 2 }), d1(RAD_I32, { 2 });
        T d(RAD_I32, { 1, 8 }), v(RAD_F32, { 1, 8 });
        d.n(tg_ids(10)); v.f(root);
        CHECK_OK(tg_route(1, 1, d, v, st1, tk, pr, s1, d1, N3, R3));
        T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
        std::vector<int64_t> di = tg_ids(20);
        const auto none = std::vector<int64_t>(8, -1);
        di.insert(di.end(), none.begin(), none.end());
        std::vector<float> vf = peak;
        vf.insert(vf.end(), 8, 0.0f);
        d2.n(di); v2.f(vf);
        CHECK_OK(tg_route(2, 1, d2, v2, st1, tk, pr, s1, d1, N3, R3));
        CHECK_EQ(s1.iat(0), 1); CHECK_EQ(s1.iat(1), 0);
        CHECK_EQ(d1.iat(0), 3); CHECK_EQ(d1.iat(1), -1);
    }
}

TEST(mtp_tree_grow_writes_the_next_passs_tree_attention_operands) {
    const int64_t sw = tg_state_words(kTgN, kTgW, kTgR);
    T st(RAD_I32, { 1, sw });
    T tok(RAD_I32, { 1, kTgN }), par(RAD_I32, { 1, kTgN });
    T d1(RAD_I32, { 1, 8 }), v1(RAD_F32, { 1, 8 });
    d1.n({ 10, 11, 12, 13, 14, 15, 16, 17 });
    v1.f({ 3, 2, 1, 0, -1, -2, -3, -4 });
    /* The pass's rows 0..3 at positions 4..7; the sequence's draft row is row 3, head index 9. */
    T hpos(RAD_I32, { 4 }), hrow(RAD_I32, { 1 });
    hpos.n({ 6, 7, 8, 9 });
    hrow.n({ 3 });
    T tab(RAD_I32, { 1, 8 });
    tab.n({ 40, 41, 42, 43, 44, 45, 46, 47 });
    T sel(RAD_I32, { 2, 4 }), nsel(RAD_I32, { 2 }), vmask(RAD_I64, { 2 }), rb(RAD_I32, { 1 });
    CHECK_OK(tg_launch(1, d1, v1, st, tok, par, &hpos, &hrow, &tab, &sel, &nsel, &vmask, &rb));
    /* head 9, block 4: rb = 8, two committed pages below it. */
    CHECK_EQ(rb.iat(0), 8);
    CHECK_EQ(nsel.iat(0), 2); CHECK_EQ(nsel.iat(1), 2);
    CHECK_EQ(sel.iat(0), 40); CHECK_EQ(sel.iat(1), 41); CHECK_EQ(sel.iat(2), -1);
    /* keys at 8 and 9 (the committed tail, the root among them) and the node's own flat slot
     * head + id: bit (head - rb) + id */
    CHECK_EQ(vmask.iat(0), 3 | (1 << 2));
    CHECK_EQ(vmask.iat(1), 3 | (1 << 3));
    /* More pages than the list holds: the last four. */
    T tab2(RAD_I32, { 1, 16 });
    tab2.n({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
    hpos.n({ 6, 7, 8, 33 });
    CHECK_OK(tg_launch(1, d1, v1, st, tok, par, &hpos, &hrow, &tab2, &sel, &nsel, &vmask, &rb));
    CHECK_EQ(rb.iat(0), 32);      /* eight pages below it, four listed */
    CHECK_EQ(nsel.iat(0), 4);
    CHECK_EQ(sel.iat(0), 4); CHECK_EQ(sel.iat(3), 7);
    CHECK_EQ(vmask.iat(0), 3 | (1 << 2));
    /* A bad configuration is refused: a width of zero, a round past the last, a pitch under N. */
    T tok2(RAD_I32, { 1, 2 }), par2(RAD_I32, { 1, 2 });
    CHECK(tg_launch(1, d1, v1, st, tok2, par2) != RAD_OK);
    CHECK(tg_launch(3, d1, v1, st, tok, par) != RAD_OK);
}

/* ================================================================== mtp_tree_grow, confidence pruning
 * abi/rad_mtp_tree.h: `prune_abs` / `prune_rel` (both optional, 0 = off). A candidate becomes a pool
 * entry only if its cumulative probability >= prune_abs and its probability >= prune_rel * its
 * parent's best candidate's; every frontier row's best candidate (the trunk) is never pruned. The
 * pruned tree's unfilled entries are inert: token -1, parent 0. Logits 3,2,1,0,-1,-2,-3,-4 are
 * probabilities .632 .233 .0855 .0315 ... (ratio to the best e^-1 = .368, e^-2 = .135, ...). */
namespace {
struct TpGrow {
    int64_t M, N, R, W;
    T st, tok, par;
    TpGrow(int64_t m, int64_t n, int64_t r, int64_t w)
        : M(m), N(n), R(r), W(w), st(RAD_I32, { m, tg_state_words(n, w, r) }),
          tok(RAD_I32, { m, n }), par(RAD_I32, { m, n }) {}
    /* One round over the candidate rows (M for round 1, M * W after); a negative threshold passes
     * no parameter at all. */
    int run_round(int round, T& didx, T& dval, double pa = -1, double pr = -1) {
        A b;
        b.t(didx).t(dval).t(st).tnull().tnull().tnull().t(tok).t(par);
        b.i("M", (int)M).i("round", round).i("rounds", (int)R).i("nodes", (int)N).i("width", (int)W);
        if (pa >= 0) b.d("prune_abs", pa);
        if (pr >= 0) b.d("prune_rel", pr);
        return run("mtp_tree_grow", b);
    }
};
/* Candidate rows of ids base + 10 * row + c, row r's logits from lg[r]. */
void tp_rows(T& d, T& v, const std::vector<std::vector<float>>& lg, int base) {
    std::vector<int64_t> di;
    std::vector<float> vf;
    for (size_t r = 0; r < lg.size(); ++r)
        for (int c = 0; c < 8; ++c) {
            di.push_back(base + 10 * (int64_t)r + c);
            vf.push_back(lg[r][(size_t)c]);
        }
    d.n(di); v.f(vf);
}
const std::vector<float> kSteep = { 3, 2, 1, 0, -1, -2, -3, -4 };
const std::vector<float> kPeak  = { 5, -40, -40, -40, -40, -40, -40, -40 };
}  /* namespace */

TEST(mtp_tree_grow_with_the_thresholds_off_is_the_same_walk) {
    /* Random candidate rows over three rounds, W = 2 and 3: passing no threshold and passing zeros
     * give the same tree and the same state words, bit for bit. */
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (int)(seed >> 8); };
    for (int64_t W = 2; W <= 3; ++W)
        for (int trial = 0; trial < 40; ++trial) {
            const int64_t N = 9, R = 3, M = 2;
            TpGrow a(M, N, R, W), b(M, N, R, W);
            for (int round = 1; round <= (int)R; ++round) {
                const int64_t rows = round == 1 ? M : M * W;
                std::vector<std::vector<float>> lg;
                for (int64_t r = 0; r < rows; ++r) {
                    std::vector<float> row;
                    for (int c = 0; c < 8; ++c) row.push_back((float)(rnd() % 400) / 50.0f - 4.0f);
                    lg.push_back(row);
                }
                T d(RAD_I32, { rows, 8 }), v(RAD_F32, { rows, 8 });
                tp_rows(d, v, lg, 100 * round);
                CHECK_OK(a.run_round(round, d, v));
                CHECK_OK(b.run_round(round, d, v, 0.0, 0.0));
            }
            CHECK(std::memcmp(a.st.t.data, b.st.t.data, a.st.mem.size() - 8) == 0);
            CHECK(std::memcmp(a.tok.t.data, b.tok.t.data, a.tok.mem.size() - 8) == 0);
            CHECK(std::memcmp(a.par.t.data, b.par.t.data, a.par.mem.size() - 8) == 0);
        }
}

TEST(mtp_tree_grow_pruned_hard_is_the_chain_of_best_children) {
    /* A threshold nothing but the best candidate meets: node j hangs off node j - 1 and carries
     * column 0 (the largest logit) of its parent's row; the nodes past the chain are inert. */
    const int64_t N = 6, R = 3, W = 2;
    for (int mode = 0; mode < 3; ++mode) {
        const double pa = mode == 0 ? 0.99 : 0.0, pr = mode == 1 ? 1.0 : mode == 2 ? 0.99 : 0.0;
        TpGrow g(1, N, R, W);
        T d1(RAD_I32, { 1, 8 }), v1(RAD_F32, { 1, 8 });
        tp_rows(d1, v1, { kSteep }, 10);
        CHECK_OK(g.run_round(1, d1, v1, pa, pr));
        /* one new node, the list ends after it: a token of -1 */
        CHECK_EQ(g.tok.iat(0), 10); CHECK_EQ(g.par.iat(0), 0);
        CHECK_EQ(g.tok.iat(1), -1);
        /* round 2 reads one live frontier row (node 1) and one inert */
        T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
        tp_rows(d2, v2, { kSteep, kSteep }, 20);
        CHECK_OK(g.run_round(2, d2, v2, pa, pr));
        CHECK_EQ(g.tok.iat(0), 20); CHECK_EQ(g.par.iat(0), 1);
        CHECK_EQ(g.tok.iat(1), -1);
        T d3(RAD_I32, { 2, 8 }), v3(RAD_F32, { 2, 8 });
        tp_rows(d3, v3, { kSteep, kSteep }, 30);
        CHECK_OK(g.run_round(3, d3, v3, pa, pr));
        const int64_t chain_tok[3] = { 10, 20, 30 };
        for (int64_t j = 0; j < 3; ++j) { CHECK_EQ(g.tok.iat(j), chain_tok[j]); CHECK_EQ(g.par.iat(j), j); }
        for (int64_t j = 3; j < N; ++j) { CHECK_EQ(g.tok.iat(j), -1); CHECK_EQ(g.par.iat(j), 0); }
    }
}

TEST(mtp_tree_grow_pruned_keeps_the_candidates_both_thresholds_pass) {
    const int64_t N = 5, R = 2, W = 2;
    /* Round 2's rows: node 1's children are peaked (only the trunk), node 2's likewise. */
    T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
    tp_rows(d2, v2, { kPeak, kPeak }, 20);
    T d1(RAD_I32, { 1, 8 }), v1(RAD_F32, { 1, 8 });
    tp_rows(d1, v1, { kSteep }, 10);
    /* Unpruned: nodes 1, 2 (the root's two best), then node 1's child (cum -.458), node 2's child
     * (-1.458), then the root's third (-2.458, token 12): all five. */
    {
        TpGrow g(1, N, R, W);
        CHECK_OK(g.run_round(1, d1, v1));
        CHECK_OK(g.run_round(2, d2, v2));
        const int64_t t[5] = { 10, 11, 20, 30, 12 }, p[5] = { 0, 0, 1, 2, 0 };
        for (int j = 0; j < 5; ++j) { CHECK_EQ(g.tok.iat(j), t[j]); CHECK_EQ(g.par.iat(j), p[j]); }
    }
    /* rel .3: the root's second (ratio .368) stays, its third (.135) goes: five nodes, the fifth
     * inert. abs .2 says the same by cumulative probability (.233 stays, .0855 goes). */
    for (int mode = 0; mode < 2; ++mode) {
        TpGrow g(1, N, R, W);
        const double pa = mode == 1 ? 0.2 : 0.0, pr = mode == 0 ? 0.3 : 0.0;
        CHECK_OK(g.run_round(1, d1, v1, pa, pr));
        CHECK_EQ(g.tok.iat(0), 10); CHECK_EQ(g.tok.iat(1), 11);
        CHECK_OK(g.run_round(2, d2, v2, pa, pr));
        const int64_t t[5] = { 10, 11, 20, 30, -1 }, p[5] = { 0, 0, 1, 2, 0 };
        for (int j = 0; j < 5; ++j) { CHECK_EQ(g.tok.iat(j), t[j]); CHECK_EQ(g.par.iat(j), p[j]); }
    }
    /* abs .25 prunes the root's second (.233): node 2 is node 1's child, the rest inert. The
     * mid-round list is [10, -1]. Both thresholds together need both: rel .3 alone keeps the
     * second, abs .25 alone drops it, so together it is dropped. */
    for (int mode = 0; mode < 2; ++mode) {
        TpGrow g(1, N, R, W);
        const double pa = 0.25, pr = mode == 1 ? 0.3 : 0.0;
        CHECK_OK(g.run_round(1, d1, v1, pa, pr));
        CHECK_EQ(g.tok.iat(0), 10); CHECK_EQ(g.tok.iat(1), -1);
        CHECK_OK(g.run_round(2, d2, v2, pa, pr));
        const int64_t t[5] = { 10, 20, -1, -1, -1 }, p[5] = { 0, 1, 0, 0, 0 };
        for (int j = 0; j < 5; ++j) { CHECK_EQ(g.tok.iat(j), t[j]); CHECK_EQ(g.par.iat(j), p[j]); }
    }
}

TEST(mtp_tree_grow_pruned_never_drops_the_trunk_even_at_a_tiny_probability) {
    /* A flat row: eight candidates of probability 1/8. abs .5 prunes seven of them, but the best
     * (the first column of the largest logit) is kept, so node 1 exists and carries column 0; a
     * row whose largest logit is column 3 trunks through column 3. */
    const int64_t N = 3, R = 1, W = 1;
    {
        TpGrow g(1, N, R, W);
        T d(RAD_I32, { 1, 8 }), v(RAD_F32, { 1, 8 });
        tp_rows(d, v, { std::vector<float>(8, 0.0f) }, 10);
        CHECK_OK(g.run_round(1, d, v, 0.5, 0.0));
        CHECK_EQ(g.tok.iat(0), 10); CHECK_EQ(g.par.iat(0), 0);
        CHECK_EQ(g.tok.iat(1), -1); CHECK_EQ(g.tok.iat(2), -1);
    }
    {
        TpGrow g(1, N, R, W);
        T d(RAD_I32, { 1, 8 }), v(RAD_F32, { 1, 8 });
        tp_rows(d, v, { { 0, 0, 0, 9, 0, 0, 0, 0 } }, 10);
        CHECK_OK(g.run_round(1, d, v, 0.99, 1.0));
        CHECK_EQ(g.tok.iat(0), 13); CHECK_EQ(g.tok.iat(1), -1);
    }
    /* No valid candidate at all: node 1 is still made (token 0 under the root, as unpruned), so the
     * tree has its trunk's first node; the rest is inert. */
    {
        TpGrow g(1, N, R, W);
        T d(RAD_I32, { 1, 8 }), v(RAD_F32, { 1, 8 });
        d.n(std::vector<int64_t>(8, -1)); v.f(std::vector<float>(8, 0.0f));
        CHECK_OK(g.run_round(1, d, v, 0.5, 0.0));
        CHECK_EQ(g.tok.iat(0), 0); CHECK_EQ(g.par.iat(0), 0);
        CHECK_EQ(g.tok.iat(1), -1); CHECK_EQ(g.tok.iat(2), -1);
    }
}

TEST(mtp_tree_grow_pruned_pool_keeps_the_trunk_of_a_kept_node) {
    /* A frontier node that passed pruning is itself a parent: its own best child is trunk and
     * enters the pool whatever its cumulative probability. Node 2 (cum .233) has two children of
     * conditional probability .5; with abs .2 the best (cum .117) is kept as trunk and its twin
     * (same cum, not the best) is pruned. */
    const int64_t N = 6, R = 2, W = 2;
    TpGrow g(1, N, R, W);
    T d1(RAD_I32, { 1, 8 }), v1(RAD_F32, { 1, 8 });
    tp_rows(d1, v1, { kSteep }, 10);
    CHECK_OK(g.run_round(1, d1, v1, 0.2, 0.0));
    CHECK_EQ(g.tok.iat(0), 10); CHECK_EQ(g.tok.iat(1), 11);
    T d2(RAD_I32, { 2, 8 }), v2(RAD_F32, { 2, 8 });
    const std::vector<float> two = { 2, 2, -40, -40, -40, -40, -40, -40 };
    tp_rows(d2, v2, { kPeak, two }, 20);
    CHECK_OK(g.run_round(2, d2, v2, 0.2, 0.0));
    /* node 3: node 1's best child (cum .632), node 4: node 2's best child (trunk), then nothing
     * the thresholds pass: N = 6 leaves two inert. */
    const int64_t t[6] = { 10, 11, 20, 30, -1, -1 }, p[6] = { 0, 0, 1, 2, 0, 0 };
    for (int j = 0; j < 6; ++j) { CHECK_EQ(g.tok.iat(j), t[j]); CHECK_EQ(g.par.iat(j), p[j]); }
}

/* The pruned grower's structural guarantees over random widths, rounds, budgets, thresholds and
 * rows (some candidates invalid): the created nodes are a prefix and the entries past them are
 * inert (-1 under the root), a parent precedes its child within the head's depth, no two siblings
 * share a token, node 1 always exists, a frontier list ends at its first -1 (no hole in a round's
 * frontier), and thresholds at 1 give exactly the chain of every round's best candidate. */
TEST(mtp_tree_grow_pruned_fuzz_keeps_the_trees_structure_and_the_trunk) {
    uint32_t s = 0x9e3779b9u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (int)(s >> 8); };
    long heavy_chains = 0;
    for (int trial = 0; trial < 6000; ++trial) {
        const int W = 1 + rnd() % 4, R = 1 + rnd() % 5;
        const int N = std::max(1, R + rnd() % 12), K = RAD_MTP_TREE_K, PITCH = N, SP = 5;
        const bool heavy = rnd() % 4 == 0;
        const double pa = heavy ? 1.0 : (rnd() % 3 == 0 ? 0.0 : (rnd() % 1000) / 1000.0);
        double pr = heavy ? 1.0 : (rnd() % 3 == 0 ? 0.0 : (rnd() % 1000) / 1000.0);
        if (pa == 0.0 && pr == 0.0) pr = 0.3;     /* off is the unpruned walk, pinned elsewhere */
        std::vector<int> st((size_t)rad_mtp_tree_state_words(N, W, R), 0), table(64, 1);
        std::vector<int> tok((size_t)PITCH), par((size_t)PITCH);
        std::vector<int> best_tok;       /* heavy: the best candidate of each round's first row */
        for (int round = 1; round <= R; ++round) {
            const int rows = round == 1 ? 1 : W;
            std::vector<int> idx((size_t)rows * K);
            std::vector<float> val((size_t)rows * K);
            for (int r = 0; r < rows; ++r) {
                std::set<int> used;
                for (int c = 0; c < K; ++c) {
                    int id;
                    do id = rnd() % 5000; while (used.count(id));
                    used.insert(id);
                    idx[(size_t)r * K + c] = rnd() % 9 == 0 ? -1 : id;
                    val[(size_t)r * K + c] = (float)(rnd() % 800) / 60.0f - 6.0f;
                }
            }
            RadMtpTreeCfg c{};
            c.round = round; c.rounds = R; c.nodes = N; c.width = W; c.block = 4;
            c.sel_pitch = SP; c.pitch = PITCH;
            rad_mtp_tree_set_prune(&c, pa, pr);
            std::vector<int> sel((size_t)W * SP), ns((size_t)W), src((size_t)W), dst((size_t)W);
            std::vector<long long> vm((size_t)W);
            int rb = 0;
            CHECK_EQ(rad_mtp_tree_grow_seq(&c, st.data(), idx.data(), val.data(), 0, K, 1,
                                           round == 1 ? 7 : -1, table.data(), tok.data(),
                                           par.data(), sel.data(), ns.data(), vm.data(), &rb, 0,
                                           src.data(), dst.data()), 0);
            if (round < R) {
                bool hole = false;
                int live = 0;
                for (int w = 0; w < W; ++w) {
                    if (tok[(size_t)w] < 0) hole = true;
                    else { CHECK(!hole); ++live; }
                }
                if (heavy) CHECK(live <= 1);
            }
            if (heavy && round == 1) {
                int best = -1;
                float mx = -INFINITY;
                for (int cc = 0; cc < K; ++cc)
                    if (idx[(size_t)cc] >= 0 && val[(size_t)cc] > mx) { mx = val[(size_t)cc]; best = cc; }
                CHECK_EQ(tok[0], best >= 0 ? idx[(size_t)best] : 0);
            }
        }
        int n = 0;
        while (n < N && tok[(size_t)n] >= 0) ++n;
        CHECK(n >= 1);
        for (int j = n; j < N; ++j) { CHECK_EQ(tok[(size_t)j], -1); CHECK_EQ(par[(size_t)j], 0); }
        std::set<std::pair<int, int>> sib;
        for (int j = 1; j <= n; ++j) {
            const int p = par[(size_t)j - 1];
            CHECK(p >= 0 && p < j);
            int depth = 0;
            for (int x = j; x > 0; x = par[(size_t)x - 1]) ++depth;
            CHECK(depth <= R);
            CHECK(sib.insert({ p, tok[(size_t)j - 1] }).second);
            if (heavy) CHECK_EQ(p, j - 1);
        }
        if (heavy) { CHECK(n <= R); ++heavy_chains; }
    }
    CHECK(heavy_chains > 1000);
}

/* --draft-adaptive's grower (abi/rad_mtp_adapt.h) over random shapes, rows, margins, floors and
 * calibrations: the same structure the pruned grower guarantees (created nodes a prefix, inert
 * tail, parent before child, no repeated sibling, node 1 exists), the floor is honoured when the
 * pool has the candidates, and a stricter margin on the SAME rows creates fewer nodes on average. */
TEST(mtp_tree_grow_adaptive_fuzz_keeps_the_structure_and_shrinks_with_the_margin) {
    uint32_t s = 0x2545f491u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (int)(s >> 8); };
    long shrunk = 0, loose = 0, strict = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int W = 1 + rnd() % 4, R = 1 + rnd() % 5;
        const int N = std::max(1, R + rnd() % 12), K = RAD_MTP_TREE_K, PITCH = N;
        const int floor_n = 1 + rnd() % 3;
        char txt[96];
        std::vector<std::vector<int>> idxs((size_t)R);
        std::vector<std::vector<float>> vals((size_t)R);
        for (int round = 1; round <= R; ++round) {
            const int rows = round == 1 ? 1 : W;
            idxs[(size_t)round - 1].resize((size_t)rows * K);
            vals[(size_t)round - 1].resize((size_t)rows * K);
            for (int r = 0; r < rows; ++r) {
                std::set<int> used;
                for (int c = 0; c < K; ++c) {
                    int id;
                    do id = rnd() % 5000; while (used.count(id));
                    used.insert(id);
                    idxs[(size_t)round - 1][(size_t)r * K + c] = rnd() % 9 == 0 ? -1 : id;
                    vals[(size_t)round - 1][(size_t)r * K + c] = (float)(rnd() % 800) / 60.0f - 6.0f;
                }
            }
        }
        const bool temp = rnd() % 2 == 0;
        int made[2] = { 0, 0 };
        int mi = 0;
        for (double margin : { 0.5, 3.0 }) {
            std::snprintf(txt, sizeof txt, "on=1;floor=%d;margin=%g%s", floor_n, margin,
                          temp ? ";temp=1.4,1.1" : "");
            std::vector<int> st((size_t)rad_mtp_tree_state_words(N, W, R), 0), table(64, 1);
            std::vector<int> tok((size_t)PITCH), par((size_t)PITCH);
            for (int round = 1; round <= R; ++round) {
                RadMtpTreeCfg c{};
                c.round = round; c.rounds = R; c.nodes = N; c.width = W; c.block = 4;
                c.sel_pitch = 5; c.pitch = PITCH;
                CHECK(rad_mtp_adapt_parse(txt, &c.ad));
                std::vector<int> sel((size_t)W * 5), ns((size_t)W), src((size_t)W), dst((size_t)W);
                std::vector<long long> vm((size_t)W);
                int rb = 0;
                CHECK_EQ(rad_mtp_tree_grow_seq(&c, st.data(), idxs[(size_t)round - 1].data(),
                                               vals[(size_t)round - 1].data(), 0, K, 1,
                                               round == 1 ? 7 : -1, table.data(), tok.data(),
                                               par.data(), sel.data(), ns.data(), vm.data(), &rb, 0,
                                               src.data(), dst.data()), 0);
            }
            int n = 0;
            while (n < N && tok[(size_t)n] >= 0) ++n;
            CHECK(n >= 1);
            for (int j = n; j < N; ++j) { CHECK_EQ(tok[(size_t)j], -1); CHECK_EQ(par[(size_t)j], 0); }
            std::set<std::pair<int, int>> sib;
            for (int j = 1; j <= n; ++j) {
                const int p = par[(size_t)j - 1];
                CHECK(p >= 0 && p < j);
                CHECK(sib.insert({ p, tok[(size_t)j - 1] }).second);
            }
            made[mi++] = n;
        }
        /* not a per-input law (fewer nodes also means a smaller expectation), a law in the mean */
        shrunk += made[1] < made[0];
        loose += made[0];
        strict += made[1];
    }
    CHECK(shrunk > 100);
    CHECK(strict < loose);
}

static void qsa_draftless_row_beside_a_tree(bool one_node, int* moved_tree_row, int* diff_out) {
    Rng r(7);
    const int64_t topk = 40, ratio = 4;
    QWorld bw(16, topk, 0, 2);
    std::vector<std::unique_ptr<QWorld>> alone;
    const int64_t bases[2] = { 20, 21 };
    for (int s = 0; s < 2; ++s) {
        alone.push_back(std::make_unique<QWorld>(16, topk, 0));
        std::vector<std::vector<float>> pk;
        for (int64_t t = 0; t < bases[s]; ++t) pk.push_back(qkey(r, bw.n, false));
        qsa_chain(*alone.back(), pk, {}, 0, false, nullptr);
        qsa_place(bw, s, *alone.back());
    }
    auto rows = [&](int64_t R, QSeq& q) {
        for (int64_t j = 0; j < R; ++j) {
            q.K.push_back(qkey(r, bw.n, false));
            std::vector<float> x;
            for (int64_t h = 0; h < bw.heads; ++h) { auto y = qkey(r, bw.n, false); x.insert(x.end(), y.begin(), y.end()); }
            q.Q.push_back(x);
        }
    };
    QSeq s0; s0.tree = true;  s0.base = 20; s0.na = 1; s0.par = { -1, 0, 0 }; rows(3, s0);
    QSeq s1; s1.tree = one_node; s1.base = 21; s1.na = 1; if (one_node) s1.par = { -1 }; rows(1, s1);   /* one row, no drafts */
    BOut B;
    qsa_batch_step(bw, { s0, s1 }, true, &B);
    const int64_t row = 3;                              /* the draftless decode row */
    /* What the same row gets on a chain step (the sequential decode): the classic table, the committed
     * pages and then the partial page (page id == 40 * sequence + block here), and its causal bound
     * in seqused. As a one-node tree the row's list has no partial page and `vmask` names its keys
     * there. */
    std::vector<int32_t> classic((size_t)topk + 1, -1);
    int32_t classic_su = (int32_t)B.su[(size_t)row];
    if (one_node) {
        int k = 0;
        for (; k < B.nsel[(size_t)row]; ++k) classic[(size_t)k] = (int32_t)B.sel[(size_t)row][(size_t)k];
        classic[(size_t)k] = (int32_t)(40 * 1 + B.nc[1]);
        classic_su = (int32_t)(s1.base + 1);
    } else {
        for (int64_t i = 0; i <= topk; ++i) classic[(size_t)i] = (int32_t)B.sel[(size_t)row][(size_t)i];
    }
    const int64_t crow = row;
    const BOut& CR = B;
    /* The attention cache: page id == the table's (s * 40 + block), random K and V everywhere,
     * the never-written slots of the partial page included (stale data in a real cache). */
    const int64_t HD = 8, G = 2, KVH = 1, NH = G * KVH, NB = 80, Rn = 4;
    uint32_t rng = 4242u;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 9) & 0x7fff) / 16384.0f - 1.0f; };
    T kv(RAD_F32, { NB, KVH, ratio, 2 * HD });
    for (int64_t i = 0; i < NB * KVH * ratio * 2 * HD; ++i) st_dt(RAD_F32, kv.t.data, i, rnd());
    T q(RAD_F32, { Rn, NH, HD });
    for (int64_t i = 0; i < Rn * NH * HD; ++i) st_dt(RAD_F32, q.t.data, i, rnd() * 2.0f);
    T sel(RAD_I32, { Rn, topk + 1 }), nsel(RAD_I32, { Rn }), vm(RAD_I64, { Rn });
    for (int64_t j = 0; j < Rn; ++j) {
        for (int64_t i = 0; i <= topk; ++i) st_int(RAD_I32, sel.t.data, j * (topk + 1) + i, B.sel[(size_t)j][(size_t)i]);
        st_int(RAD_I32, nsel.t.data, j, B.nsel[(size_t)j]);
        st_int(RAD_I64, vm.t.data, j, B.vmask[(size_t)j]);
    }
    T rb(RAD_I32, { 2 }); rb.n({ ratio * B.nc[0], ratio * B.nc[1] });      /* op_rb: rb = ratio * nc */
    T cu(RAD_I32, { 3 }); cu.n({ 0, 3, 4 });
    auto params = [&](A& a, int64_t qlen, bool tree) {
        a.i("q_len", qlen).i("head_dim", HD).i("gqa", G).i("block_size", ratio).i("causal", 1)
         .i("window", 0).s("q_dtype", "f32").s("kv_dtype", "f32").i("n_head", NH);
        if (tree) a.i("sel_pitch", topk + 1);
    };
    /* (1) what AttnGatedFP8::step issues for the decode half of a tree step. */
    auto tree_issue = [&](T& out) {
        A a; a.t(q).t(kv).t(sel).t(nsel).tnull().tnull().tnull().t(out).t(vm).t(bw.tab).t(rb).t(cu);
        params(a, 1, true);
        REQUIRE_EQ(run("attn_paged_tree_sel", a), RAD_OK);
    };
    /* (2) what the same row gets on a chain step: attn_paged over (qsa_sel, qsa_sequ). */
    auto chain_issue = [&](T& o1) {
        T tab(RAD_I32, { 1, topk + 1 }), su(RAD_I32, { 1 }), q1(RAD_F32, { 1, NH, HD });
        for (int64_t i = 0; i <= topk; ++i) st_int(RAD_I32, tab.t.data, i, classic[(size_t)i]);
        su.n({ classic_su });
        std::memcpy(q1.t.data, (const float*)q.t.data + row * NH * HD, sizeof(float) * NH * HD);
        A b; b.t(q1).t(kv).t(tab).t(su).tnull().tnull().tnull().t(o1);
        params(b, 1, false);
        REQUIRE_EQ(run("attn_paged", b), RAD_OK);
    };
    T out_t(RAD_F32, { Rn, NH, HD }), out_c(RAD_F32, { 1, NH, HD });
    tree_issue(out_t);
    chain_issue(out_c);
    int diff = 0;
    double maxd = 0.0, maxv = 0.0;
    for (int e = 0; e < NH * HD; ++e) {
        const double a_ = out_t.at(row * NH * HD + e), b_ = out_c.at(e);
        /* the same key set; the order the keys accumulate in differs (tree: committed keys then the
         * region, sequential: the region's page among them), so equal to the last bits, not bytes */
        diff += std::fabs(a_ - b_) > 1e-5 * (1.0 + std::fabs(b_));
        maxd = std::max(maxd, std::fabs(a_ - b_)); maxv = std::max(maxv, std::fabs(b_));
    }
    fprintf(stderr, "    max |tree - chain| = %.3g (max |out| %.3g)\n", maxd, maxv);
    fprintf(stderr, "    tree-step issue vs chain issue of the same row: %d of %d outputs differ\n",
            diff, (int)(NH * HD));
    *diff_out = diff;
    /* The keys after the row's position in its partial page (positions 22, 23: never written by
     * this sequence) must not matter. */
    const int64_t part = one_node ? 40 * 1 + B.nc[1] : B.sel[(size_t)row][(size_t)(B.nsel[(size_t)row] - 1)];
    (void)crow; (void)CR;
    for (int64_t slot = 2; slot < 4; ++slot)
        for (int64_t d = 0; d < 2 * HD; ++d)
            st_dt(RAD_F32, kv.t.data, ((part * KVH) * ratio + slot) * 2 * HD + d, 5.0f + (float)d);
    T out_t2(RAD_F32, { Rn, NH, HD }), out_c2(RAD_F32, { 1, NH, HD });
    tree_issue(out_t2);
    chain_issue(out_c2);
    int moved_t = 0, moved_c = 0;
    for (int e = 0; e < NH * HD; ++e) {
        moved_t += out_t2.at(row * NH * HD + e) != out_t.at(row * NH * HD + e);
        moved_c += out_c2.at(e) != out_c.at(e);
    }
    fprintf(stderr, "    after rewriting the unwritten slots: tree-step row moved in %d outputs, chain row in %d\n",
            moved_t, moved_c);
    *moved_tree_row = moved_t;
    CHECK_EQ(moved_c, 0);
}

/* A decode row without drafts beside a tree verify goes through attn_paged_tree_sel with the tree
 * step's (sel, nsel, vmask): that form reads every listed page in full and ignores seqused. Staged as
 * a chain row (parent -2) its table is the classic one, partial page last, and it attends over the
 * 1-3 slots behind its own position (stale K/V: 16 of 16 outputs differ from the chain's issue of the
 * same row and move when those slots change). The core stages it as the one-node tree (parent -1):
 * no partial page, vmask the committed prefix, the row's output is the chain's and independent of the
 * slots past its position. The first call shows the test's sensitivity, the second is the contract. */
TEST(qsa_draftless_decode_row_beside_a_tree_is_a_one_node_tree_and_reads_nothing_past_its_position) {
    int moved = -1, diff = -1;
    qsa_draftless_row_beside_a_tree(false, &moved, &diff);
    CHECK(diff > 0);          /* a chain row through the tree form: wrong keys */
    CHECK(moved > 0);
    moved = -1; diff = -1;
    qsa_draftless_row_beside_a_tree(true, &moved, &diff);
    CHECK_EQ(diff, 0);
    CHECK_EQ(moved, 0);
}

// The tree verify's folded delta-net op (gdn_conv_recurrent_update with spec_parent / spec_path) in
// its chained form on a real card: launched with `chain` set on the AQL backend, back to back, it
// must finish (a chained launch that never reports completion would stall the next one) and give
// the bytes the plain launch gives. Skips without a device.
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_device.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "rad_plugin.h"
#include <dlfcn.h>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool device() {
    RadDeviceProps p{};
    return rad_dev_props(0, &p) == RAD_OK && !p.is_host_backend && rad_dev_set(0) == RAD_OK;
}
struct Library {
    void* h;
    Library() {
        const char* home = std::getenv("RADIANCE_HOME");
        h = dlopen((std::string(home ? home : "") + "/kernels/libr4d.so").c_str(),
                   RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(h); }
    const RadKernelInfo* row(const char* name) {
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        for (int i = 0; i < count(); ++i)
            if (!std::strcmp(at(i)->name, name)) return at(i);
        throw std::runtime_error(name);
    }
};
RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape, int64_t row_pitch = 0) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    if (row_pitch) t.stride[0] = row_pitch;
    return t;
}
RadParam pint(const char* k, long long v) {
    RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
RadParam pf64(const char* k, double v) {
    RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
uint16_t bf(float f) { return rad_f32_to_bf16(f); }
float val(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}
}  // namespace

TEST(gdn_tree_fold_chained_matches_plain) {
    if (!device()) SKIP("requires real device");
    Library lib;
    const RadKernelInfo* fold = lib.row("gdn_conv_recurrent_update_k128_v128_bf16");
    constexpr int64_t K = 128, V = 128, H = 24, Hg = 8, W = 4, NS = 2, Q = 7, PITCH = 16;
    constexpr int64_t CD = 2 * Hg * K + H * V, INW = CD + H * V, SL = W - 2 + 8;
    uint32_t seed = 12345u;
    std::vector<uint16_t> hin(Q * INW), hab(Q * 2 * H), hw(CD * W), hcs(NS * CD * SL);
    for (auto& e : hin) e = bf(val(seed));
    for (auto& e : hab) e = bf(val(seed));
    for (auto& e : hw) e = bf(0.5f * val(seed));
    for (auto& e : hcs) e = bf(val(seed));
    std::vector<float> halog(H, -2.0f), hdtb(H, -1.0f), hnw(V, 1.0f), hst0(2 * NS * H * V * K, 0.0f);
    for (size_t i = 0; i < (size_t)(NS * H * V * K); ++i) hst0[i] = 0.1f * val(seed);
    std::vector<int32_t> par = {-1, 0, 1, 2, 0, 0, 0}, path(PITCH, 0), cu = {0, (int32_t)Q},
                         cidx = {0}, sidx = {0, 0, 0, (int32_t)NS, (int32_t)NS, (int32_t)NS},
                         nacc = {3};
    for (int i = 0; i < 3; ++i) path[i] = i;

    auto up = [](const void* h, size_t b) -> void* {
        void* d = rad_dev_alloc(b, RAD_MEM_DEVICE);
        if (!d) return nullptr;
        std::vector<uint8_t> tmp((const uint8_t*)h, (const uint8_t*)h + b);
        RadStream s = nullptr;
        if (rad_stream_create(&s, 0) != RAD_OK) return nullptr;
        const int e = rad_memcpy_async(d, tmp.data(), b, s);
        const int e2 = rad_stream_sync(s);
        rad_stream_destroy(s);
        return e == RAD_OK && e2 == RAD_OK ? d : nullptr;
    };
    void *d_in = up(hin.data(), hin.size() * 2), *d_ab = up(hab.data(), hab.size() * 2),
         *d_w = up(hw.data(), hw.size() * 2), *d_cs = up(hcs.data(), hcs.size() * 2),
         *d_alog = up(halog.data(), H * 4), *d_dtb = up(hdtb.data(), H * 4),
         *d_nw = up(hnw.data(), V * 4), *d_st = up(hst0.data(), hst0.size() * 4),
         *d_cu = up(cu.data(), 8), *d_cidx = up(cidx.data(), 4), *d_sidx = up(sidx.data(), 24),
         *d_nacc = up(nacc.data(), 4), *d_par = up(par.data(), Q * 4),
         *d_path = up(path.data(), PITCH * 4);
    void* d_o = rad_dev_alloc(Q * H * V * 2, RAD_MEM_DEVICE);
    REQUIRE(d_in && d_ab && d_w && d_cs && d_alog && d_dtb && d_nw && d_st && d_cu && d_cidx &&
            d_sidx && d_nacc && d_par && d_path && d_o);

    RadTensor none{};
    std::vector<RadTensor> ts = {
        tens(d_in, RAD_BF16, {Q, CD}, INW), tens(d_w, RAD_BF16, {CD, W}), none,
        tens(d_cs, RAD_BF16, {NS, CD, SL}), tens(d_cidx, RAD_I32, {1}),
        tens(d_ab, RAD_BF16, {Q, H}, 2 * H),
        tens((char*)d_ab + H * 2, RAD_BF16, {Q, H}, 2 * H),
        tens(d_alog, RAD_F32, {H}), tens(d_dtb, RAD_F32, {H}),
        tens(d_st, RAD_F32, {2 * NS, H, V, K}), tens(d_sidx, RAD_I32, {1, 6}),
        tens(d_nacc, RAD_I32, {1}), tens(d_cu, RAD_I32, {2}),
        tens((char*)d_in + CD * 2, RAD_BF16, {Q, H * V}, INW), tens(d_nw, RAD_F32, {V}),
        tens(d_o, RAD_BF16, {Q, H, V}), none, none, tens(d_par, RAD_I32, {Q}),
        tens(d_path, RAD_I32, {1, PITCH})};
    std::vector<RadParam> ps = {pint("q_len", Q), pint("head_k", K), pint("head_v", V),
                                pint("n_head_v", H), pf64("scale", 0.088388), pf64("eps", 1e-6),
                                pstr("act", "sigmoid"), pstr("state_form", "anchor"),
                                pint("conv_width", W), pint("chain", 0)};
    RadArgs a{};
    a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    RadStream s = nullptr;
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);

    auto reset = [&]() {
        CHECK_EQ(rad_memcpy_async(d_st, hst0.data(), hst0.size() * 4, s), RAD_OK);
        CHECK_EQ(rad_memcpy_async(d_cs, hcs.data(), hcs.size() * 2, s), RAD_OK);
    };
    auto fetch = [&]() {
        std::vector<uint16_t> o(Q * H * V);
        CHECK_EQ(rad_memcpy_async(o.data(), d_o, o.size() * 2, s), RAD_OK);
        CHECK_EQ(rad_stream_sync(s), RAD_OK);
        return o;
    };
    reset();
    REQUIRE_EQ(fold->launch(&a, s), RAD_OK);
    REQUIRE_EQ(fold->launch(&a, s), RAD_OK);
    const std::vector<uint16_t> plain = fetch();

    ps.back() = pint("chain", 1);
    rad_dev_chain_infer(1);
    REQUIRE_EQ(rad_dev_chain_enabled(), 1);
    reset();
    for (int i = 0; i < 2; ++i) REQUIRE_EQ(fold->launch(&a, s), RAD_OK);
    REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
    rad_dev_chain_infer(0);
    const std::vector<uint16_t> chained = fetch();
    CHECK(plain == chained);
    for (void* v : {d_in, d_ab, d_w, d_cs, d_alog, d_dtb, d_nw, d_st, d_cu, d_cidx, d_sidx, d_nacc,
                    d_par, d_path, d_o})
        rad_dev_free(v, RAD_MEM_DEVICE);
    rad_stream_destroy(s);
}
RAD_TEST_MAIN()

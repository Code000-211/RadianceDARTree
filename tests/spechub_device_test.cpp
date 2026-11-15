/* tests/spechub_device_test.cpp -- the spechub plugin's DEVICE rows against its host rows, for the
 * draft tree's coupled ops (abi/rad_mtp_rrs.h), on a card.
 *
 *   mtp_couple_propose with `leaves`   the trunk token and its siblings drawn without replacement
 *                                      (Gumbel-top-k) into idx, the state in draft order
 *   mtp_couple_verify                  rows whose state drafted 1..4 children: recursive rejection
 *                                      sampling, the chain's maximal coupling for one child
 *
 * Both rows run the same shared header code (the device verify hands a tree row to the host helper
 * on one thread), so the outputs must agree: tokens, ids and counts exactly, Q's FP64 weights to the
 * last bits of the device's double exp. Skipped without a card (the host backend), or without the
 * device rows in the plugin. Run it on a card under the shared lock:
 *   flock gpu.lock build/tests/spechub_device_test
 */
#include "rad_test.h"

#include "rad_abi.h"
#include "rad_device.h"
#include "rad_mtp_rrs.h"
#include "rad_sample.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef RAD_SPECHUB_LIB
#define RAD_SPECHUB_LIB ""
#endif

namespace {

struct Plugin {
    void* handle = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    bool open(const char* path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        count = (int (*)())dlsym(handle, "rad_kernel_count");
        at    = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return count && at;
    }
    const RadKernelInfo* row(const char* name) const {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* r = at(i);
            if (r && r->name && std::string(r->name) == name) return r;
        }
        return nullptr;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

/* A tensor over a host buffer or its device twin. */
struct Buf {
    std::vector<uint8_t> host;
    void* dev = nullptr;
    RadTensor t{};
    Buf(uint32_t dtype, std::vector<int64_t> shape, size_t elem) {
        t.dtype = dtype;
        t.rank = (uint32_t)shape.size();
        int64_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; n *= shape[i]; }
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
        host.assign((size_t)n * elem + 16, 0);
    }
    ~Buf() { if (dev) rad_dev_free(dev, RAD_MEM_DEVICE); }
    RadTensor on_host() { RadTensor x = t; x.data = host.data(); return x; }
    RadTensor on_dev(RadStream s) {
        if (!dev) dev = rad_dev_alloc((int64_t)host.size(), RAD_MEM_DEVICE);
        rad_memcpy_async(dev, host.data(), (int64_t)host.size(), s);
        RadTensor x = t; x.data = dev; return x;
    }
    void fetch(RadStream s) { rad_memcpy_async(host.data(), dev, (int64_t)host.size(), s); }
};

RadParam P(const char* k, long long v) { return RAD_INT(k, v); }

uint32_t g_seed = 424242;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return (g_seed >> 8) & 0xffffff; }

RadMtpCoupleDraw make_draw(uint64_t seed, uint64_t base, uint64_t seq, int card, uint32_t mode) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = seq; d.card = card; d.mode = mode;
    return d;
}

}  /* namespace */

TEST(the_device_rows_of_the_trees_coupled_ops_agree_with_the_host_rows) {
    RadDeviceProps dp{};
    if (rad_dev_count() < 1 || rad_dev_props(0, &dp) != RAD_OK || dp.is_host_backend) {
        fprintf(stderr, "  SKIP %s: no card\n", ::radtest::current());
        return;
    }
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB)) { fprintf(stderr, "  SKIP: %s did not load\n", RAD_SPECHUB_LIB); return; }
    const RadKernelInfo* dpro = pl.row("spechub_mtp_couple_propose");
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    const RadKernelInfo* hpro = pl.row("ref_mtp_couple_propose");
    const RadKernelInfo* hver = pl.row("ref_mtp_couple_verify");
    if (!dpro || !dver) { fprintf(stderr, "  SKIP: the plugin has no device rows\n"); return; }
    REQUIRE(hpro && hver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS, DW = (int)RAD_MTP_COUPLE_DRAW_WORDS;
    const int64_t VOC = 50000;

    /* ---- propose, round 1 of a 2-round head, leaves 3; then the same with confidence pruning
     * (prune_abs / prune_rel, decided by the slot priorities of Q) */
    for (int pass = 0; pass < 2; ++pass) {
        const int M = 8, LEAVES = 3;
        Buf idx(RAD_I32, { M, 8 }, 4), val(RAD_F32, { M, 8 }, 4), card(RAD_I32, { M }, 4);
        Buf draw(RAD_U32, { M, DW }, 4), st(RAD_U32, { 2, M, SW }, 4);
        int32_t* pi = (int32_t*)idx.host.data();
        float* pv = (float*)val.host.data();
        int32_t* pc = (int32_t*)card.host.data();
        RadMtpCoupleDraw* pd = (RadMtpCoupleDraw*)draw.host.data();
        for (int m = 0; m < M; ++m) {
            const int nv = m == 3 ? 2 : 4 + (int)(rnd() % 5);
            for (int c = 0; c < 8; ++c) {
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % VOC);
                    dup = false;
                    for (int q = 0; q < c; ++q) dup = dup || pi[m * 8 + q] == id;
                } while (dup);
                pi[m * 8 + c] = c < nv ? id : -1;
                pv[m * 8 + c] = c < nv ? 5.0f - 0.8f * c - (rnd() % 100) * 0.01f : -INFINITY;
            }
            pd[m] = make_draw(0x500ull + 31ull * (uint64_t)m, 90ull + (uint64_t)m, (uint64_t)m + 1, m, 1u);
            pc[m] = m;
        }
        pd[4].mode = 0;     /* greedy: idx untouched */
        pc[5] = 77;         /* another card's draw: status -1 */
        std::vector<RadParam> ps = { P("M", M), P("n_vocab", (int)VOC), P("round", 1),
                                     P("max_rounds", 2), P("leaves", LEAVES) };
        if (pass == 1) { ps.push_back(RAD_F64("prune_abs", 0.08)); ps.push_back(RAD_F64("prune_rel", 0.35)); }
        /* host: on copies */
        Buf hidx = idx, hst = st;
        {
            RadTensor ts[5] = { hidx.on_host(), val.on_host(), card.on_host(), draw.on_host(), hst.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = ps.data(); a.n_p = (int)ps.size();
            REQUIRE_EQ(hpro->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[5] = { idx.on_dev(s), val.on_dev(s), card.on_dev(s), draw.on_dev(s), st.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = ps.data(); a.n_p = (int)ps.size();
            REQUIRE_EQ(dpro->launch(&a, s), RAD_OK);
            idx.fetch(s); st.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        int coupled = 0;
        for (int m = 0; m < M; ++m) {
            for (int c = 0; c < 8; ++c) CHECK_EQ(((int32_t*)idx.host.data())[m * 8 + c], ((int32_t*)hidx.host.data())[m * 8 + c]);
            RadMtpCoupleState g, w;
            std::memcpy(&g, st.host.data() + (size_t)m * SW * 4, sizeof g);
            std::memcpy(&w, hst.host.data() + (size_t)m * SW * 4, sizeof w);
            CHECK_EQ(g.status, w.status); CHECK_EQ(g.round, w.round); CHECK_EQ(g.x, w.x);
            CHECK_EQ(g.count, w.count);   CHECK_EQ(g.nkids, w.nkids);
            coupled += w.status == 1 && w.nkids >= 2;
            for (int i = 0; i < 8; ++i) {
                CHECK_EQ(g.ids[i], w.ids[i]);
                CHECK(std::fabs(g.q[i] - w.q[i]) <= 1e-12 * std::fabs(w.q[i]));
            }
        }
        CHECK(coupled >= (pass == 0 ? 4 : 1));
        if (pass == 1) {
            /* the pruning cut something: fewer children than the 1 + leaves an unpruned row draws */
            int fewer = 0;
            for (int m = 0; m < M; ++m) {
                RadMtpCoupleState g;
                std::memcpy(&g, st.host.data() + (size_t)m * SW * 4, sizeof g);
                fewer += g.status == 1 && g.nkids < 1 + LEAVES && g.count > g.nkids;
            }
            CHECK(fewer >= 1);
        }
    }

    /* ---- verify over states with 1..4 children */
    {
        const int M = 64, NC = 24;
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        for (int m = 0; m < M; ++m) {
            for (int c = 0; c < NC; ++c) {
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % 40);
                    dup = false;
                    for (int q = 0; q < c; ++q) dup = dup || pi[m * NC + q] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = 2.0f - 0.25f * c - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) { qid[c] = (int)(rnd() % 40); qv[c] = 2.0f - 0.3f * c; }
            for (int c = 0; c < 8; ++c)
                for (int q = 0; q < c; ++q) if (qid[q] == qid[c]) qid[c] = 100 + c;
            RadMtpCoupleState s0{};
            rad_mtp_couple_build_q(qid, qv, VOC, &s0);
            s0.draw = make_draw(0x900ull + (uint64_t)m, 3, 1, 0, 1u);
            s0.status = 1; s0.round = 1; s0.pos = 11 + (uint64_t)m;
            const int nk = 1 + m % 4;
            if (nk >= 2) rad_mtp_rrs_draw(&s0, 0, nk); else s0.x = s0.ids[0];
            ps_[m] = s0;
            pp[m].seed = s0.draw.seed;
            pp[m].pos = s0.pos;
            pp[m].flags = m % 16 == 15 ? RAD_SP_GREEDY : 0;
            pt[m] = -9;
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        Buf htok = tok;
        {
            RadTensor ts[5] = { cid.on_host(), cval.on_host(), prm.on_host(), sts.on_host(), htok.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(hver->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[5] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dver->launch(&a, s), RAD_OK);
            tok.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        int multi = 0, kept = 0;
        for (int m = 0; m < M; ++m) {
            const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
            CHECK_EQ(g, w);
            multi += ps_[m].nkids >= 2 && w != -9;
            kept += w == -9;
        }
        CHECK(multi >= 20);
        CHECK_EQ(kept, 4);   /* the greedy rows keep the ordinary token */
    }
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()

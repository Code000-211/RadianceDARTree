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
#include "rad_mtp_traversal.h"
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
    for (int pass = 0; pass < 3; ++pass) {
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
            if (pass == 2) {
                pd[m].dscale = 1.0f / 0.7f; pd[m].dtop_p = 0.9f; pd[m].dtop_k = 4;   /* the draft law recipe */
                /* the online word: a per-depth tau code on half the rows (round 1 reads depth 1's) */
                if (m % 2) pd[m].dpad_ = (unsigned)(4 + m % 7) | (3u << 4) | (8u << 8) | (11u << 16);
            }
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
            if (m % 2 == 0) s0.draw.dpad_ |= RAD_MTP_DRAW_EXPORT_BIT;   /* the verify reports P on these rows */
            const int nk = 1 + m % 4;
            if (nk >= 2) rad_mtp_rrs_draw(&s0, 0, nk); else s0.x = s0.ids[0];
            ps_[m] = s0;
            pp[m].seed = s0.draw.seed;
            pp[m].pos = s0.pos;
            pp[m].flags = m % 16 == 15 ? RAD_SP_GREEDY : 0;
            pt[m] = -9;
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        Buf htok = tok, hst2 = sts;
        {
            RadTensor ts[5] = { cid.on_host(), cval.on_host(), prm.on_host(), hst2.on_host(), htok.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(hver->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[5] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dver->launch(&a, s), RAD_OK);
            tok.fetch(s);
            sts.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        int multi = 0, kept = 0, exported = 0;
        for (int m = 0; m < M; ++m) {
            const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
            CHECK_EQ(g, w);
            /* the export (rad_mtp_couple_export_p): the device wrote what the host reference wrote */
            RadMtpCoupleState dg, hw;
            std::memcpy(&dg, sts.host.data() + (size_t)m * SW * 4, sizeof dg);
            std::memcpy(&hw, hst2.host.data() + (size_t)m * SW * 4, sizeof hw);
            CHECK_EQ(dg.status, hw.status);
            for (int i = 0; i < 8; ++i) CHECK(std::fabs(dg.q[i] - hw.q[i]) <= 1e-12 * std::fabs(hw.q[i]) + 1e-300);
            exported += hw.status == RAD_MTP_COUPLE_STATUS_EXPORTED;
            if (m % 2 == 1) CHECK_EQ(hw.status, 1);   /* no export bit: the state is left alone */
            multi += ps_[m].nkids >= 2 && w != -9;
            kept += w == -9;
        }
        CHECK(multi >= 20);
        CHECK(exported >= 20);
        CHECK_EQ(kept, 4);   /* the greedy rows keep the ordinary token */
    }
    rad_stream_destroy(s);
}

/* --draft-verify block (abi/rad_mtp_block.h): the device verify row decides a whole chain from its
 * round-1 row and writes the tokens of rows 0 .. tau; the host row is the same shared code over the
 * dense plane. Blocks of 1..8 rows, planes of 24 (the common case), 1000 (live slots compacted over
 * several chunks of the workgroup) and 1100 (past the LDS plane: one thread runs the helper) slots,
 * with holes; tokens must agree row for row, rows the block does not decide keep their sentinel. */
TEST(the_device_block_verify_agrees_with_the_host_row) {
    RadDeviceProps dp{};
    if (rad_dev_count() < 1 || rad_dev_props(0, &dp) != RAD_OK || dp.is_host_backend) {
        fprintf(stderr, "  SKIP %s: no card\n", ::radtest::current());
        return;
    }
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB)) { fprintf(stderr, "  SKIP: %s did not load\n", RAD_SPECHUB_LIB); return; }
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    const RadKernelInfo* hver = pl.row("ref_mtp_couple_verify");
    if (!dver) { fprintf(stderr, "  SKIP: the plugin has no device rows\n"); return; }
    REQUIRE(hver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;
    int shortened = 0, whole = 0, writes = 0;
    for (int NC : { 24, 1000, 1100 }) {
        const int NSEQ = 40, MAXROWS = 9;
        const int M = NSEQ * MAXROWS;
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        for (int m = 0; m < M; ++m) { pt[m] = -9; for (int c = 0; c < NC; ++c) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; } }
        int row = 0;
        for (int q = 0; q < NSEQ; ++q) {
            const int g = 1 + q % 8;
            const uint64_t seed = 0xb10c0000ull + (uint64_t)q, pos0 = 20 + 3 * (uint64_t)q;
            for (int i = 0; i <= g; ++i, ++row) {
                /* live candidates at random slots */
                const int nlive = 6 + (int)(rnd() % 19);
                int ids[24];
                for (int c = 0; c < nlive; ++c) {
                    bool dup;
                    do {
                        ids[c] = (int)(rnd() % 40);
                        dup = false;
                        for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c];
                    } while (dup);
                    int slot;
                    do { slot = (int)(rnd() % NC); } while (pi[row * NC + slot] >= 0);
                    pi[row * NC + slot] = ids[c];
                    pv[row * NC + slot] = 3.0f - 0.35f * c - (rnd() % 100) * 0.01f;
                }
                pp[row].seed = seed;
                pp[row].pos = pos0 + (uint64_t)i;
                if (i == g) continue;
                int32_t qid[8];
                float qv[8];
                const int nq = 3 + (int)(rnd() % 6);
                for (int c = 0; c < 8; ++c) {
                    qid[c] = c < nq ? (c < nlive && rnd() % 4 ? ids[c] : (int)(rnd() % 40)) : -1;
                    qv[c] = c < nq ? 2.5f - 0.4f * c - (rnd() % 100) * 0.01f : -INFINITY;
                    for (int k = 0; k < c; ++k) if (qid[c] >= 0 && qid[k] == qid[c]) qid[c] = 100 + c;
                }
                RadMtpCoupleState st{};
                REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
                st.draw = make_draw(seed, pos0, 7 + (uint64_t)q, 0, 1u);
                st.status = 1; st.round = i + 1; st.pos = pos0 + (uint64_t)i; st.blk = g;
                st.x = st.ids[rnd() % (uint32_t)st.count];
                ps_[row] = st;
            }
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
        row = 0;
        for (int q = 0; q < NSEQ; ++q) {
            const int g = 1 + q % 8;
            int written = 0;
            for (int i = 0; i <= g; ++i) {
                const int32_t gv = ((int32_t*)tok.host.data())[row + i], wv = ((int32_t*)htok.host.data())[row + i];
                CHECK_EQ(gv, wv);
                written += wv != -9;
            }
            writes += written > 0;
            whole += written == g;      /* tau == g: every draft written, the bonus row alone untouched */
            shortened += written > 0 && written < g;
            row += g + 1;
        }
    }
    CHECK(writes >= 100);
    CHECK(shortened >= 10);
    CHECK(whole >= 10);
    rad_stream_destroy(s);
}

/* Traversal verification (abi/rad_mtp_traversal.h): the device row against the host row, over
 * requests whose rows form a small tree (a root with two children, one of them with two, and a
 * chain below the other). The device row runs the same header code on one thread, so the tokens
 * must be identical; the check is that the operand plumbing, the topology plane and the writes
 * of the accepted path agree. */
TEST(the_device_traversal_verify_agrees_with_the_host_row) {
    RadDeviceProps dp{};
    if (rad_dev_count() < 1 || rad_dev_props(0, &dp) != RAD_OK || dp.is_host_backend) {
        fprintf(stderr, "  SKIP %s: no card\n", ::radtest::current());
        return;
    }
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB)) { fprintf(stderr, "  SKIP: %s did not load\n", RAD_SPECHUB_LIB); return; }
    const RadKernelInfo* dtr = pl.row("spechub_mtp_trav_verify");
    const RadKernelInfo* htr = pl.row("ref_mtp_trav_verify");
    if (!dtr) { fprintf(stderr, "  SKIP: the plugin has no traversal device row\n"); return; }
    REQUIRE(htr);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;
    const int RQ = 12, NR = 7, M = RQ * NR, NC = 12;
    const int par[NR] = { -1, 0, 0, 1, 1, 2, 5 };
    const int nkid[NR] = { 2, 2, 1, 0, 0, 1, 0 };
    Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
    Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4), topo(RAD_I32, { M, 2 }, 4);
    int32_t* pi = (int32_t*)cid.host.data();
    float* pv = (float*)cval.host.data();
    RadSampleParams* pp = (RadSampleParams*)prm.host.data();
    RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
    int32_t* pt = (int32_t*)tok.host.data();
    int32_t* ptopo = (int32_t*)topo.host.data();
    for (int q = 0; q < RQ; ++q) {
        const uint64_t seed = 0x700ull + (uint64_t)q, p0 = 20 + (uint64_t)q;
        int depth[NR] = {};
        int32_t childtok[NR][4] = {};
        for (int n = 0; n < NR; ++n) {
            const int m = q * NR + n;
            depth[n] = n ? depth[par[n]] + 1 : 0;
            for (int c = 0; c < NC; ++c) {
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % 14);
                    dup = false;
                    for (int z = 0; z < c; ++z) dup = dup || pi[m * NC + z] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = (c < 10 ? 2.0f - 0.25f * c : -INFINITY) - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) { qid[c] = (int)(rnd() % 14); qv[c] = 2.0f - 0.3f * c - (rnd() % 50) * 0.01f; }
            for (int c = 0; c < 8; ++c)
                for (int z = 0; z < c; ++z) if (qid[z] == qid[c]) qid[c] = 100 + c;
            RadMtpCoupleState s0{};
            rad_mtp_couple_build_q(qid, qv, VOC, &s0);
            s0.draw = make_draw(seed, 3, 1, 0, 1u);
            s0.status = 1; s0.round = depth[n] + 1; s0.pos = p0 + (uint64_t)depth[n];
            if (nkid[n] >= 2) rad_mtp_rrs_draw(&s0, (uint64_t)n, nkid[n]);
            else { s0.x = s0.ids[0]; s0.nkids = 0; }
            for (int c = 0; c < nkid[n]; ++c) childtok[n][c] = s0.ids[c];
            if (nkid[n] > 0) ps_[m] = s0; else ps_[m].status = 0;
            pp[m].seed = seed;
            pp[m].pos = p0 + (uint64_t)depth[n];
            pp[m].flags = q == 5 ? RAD_SP_GREEDY : 0;   /* one request is greedy: untouched */
            pt[m] = -9;
        }
        int seen[NR] = {};
        for (int n = 0; n < NR; ++n) {
            const int m = q * NR + n;
            ptopo[2 * m] = n ? q * NR + par[n] : -1;
            ptopo[2 * m + 1] = n ? childtok[par[n]][seen[par[n]]++] : 0;
        }
    }
    const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
    Buf htok = tok;
    {
        RadTensor ts[6] = { cid.on_host(), cval.on_host(), prm.on_host(), sts.on_host(), htok.on_host(),
                            topo.on_host() };
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
        REQUIRE_EQ(htr->launch(&a, s), RAD_OK);
    }
    {
        RadTensor ts[6] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s),
                            topo.on_dev(s) };
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
        REQUIRE_EQ(dtr->launch(&a, s), RAD_OK);
        tok.fetch(s);
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
    }
    int written = 0, greedy_kept = 0;
    for (int m = 0; m < M; ++m) {
        const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
        CHECK_EQ(g, w);
        written += w != -9;
        if (m / NR == 5) greedy_kept += w == -9;
    }
    CHECK(written >= RQ);
    CHECK_EQ(greedy_kept, NR);
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()

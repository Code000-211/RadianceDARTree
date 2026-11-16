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
#include "device/aql.h"

#include "rad_abi.h"
#include "rad_device.h"
#include "rad_mtp_rrs.h"
#include "rad_mtp_traversal.h"
#include "rad_sample.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
        if (const char* o = std::getenv("RAD_SPECHUB_LIB_OVERRIDE")) path = o;   /* a baseline build, for RAD_BENCH */
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

/* The queue's scratch is sized by a kernel's private segment times every lane it may run: couple_verify
 * once carried the block row's planes (17.8 KB a lane, with a workgroup of 128 lanes of which one
 * works) and the first sampled tree (>= 5 rows) faulted the queue with OUT_OF_RESOURCES once
 * HSA_NO_SCRATCH_RECLAIM kept the scratch resident. Pin the code object: the workgroup kernel keeps
 * a small private segment (the serial helper that needs the big one is a kernel of its own). Runs
 * without a card: it reads the plugin's own code object. */
TEST(the_tree_verify_kernel_asks_for_little_scratch) {
    FILE* f = std::fopen(RAD_SPECHUB_LIB, "rb");
    if (!f) { fprintf(stderr, "  SKIP: %s not readable\n", RAD_SPECHUB_LIB); return; }
    std::vector<unsigned char> img;
    unsigned char buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) img.insert(img.end(), buf, buf + n);
    std::fclose(f);
    static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
    auto at = std::search(img.begin(), img.end(), kMagic, kMagic + sizeof kMagic - 1);
    if (at == img.end()) { fprintf(stderr, "  SKIP: no device code in the plugin\n"); return; }
    const void* co = nullptr;
    size_t cosz = 0;
    std::string why;
    REQUIRE(rad::aql::bundle_code_object(&*at, { "hipv4-amdgcn-amd-amdhsa--gfx1201", "hip-amdgcn-amd-amdhsa--gfx1201",
                                                 "amdgcn-amd-amdhsa--gfx1201" }, &co, &cosz, &why));
    std::vector<rad::aql::KernelMeta> ks;
    REQUIRE(rad::aql::code_object_kernels(co, cosz, &ks, &why));
    int seen = 0;
    for (const auto& k : ks) {
        if (k.symbol.find("couple_verify") == std::string::npos || k.symbol.find("wide") != std::string::npos)
            continue;
        ++seen;
        CHECK(k.private_size <= 256u);
        fprintf(stderr, "  couple_verify private segment %u bytes\n", (unsigned)k.private_size);
    }
    CHECK_EQ(seen, 1);
}

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

    /* ---- verify over states with 1..4 children, at a 24-slot plane and at the sampler's full 1024-slot
     * plane (top_k 0: every slot live on even rows, 40 live slots and holes on odd rows) -- the
     * workgroup's compaction and weights over several chunks, then the host's decision over them */
    for (const int NC : { 24, 1024, 1100 }) {
        const int M = 64;
        const bool wide = NC > 64;      /* 1100: past the workgroup's plane, one lane decides */
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        for (int m = 0; m < M; ++m) {
            const int live = !wide || m % 2 == 0 ? NC : 40;
            for (int c = 0; c < NC; ++c) {
                if (c >= live) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; continue; }
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % (wide ? VOC : 40));
                    dup = false;
                    for (int q = 0; q < c; ++q) dup = dup || pi[m * NC + q] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = 2.0f - (wide ? 0.01f : 0.25f) * c - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) {
                /* Q's support among the candidates (wide) or the 40-token universe */
                qid[c] = wide ? pi[m * NC + (int)(rnd() % (uint32_t)(live < 60 ? live : 60))] : (int)(rnd() % 40);
                qv[c] = 2.0f - 0.3f * c;
            }
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
    for (int NC : { 24, 1000, 1100, 1001 }) {   /* 1001: wide, 300..500 live slots a row */
        const bool wide = NC == 1001;
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
                const int nlive = wide ? 300 + (int)(rnd() % 200) : 6 + (int)(rnd() % 19);
                int ids[512];
                for (int c = 0; c < nlive; ++c) {
                    bool dup;
                    do {
                        ids[c] = (int)(rnd() % (wide ? 20000u : 40u));
                        dup = false;
                        for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c];
                    } while (dup);
                    int slot;
                    do { slot = (int)(rnd() % NC); } while (pi[row * NC + slot] >= 0);
                    pi[row * NC + slot] = ids[c];
                    pv[row * NC + slot] = wide ? 3.0f - 0.02f * c - (rnd() % 100) * 0.01f : 3.0f - 0.35f * c - (rnd() % 100) * 0.01f;
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
  for (const int NC : { 12, 1024, 1100 }) {
    const bool wide = NC > 64;
    const int RQ = 12, NR = 7, M = RQ * NR;
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
            /* wide: the sampler's full plane at top_k 0 on even requests, 40 live slots and holes on odd */
            const int live = !wide || q % 2 == 0 ? NC : 40;
            for (int c = 0; c < NC; ++c) {
                if (wide && c >= live) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; continue; }
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % (wide ? VOC : 14));
                    dup = false;
                    for (int z = 0; z < c; ++z) dup = dup || pi[m * NC + z] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = wide ? 2.0f - 0.01f * c - (rnd() % 100) * 0.01f
                                      : (c < 10 ? 2.0f - 0.25f * c : -INFINITY) - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) {
                qid[c] = wide ? pi[m * NC + (int)(rnd() % (uint32_t)(live < 60 ? live : 60))] : (int)(rnd() % 14);
                qv[c] = 2.0f - 0.3f * c - (rnd() % 50) * 0.01f;
            }
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
  }
    rad_stream_destroy(s);
}

/* RAD_BENCH=1: microseconds per call (launch + sync included, the same constant before and after a
 * change) of the verify at a node of 3 drafted children (recursive rejection sampling) and of the
 * traversal walk of a 7-row tree, on the sampler's 1024-slot plane with 20 live slots (top_k 20) and
 * with every slot live (top_k 0). Prints; checks nothing. */
TEST(spechub_bench_tree_verify_and_traversal) {
    if (!std::getenv("RAD_BENCH")) return;
    RadDeviceProps dp{};
    if (rad_dev_count() < 1 || rad_dev_props(0, &dp) != RAD_OK || dp.is_host_backend) return;
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB)) return;
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    const RadKernelInfo* dtr = pl.row("spechub_mtp_trav_verify");
    if (!dver || !dtr) return;
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;
    const int NC = 1024, NR = 7, M = 16;
    const int par[NR] = { -1, 0, 0, 1, 1, 2, 5 };
    const int nkid[NR] = { 2, 2, 1, 0, 0, 1, 0 };
    for (const int live : { 20, 1024 }) {
        /* rows of requests of NR rows: M rows for the verify, one request for the traversal */
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4), topo(RAD_I32, { M, 2 }, 4);
        Buf tcid(RAD_I32, { NR, NC }, 4), tcval(RAD_F32, { NR, NC }, 4), tprm(RAD_U32, { NR, 24 }, 4);
        Buf tsts(RAD_U32, { NR, SW }, 4), ttok(RAD_I32, { NR }, 4);
        auto fill = [&](int32_t* pi, float* pv, int m) {
            for (int c = 0; c < NC; ++c) {
                if (c >= live) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; continue; }
                pi[m * NC + c] = (int)((rnd() % 40) * 1000 + (unsigned)c);
                pv[m * NC + c] = 2.0f - 0.005f * c - (rnd() % 100) * 0.01f;
            }
        };
        auto state = [&](const int32_t* pi, int m, int depth, int nk, RadSampleParams* pp, RadMtpCoupleState* ps_) {
            int32_t qid[8]; float qv[8];
            for (int c = 0; c < 8; ++c) { qid[c] = pi[m * NC + c * 2]; qv[c] = 2.0f - 0.3f * c; }
            RadMtpCoupleState s0{};
            rad_mtp_couple_build_q(qid, qv, VOC, &s0);
            s0.draw = make_draw(0x900ull + (uint64_t)m, 3, 1, 0, 1u);
            s0.status = 1; s0.round = depth + 1; s0.pos = 11 + (uint64_t)depth;
            if (nk >= 2) rad_mtp_rrs_draw(&s0, (uint64_t)m, nk); else { s0.x = s0.ids[0]; s0.nkids = 0; }
            ps_[m] = s0;
            pp[m].seed = s0.draw.seed; pp[m].pos = s0.pos; pp[m].flags = 0;
        };
        int32_t* pi = (int32_t*)cid.host.data(); float* pv = (float*)cval.host.data();
        for (int m = 0; m < M; ++m) {
            fill(pi, pv, m);
            state(pi, m, 0, 3, (RadSampleParams*)prm.host.data(), (RadMtpCoupleState*)sts.host.data());
            ((int32_t*)tok.host.data())[m] = -9;
        }
        int32_t* ti = (int32_t*)tcid.host.data(); float* tv = (float*)tcval.host.data();
        int32_t* ptopo = (int32_t*)topo.host.data();
        int depth[NR] = {}, seen[NR] = {};
        int32_t childtok[NR][4] = {};
        for (int n = 0; n < NR; ++n) {
            fill(ti, tv, n);
            depth[n] = n ? depth[par[n]] + 1 : 0;
            if (nkid[n] > 0) {
                state(ti, n, depth[n], nkid[n], (RadSampleParams*)tprm.host.data(), (RadMtpCoupleState*)tsts.host.data());
                for (int c = 0; c < nkid[n]; ++c) childtok[n][c] = ((RadMtpCoupleState*)tsts.host.data())[n].ids[c];
            } else {
                ((RadMtpCoupleState*)tsts.host.data())[n].status = 0;
                ((RadSampleParams*)tprm.host.data())[n].seed = 0x900ull + (uint64_t)n;
                ((RadSampleParams*)tprm.host.data())[n].pos = 11 + (uint64_t)depth[n];
            }
            ((int32_t*)ttok.host.data())[n] = -9;
        }
        for (int n = 0; n < M; ++n) { ptopo[2 * n] = -1; ptopo[2 * n + 1] = 0; }
        for (int n = 0; n < NR; ++n) {
            ptopo[2 * n] = n ? par[n] : -1;
            ptopo[2 * n + 1] = n ? childtok[par[n]][seen[par[n]]++] : 0;
        }
        const std::vector<RadParam> pv_ = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        const std::vector<RadParam> pt_ = { P("M", NR), P("n_cand", NC), P("n_vocab", (int)VOC) };
        RadTensor tv_[5] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s) };
        RadTensor tt_[6] = { tcid.on_dev(s), tcval.on_dev(s), tprm.on_dev(s), tsts.on_dev(s), ttok.on_dev(s), topo.on_dev(s) };
        RadArgs av{}; av.t = tv_; av.n_t = 5; av.p = pv_.data(); av.n_p = (int)pv_.size();
        RadArgs at{}; at.t = tt_; at.n_t = 6; at.p = pt_.data(); at.n_p = (int)pt_.size();
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        for (int i = 0; i < 20; ++i) { dver->launch(&av, s); dtr->launch(&at, s); }
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        const int iters = 300;
        double tv_us = 0, tt_us = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            dver->launch(&av, s); rad_stream_sync(s);
            auto t1 = std::chrono::steady_clock::now();
            dtr->launch(&at, s); rad_stream_sync(s);
            auto t2 = std::chrono::steady_clock::now();
            tv_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            tt_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        std::printf("BENCH live=%4d: rrs-node verify (%d rows x 3 children) %.1f us, traversal (%d-row tree) %.1f us per call\n",
                    live, M, tv_us / iters, NR, tt_us / iters);
    }
    rad_stream_destroy(s);
}


/* Microbenchmark of the block verify op (set RAD_BLOCK_BENCH=1; RAD_BENCH_LIB=<spechub.so> to time
 * another build of the plugin). Wall time per launch over back-to-back launches of one block (g
 * drafts plus the bonus row, one sequence as at max-num-seqs 1), minus the same launch over rows the
 * op ignores (the launch floor). Planes of 1024 slots with `live` live candidates: 20 (top_k 20)
 * or 1024 (top_k 0). Eight different blocks are cycled. */
TEST(block_verify_microbenchmark) {
    if (!getenv("RAD_BLOCK_BENCH")) return;
    RadDeviceProps dp{};
    if (rad_dev_count() < 1 || rad_dev_props(0, &dp) != RAD_OK || dp.is_host_backend) return;
    Plugin pl;
    const char* lib = getenv("RAD_BENCH_LIB") ? getenv("RAD_BENCH_LIB") : RAD_SPECHUB_LIB;
    REQUIRE(pl.open(lib));
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    REQUIRE(dver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 150000;
    const int NC = 1024, NV = 8;
    const int nrep = getenv("RAD_BENCH_REPS") ? atoi(getenv("RAD_BENCH_REPS")) : 400;
    const int divide = getenv("RAD_BENCH_DIV") ? atoi(getenv("RAD_BENCH_DIV")) : 1;   /* the plugin's in-kernel repeat count */
    for (int live : { 20, 1024 }) for (int g : { 3, 5, 7 }) {
        const int M = g + 1;
        struct Var { Buf cid, cval, prm, sts, tok; Var(int M, int NC, int SW) :
            cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4),
            sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4) {} };
        std::vector<Var*> vars;
        for (int v = 0; v < NV; ++v) {
            Var* x = new Var(M, NC, SW);
            vars.push_back(x);
            int32_t* pi = (int32_t*)x->cid.host.data();
            float* pv = (float*)x->cval.host.data();
            RadSampleParams* pp = (RadSampleParams*)x->prm.host.data();
            RadMtpCoupleState* ps_ = (RadMtpCoupleState*)x->sts.host.data();
            for (int m = 0; m < M; ++m) for (int c = 0; c < NC; ++c) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; }
            const uint64_t seed = 0xbe00ull + (uint64_t)v, pos0 = 100;
            for (int i = 0; i < M; ++i) {
                int ids[1024];
                for (int c = 0; c < live; ++c) {
                    bool dup;
                    do { ids[c] = (int)(rnd() % VOC); dup = false;
                         for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c]; } while (dup);
                    pi[i * NC + c] = ids[c];
                    pv[i * NC + c] = 12.0f - 0.9f * (float)c * (20.0f / (float)live) - (rnd() % 100) * 0.01f;
                }
                pp[i].seed = seed; pp[i].pos = pos0 + (uint64_t)i;
                if (i == g) continue;
                int32_t qid[8]; float qv[8];
                for (int c = 0; c < 8; ++c) { qid[c] = ids[c]; qv[c] = 11.0f - 0.9f * c - (rnd() % 100) * 0.01f; }
                RadMtpCoupleState st{};
                REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
                st.draw = make_draw(seed, pos0, 7 + (uint64_t)v, 0, 1u);
                st.status = 1; st.round = i + 1; st.pos = pos0 + (uint64_t)i; st.blk = g;
                st.x = st.ids[rnd() % 3u];
                ps_[i] = st;
            }
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        std::vector<RadTensor> ts[NV];
        std::vector<RadTensor> tn[NV];
        for (int v = 0; v < NV; ++v) {
            Var* x = vars[(size_t)v];
            ts[v] = { x->cid.on_dev(s), x->cval.on_dev(s), x->prm.on_dev(s), x->sts.on_dev(s), x->tok.on_dev(s) };
        }
        /* the floor: the same launch with every state inert (status 0) */
        std::vector<Var*> inert;
        for (int v = 0; v < NV; ++v) {
            Var* x = new Var(M, NC, SW);
            memcpy(x->cid.host.data(), vars[(size_t)v]->cid.host.data(), x->cid.host.size());
            memcpy(x->cval.host.data(), vars[(size_t)v]->cval.host.data(), x->cval.host.size());
            memcpy(x->prm.host.data(), vars[(size_t)v]->prm.host.data(), x->prm.host.size());
            inert.push_back(x);
            tn[v] = { x->cid.on_dev(s), x->cval.on_dev(s), x->prm.on_dev(s), x->sts.on_dev(s), x->tok.on_dev(s) };
        }
        auto run = [&](std::vector<RadTensor>* t, int reps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                RadArgs a{}; a.t = t[r % NV].data(); a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
                if (dver->launch(&a, s) != RAD_OK) return -1.0;
            }
            rad_stream_sync(s);
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        run(ts, 50); run(tn, 50);
        double best = 1e9, floor_ = 1e9;
        for (int rep = 0; rep < 5; ++rep) { best = std::min(best, run(ts, nrep)); floor_ = std::min(floor_, run(tn, nrep)); }
        fprintf(stderr, "  BENCH live=%4d g=%d  block %.1f us/call  floor %.1f us  net per block decision %.1f us\n", live, g, best, floor_, (best - floor_) / divide);
        for (auto* x : vars) delete x;
        for (auto* x : inert) delete x;
    }
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()

/* advance_test.cpp -- the card's bookkeeping between passes (core/sched/advance_impl.h), run on
 * host arrays, against the host's own acceptance rule and hand-computed rows.
 *
 * advance_impl.h is shared by the kernel and the host backend, so these rows are the arithmetic
 * the card runs. The slab below is carved the way BatchBuilder carves a build: one int32 array
 * per field, a row descriptor block, AdvGroup records and per-group tables.
 */
#include "rad_test.h"

#include "sample/accept.h"
#include "sched/advance_impl.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace rad;

namespace {

static_assert(sizeof(AdvGroup) == 72, "AdvGroup's layout is part of the card's slab");

/* Word offsets of every array the advance reads or writes. Up to kSeq rows, kTok tokens and
 * kGroups KV groups, each group with a kPitch-entry block table per row. */
struct Slab {
    static constexpr int kSeq = 8, kTok = 64, kPitch = 16, kGroups = 2, kPathPitch = 12;
    enum : int32_t {
        ROWS  = 0,                              /* [4][kSeq] row descriptors */
        TOK   = ROWS + 4 * kSeq,
        POS   = TOK + kTok,
        CTX   = POS + kTok,
        ACC   = CTX + kSeq,
        OUT   = ACC + kSeq,
        SID   = OUT + kTok,
        CU    = SID + kSeq,
        Q     = CU + kSeq + 8,
        ROPE  = Q + kSeq,                       /* [3][kTok] */
        GRP   = ROPE + 3 * kTok,                /* [kGroups] AdvGroup, sizeof/4 words each */
        SLOT  = GRP + (int)(sizeof(AdvGroup) / 4) * kGroups,  /* [kGroups][kTok] */
        USED  = SLOT + kGroups * kTok,          /* [kGroups][kSeq] */
        BT    = USED + kGroups * kSeq,          /* [kGroups][kSeq][kPitch] */
        FIRST = BT + kGroups * kSeq * kPitch,   /* [kGroups][kSeq] */
        PAR   = FIRST + kGroups * kSeq,         /* [kTok] spec_parent, when a test stages trees */
        PATH  = PAR + kTok,                     /* [kSeq][kPathPitch] spec_path */
        END   = PATH + kSeq * kPathPitch
    };
    alignas(16) int32_t w[END] = {};
    StepSlot st[kSeq];
    int32_t  src[3 * kTok] = {};
    int32_t  eog[8] = {};
    AdvanceArgs a;

    /* `n_groups` groups: group 0 paged with `block` tokens a block, group 1 (when present) a
     * state group. Every block table starts as block b at entry b, so slot == position. */
    Slab(const Slab&) = delete;   /* `a` points into the object */
    explicit Slab(int mode, int n_groups = 1, int block = 16) {
        std::memset(st, 0x7f, sizeof st);
        a.mode = mode;
        a.n_groups = n_groups;
        a.slab = reinterpret_cast<char*>(w);
        a.row_off = ROWS * 4;  a.row_pitch = kSeq;
        a.tok_off = TOK * 4;   a.pos_off = POS * 4; a.ctx_off = CTX * 4; a.acc_off = ACC * 4;
        a.out_off = OUT * 4;   a.sid_off = SID * 4; a.cu_off = CU * 4;   a.q_off = Q * 4;
        a.grp_off = GRP * 4;
        a.st = st;
        a.src = src;
        a.eog = eog;
        for (int g = 0; g < n_groups; ++g) {
            AdvGroup G{};
            G.block_size = g == 0 ? block : 1;
            G.paged = g == 0 ? 1 : 0;
            G.pitch = g == 0 ? kPitch : 1;
            G.slot_off  = (int64_t)(SLOT + g * kTok) * 4;
            G.used_off  = (int64_t)(USED + g * kSeq) * 4;
            G.bt_off    = (int64_t)(BT + g * kSeq * kPitch) * 4;
            G.first_off = (int64_t)(FIRST + g * kSeq) * 4;
            std::memcpy(reinterpret_cast<char*>(w) + GRP * 4 + g * (int)sizeof(AdvGroup), &G,
                        sizeof G);
            for (int i = 0; i < kSeq; ++i)
                for (int b = 0; b < kPitch; ++b) bt(g, i)[b] = b;
        }
    }

    int32_t* bt(int g, int i) { return &w[BT + (g * kSeq + i) * kPitch]; }
    int32_t& first(int g, int i) { return w[FIRST + g * kSeq + i]; }
    int32_t& used(int g, int i) { return w[USED + g * kSeq + i]; }
    int32_t& slot(int g, int t) { return w[SLOT + g * kTok + t]; }

    /* Row i: a device row or not, its tokens [k, k + n), its context, the StepSlot it names, and
     * the advance's row descriptor (first verified token row, first sampled row, mode number). */
    void row(int i, bool dev, int k, int n, int ctx, int card, int rtok, int rout, int aux) {
        w[ROWS + i] = dev ? 1 : 0;
        w[ROWS + kSeq + i] = rtok;
        w[ROWS + 2 * kSeq + i] = rout;
        w[ROWS + 3 * kSeq + i] = aux;
        w[CU + i] = k;
        w[CU + i + 1] = k + n;
        w[Q + i] = n;
        w[CTX + i] = ctx;
        w[SID + i] = card;
        if (i + 1 > a.n_seq) a.n_seq = i + 1;
    }
    void set_eog(std::initializer_list<int32_t> ids) {
        int n = 0;
        for (int32_t t : ids) eog[n++] = t;
        std::sort(eog, eog + n);
        a.n_eog = n;
    }
    void run() { for (int i = 0; i < a.n_seq; ++i) adv_row(a, i); }
};

uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

}  /* namespace */

/* ================================================================== the end-of-generation cut */

/* The host's cut: the first end-of-generation token among the accepted drafts and the target's
 * own token is emitted and ends the run; a run without one is kept whole. */
TEST(the_end_of_generation_cut_keeps_the_stop_token_and_drops_what_follows) {
    const int32_t out[3] = { 5, 9, 7 };            /* two accepted drafts, then the target's */
    auto in = [](std::initializer_list<int32_t> s) {
        return [s](int32_t t) { return std::find(s.begin(), s.end(), t) != s.end(); };
    };
    CHECK_EQ(accept_cut_at_eog(out, 2, in({})), 2);
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 5 })), 0);      /* the first draft ends it */
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 9 })), 1);
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 7 })), 2);      /* the target's own token */
    CHECK_EQ(accept_cut_at_eog(out, 2, in({ 9, 5 })), 0);   /* the earliest wins */
    CHECK_EQ(accept_cut_at_eog(out, 0, in({ 9 })), 0);      /* nothing accepted past out[0] */
}

/* ================================================================== ADV_ACCEPT */

/* The card's acceptance is the host's: for any drafts, targets and end-of-generation set, a row's
 * StepSlot holds the drafts accept_greedy accepts, cut at the first end-of-generation token
 * (accept_cut_at_eog), and the token after them. The engine refuses a step where the two
 * disagree, so a drift between the copies stops serving; and the card's count positions every
 * later pass, so a wrong one writes K/V at the wrong positions. Rows that sampled nothing keep
 * their StepSlot. Seeded; a failure prints the case. */
TEST(the_cards_acceptance_is_accept_greedy_and_the_end_of_generation_cut) {
    const uint32_t seed = 0x5eed2a2u;
    uint32_t rng = seed;
    int bad = 0, cut_cases = 0, partial = 0;
    for (int c = 0; c < 10000 && !bad; ++c) {
        Slab s(ADV_ACCEPT);
        /* Tokens from {1, 2, 3}, so drafts often agree; stop tokens a subset of {2, 3, 8}. */
        int n_eog = 0;
        int32_t eog_ids[3];
        for (int32_t t : { 2, 3, 8 }) if (lcg(rng) % 3 == 0) eog_ids[n_eog++] = t;
        for (int j = 0; j < n_eog; ++j) s.eog[j] = eog_ids[j];
        s.a.n_eog = n_eog;
        auto is_eog = [&](int32_t t) {
            return std::find(eog_ids, eog_ids + n_eog, t) != eog_ids + n_eog;
        };

        const int n_rows = 1 + (int)(lcg(rng) % 4);
        int k = 0, out_row = 0;
        int32_t cards[Slab::kSeq];
        for (int i = 0; i < Slab::kSeq; ++i) cards[i] = i;
        for (int i = Slab::kSeq - 1; i > 0; --i) std::swap(cards[i], cards[lcg(rng) % (i + 1)]);
        struct Want { int card, rout, acc, tok, head, hrow; } want[4];
        for (int i = 0; i < n_rows; ++i) {
            const int kind = (int)(lcg(rng) % 4);      /* 0: mid-prefill, 1: last chunk, else decode */
            const int n_spec = kind >= 2 ? (int)(lcg(rng) % 9) : 0;
            const int n_tok = kind >= 2 ? 1 + n_spec : 2 + (int)(lcg(rng) % 5);
            const int ctx = (int)(lcg(rng) % 1000);
            const int rtok = k + n_tok - 1 - n_spec;
            const int rout = kind == 0 ? -1 : out_row;
            s.row(i, kind >= 2, k, n_tok, ctx, cards[i], rtok, rout, n_spec);
            int32_t draft[8] = {}, target[9] = {};
            for (int j = 0; j < n_tok; ++j) s.w[Slab::TOK + k + j] = 50 + j;
            for (int j = 0; j < n_spec; ++j) {
                draft[j] = 1 + (int32_t)(lcg(rng) % 3);
                s.w[Slab::TOK + rtok + 1 + j] = draft[j];
            }
            for (int j = 0; j <= n_spec; ++j) {
                target[j] = (j < n_spec && lcg(rng) % 10 < 7) ? draft[j]
                                                              : 1 + (int32_t)(lcg(rng) % 3);
                if (rout >= 0) s.src[rout + j] = target[j];
            }
            want[i] = { cards[i], rout, -1, 0, 0, 0 };
            if (rout >= 0) {
                int32_t out[9] = {}, acc = -1;
                CHECK_OK(accept_greedy(draft, n_spec, target, out, &acc));
                const int32_t cut = accept_cut_at_eog(out, acc, is_eog);
                cut_cases += cut < acc;
                partial += cut > 0 && cut < n_spec;
                want[i].acc = cut;
                want[i].tok = out[cut];
                want[i].head = ctx + (rtok - k) + cut;
                want[i].hrow = rtok + cut;   /* the row of the last accepted position */
                out_row += 1 + n_spec;
            }
            k += n_tok;
        }
        s.run();
        for (int i = 0; i < n_rows && !bad; ++i) {
            const StepSlot& S = s.st[want[i].card];
            bool ok;
            if (want[i].rout < 0) {
                StepSlot untouched;
                std::memset(&untouched, 0x7f, sizeof untouched);
                ok = std::memcmp(&S, &untouched, sizeof S) == 0;
            } else {
                ok = S.acc == want[i].acc && S.tok[0] == want[i].tok && S.n_acc == want[i].acc + 1 &&
                     S.head == want[i].head && S.ctx == want[i].head + 1 &&
                     S.hrow == want[i].hrow;
            }
            if (!ok) {
                bad = 1;
                CHECK(ok);
                std::fprintf(stderr, "    seed 0x%x, case %d, row %d: card acc %d tok %d head %d "
                             "ctx %d n_acc %d; host acc %d tok %d head %d (sampled row %d)\n",
                             seed, c, i, S.acc, S.tok[0], S.head, S.ctx, S.n_acc, want[i].acc,
                             want[i].tok, want[i].head, want[i].rout);
            }
        }
    }
    CHECK_EQ(bad, 0);
    /* The draw reaches the interesting branches, not only full acceptance or none. */
    CHECK(cut_cases > 100);
    CHECK(partial > 100);
}

/* The same over trees (rad_drafter_tree): a row whose spec_parent marks a tree verify is accepted
 * by accept_tree's walk (the lowest-numbered child that is the target's pick at its parent, from
 * the bonus token down), cut at the first end-of-generation token, and the StepSlot carries the
 * accepted nodes for ADV_COMPACT. Random trees of 0-8 nodes over {1, 2, 3}, so siblings often
 * share a token and the lowest-numbered one must win. Seeded; a failure prints the case. */
TEST(the_cards_tree_acceptance_is_accept_tree_and_the_end_of_generation_cut) {
    const uint32_t seed = 0x7ee5eedu;
    uint32_t rng = seed;
    int bad = 0, deep = 0, branched = 0, cut_cases = 0;
    for (int c = 0; c < 10000 && !bad; ++c) {
        Slab s(ADV_ACCEPT);
        s.a.parent_off = Slab::PAR * 4;
        for (int t = 0; t < Slab::kTok; ++t) s.w[Slab::PAR + t] = -2;
        int n_eog = 0;
        int32_t eog_ids[3];
        for (int32_t t : { 2, 3, 8 }) if (lcg(rng) % 4 == 0) eog_ids[n_eog++] = t;
        for (int j = 0; j < n_eog; ++j) s.eog[j] = eog_ids[j];
        s.a.n_eog = n_eog;
        auto is_eog = [&](int32_t t) {
            return std::find(eog_ids, eog_ids + n_eog, t) != eog_ids + n_eog;
        };

        const int n_rows = 1 + (int)(lcg(rng) % 4);
        int k = 0, out_row = 0;
        struct Want { int card, acc, tok, head, hrow; int32_t path[8]; } want[4];
        for (int i = 0; i < n_rows; ++i) {
            const int n_spec = (int)(lcg(rng) % 9);
            const int n_tok = 1 + n_spec;
            const int ctx = (int)(lcg(rng) % 1000);
            const int rtok = k;                       /* a decode row: its first token is the bonus */
            s.row(i, true, k, n_tok, ctx, i, rtok, out_row, n_spec);
            int32_t draft[8] = {}, parent[8] = {}, target[9] = {};
            s.w[Slab::TOK + rtok] = 50;
            s.w[Slab::PAR + rtok] = -1;                /* the root: this row is a tree verify */
            for (int j = 1; j <= n_spec; ++j) {
                parent[j - 1] = (int32_t)(lcg(rng) % (uint32_t)j);
                draft[j - 1] = 1 + (int32_t)(lcg(rng) % 3);
                s.w[Slab::TOK + rtok + j] = draft[j - 1];
                s.w[Slab::PAR + rtok + j] = (rtok - k) + parent[j - 1];
                branched += parent[j - 1] != j - 1;
            }
            for (int r = 0; r <= n_spec; ++r) {
                target[r] = 1 + (int32_t)(lcg(rng) % 3);
                s.src[out_row + r] = target[r];
            }
            int32_t out[9] = {}, path[8] = {}, acc = -1;
            CHECK_OK(accept_tree(draft, parent, n_spec, target, out, path, &acc));
            const int32_t cut = accept_cut_at_eog(out, acc, is_eog);
            deep += cut >= 2;
            cut_cases += cut < acc;
            /* the row of the last node on the accepted path: rows are nodes, in order */
            want[i] = { i, cut, out[cut], ctx + cut, rtok + (cut == 0 ? 0 : path[cut - 1]), {} };
            std::memcpy(want[i].path, path, sizeof path);
            out_row += 1 + n_spec;
            k += n_tok;
        }
        s.run();
        for (int i = 0; i < n_rows && !bad; ++i) {
            const StepSlot& S = s.st[want[i].card];
            bool ok = S.acc == want[i].acc && S.tok[0] == want[i].tok && S.n_acc == want[i].acc + 1 &&
                      S.head == want[i].head && S.ctx == want[i].head + 1 && S.npath == want[i].acc &&
                      S.hrow == want[i].hrow;
            for (int t = 0; ok && t < want[i].acc; ++t) ok = S.path[t] == want[i].path[t];
            if (!ok) {
                bad = 1;
                CHECK(ok);
                std::fprintf(stderr, "    seed 0x%x, case %d, row %d: card acc %d tok %d npath %d; "
                             "host acc %d tok %d\n", seed, c, i, S.acc, S.tok[0], S.npath,
                             want[i].acc, want[i].tok);
            }
        }
    }
    CHECK_EQ(bad, 0);
    /* The draw reaches branching trees, accepted paths of two nodes or more, and the cut. */
    CHECK(branched > 1000);
    CHECK(deep > 100);
    CHECK(cut_cases > 100);
}

/* ================================================================== ADV_TRUNK */

/* A verify step issued ahead takes its rows from the StepSlots: the bonus token and the head's
 * proposals as tokens, positions from the committed context, the rollback count, each group's
 * length and every slot through the row's own block table. Rows differ in offset, length,
 * context and StepSlot; a host row keeps what the host staged; a position past the table's pitch
 * or on an unassigned block gets no slot. A wrong row here writes K/V into another sequence's
 * blocks or reads the wrong token, with no error anywhere. */
TEST(a_step_issued_ahead_takes_every_rows_tokens_positions_and_slots_from_its_step_slot) {
    Slab s(ADV_TRUNK, /*n_groups=*/2, /*block=*/4);
    /* Row 0: three positions (two drafts) for card 2, block b of its table at 100 + 3b. */
    s.row(0, true, 0, 3, /*ctx bound=*/45, /*card=*/2, 0, 0, 2);
    for (int b = 0; b < Slab::kPitch; ++b) s.bt(0, 0)[b] = 100 + 3 * b;
    s.st[2].ctx = 37; s.st[2].n_acc = 2;
    s.st[2].tok[0] = 501; s.st[2].tok[1] = 502; s.st[2].tok[2] = 503;
    /* Row 1: a host-staged prefill row, which no advance may touch. */
    s.row(1, false, 3, 2, 70, 4, 4, -1, 0);
    s.w[Slab::TOK + 3] = 900; s.w[Slab::TOK + 4] = 901;
    s.w[Slab::POS + 3] = 70;  s.w[Slab::POS + 4] = 71;
    s.slot(0, 3) = 1234;      s.slot(0, 4) = 1235;
    s.w[Slab::ACC + 1] = 1;
    s.used(0, 1) = 72;
    /* Row 2: one position for card 0, whose second block was never assigned. */
    s.row(2, true, 5, 1, 64, 0, 5, 3, 0);
    s.bt(0, 2)[1] = -1;
    s.st[0].ctx = 6; s.st[0].n_acc = 1; s.st[0].tok[0] = 777;
    /* Row 3: three positions for card 5 in a table rebased at 8; the last falls past the
     * pitch (position 72 is block (72 - 8) / 4 = 16). */
    s.row(3, true, 6, 3, 80, 5, 6, 4, 2);
    s.first(0, 3) = 8;
    s.st[5].ctx = 70; s.st[5].n_acc = 3;
    s.st[5].tok[0] = 601; s.st[5].tok[1] = 602; s.st[5].tok[2] = 603;
    s.run();

    const int32_t tok[9]  = { 501, 502, 503, 900, 901, 777, 601, 602, 603 };
    const int32_t pos[9]  = { 37, 38, 39, 70, 71, 6, 70, 71, 72 };
    /* Paged slot = table[(p - first) / 4] * 4 + p % 4. */
    const int32_t slot[9] = { (100 + 3 * 9) * 4 + 1, (100 + 3 * 9) * 4 + 2, (100 + 3 * 9) * 4 + 3,
                              1234, 1235, -1, 15 * 4 + 2, 15 * 4 + 3, -1 };
    for (int t = 0; t < 9; ++t) {
        CHECK_EQ(s.w[Slab::TOK + t], tok[t]);
        CHECK_EQ(s.w[Slab::POS + t], pos[t]);
        CHECK_EQ(s.slot(0, t), slot[t]);
    }
    const int32_t ctx[4] = { 37, 70, 6, 70 }, acc[4] = { 2, 1, 1, 3 };
    const int32_t used0[4] = { 40, 72, 7, 65 }, used1[4] = { 40, 0, 7, 73 };
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(s.w[Slab::CTX + i], ctx[i]);
        CHECK_EQ(s.w[Slab::ACC + i], acc[i]);
        CHECK_EQ(s.used(0, i), used0[i]);   /* from the table's first position */
        CHECK_EQ(s.used(1, i), used1[i]);   /* a state group counts from 0 and has no slots */
    }
    for (int t = 0; t < 9; ++t) CHECK_EQ(s.slot(1, t), 0);
}

/* A row the card moves keeps the host's offset between its position and its rotary position on
 * all three planes (a text row after an image rotates at its index plus a shift); a row whose
 * position does not move keeps the host's three components, which differ for a media token. A
 * wrong shift misplaces every M-RoPE text row after a picture during speculation. */
TEST(a_moved_row_keeps_its_rotary_shift_and_an_unmoved_row_keeps_its_components) {
    Slab s(ADV_TRUNK);
    s.a.rope_off = Slab::ROPE * 4;
    s.a.rope_pitch = 3;
    int32_t* r = &s.w[Slab::ROPE];
    /* Row 0 (one position) moves from the bound 120 to 100; its rotary index was 120 + 7. */
    s.row(0, true, 0, 1, 120, 0, 0, 0, 0);
    s.w[Slab::POS + 0] = 120;
    r[0] = 127; r[3 + 0] = 127; r[6 + 0] = 127;
    s.st[0].ctx = 100; s.st[0].n_acc = 1; s.st[0].tok[0] = 11;
    /* Row 1 (two positions) is already where the card puts it: a media row's three components
     * stay as the host staged them. */
    s.row(1, true, 1, 2, 50, 1, 1, 1, 1);
    s.w[Slab::POS + 1] = 50; s.w[Slab::POS + 2] = 51;
    r[1] = 10; r[3 + 1] = 20; r[6 + 1] = 30;
    r[2] = 31; r[3 + 2] = 21; r[6 + 2] = 11;
    s.st[1].ctx = 50; s.st[1].n_acc = 1; s.st[1].tok[0] = 12; s.st[1].tok[1] = 13;
    s.run();
    CHECK_EQ(s.w[Slab::POS + 0], 100);
    for (int p = 0; p < 3; ++p) CHECK_EQ(r[3 * p + 0], 107);
    CHECK_EQ(r[1], 10); CHECK_EQ(r[3 + 1], 20); CHECK_EQ(r[6 + 1], 30);
    CHECK_EQ(r[2], 31); CHECK_EQ(r[3 + 2], 21); CHECK_EQ(r[6 + 2], 11);
}

/* ================================================================== the draft passes */

/* Round r reads row i's proposal at i * src_pitch: a head with a proposal buffer lays rows out at
 * its own pitch. Reading at i instead gives sequence i the proposal of sequence i / pitch, and
 * acceptance collapses with no error. Both the round's token and the final take follow it. */
TEST(draft_rounds_read_each_rows_proposal_at_the_buffers_pitch) {
    for (int mode : { (int)ADV_ROUND, (int)ADV_TAKE }) {
        Slab s(mode);
        s.a.round = 3;
        s.a.src_pitch = 3;
        for (int i = 0; i < 3; ++i) {
            s.row(i, true, i, 1, 0, /*card=*/2 - i, i, i, 0);
            s.st[2 - i].head = 100 * (i + 1);
            for (int c = 0; c < 3; ++c) s.src[i * 3 + c] = 1000 * (i + 1) + c;
        }
        s.run();
        for (int i = 0; i < 3; ++i) {
            const StepSlot& S = s.st[2 - i];
            if (mode == ADV_ROUND) {
                CHECK_EQ(S.tok[2], 1000 * (i + 1));
                CHECK_EQ(s.w[Slab::TOK + i], 1000 * (i + 1));
                CHECK_EQ(s.w[Slab::POS + i], 100 * (i + 1) + 2);   /* head + round - 1 */
                CHECK_EQ(s.w[Slab::CTX + i], 100 * (i + 1) + 2);
            } else {
                CHECK_EQ(S.tok[3], 1000 * (i + 1));
            }
        }
    }
}

/* A history pass with `aux` host-staged positions before the verified ones: those keep the host's
 * tokens, the verified positions up to the last committed one take the sampled tokens, and rows
 * past it repeat it, so no rejected draft's hidden state reaches the head's cache. */
TEST(a_history_pass_keeps_the_hosts_rows_and_repeats_the_last_committed_one) {
    Slab s(ADV_HIST);
    s.row(0, true, 0, 5, 0, 1, 0, 6, /*aux=*/2);
    s.w[Slab::TOK + 0] = 41; s.w[Slab::TOK + 1] = 42;
    s.w[Slab::OUT + 0] = 20;
    s.st[1].head = 103; s.st[1].acc = 1;
    s.src[6] = 7001; s.src[7] = 7002; s.src[8] = 7003;
    s.run();
    /* base = head - acc - aux = 100; rows reach index aux + acc = 3. */
    const int32_t pos[5] = { 100, 101, 102, 103, 103 }, tok[5] = { 41, 42, 7001, 7002, 7002 };
    for (int j = 0; j < 5; ++j) {
        CHECK_EQ(s.w[Slab::POS + j], pos[j]);
        CHECK_EQ(s.w[Slab::TOK + j], tok[j]);
        CHECK_EQ(s.w[Slab::OUT + j], 20 + (j < 3 ? j : 3));
        CHECK_EQ(s.slot(0, j), pos[j]);
    }
    CHECK_EQ(s.w[Slab::CTX], 100);
    CHECK_EQ(s.used(0, 0), 104);        /* ends at the last committed position, 103 */
}

/* Attention places row j of a q_len-row run at seqused - q_len + j and reads keys up to there
 * (attn_paged's contract, libref/ref_attn.cpp; libr4d's decode kernel computes the same `pos`).
 * So on every pass the card stages, the last row of a run -- round 1's input on a history pass,
 * the only row whose output the head goes on with -- must sit exactly at its own position, and no
 * row may count past its own: a key past it is a position the head's layer has not written for
 * this sequence. (After a verify that accepted acc < n_spec drafts, a history pass's whole run
 * length would put its last row n_spec - acc positions too far.) */
TEST(no_draft_pass_row_reads_keys_past_its_own_position) {
    const int n_spec = 3;
    for (int aux = 0; aux <= 2; ++aux)
        for (int acc = 0; acc <= n_spec; ++acc) {
            Slab s(ADV_HIST);
            const int n = aux + 1 + n_spec, head = 200 + aux + acc;
            s.row(0, true, 0, n, 0, /*card=*/3, 0, 0, aux);
            s.st[3].head = head; s.st[3].acc = acc;
            s.run();
            const int32_t used = s.used(0, 0);
            for (int j = 0; j < n; ++j)
                CHECK(used - n + j <= s.w[Slab::POS + j]);
            CHECK_EQ(s.w[Slab::POS + n - 1], head);
            CHECK_EQ(used - n + (n - 1), head);
        }
    for (int round = 2; round <= n_spec; ++round) {
        Slab s(ADV_ROUND);
        s.a.round = round;
        s.row(0, true, 0, 1, 0, /*card=*/1, 0, 0, 0);
        s.st[1].head = 300;
        s.run();
        CHECK_EQ(s.used(0, 0) - 1, s.w[Slab::POS]);
        CHECK_EQ(s.w[Slab::POS], 300 + round - 1);
    }
}

/* The same pass right after a one- or two-token prompt, where the run is longer than the context
 * it ends at (head + 1 < aux + 1 + n_spec): its first rows then count before position 0. Nothing
 * reads their output and attention gives such a row zeros (op_attn_paged_rows_before_the_context
 * in ref_test); what must hold is that every row still stores at a real position of its own
 * table, the context is at least one key, and the last row sits at the committed position. */
TEST(a_history_pass_right_after_a_tiny_prompt_stores_every_row_at_a_real_position) {
    const int n_spec = 3;
    int negative = 0;
    for (int base = 1; base <= 2; ++base)            /* the bonus token's position: a 1-2 token prompt */
        for (int aux = 0; aux <= 2; ++aux)
            for (int acc = 0; acc <= n_spec; ++acc) {
                Slab s(ADV_HIST, /*n_groups=*/1, /*block=*/4);
                const int n = aux + 1 + n_spec, head = base + aux + acc;
                s.row(0, true, 0, n, 0, /*card=*/2, 0, 0, aux);
                s.st[2].head = head; s.st[2].acc = acc;
                s.run();
                const int32_t used = s.used(0, 0);
                CHECK_EQ(used, head + 1);
                CHECK(used >= 1);
                for (int j = 0; j < n; ++j) {
                    const int32_t p = s.w[Slab::POS + j];
                    CHECK(p >= base && p <= head);
                    CHECK_EQ(s.slot(0, j), p);         /* block b at entry b: slot == position */
                }
                CHECK_EQ(used - n + (n - 1), head);
                if (used - n < 0) ++negative;
            }
    CHECK(negative > 0);                            /* the cases the test is for were reached */
}

/* What the card writes into a draft pass, for both head shapes. A verify at positions 101..105
 * accepted two drafts: committed position 103, trunk token 7003; slot p is position p. A head
 * with its own cache gets the verified run, tail repeated, and a cacheless head the committed
 * position; both get a context ending there, every round. */
TEST(the_card_stages_a_cacheless_head_at_the_committed_position) {
    auto slab = [](Slab& s, int n_rows, int round, bool shared, int shift) {
        s.a.round = round;
        s.a.shared_kv = shared ? 1 : 0;
        s.a.rot_shift = shift;
        s.row(0, true, 0, n_rows, 0, 0, 0, 0, 0);
        s.w[Slab::OUT] = 40;   /* the host staged the run's first trunk row */
        s.src[0] = 7001; s.src[1] = 7002; s.src[2] = 7003; s.src[3] = 7004; s.src[4] = 7005;
        s.st[0].head = 103; s.st[0].acc = 2; s.st[0].ctx = 104; s.st[0].n_acc = 3;
        s.st[0].tok[0] = 7003;
    };
    /* The history pass. */
    {
        Slab o(ADV_HIST);
        slab(o, 5, 0, false, 0);
        o.run();
        const int32_t pos[5] = { 101, 102, 103, 103, 103 }, tok[5] = { 7001, 7002, 7003, 7003, 7003 };
        for (int j = 0; j < 5; ++j) {
            CHECK_EQ(o.w[Slab::POS + j], pos[j]);
            CHECK_EQ(o.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(o.w[Slab::OUT + j], 40 + (j < 2 ? j : 2));
            CHECK_EQ(o.slot(0, j), pos[j]);
        }
        CHECK_EQ(o.used(0, 0), 104);        /* the committed positions and no more */
        CHECK_EQ(o.w[Slab::CTX], 101);
    }
    for (int shift = 0; shift < 2; ++shift) {
        Slab s(ADV_HIST);
        slab(s, 1, 0, true, shift);
        s.run();
        CHECK_EQ(s.w[Slab::POS], 103 + shift);
        CHECK_EQ(s.w[Slab::TOK], 7003);
        CHECK_EQ(s.w[Slab::OUT], 42);
        CHECK_EQ(s.used(0, 0), 104);        /* the committed positions and no more */
        CHECK_EQ(s.w[Slab::CTX], 103);
    }
    /* Round 3: the old head two positions on, the cacheless one where it was. */
    {
        Slab o(ADV_ROUND);
        slab(o, 1, 3, false, 0);
        o.src[0] = 9001;
        o.run();
        CHECK_EQ(o.w[Slab::POS], 105);
        CHECK_EQ(o.used(0, 0), 106);
        CHECK_EQ(o.w[Slab::TOK], 9001);
        CHECK_EQ(o.st[0].tok[2], 9001);
    }
    for (int shift = 0; shift < 2; ++shift) {
        Slab s(ADV_ROUND);
        slab(s, 1, 3, true, shift);
        s.src[0] = 9001;
        s.run();
        CHECK_EQ(s.w[Slab::POS], 103 + shift);
        CHECK_EQ(s.used(0, 0), 104);
        CHECK_EQ(s.w[Slab::CTX], 103);
        CHECK_EQ(s.w[Slab::TOK], 9001);
        CHECK_EQ(s.st[0].tok[2], 9001);
    }
}

/* ================================================================== the block drafter */

/* A block drafter's query row on the card: the anchor (the verify's chosen token) at the
 * committed end, the row's mask id (`aux`) at every position after it, slots through the row's
 * own table and the group's length to the block's end. The host staged the furthest the anchor
 * could be; a host row is left as staged. A wrong base drafts from the wrong position and
 * acceptance collapses with no error; a wrong slot writes the drafter's K/V into another block. */
TEST(a_block_query_row_takes_its_anchor_and_position_from_the_step_slot) {
    Slab s(ADV_BLOCK, /*n_groups=*/1, /*block=*/4);
    /* Row 0: a block of four for card 3, staged at 60; the card has it at 50. */
    s.row(0, true, 0, 4, 60, 3, 0, -1, /*mask=*/4);
    for (int b = 0; b < Slab::kPitch; ++b) s.bt(0, 0)[b] = 100 + b;
    s.st[3].ctx = 50; s.st[3].tok[0] = 777;
    /* Row 1: host-staged, untouched. */
    s.row(1, false, 4, 2, 30, 4, 0, -1, 9);
    s.w[Slab::TOK + 4] = 900; s.w[Slab::TOK + 5] = 901;
    s.w[Slab::POS + 4] = 30;  s.w[Slab::POS + 5] = 31;
    /* Row 2: card 1 at 7 with mask 9; its block 1 (positions 4..7) was never assigned. */
    s.row(2, true, 6, 4, 20, 1, 0, -1, 9);
    s.bt(0, 2)[1] = -1;
    s.st[1].ctx = 7; s.st[1].tok[0] = 555;
    s.run();

    const int32_t tok[10] = { 777, 4, 4, 4, 900, 901, 555, 9, 9, 9 };
    const int32_t pos[10] = { 50, 51, 52, 53, 30, 31, 7, 8, 9, 10 };
    const int32_t slot[10] = { 112 * 4 + 2, 112 * 4 + 3, 113 * 4 + 0, 113 * 4 + 1, 0, 0,
                               -1, 8, 9, 10 };
    for (int t = 0; t < 10; ++t) {
        CHECK_EQ(s.w[Slab::TOK + t], tok[t]);
        CHECK_EQ(s.w[Slab::POS + t], pos[t]);
        CHECK_EQ(s.slot(0, t), slot[t]);
    }
    CHECK_EQ(s.w[Slab::CTX + 0], 50);
    CHECK_EQ(s.w[Slab::CTX + 1], 30);
    CHECK_EQ(s.w[Slab::CTX + 2], 7);
    CHECK_EQ(s.used(0, 0), 54);
    CHECK_EQ(s.used(0, 2), 11);
    CHECK_EQ(s.st[3].tok[0], 777);   /* the anchor stays the next verify's first token */
}

/* The block's proposals go into the slot behind the anchor, row i's from i * src_pitch: exactly
 * the tokens the next verify reads after its first. The anchor and host rows are untouched. */
TEST(a_block_take_puts_each_rows_proposals_behind_its_anchor) {
    Slab s(ADV_BTAKE);
    s.a.round = 3;
    s.a.src_pitch = 3;
    for (int i = 0; i < 3; ++i) {
        s.row(i, true, 4 * i, 4, 0, /*card=*/2 - i, 0, -1, 4);
        s.st[2 - i].tok[0] = 50 + i;
        for (int c = 0; c < 3; ++c) s.src[i * 3 + c] = 1000 * (i + 1) + c;
    }
    s.row(3, false, 12, 4, 0, 5, 0, -1, 4);
    s.src[9] = 4242;
    s.run();
    for (int i = 0; i < 3; ++i) {
        const StepSlot& S = s.st[2 - i];
        CHECK_EQ(S.tok[0], 50 + i);
        for (int c = 0; c < 3; ++c) CHECK_EQ(S.tok[1 + c], 1000 * (i + 1) + c);
    }
    CHECK(s.st[5].tok[1] != 4242);
}

/* The chain a pipelined block-drafter step runs, row for row: the verify's acceptance, the query
 * block at the committed end, the take, then the step issued ahead. A verify at 101..105 (four
 * drafts) accepts two, so the anchor is the target's third pick at 104, and the next verify
 * starts there with the anchor and the block's proposals. */
TEST(a_block_drafter_step_on_the_card_continues_where_the_verify_left_off) {
    StepSlot st;
    /* The verify: 7000 then drafts 7101..7104; the target picks 7101, 7102, 9999, ... */
    {
        Slab v(ADV_ACCEPT);
        v.row(0, false, 0, 5, 101, 1, 0, 0, 4);
        const int32_t t[5] = { 7000, 7101, 7102, 7103, 7104 };
        const int32_t y[5] = { 7101, 7102, 9999, 1, 2 };
        for (int j = 0; j < 5; ++j) { v.w[Slab::TOK + j] = t[j]; v.src[j] = y[j]; }
        v.run();
        st = v.st[1];
    }
    CHECK_EQ(st.acc, 2);
    CHECK_EQ(st.ctx, 104);
    CHECK_EQ(st.tok[0], 9999);
    /* The query block of four, staged at the furthest anchor, 106. */
    {
        Slab q(ADV_BLOCK);
        q.st[1] = st;
        q.row(0, true, 0, 4, 106, 1, 0, -1, 4);
        q.run();
        const int32_t tok[4] = { 9999, 4, 4, 4 };
        for (int j = 0; j < 4; ++j) {
            CHECK_EQ(q.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(q.w[Slab::POS + j], 104 + j);
        }
        st = q.st[1];
    }
    /* Its three proposals, then the next verify of 1 + 3. */
    {
        Slab t(ADV_BTAKE);
        t.st[1] = st;
        t.a.round = 3;
        t.a.src_pitch = 3;
        t.row(0, true, 0, 4, 106, 1, 0, -1, 4);
        t.src[0] = 8001; t.src[1] = 8002; t.src[2] = 8003;
        t.run();
        st = t.st[1];
    }
    {
        Slab n(ADV_TRUNK);
        n.st[1] = st;
        n.row(0, true, 0, 4, 106, 1, 0, 0, 3);
        n.run();
        const int32_t tok[4] = { 9999, 8001, 8002, 8003 };
        for (int j = 0; j < 4; ++j) {
            CHECK_EQ(n.w[Slab::TOK + j], tok[j]);
            CHECK_EQ(n.w[Slab::POS + j], 104 + j);
        }
        CHECK_EQ(n.w[Slab::CTX], 104);
        CHECK_EQ(n.w[Slab::ACC], 3);
    }
}

/* ================================================================== a serial head's tree */

namespace {

/* A serial tree's advance as the engine drives it: R rounds of a head whose proposal rows are
 * [pick, runner-ups...] at pitch 8, leaves L, budget N. Row i's round-r proposal row is
 * rows[r][i]. Returns the StepSlots after ADV_TAKE. */
struct SerialRun {
    static constexpr int kRounds = 3;
    int32_t rows[kRounds + 1][2][8] = {};   /* [round][drafting row][column] */
    StepSlot st[2];
    void go(int rounds, int leaves, int nodes, int vocab, int n_rows = 2) {
        for (int i = 0; i < 2; ++i) std::memset(&st[i], 0, sizeof st[i]);
        for (int i = 0; i < n_rows; ++i) st[i].head = 100 + 10 * i;
        for (int r = 2; r <= rounds; ++r) {
            Slab s(ADV_ROUND);
            s.a.round = r;
            s.a.src_pitch = 8;
            s.a.tree_nodes = nodes; s.a.leaves = leaves; s.a.vocab = vocab;
            std::memcpy(s.st, st, sizeof st);
            for (int i = 0; i < n_rows; ++i) {
                s.row(i, true, i, 1, 0, i, i, i, 0);
                std::memcpy(&s.src[i * 8], rows[r - 1][i], 8 * 4);
            }
            s.run();
            std::memcpy(st, s.st, sizeof st);
        }
        Slab s(ADV_TAKE);
        s.a.round = rounds;
        s.a.src_pitch = 8;
        s.a.tree_nodes = nodes; s.a.leaves = leaves; s.a.vocab = vocab;
        std::memcpy(s.st, st, sizeof st);
        for (int i = 0; i < n_rows; ++i) {
            s.row(i, true, i, 1, 0, i, i, i, 0);
            std::memcpy(&s.src[i * 8], rows[rounds][i], 8 * 4);
        }
        s.run();
        std::memcpy(st, s.st, sizeof st);
    }
};

}  /* namespace */

/* Stage 1: the first round's runner-up columns become leaves hanging off the bonus token, behind
 * the chain, before round 2's pass overwrites them. An id that is the pick, repeated, negative or
 * past the vocabulary is no leaf; a short list is padded with the chain's first token so the
 * node count is the budget's alone (the host planned the step issued ahead by it). */
TEST(a_serial_tree_takes_the_first_rounds_runner_ups_as_leaves_behind_the_chain) {
    SerialRun r;
    /* round 1: pick 500, then 500 (the pick), 601, 601 (repeat), -1, 999999 (past the vocab), 602,
     * 603; later rounds' candidates are not leaves. */
    const int32_t c0[8] = { 500, 500, 601, 601, -1, 999999, 602, 603 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    const int32_t c1[8] = { 510, 701, 702, 703, 704, 705, 706, 707 };
    std::memcpy(r.rows[1][1], c1, sizeof c1);
    for (int i = 0; i < 2; ++i) {
        r.rows[2][i][0] = 520 + i;
        r.rows[3][i][0] = 530 + i;
    }
    r.go(/*rounds=*/3, /*leaves=*/3, /*nodes=*/6, /*vocab=*/1000);
    /* row 0: chain 500 520 530, leaves 601 602 603 (valid ones, in column order) */
    const StepSlot& a = r.st[0];
    CHECK_EQ(a.ntree, 6);
    const int32_t at[7] = { 0, 500, 520, 530, 601, 602, 603 };
    for (int j = 1; j <= 6; ++j) CHECK_EQ(a.tok[j], at[j]);
    const int32_t ap[6] = { 0, 1, 2, 0, 0, 0 };      /* node j + 1's parent */
    for (int j = 0; j < 6; ++j) CHECK_EQ(a.par[j], ap[j]);
    /* row 1: leaves 701 702 703 */
    const StepSlot& b = r.st[1];
    CHECK_EQ(b.ntree, 6);
    const int32_t bt[7] = { 0, 510, 521, 531, 701, 702, 703 };
    for (int j = 1; j <= 6; ++j) CHECK_EQ(b.tok[j], bt[j]);
}

TEST(a_serial_tree_pads_a_short_leaf_list_with_the_chains_first_token_and_caps_at_the_budget) {
    SerialRun r;
    /* Only one valid runner-up, three leaves asked for: the other two repeat node 1's token. */
    const int32_t c0[8] = { 500, 500, 601, -5, 1000, 500, -1, -1 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    for (int i = 0; i < 2; ++i) { r.rows[2][i][0] = 520; r.rows[3][i][0] = 530; }
    r.go(3, 3, 6, 1000, 1);
    const StepSlot& a = r.st[0];
    CHECK_EQ(a.ntree, 6);
    CHECK_EQ(a.tok[4], 601);
    CHECK_EQ(a.tok[5], 500);
    CHECK_EQ(a.tok[6], 500);
    for (int q = 3; q < 6; ++q) CHECK_EQ(a.par[q], 0);
    /* A budget of 4 over 3 rounds holds one leaf only. */
    SerialRun r2 = r;
    r2.go(3, 3, 4, 1000, 1);
    CHECK_EQ(r2.st[0].ntree, 4);
    CHECK_EQ(r2.st[0].tok[4], 601);
    /* A budget equal to the rounds is the chain, still a (chain-shaped) tree. */
    SerialRun r3 = r;
    r3.go(3, 3, 3, 1000, 1);
    CHECK_EQ(r3.st[0].ntree, 3);
    for (int j = 0; j < 3; ++j) CHECK_EQ(r3.st[0].par[j], j);
}

/* One round: ADV_TAKE has the candidates before anything overwrites them. */
TEST(a_serial_tree_of_one_round_takes_its_leaves_in_the_take) {
    SerialRun r;
    const int32_t c0[8] = { 500, 601, 602, 500, 603, 604, 605, 606 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    r.go(1, 2, 3, 1000, 1);
    const StepSlot& a = r.st[0];
    CHECK_EQ(a.ntree, 3);
    CHECK_EQ(a.tok[1], 500);
    CHECK_EQ(a.tok[2], 601);
    CHECK_EQ(a.tok[3], 602);
    CHECK_EQ(a.par[0], 0);
    CHECK_EQ(a.par[1], 0);
    CHECK_EQ(a.par[2], 0);
}

/* Without leaves (or the tree fields) a chain round and take behave as before: no capture, no
 * parents written, ntree untouched. */
TEST(a_chain_take_and_round_ignore_the_tree_fields_when_there_is_none) {
    StepSlot keep{};
    keep.ntree = 77; keep.nalt = 5; keep.par[0] = 9;
    SerialRun r;
    for (int i = 0; i < 2; ++i) { r.rows[1][i][0] = 11; r.rows[2][i][0] = 12; }
    r.rows[1][0][1] = 13;
    for (int i = 0; i < 2; ++i) r.st[i] = keep;
    Slab s(ADV_ROUND);
    s.a.round = 2; s.a.src_pitch = 8;
    s.st[0] = keep;
    s.row(0, true, 0, 1, 0, 0, 0, 0, 0);
    std::memcpy(s.src, r.rows[1][0], 8 * 4);
    s.run();
    CHECK_EQ(s.st[0].ntree, 77);
    CHECK_EQ(s.st[0].nalt, 5);
    CHECK_EQ(s.st[0].par[0], 9);
    CHECK_EQ(s.st[0].tok[1], 11);
    Slab t(ADV_TAKE);
    t.a.round = 2; t.a.src_pitch = 8;
    t.st[0] = keep;
    t.row(0, true, 0, 1, 0, 0, 0, 0, 0);
    std::memcpy(t.src, r.rows[2][0], 8 * 4);
    t.run();
    CHECK_EQ(t.st[0].ntree, 77);
    CHECK_EQ(t.st[0].tok[2], 12);
}

/* The verify of that tree on the card, then its acceptance and the next step's path plane. Row 0
 * of a step with context 100: bonus 9, chain 500 520 530 (rows 1-3), leaves 601 602 603 (rows
 * 4-6). A leaf accepted is a path of one node at row 5 whose K/V ADV_COMPACT moves to
 * base + 1; the plane the next trunk step reads is [0, 5, identity...]. A chain-prefix path
 * moves nothing and is the identity. */
TEST(a_serial_tree_verifies_in_flat_slots_and_the_path_plane_follows_the_accepted_leaf) {
    SerialRun r;
    const int32_t c0[8] = { 500, 500, 601, 602, 603, 604, 605, 606 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    r.rows[2][0][0] = 520; r.rows[3][0][0] = 530;
    r.go(3, 3, 6, 1000, 1);
    StepSlot st = r.st[0];
    st.tok[0] = 9;
    st.ctx = 100; st.n_acc = 1; st.npath = 0;
    /* ADV_TRUNK: 7 rows */
    Slab v(ADV_TRUNK);
    v.a.parent_off = Slab::PAR * 4;
    v.a.path_off = Slab::PATH * 4; v.a.path_pitch = Slab::kPathPitch;
    v.st[0] = st;
    v.row(0, true, 0, 7, 100, 0, 0, 0, 6);
    v.run();
    const int32_t depth[7] = { 0, 1, 2, 3, 1, 1, 1 };
    const int32_t par[7] = { -1, 0, 1, 2, 0, 0, 0 };
    for (int j = 0; j < 7; ++j) {
        CHECK_EQ(v.w[Slab::TOK + j], j == 0 ? 9 : st.tok[j]);
        CHECK_EQ(v.w[Slab::PAR + j], par[j]);
        CHECK_EQ(v.w[Slab::POS + j], 100 + depth[j]);   /* rotates at its depth */
        CHECK_EQ(v.slot(0, j), 100 + j);                /* stores at its row */
    }
    for (int t = 0; t < Slab::kPathPitch; ++t) CHECK_EQ(v.w[Slab::PATH + t], t);   /* identity */

    /* The target picks leaf 602 after the bonus token: node 5 is accepted, its own pick is 777. */
    Slab acc(ADV_ACCEPT);
    acc.a.parent_off = Slab::PAR * 4;
    for (int j = 0; j < Slab::kTok; ++j) acc.w[Slab::PAR + j] = v.w[Slab::PAR + j];
    for (int j = 0; j < 7; ++j) acc.w[Slab::TOK + j] = v.w[Slab::TOK + j];
    acc.st[0] = st;
    acc.row(0, true, 0, 7, 100, 0, 0, 0, 6);
    const int32_t y[7] = { 602, 1, 2, 3, 4, 777, 5 };
    for (int j = 0; j < 7; ++j) acc.src[j] = y[j];
    acc.run();
    st = acc.st[0];
    CHECK_EQ(st.acc, 1);
    CHECK_EQ(st.npath, 1);
    CHECK_EQ(st.path[0], 5);
    CHECK_EQ(st.tok[0], 777);
    CHECK_EQ(st.head, 101);
    CHECK_EQ(st.hrow, 5);

    /* ADV_COMPACT: node 5 at slot 105 moves to 101, and nothing else moves (a chain-prefix
     * would not move at all). */
    {
        constexpr int kRow = 16;
        alignas(16) char cache[256 * kRow];
        for (int p = 0; p < 256; ++p) std::memset(cache + p * kRow, p, kRow);
        Slab c(ADV_ACCEPT);
        c.st[0] = st;
        c.row(0, true, 0, 7, 100, 0, 0, 0, 6);
        c.a.mode = 0;
        c.a.n_pool = 1;
        c.a.pool[0] = AdvPool{ cache, 0, 1, 1, kRow, 0 };
        c.a.pool[0].grp = 0;
        adv_compact(c.a, 0, 0, 0, 0, 0, 1);
        for (int p = 0; p < 256; ++p) {
            const int want = p == 101 ? 105 : p;
            CHECK_EQ((int)(unsigned char)cache[p * kRow], want);
        }
        /* a chain-prefix path of 2 nodes: path {1, 2}: src == dst, nothing moves */
        for (int p = 0; p < 256; ++p) std::memset(cache + p * kRow, p, kRow);
        StepSlot pre = st;
        pre.acc = 2; pre.npath = 2; pre.path[0] = 1; pre.path[1] = 2; pre.head = 102;
        c.st[0] = pre;
        adv_compact(c.a, 0, 0, 0, 0, 0, 1);
        for (int p = 0; p < 256; ++p) CHECK_EQ((int)(unsigned char)cache[p * kRow], p);
    }

    /* The next trunk step: its path plane is the previous verify's committed rows. */
    Slab nx(ADV_TRUNK);
    nx.a.parent_off = Slab::PAR * 4;
    nx.a.path_off = Slab::PATH * 4; nx.a.path_pitch = Slab::kPathPitch;
    nx.st[0] = st;
    nx.st[0].ntree = 0;       /* a chain step */
    nx.row(0, true, 0, 3, 101, 0, 0, 0, 2);
    nx.run();
    CHECK_EQ(nx.w[Slab::PATH + 0], 0);
    CHECK_EQ(nx.w[Slab::PATH + 1], 5);
    for (int t = 2; t < Slab::kPathPitch; ++t) CHECK_EQ(nx.w[Slab::PATH + t], t);
    CHECK_EQ(nx.w[Slab::ACC], 2);                         /* num_accepted = path length + 1 */
}

/* A path of several nodes, the plane a chain step writes and a row the card does not own. */
TEST(the_path_plane_is_the_identity_for_a_chain_and_untouched_for_a_host_row) {
    Slab s(ADV_TRUNK);
    s.a.path_off = Slab::PATH * 4; s.a.path_pitch = Slab::kPathPitch;
    for (int i = 0; i < Slab::kSeq * Slab::kPathPitch; ++i) s.w[Slab::PATH + i] = -7;
    s.st[0].ctx = 50; s.st[0].n_acc = 4; s.st[0].npath = 0; s.st[0].ntree = 0;
    s.st[1].ctx = 60; s.st[1].n_acc = 3; s.st[1].acc = 2; s.st[1].npath = 2;
    s.st[1].path[0] = 2; s.st[1].path[1] = 5; s.st[1].ntree = 0;
    s.row(0, true, 0, 2, 50, 0, 0, 0, 1);
    s.row(1, true, 2, 2, 60, 1, 2, 2, 1);
    s.row(2, false, 4, 2, 70, 2, 4, 4, 1);     /* a host row */
    s.run();
    for (int t = 0; t < Slab::kPathPitch; ++t) CHECK_EQ(s.w[Slab::PATH + t], t);
    const int32_t want[4] = { 0, 2, 5, 3 };
    for (int t = 0; t < 4; ++t) CHECK_EQ(s.w[Slab::PATH + Slab::kPathPitch + t], want[t]);
    CHECK_EQ(s.w[Slab::PATH + Slab::kPathPitch + 5], 5);
    CHECK_EQ(s.w[Slab::PATH + 2 * Slab::kPathPitch], -7);
}

/* ADV_HIST after a tree verify follows the accepted path: committed position t is node
 * path[t - 1]'s, so its token is the target's pick at that node's row and its hidden row that
 * node's, where a chain verify of the same committed tokens takes the first rows. Everything else
 * (positions, slots, the last-committed repeat, the context's end) is the chain's, row for row. */
TEST(a_history_pass_after_a_tree_verify_follows_the_accepted_path_like_a_sequential_chain) {
    /* tree: bonus + 3 chain + 3 leaves; the leaf at row 5 is accepted, then 777 */
    const int32_t y[7] = { 602, 1, 2, 3, 4, 777, 5 };
    Slab tr(ADV_HIST);
    tr.row(0, true, 0, 4, 0, 1, 0, 0, /*aux=*/0);
    tr.w[Slab::OUT + 0] = 40;
    tr.st[1].head = 101; tr.st[1].acc = 1; tr.st[1].npath = 1; tr.st[1].path[0] = 5;
    for (int j = 0; j < 7; ++j) tr.src[j] = y[j];
    tr.run();
    /* the sequential decode of the committed path: one chain verify [bonus, 602] whose target
     * picks are 602 then 777 */
    Slab ch(ADV_HIST);
    ch.row(0, true, 0, 4, 0, 1, 0, 0, 0);
    ch.w[Slab::OUT + 0] = 40;
    ch.st[1].head = 101; ch.st[1].acc = 1; ch.st[1].npath = 0;
    ch.src[0] = 602; ch.src[1] = 777;
    ch.run();
    for (int j = 0; j < 4; ++j) {
        CHECK_EQ(tr.w[Slab::TOK + j], ch.w[Slab::TOK + j]);
        CHECK_EQ(tr.w[Slab::POS + j], ch.w[Slab::POS + j]);
        CHECK_EQ(tr.slot(0, j), ch.slot(0, j));
    }
    CHECK_EQ(tr.w[Slab::CTX], ch.w[Slab::CTX]);
    CHECK_EQ(tr.used(0, 0), ch.used(0, 0));
    /* the hidden rows differ by the path: the anchor's row 0, then node 5's, repeated */
    const int32_t rows_tree[4] = { 40, 45, 45, 45 }, rows_chain[4] = { 40, 41, 41, 41 };
    for (int j = 0; j < 4; ++j) {
        CHECK_EQ(tr.w[Slab::OUT + j], rows_tree[j]);
        CHECK_EQ(ch.w[Slab::OUT + j], rows_chain[j]);
    }
    /* A deeper path with host-known rows first (aux 2): path {1, 4}: nodes 0, 1, 4. */
    Slab dp(ADV_HIST);
    dp.row(0, true, 0, 5, 0, 1, 0, 8, /*aux=*/2);
    dp.w[Slab::OUT + 0] = 20;
    dp.w[Slab::TOK + 0] = 41; dp.w[Slab::TOK + 1] = 42;
    dp.st[1].head = 110; dp.st[1].acc = 2; dp.st[1].npath = 2;
    dp.st[1].path[0] = 1; dp.st[1].path[1] = 4;
    for (int j = 0; j < 7; ++j) dp.src[8 + j] = 9000 + j;
    dp.run();
    /* base = 110 - 2 - 2 = 106; j: 0,1 host; 2 -> node 0; 3 -> node 1; 4 -> node 4 */
    const int32_t tok[5] = { 41, 42, 9000, 9001, 9004 };
    const int32_t row[5] = { 20, 21, 22, 23, 26 };
    for (int j = 0; j < 5; ++j) {
        CHECK_EQ(dp.w[Slab::TOK + j], tok[j]);
        CHECK_EQ(dp.w[Slab::POS + j], 106 + j);
        CHECK_EQ(dp.w[Slab::OUT + j], row[j]);
    }
}

/* ADV_ROUND_TREE with one frontier row is ADV_ROUND to the bit: the token, the position, the
 * context, the length and the slot, round after round. */
TEST(a_tree_round_of_width_one_is_a_chain_round) {
    for (int r = 2; r <= 5; ++r) {
        Slab c(ADV_ROUND), t(ADV_ROUND_TREE);
        for (Slab* s : { &c, &t }) {
            s->a.round = r; s->a.src_pitch = 4;
            s->a.tree_nodes = 8; s->a.width = 1;
            s->st[0].head = 200; s->st[0].tok[0] = 3;
            s->st[0].ntree = 0;
            s->row(0, true, 0, 1, 0, 0, 0, 0, 0);
            s->src[0] = 4000 + r;
        }
        int32_t par[4] = { 0, 0, 0, 0 };
        t.a.par = par;
        /* the chain's earlier rounds left r - 2 nodes in a line */
        t.st[0].ntree = r - 2;
        for (int j = 0; j < r - 2; ++j) t.st[0].par[j] = j;
        par[0] = r - 2;            /* the new node hangs off the last one */
        c.run();
        t.run();
        CHECK_EQ(t.w[Slab::TOK], c.w[Slab::TOK]);
        CHECK_EQ(t.w[Slab::POS], c.w[Slab::POS]);
        CHECK_EQ(t.w[Slab::CTX], c.w[Slab::CTX]);
        CHECK_EQ(t.used(0, 0), c.used(0, 0));
        CHECK_EQ(t.slot(0, 0), c.slot(0, 0));
    }
}

/* Width 3: the pass's children become nodes numbered on from the last round's, each embedded at
 * the head + its depth, stored at the flat slot head + its node, the keys ending past the last of
 * them; a row beyond the list is inert; the node budget stops the list. */
TEST(a_tree_round_stages_its_frontier_rows_by_depth_and_flat_slot) {
    Slab s(ADV_ROUND_TREE);
    s.a.round = 3; s.a.src_pitch = 4;
    s.a.tree_nodes = 5; s.a.width = 3;
    int32_t par[4] = { 1, 2, 1, 0 };
    s.a.par = par;
    s.a.parent_off = Slab::PAR * 4;
    for (int j = 0; j < Slab::kTok; ++j) s.w[Slab::PAR + j] = -2;
    /* the previous round made nodes 1 (under 0) and 2 (under 0): depth 1 each */
    StepSlot& S = s.st[0];
    S.head = 100; S.ntree = 2; S.par[0] = 0; S.par[1] = 0; S.tok[1] = 11; S.tok[2] = 12;
    s.row(0, true, 0, 3, 0, 0, 0, 0, 0);
    s.src[0] = 21; s.src[1] = 22; s.src[2] = 23; s.src[3] = -1;
    s.run();
    /* new nodes 3 (par 1), 4 (par 2), 5 (par 1) -- the budget of 5 takes all three */
    CHECK_EQ(S.ntree, 5);
    CHECK_EQ(S.tok[3], 21); CHECK_EQ(S.tok[4], 22); CHECK_EQ(S.tok[5], 23);
    CHECK_EQ(S.par[2], 1); CHECK_EQ(S.par[3], 2); CHECK_EQ(S.par[4], 1);
    for (int w = 0; w < 3; ++w) {
        CHECK_EQ(s.w[Slab::TOK + w], 21 + w);
        CHECK_EQ(s.w[Slab::POS + w], 102);               /* head + depth 2 */
        CHECK_EQ(s.slot(0, w), 100 + 3 + w);             /* head + node */
        CHECK_EQ(s.w[Slab::PAR + w], par[w]);
    }
    CHECK_EQ(s.w[Slab::CTX], 103);
    CHECK_EQ(s.used(0, 0), 106);                         /* keys through node 5's slot */
    /* the budget cuts a longer list; a short one leaves inert rows */
    Slab u(ADV_ROUND_TREE);
    u.a.round = 4; u.a.src_pitch = 4; u.a.tree_nodes = 5; u.a.width = 3;
    int32_t par2[4] = { 3, 0, 0, 0 };
    u.a.par = par2;
    u.a.parent_off = Slab::PAR * 4;
    u.st[0] = S;
    u.st[0].ntree = 4;
    u.row(0, true, 0, 3, 0, 0, 0, 0, 0);
    u.src[0] = 31; u.src[1] = 32; u.src[2] = 33;
    u.run();
    CHECK_EQ(u.st[0].ntree, 5);                          /* one fit */
    CHECK_EQ(u.w[Slab::TOK], 31);
    CHECK_EQ(u.w[Slab::POS], 100 + 3);                   /* node 3 is at depth 2: node 5 at 3 */
    CHECK_EQ(u.w[Slab::TOK + 1], 0);
    CHECK_EQ(u.slot(0, 1), -1);
    CHECK_EQ(u.w[Slab::PAR + 1], -2);
}

/* The plugin's final tree (ADV_BTAKE) is bounded by what the host reserved: depth <= R and every
 * parent an earlier node. A tree that breaks either falls back to the chain prefix with the same
 * node count (nodes 1..R chained, the rest siblings of node 1 that the walk never takes). */
TEST(a_serial_btake_keeps_a_valid_tree_and_falls_back_to_the_chain_prefix_for_a_bad_one) {
    const int32_t good[5] = { 0, 1, 0, 2, 1 };        /* depth 3 */
    const int32_t deep[5] = { 0, 1, 2, 3, 0 };        /* depth 4 > R = 3 */
    const int32_t fwd[5]  = { 0, 1, 4, 0, 0 };        /* node 3's parent (4) is a later node */
    const int32_t neg[5]  = { 0, -1, 0, 0, 0 };
    const int32_t* cases[4] = { good, deep, fwd, neg };
    for (int c = 0; c < 4; ++c) {
        Slab s(ADV_BTAKE);
        s.a.round = 5; s.a.src_pitch = 5;
        s.a.tree_nodes = 5; s.a.depth = 3;
        int32_t par[5];
        for (int j = 0; j < 5; ++j) par[j] = cases[c][j];
        s.a.par = par;
        s.row(0, true, 0, 6, 0, 0, 0, -1, 5);
        for (int j = 0; j < 5; ++j) s.src[j] = 70 + j;
        s.run();
        const StepSlot& S = s.st[0];
        CHECK_EQ(S.ntree, 5);
        for (int j = 0; j < 5; ++j) {
            if (c == 0) {
                CHECK_EQ(S.par[j], good[j]);
                CHECK_EQ(S.tok[1 + j], 70 + j);
            } else if (j < 3) {
                CHECK_EQ(S.par[j], j);                  /* the chain prefix, depth R */
                CHECK_EQ(S.tok[1 + j], 70 + j);
            } else {
                CHECK_EQ(S.par[j], 0);                  /* padding siblings of node 1 */
                CHECK_EQ(S.tok[1 + j], 70);
            }
        }
        CHECK(adv_tree_valid(S.par, S.ntree, 3));
    }
    /* depth 0: unchecked, as a block drafter's or an older caller's take */
    Slab u(ADV_BTAKE);
    u.a.round = 5; u.a.src_pitch = 5; u.a.tree_nodes = 5;
    int32_t par[5] = { 0, 1, 2, 3, 4 };
    u.a.par = par;
    u.row(0, true, 0, 6, 0, 0, 0, -1, 5);
    u.run();
    for (int j = 0; j < 5; ++j) CHECK_EQ(u.st[0].par[j], j);
}

/* A frontier row's parent must be a node of an earlier round: the same pass's rows have no K/V. */
TEST(a_tree_round_roots_a_parent_made_in_the_same_pass) {
    Slab s(ADV_ROUND_TREE);
    s.a.round = 3; s.a.src_pitch = 3; s.a.tree_nodes = 6; s.a.width = 3;
    int32_t par[3] = { 1, 3, 4 };          /* nodes 3, 4, 5: node 4's parent 3 is the same pass's */
    s.a.par = par;
    s.a.parent_off = Slab::PAR * 4;
    StepSlot& S = s.st[0];
    S.head = 100; S.ntree = 2; S.par[0] = 0; S.par[1] = 1; S.tok[1] = 11; S.tok[2] = 12;
    s.row(0, true, 0, 3, 0, 0, 0, 0, 0);
    s.src[0] = 21; s.src[1] = 22; s.src[2] = 23;
    s.run();
    CHECK_EQ(S.par[2], 1);
    CHECK_EQ(S.par[3], 0);
    CHECK_EQ(S.par[4], 0);
}

/* ================================================================== confidence pruning
 * abi/rad_mtp_tree.h's rule on stage 1's depth-1 leaves, and the inert slot a pruned tree leaves
 * behind: a duplicate of node 1 under the root, which the acceptance walk (lowest-index child that
 * matches) always meets node 1 first and so never takes. */
namespace {

/* ADV_TAKE of a one-round serial tree over one proposal row with the candidates' logits beside it,
 * thresholds as probabilities (0 = off). Returns the StepSlot. */
StepSlot prune_take(const int32_t (&ids)[8], const float (&val)[8], double pabs, double prel,
                    int leaves = 3, int nodes = 4, bool with_val = true) {
    Slab s(ADV_TAKE);
    s.a.round = 1; s.a.src_pitch = 8;
    s.a.tree_nodes = nodes; s.a.leaves = leaves; s.a.vocab = 1000;
    RadMtpTreeCfg pc{};
    rad_mtp_tree_set_prune(&pc, pabs, prel);
    s.a.prune = pc.prune; s.a.prune_log_abs = pc.log_abs; s.a.prune_log_rel = pc.log_rel;
    float v[8];
    std::memcpy(v, val, sizeof v);
    if (with_val) s.a.src_val = v;
    std::memset(&s.st[0], 0, sizeof s.st[0]);
    s.row(0, true, 0, 1, 0, 0, 0, 0, 0);
    std::memcpy(s.src, ids, sizeof ids);
    s.run();
    return s.st[0];
}

}  /* namespace */

/* Logits 5 4 3 0 -1 -2 -3 -4: probabilities .66 .243 .0895 .0045 ... (ratio to the best .368, .135,
 * .007); a leaf is a column past the pick. Thresholds off: all three leaves, as before. */
TEST(a_pruned_serial_tree_keeps_only_the_leaves_the_distribution_backs) {
    const int32_t ids[8] = { 500, 601, 602, 603, 604, 605, 606, 607 };
    const float   val[8] = { 5, 4, 3, 0, -1, -2, -3, -4 };
    {
        const StepSlot a = prune_take(ids, val, 0.0, 0.0);
        CHECK_EQ(a.ntree, 4);
        CHECK_EQ(a.nalt, 3);
        CHECK_EQ(a.tok[2], 601); CHECK_EQ(a.tok[3], 602); CHECK_EQ(a.tok[4], 603);
    }
    {   /* no value plane: pruning cannot apply, the capture is the old one */
        const StepSlot a = prune_take(ids, val, 0.5, 0.0, 3, 4, /*with_val=*/false);
        CHECK_EQ(a.nalt, 3);
    }
    {   /* abs .1: only .243 passes; the others are pruned and the list is padded with node 1's
         * token (the inert slot) */
        const StepSlot a = prune_take(ids, val, 0.1, 0.0);
        CHECK_EQ(a.ntree, 4);
        CHECK_EQ(a.nalt, 1);
        CHECK_EQ(a.tok[2], 601); CHECK_EQ(a.tok[3], 500); CHECK_EQ(a.tok[4], 500);
        for (int j = 1; j < 4; ++j) CHECK_EQ(a.par[j], 0);
    }
    {   /* rel .3: ratios .368 stays, .135 goes */
        const StepSlot a = prune_take(ids, val, 0.0, 0.3);
        CHECK_EQ(a.nalt, 1);
        CHECK_EQ(a.tok[2], 601);
    }
    {   /* both: abs .05 (keeps the first two) and rel .3 (keeps the first): one */
        const StepSlot a = prune_take(ids, val, 0.05, 0.3);
        CHECK_EQ(a.nalt, 1);
    }
    {   /* abs .25 drops .243 too: no leaf, the chain with three inert slots */
        const StepSlot a = prune_take(ids, val, 0.25, 0.0);
        CHECK_EQ(a.nalt, 0);
        CHECK_EQ(a.ntree, 4);
        for (int j = 2; j <= 4; ++j) CHECK_EQ(a.tok[j], 500);
    }
}

/* --draft-adaptive's stage 1 without coupling (abi/rad_mtp_adapt.h): a leaf is kept while its
 * calibrated probability beats the marginal verify-row cost, with the one-round chain it joins. Same
 * row as above (.66 .243 .0895 .0045): default cost, depth 1: E 1.66, thresholds .141 / .148 / ... */
TEST(an_adaptive_serial_tree_keeps_the_leaves_that_pay_for_their_rows) {
    const int32_t ids[8] = { 500, 601, 602, 603, 604, 605, 606, 607 };
    const float   val[8] = { 5, 4, 3, 0, -1, -2, -3, -4 };
    auto run = [&](const char* txt, int depth = 1) {
        Slab s(ADV_TAKE);
        s.a.round = 1; s.a.src_pitch = 8;
        s.a.tree_nodes = 4; s.a.leaves = 3; s.a.vocab = 1000; s.a.depth = depth;
        CHECK(rad_mtp_adapt_parse(txt, &s.a.adapt));
        float v[8];
        std::memcpy(v, val, sizeof v);
        s.a.src_val = v;
        std::memset(&s.st[0], 0, sizeof s.st[0]);
        s.row(0, true, 0, 1, 0, 0, 0, 0, 0);
        std::memcpy(s.src, ids, sizeof ids);
        s.run();
        return s.st[0];
    };
    CHECK_EQ(run("on=1").nalt, 1);                    /* .243 passes .141, .0895 fails .148 */
    CHECK_EQ(run("on=1").tok[2], 601);
    CHECK_EQ(run("on=1;margin=0.5").nalt, 2);         /* thresholds halve */
    CHECK_EQ(run("on=1;margin=3").nalt, 0);
    CHECK_EQ(run("on=1;margin=0.01").nalt, 3);        /* every valid leaf, as unpruned */
    CHECK_EQ(run("on=1;cost=10,10,10,10,10,10;row=0;draft=0").nalt, 3);   /* free rows: all */
    /* calibration: a flatter head (T 2) raises the runner-ups' probabilities */
    CHECK(run("on=1;margin=1;temp=2.0").nalt >= run("on=1;margin=1").nalt);
    /* off: untouched */
    CHECK_EQ(run("on=0").nalt, 3);
    /* the leaves are the prefix of the columns, never a later one over an earlier */
    const StepSlot a = run("on=1;margin=0.5");
    CHECK_EQ(a.tok[2], 601); CHECK_EQ(a.tok[3], 602);
}

TEST(a_pruned_serial_tree_never_prunes_the_rows_best_candidate_and_drops_invalid_ones) {
    const int32_t ids[8] = { 500, 601, 602, 603, 604, 605, 606, 607 };
    /* the largest logit is column 2, not the pick's column 0: it is the row's best, so a leaf the
     * hardest thresholds keep; column 1 is pruned */
    const float val[8] = { 1, 0, 9, 0, 0, 0, 0, 0 };
    const StepSlot a = prune_take(ids, val, 0.999, 1.0);
    CHECK_EQ(a.nalt, 1);
    CHECK_EQ(a.tok[2], 602);
    /* a candidate whose logit is not a number (or -inf) is no leaf once pruning is on, but is
     * still taken with the thresholds off (the old capture ignores values) */
    const float bad[8] = { 5, -INFINITY, 4, NAN, 3, 2, 1, 0 };
    const StepSlot b = prune_take(ids, bad, 0.001, 0.0);
    CHECK_EQ(b.tok[2], 602); CHECK_EQ(b.tok[3], 604); CHECK_EQ(b.tok[4], 605);
    const StepSlot c = prune_take(ids, bad, 0.0, 0.0);
    CHECK_EQ(c.tok[2], 601); CHECK_EQ(c.tok[3], 602); CHECK_EQ(c.tok[4], 603);
}

/* Round 2 of a longer head captures the same way (the proposal row of round 1 is intact then). */
TEST(a_pruned_serial_tree_captures_in_round_two_with_the_same_rule) {
    const int32_t ids[8] = { 500, 601, 602, 603, 604, 605, 606, 607 };
    const float   val[8] = { 5, 4, 3, 0, -1, -2, -3, -4 };
    Slab s(ADV_ROUND);
    s.a.round = 2; s.a.src_pitch = 8;
    s.a.tree_nodes = 6; s.a.leaves = 3; s.a.vocab = 1000;
    RadMtpTreeCfg pc{};
    rad_mtp_tree_set_prune(&pc, 0.1, 0.0);
    s.a.prune = pc.prune; s.a.prune_log_abs = pc.log_abs; s.a.prune_log_rel = pc.log_rel;
    float v[8];
    std::memcpy(v, val, sizeof v);
    s.a.src_val = v;
    std::memset(&s.st[0], 0, sizeof s.st[0]);
    s.row(0, true, 0, 1, 0, 0, 0, 0, 0);
    std::memcpy(s.src, ids, sizeof ids);
    s.run();
    CHECK_EQ(s.st[0].nalt, 1);
    CHECK_EQ(s.st[0].alt[0], 601);
}

/* ADV_BTAKE of a pruned grower's final tree: an inert entry (token -1, or past the vocabulary)
 * becomes a duplicate of node 1 under the root; a plugin's valid tree is taken as it was. */
TEST(a_btake_of_a_pruned_tree_turns_inert_entries_into_duplicates_of_node_one) {
    Slab s(ADV_BTAKE);
    s.a.round = 5; s.a.src_pitch = 5;
    s.a.tree_nodes = 5; s.a.depth = 3; s.a.vocab = 1000;
    int32_t par[5] = { 0, 0, 1, 0, 0 };
    s.a.par = par;
    s.row(0, true, 0, 6, 0, 0, 0, -1, 5);
    const int32_t tok[5] = { 70, 71, 72, -1, 1000 };
    for (int j = 0; j < 5; ++j) s.src[j] = tok[j];
    s.run();
    const StepSlot& S = s.st[0];
    CHECK_EQ(S.ntree, 5);
    const int32_t want_t[5] = { 70, 71, 72, 70, 70 }, want_p[5] = { 0, 0, 1, 0, 0 };
    for (int j = 0; j < 5; ++j) { CHECK_EQ(S.tok[1 + j], want_t[j]); CHECK_EQ(S.par[j], want_p[j]); }
    CHECK(adv_tree_valid(S.par, S.ntree, 3));
    /* the tree's first node itself inert (a grower that made nothing): token 0 under the root */
    Slab u(ADV_BTAKE);
    u.a.round = 3; u.a.src_pitch = 3;
    u.a.tree_nodes = 3; u.a.depth = 2; u.a.vocab = 1000;
    int32_t par2[3] = { 0, 0, 0 };
    u.a.par = par2;
    u.row(0, true, 0, 4, 0, 0, 0, -1, 3);
    for (int j = 0; j < 3; ++j) u.src[j] = -1;
    u.run();
    for (int j = 0; j < 3; ++j) { CHECK_EQ(u.st[0].tok[1 + j], 0); CHECK_EQ(u.st[0].par[j], 0); }
    /* a block drafter (no tree_nodes) keeps whatever it wrote */
    Slab b(ADV_BTAKE);
    b.a.round = 2; b.a.src_pitch = 2;
    int32_t par3[2] = { 0, 1 };
    b.a.par = par3;
    b.row(0, true, 0, 3, 0, 0, 0, -1, 2);
    b.src[0] = -1; b.src[1] = 5;
    b.run();
    CHECK_EQ(b.st[0].tok[1], -1);
}

/* An inert node is never accepted: over random trees whose active nodes are a prefix 1..k and
 * whose remaining nodes are the pruned duplicates (node 1's token, parent the root), whatever the
 * target picks (including the duplicated token), the accepted path lies in the active prefix and
 * is the path the tree without them gives. */
TEST(an_inert_node_is_never_accepted_whatever_the_target_picks) {
    uint32_t rng = 0x1ae27u;
    int inert_token_picked = 0, deep = 0;
    for (int c = 0; c < 6000; ++c) {
        Slab s(ADV_ACCEPT);
        s.a.parent_off = Slab::PAR * 4;
        for (int t = 0; t < Slab::kTok; ++t) s.w[Slab::PAR + t] = -2;
        const int n_spec = 2 + (int)(lcg(rng) % 7);       /* 2..8 */
        const int k = 1 + (int)(lcg(rng) % (uint32_t)n_spec);       /* active nodes 1..k */
        s.row(0, true, 0, 1 + n_spec, (int)(lcg(rng) % 1000), 0, 0, 0, n_spec);
        int32_t draft[8] = {}, parent[8] = {}, target[9] = {};
        s.w[Slab::TOK] = 50;
        s.w[Slab::PAR] = -1;
        for (int j = 1; j <= n_spec; ++j) {
            if (j <= k) {
                parent[j - 1] = j == 1 ? 0 : (int32_t)(lcg(rng) % (uint32_t)j);
                draft[j - 1] = 1 + (int32_t)(lcg(rng) % 4);
                if (j > 1 && parent[j - 1] == 0 && draft[j - 1] == draft[0]) draft[j - 1] = 5;
            } else {
                parent[j - 1] = 0;          /* inert: node 1's token under the root */
                draft[j - 1] = draft[0];
            }
            s.w[Slab::TOK + j] = draft[j - 1];
            s.w[Slab::PAR + j] = parent[j - 1];
        }
        for (int r = 0; r <= n_spec; ++r) {
            /* bias toward node 1's token so the duplicates are tempting */
            target[r] = lcg(rng) % 3 == 0 ? draft[0] : 1 + (int32_t)(lcg(rng) % 5);
            s.src[r] = target[r];
        }
        inert_token_picked += target[0] == draft[0] && k < n_spec;
        int32_t out[9] = {}, path[8] = {}, acc = -1, outk[9] = {}, pathk[8] = {}, acck = -1;
        CHECK_OK(accept_tree(draft, parent, n_spec, target, out, path, &acc));
        CHECK_OK(accept_tree(draft, parent, k, target, outk, pathk, &acck));
        s.run();
        const StepSlot& S = s.st[0];
        CHECK_EQ(S.acc, acc);
        CHECK_EQ(S.acc, acck);
        for (int t = 0; t < S.acc; ++t) {
            CHECK(S.path[t] >= 1 && S.path[t] <= k);          /* never an inert node */
            CHECK_EQ(S.path[t], pathk[t]);
        }
        deep += S.acc >= 2;
    }
    CHECK(inert_token_picked > 200);
    CHECK(deep > 200);
}

RAD_TEST_MAIN()

/* ================================================================== dynamic rows (DESIGN-dynrows.md) */

/* A step issued in the chain's form (no spec_parent, no spec_path) over a tree the head grew: the
 * card verifies the first `d` nodes in node order as the chain they are -- stage 1 puts the chain
 * first -- at depth == flat position, writes nothing into the parent or path planes (there are
 * none), and its acceptance is the chain rule with npath 0, so everything after it (history pass,
 * compaction, the next step's path) reads the identity. The leaves a prediction cut are dropped,
 * which for a suffix is always valid. */
TEST(a_chain_form_verify_of_a_grown_tree_takes_its_first_nodes_as_the_chain) {
    SerialRun r;
    const int32_t c0[8] = { 500, 500, 601, 602, 603, 604, 605, 606 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    r.rows[2][0][0] = 520; r.rows[3][0][0] = 530;
    r.go(3, 3, 6, 1000, 1);
    StepSlot st = r.st[0];
    st.tok[0] = 9;
    st.ctx = 100; st.n_acc = 1; st.npath = 0;

    Slab v(ADV_TRUNK);             /* parent_off and path_off stay -1: the build staged neither */
    for (int j = 0; j < Slab::kTok; ++j) v.w[Slab::PAR + j] = -7;
    for (int j = 0; j < Slab::kSeq * Slab::kPathPitch; ++j) v.w[Slab::PATH + j] = -7;
    v.st[0] = st;
    v.row(0, true, 0, 4, 100, 0, 0, 0, 3);       /* d = 3: the chain, not the three leaves */
    v.run();
    const int32_t want[4] = { 9, 500, 520, 530 };
    for (int j = 0; j < 4; ++j) {
        CHECK_EQ(v.w[Slab::TOK + j], want[j]);
        CHECK_EQ(v.w[Slab::POS + j], 100 + j);
        CHECK_EQ(v.slot(0, j), 100 + j);
    }
    CHECK_EQ(v.w[Slab::ACC], 1);
    for (int j = 0; j < Slab::kTok; ++j) CHECK_EQ(v.w[Slab::PAR + j], -7);
    for (int j = 0; j < Slab::kSeq * Slab::kPathPitch; ++j) CHECK_EQ(v.w[Slab::PATH + j], -7);

    /* Acceptance: the target agrees with the first two drafts and picks 777 after them. */
    Slab acc(ADV_ACCEPT);
    for (int j = 0; j < 4; ++j) acc.w[Slab::TOK + j] = v.w[Slab::TOK + j];
    acc.st[0] = st;
    acc.row(0, true, 0, 4, 100, 0, 0, 0, 3);
    const int32_t y[4] = { 500, 520, 777, 1 };
    for (int j = 0; j < 4; ++j) acc.src[j] = y[j];
    acc.run();
    CHECK_EQ(acc.st[0].acc, 2);
    CHECK_EQ(acc.st[0].npath, 0);
    CHECK_EQ(acc.st[0].tok[0], 777);
    CHECK_EQ(acc.st[0].head, 102);
    CHECK_EQ(acc.st[0].hrow, 2);

    /* The next step is a tree verify again: its path plane is the identity, as after any chain. */
    Slab nx(ADV_TRUNK);
    nx.a.parent_off = Slab::PAR * 4;
    nx.a.path_off = Slab::PATH * 4; nx.a.path_pitch = Slab::kPathPitch;
    nx.st[0] = acc.st[0];
    nx.st[0].ntree = 0;
    nx.row(0, true, 0, 3, 102, 0, 0, 0, 2);
    nx.run();
    for (int t = 0; t < Slab::kPathPitch; ++t) CHECK_EQ(nx.w[Slab::PATH + t], t);
}

/* The same cut on a tree-form step: a prefix of the grown tree is the tree the verify carries, the
 * leaves past the count are dropped, and what is kept is exactly what a full verify holds for those
 * rows (tokens, parents, depth positions, flat slots). */
TEST(a_tree_form_verify_of_a_prefix_keeps_those_rows_exactly_as_the_full_verify_has_them) {
    SerialRun r;
    const int32_t c0[8] = { 500, 500, 601, 602, 603, 604, 605, 606 };
    std::memcpy(r.rows[1][0], c0, sizeof c0);
    r.rows[2][0][0] = 520; r.rows[3][0][0] = 530;
    r.go(3, 3, 6, 1000, 1);
    StepSlot st = r.st[0];
    st.tok[0] = 9; st.ctx = 100; st.n_acc = 1; st.npath = 0;
    auto run = [&](int n) {
        auto v = std::make_unique<Slab>(ADV_TRUNK);
        v->a.parent_off = Slab::PAR * 4;
        v->a.path_off = Slab::PATH * 4; v->a.path_pitch = Slab::kPathPitch;
        v->st[0] = st;
        v->row(0, true, 0, n, 100, 0, 0, 0, n - 1);
        v->run();
        return v;
    };
    const auto full = run(7);
    for (int n : { 2, 4, 5, 6 }) {
        const auto cut = run(n);
        for (int j = 0; j < n; ++j) {
            CHECK_EQ(cut->w[Slab::TOK + j], full->w[Slab::TOK + j]);
            CHECK_EQ(cut->w[Slab::PAR + j], full->w[Slab::PAR + j]);
            CHECK_EQ(cut->w[Slab::POS + j], full->w[Slab::POS + j]);
            CHECK_EQ(cut->slot(0, j), full->slot(0, j));
        }
    }
}

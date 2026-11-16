/* plugin/spechub/spechub.h -- the plugin-private contract between its host half and its kernels.
 *
 * Plugin-private ON PURPOSE: core/ must never include this, and everything the CORE needs (the wire
 * row and the coupling arithmetic) already lives in abi/rad_spechub.h. Only the packed call structs
 * and the argument readers are here, shared by the host reference and the device launch.
 */
#ifndef RAD_PLUGIN_SPECHUB_H
#define RAD_PLUGIN_SPECHUB_H

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_param_geti/getdim: the pure readers, inline and total */
#include "rad_spechub.h"
#include "rad_mtp_rrs.h"
#include "rad_mtp_block.h"
#include "rad_mtp_traversal.h"

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#define RAD_SPECHUB_HD __host__ __device__
#else
#define RAD_SPECHUB_HD
#endif

/* A draft round's operands, packed once for both the device launch and the host reference. */
struct RadMtpCoupleProposeCall {
    int32_t*       idx;    /* [M, 8] INOUT: this rank's shortlist; column 0 is the proposal */
    const float*   val;    /* [M, 8] the shortlist's scores */
    const int32_t* card;   /* [batch] RadBatch::seq_ids: the slot the advance writes */
    const uint32_t* draw;  /* [M, DRAW_WORDS] RadMtpCoupleDraw */
    uint32_t*      state;  /* [rounds, M, STATE_WORDS] RadMtpCoupleState */
    int64_t        M;
    int64_t        vocab;
    int64_t        ld_idx, ld_val, ld_draw, st_round, st_row;
    int            round;
    int            leaves;  /* round 1 of a draft tree: sibling candidates drafted beside x (0: the chain) */
    /* Confidence pruning of the leaves (rad_mtp_tree_set_prune's floats; prune 0: off). It reads
     * the slot priorities of Q only (rad_mtp_rrs_slot_keep_count), so it drops a suffix of the
     * draft order. */
    int            prune;
    float          log_abs, log_rel;
    /* --draft-adaptive's rule for the leaves (abi/rad_mtp_adapt.h) and the chain (max_rounds) they
     * join; ad.on 0: off. Slot priorities only, like the pruning. */
    RadMtpAdapt    ad;
    int            max_rounds;
    /* Sibling scheme of round 1's leaves: 0 = Gumbel-top-k (default), m + 1 = RheoSampling with m
     * deterministic top-q children and one tail-sampled one (--draft-siblings rheo). */
    int            scheme;
};

/* The verify's operands: the sampler's own candidate plane, its parameter rows, the bound state
 * rows and the token plane the pick already wrote. */
struct RadMtpCoupleVerifyCall {
    const int32_t*  idx;    /* [M, n_cand] */
    const float*    val;    /* [M, n_cand] */
    const uint32_t* params; /* [M, 24 words] RadSampleParams */
    const uint32_t* state;  /* [M, STATE_WORDS] RadMtpCoupleState */
    int32_t*        token;  /* [M] or [M, 1] */
    int64_t         M, n_cand, vocab, ld[5];
};

/* The traversal verify's operands: the row-wise verify's, plus the step's tree. `topo` is
 * [M, 2] int32 per sampled row: the parent's row in the step (-1 for a request's first row, its
 * tree's root) and the row's own draft token. The rows of one request are contiguous, the root
 * first, parents before children (core/sample/sampler.cpp stages it that way). */
struct RadMtpTravCall {
    RadMtpCoupleVerifyCall v;
    const int32_t* topo;
    int64_t        ld_topo;
};

/* DFlash2's coupled walk (`dflash_couple_propose`, abi/rad_spechub.h): one sequence a row. Its
 * operands, in schema order: cand, unary, hp?, anchor, pred w, succ w, seq_ids, draw, tokens
 * (INOUT: dflash_select's greedy walk, rewritten for a sampled row), state. */
struct RadDflashCoupleCall {
    const int32_t*  cand;     /* [M*steps, >= K] */
    const float*    unary;    /* [M*steps, >= K] */
    const uint16_t* hp;       /* [M*steps, >= R] bf16, or null for ones */
    const int32_t*  anchor;   /* the step's token ids, the anchor of sequence s at s*anchor_stride */
    const uint16_t* pred;     /* [vocab, R] bf16 */
    const uint16_t* succ;     /* [vocab, R] bf16 */
    const int32_t*  card;     /* [>= M] RadBatch::seq_ids */
    const uint32_t* draw;     /* [M, DRAW_WORDS] */
    int32_t*        tokens;   /* [M, >= steps] */
    uint32_t*       state;    /* [rounds, >= M, STATE_WORDS] */
    int64_t M, steps, K, R, vocab, anchor_stride;
    int64_t ld_cand, ld_unary, ld_hp, ld_draw, ld_tok, st_round, st_row;
    float   q_temp;
};

int rad_mtp_couple_propose_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_verify_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_dflash_couple_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_couple_propose_args(const RadArgs*, RadMtpCoupleProposeCall*);
int rad_mtp_couple_verify_args(const RadArgs*, RadMtpCoupleVerifyCall*);
int rad_mtp_trav_verify_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_trav_verify_args(const RadArgs*, RadMtpTravCall*);
int rad_dflash_couple_args(const RadArgs*, RadDflashCoupleCall*);

/* The geometry/validation the shape hooks and the arg readers share. `op` 0 proposes, 1 verifies. */
int rad_couple_validate(const RadArgs*, int op, int64_t* M, int64_t* vocab, int64_t* n_cand);

/* The op set. Three schemas, three device rows and three host-reference rows; row/schema index 2 is
 * dflash_couple_propose. */
#define RAD_SPECHUB_N_OPS 4
int rad_spechub_schema_count();
const RadOpSchema* rad_spechub_schema_at(int i);
RadKernelInfo rad_spechub_row(int i, RadLaunchFn launch);

/* The launches. */
int rad_mine_mtp_couple_propose_launch(const RadArgs*, RadStream);
int rad_mine_mtp_couple_verify_launch(const RadArgs*, RadStream);
int rad_ref_mtp_couple_propose(const RadArgs*, RadStream);
int rad_ref_mtp_couple_verify(const RadArgs*, RadStream);
int rad_mine_dflash_couple_launch(const RadArgs*, RadStream);
int rad_mine_mtp_trav_verify_launch(const RadArgs*, RadStream);
int rad_ref_mtp_trav_verify(const RadArgs*, RadStream);
int rad_ref_dflash_couple(const RadArgs*, RadStream);

/* One propose row, host and device: x ~ Q over the final eight. Column 0 is what ADV_ROUND and
 * ADV_TAKE read, and it becomes x only for a valid sampled row. */
static inline RAD_SPECHUB_HD void rad_mtp_couple_propose_row(const RadMtpCoupleProposeCall& c,
                                                             int64_t row) {
    RadMtpCoupleState s{};
    s.draw = *reinterpret_cast<const RadMtpCoupleDraw*>(c.draw + row * c.ld_draw);
    int32_t* idx = c.idx + row * c.ld_idx;
    s.round = c.round;
    s.x = idx[0];
    for (int i = 0; i < 8; ++i) s.ids[i] = -1;
    if (s.draw.mode == 1) {
        s.status = -1;
        if (s.draw.card == c.card[row] &&
            rad_mtp_couple_build_law_at(idx, c.val + row * c.ld_val, c.vocab, s.draw, c.round, &s)) {
            if (c.leaves > 0 && c.round == 1) {
                /* A tree's root: the trunk child and `leaves` siblings, drawn without replacement
                 * (Gumbel-top-k over Q, abi/rad_mtp_rrs.h) in draft order into the shortlist's
                 * first columns; the columns past them are holes, so the core's leaf capture
                 * takes exactly the drawn ones. */
                RadMtpTreeCfg pc{};
                pc.prune = c.prune; pc.log_abs = c.log_abs; pc.log_rel = c.log_rel;
                pc.ad = c.ad; pc.rounds = c.max_rounds;
                rad_mtp_calib_apply(&pc.ad, s.draw);
                if (c.scheme > 0)
                    rad_mtp_rrs_draw_rheo(&s, 0, rad_mtp_rrs_slot_keep_count(s, &pc, 1 + c.leaves, c.scheme),
                                          c.scheme - 1);
                else
                    rad_mtp_rrs_draw(&s, 0, rad_mtp_rrs_slot_keep_count(s, &pc, 1 + c.leaves));
                s.status = 1;
                for (int i = 0; i < 8; ++i) idx[i] = i < s.nkids ? s.ids[i] : -1;
            } else {
                const double u = rad_spechub_uniform(s.draw.seed, s.draw.base,
                                                     RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)c.round);
                s.x = s.ids[rad_mtp_couple_pick(s, u)];
                s.status = 1;
                idx[0] = s.x;
            }
        } else {
            s.count = 0;
            for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
        }
    }
    *reinterpret_cast<RadMtpCoupleState*>(c.state + (c.round - 1) * c.st_round + row * c.st_row) = s;
}

/* --draft-verify block (abi/rad_mtp_block.h). The sampler marks the chain rows of a request it
 * verifies as one block by `blk` = the block's length g (the rows 0 .. g-1 bound to rounds 1 .. g,
 * consecutive); the row of round 1 decides the whole block and writes the tokens of rows 0 .. tau,
 * the other rows do nothing (their tokens are written by it). Row i of the block is admitted only
 * if it is the state the sampler would have bound for it: same request, seed, round, position
 * (pos0 + i) and a valid state; any other row ends the block's coupling (the ordinary picks
 * stand, which is exact). */
static inline RAD_SPECHUB_HD bool rad_mtp_block_row_ok(const RadMtpCoupleState& s0,
                                                       const RadMtpCoupleState& s,
                                                       const RadSampleParams& p0,
                                                       const RadSampleParams& p, int i, int g,
                                                       int64_t vocab) {
    return s.status == 1 && s.blk == g && s.round == i + 1 && s.nkids <= 1 &&
           !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed && p.seed == p0.seed &&
           s.draw.seq == s0.draw.seq && s.pos == p.pos && p.pos == p0.pos + (uint64_t)i &&
           s.pos == s0.pos + (uint64_t)i && rad_mtp_couple_state_valid(s, vocab);
}

/* The block's tokens: rows 0 .. tau of the block take the drafts and then Y. */
static inline RAD_SPECHUB_HD void rad_mtp_block_write(const RadMtpCoupleVerifyCall& c, int64_t row,
                                                      const RadMtpCoupleState* st, int tau, int g,
                                                      int32_t y) {
    for (int i = 0; i < tau; ++i) c.token[(row + i) * c.ld[4]] = st[i].x;
    if (tau < g) c.token[(row + tau) * c.ld[4]] = y;
}

static inline RAD_SPECHUB_HD void rad_mtp_block_verify_row(const RadMtpCoupleVerifyCall& c,
                                                           int64_t row) {
    const auto& s0 = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const int g = s0.blk;
    if (s0.round != 1 || g < 1 || g > RAD_MTP_BLOCK_MAX || row + g > c.M) return;
    const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    RadMtpCoupleState st[RAD_MTP_BLOCK_MAX];
    RadBlockRow       rows[RAD_MTP_BLOCK_MAX];
    for (int i = 0; i < g; ++i) {
        st[i] = *reinterpret_cast<const RadMtpCoupleState*>(c.state + (row + i) * c.ld[3]);
        const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + (row + i) * c.ld[2]);
        if (!rad_mtp_block_row_ok(s0, st[i], p0, p, i, g, c.vocab)) return;
        rows[i] = RadBlockRow{ c.idx + (row + i) * c.ld[0], c.val + (row + i) * c.ld[1], nullptr,
                               c.n_cand, 0, 0.0f };
        rad_mtp_block_scan(&rows[i], c.vocab);
    }
    int tau = 0;
    int32_t y = -1;
    if (!rad_mtp_block_decide(st, rows, g, p0.seed, p0.pos, &tau, &y)) return;
    rad_mtp_block_write(c, row, st, tau, g, y);
}

/* One verify row, after sample_pick. A row without a valid bound state keeps the ordinary token. */
static inline RAD_SPECHUB_HD void rad_mtp_couple_verify_row(const RadMtpCoupleVerifyCall& c,
                                                            int64_t row) {
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    if (s.status != 1) return;
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    if ((p.flags & RAD_SP_GREEDY) || s.draw.seed != p.seed || s.pos != p.pos ||
        !rad_mtp_couple_state_valid(s, c.vocab))
        return;
    if (s.blk >= 1) { rad_mtp_block_verify_row(c, row); return; }
    int32_t out = -1;
    /* A RheoSampling node (--draft-siblings rheo): the greedy multi-draft verifier, any nkids. */
    if (s.scheme != 0) {
        if (rad_mtp_rrs_decide_rheo(s, c.idx + row * c.ld[0], c.val + row * c.ld[1], c.n_cand, c.vocab,
                                    p.seed, p.pos, &out))
            c.token[row * c.ld[4]] = out;
        return;
    }
    /* A tree node with several drafted children: recursive rejection sampling over them. */
    if (s.nkids >= 2) {
        if (rad_mtp_rrs_decide(s, c.idx + row * c.ld[0], c.val + row * c.ld[1], c.n_cand, c.vocab,
                               p.seed, p.pos, &out)) {
            c.token[row * c.ld[4]] = out;
            rad_mtp_couple_export_p(const_cast<RadMtpCoupleState*>(&s), c.idx + row * c.ld[0],
                                    c.val + row * c.ld[1], c.n_cand, c.vocab);
        }
        return;
    }
    if (rad_mtp_couple_decide(s, c.idx + row * c.ld[0], c.val + row * c.ld[1], c.n_cand, c.vocab,
                              rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT),
                              rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID), &out)) {
        c.token[row * c.ld[4]] = out;
        rad_mtp_couple_export_p(const_cast<RadMtpCoupleState*>(&s), c.idx + row * c.ld[0],
                                c.val + row * c.ld[1], c.n_cand, c.vocab);
    }
}

/* The traversal walk's view of one request's rows: node v is row `root + v`. A row is a node the
 * walk may use only when the row-wise verify would have accepted its state (same gate), so a row
 * that is greedy, unbound or stale is a leaf here and keeps the pick's token. */
struct RadMtpTravRows {
    const RadMtpTravCall& c;
    int64_t root;
    int64_t n;   /* rows of the request */
    RAD_SPECHUB_HD const RadMtpCoupleState* state(int v) const {
        const int64_t row = root + v;
        const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.v.state + row * c.v.ld[3]);
        const auto& p = *reinterpret_cast<const RadSampleParams*>(c.v.params + row * c.v.ld[2]);
        if (s.status != 1 || (p.flags & RAD_SP_GREEDY) || s.draw.seed != p.seed || s.pos != p.pos ||
            !rad_mtp_couple_state_valid(s, c.v.vocab))
            return nullptr;
        return &s;
    }
    RAD_SPECHUB_HD bool cand(int v, const int32_t** ids, const float** val, int64_t* nc) const {
        *ids = c.v.idx + (root + v) * c.v.ld[0];
        *val = c.v.val + (root + v) * c.v.ld[1];
        *nc = c.v.n_cand;
        return true;
    }
    RAD_SPECHUB_HD int child(int v, int32_t token) const {
        for (int64_t u = (int64_t)v + 1; u < n; ++u) {
            const int32_t* t = c.topo + (root + u) * c.ld_topo;
            if (t[0] == (int32_t)(root + v) && t[1] == token) return (int)u;
        }
        return -1;
    }
    RAD_SPECHUB_HD int64_t vocab() const { return c.v.vocab; }
};

/* One request's tree, if `row` is its root: walk it and write the path's tokens. */
static inline RAD_SPECHUB_HD void rad_mtp_trav_verify_root(const RadMtpTravCall& c, int64_t row) {
    if (c.topo[row * c.ld_topo] >= 0) return;
    int64_t n = 1;
    while (row + n < c.v.M && c.topo[(row + n) * c.ld_topo] >= 0) ++n;
    const RadMtpTravRows T{ c, row, n };
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.v.params + row * c.v.ld[2]);
    RadMtpTravOut o;
    if (!rad_mtp_trav_walk(T, 0, p.seed, p.pos, &o)) return;
    for (int i = 0; i + 1 < o.n; ++i)
        c.v.token[(row + o.node[i]) * c.v.ld[4]] = c.topo[(row + o.node[i + 1]) * c.ld_topo + 1];
    if (o.has_y) c.v.token[(row + o.node[o.n - 1]) * c.v.ld[4]] = o.y;
}

#undef RAD_SPECHUB_HD
#endif  /* RAD_PLUGIN_SPECHUB_H */

/* rad_mtp_tree_rrs.h -- the coupled form of mtp_tree_grow's pass (--draft-coupling on, stage-2
 * tree). A thin wrapper over rad_mtp_tree_grow_seq (abi/rad_mtp_tree.h, untouched): per frontier
 * row it builds Q from the row's reranked top 8, draws the row's children without replacement
 * (abi/rad_mtp_rrs.h), hands the grower the same row re-expressed as SLOTS (slot j = the j-th
 * largest q's priority, carrying the j-th DRAWN token), and afterwards counts the children the
 * grower created into the parent's state row. The grower's best-first budget therefore decides
 * how many children a node gets from the slot priorities alone (log of the j-th largest q of the
 * node's own law, plus the parent's cumulative score), never from the drawn tokens, and always
 * as a prefix of the draft order. The grower's confidence pruning (cfg->prune) therefore reads the
 * same slot priorities: it drops a SUFFIX of a node's draft order by the j-th largest q, never by
 * the drawn token's own probability (DESIGN-coupling.md 3.5): the two conditions under which recursive rejection sampling over
 * the created children is exact (DESIGN-coupling.md section 3).
 *
 * The state plane is [N node rows, sequences, RAD_MTP_COUPLE_STATE_WORDS]: node f's row holds the
 * law its children were drawn from, the children in draft order in ids[0 .. nkids), and `round`
 * = its depth + 1; the engine reads it back and binds it to verify row f.
 */
#ifndef RAD_MTP_TREE_RRS_H
#define RAD_MTP_TREE_RRS_H

#include "rad_mtp_tree.h"
#include "rad_mtp_rrs.h"

#ifdef __cplusplus



typedef struct RadMtpTreeCpl {
    const uint32_t* draw;      /* this sequence's RadMtpCoupleDraw row */
    int32_t         card;      /* the batch's seq_ids entry of this row */
    uint32_t*       state;     /* the plane's base */
    int64_t         st_node;   /* words between node rows */
    int64_t         st_seq;    /* words between sequences */
    int             rows;      /* node rows the plane holds */
    int64_t         vocab;     /* the whole vocabulary */
    int             scheme;    /* 0: Gumbel-top-k siblings; m + 1: RheoSampling with m top-q children
                                * and one tail-sampled one (--draft-siblings rheo, abi/rad_mtp_rrs.h) */
} RadMtpTreeCpl;

/* The coupled pass in three parts, so a device grower can run the independent middle on W lanes and
 * the sequence's serial walk on one (rad_mtp_tree_grow_seq_cpl composes them for the host).
 *   begin: whether this sequence is coupled, its draw and the calibrated cfg (false: the plain grower);
 *   row:   frontier row w's law, drawn children and state (no state shared between rows);
 *   post:  the children the grower created, counted into their parents' states. */
RAD_MTP_HD static inline bool rad_mtp_tree_cpl_active(const RadMtpTreeCfg* cfg, const RadMtpTreeCpl* cpl,
                                                      RadMtpCoupleDraw* draw) {
    *draw = cpl && cpl->draw ? *reinterpret_cast<const RadMtpCoupleDraw*>(cpl->draw)
                             : RadMtpCoupleDraw{};
    return cpl && cpl->state && draw->mode == 1 && draw->card == cpl->card &&
           cfg->width <= RAD_MTP_RRS_MAX_WIDTH && cfg->nodes >= 1 &&
           cfg->nodes <= RAD_MTP_TREE_MAX_NODES && cfg->round >= 1 && cfg->round <= cfg->rounds;
}
RAD_MTP_HD static inline bool rad_mtp_tree_cpl_begin(const RadMtpTreeCfg* cfg, const RadMtpTreeCpl* cpl,
                                                     RadMtpCoupleDraw* draw, RadMtpTreeCfg* lcfg) {
    if (!rad_mtp_tree_cpl_active(cfg, cpl, draw)) return false;
    /* --draft-calib online: this step's per-depth calibration temperatures (the draw's), for the
     * pool scores and the adaptive rule alike */
    *lcfg = *cfg;
    rad_mtp_calib_apply(&lcfg->ad, *draw);
    return true;
}
/* Frontier rows this pass reads (the grower's own count). */
RAD_MTP_HD static inline int rad_mtp_tree_cpl_nfront(const RadMtpTreeCfg* cfg, const int* st) {
    return cfg->round == 1 ? 1 : (st[2] < cfg->width ? st[2] : cfg->width);
}
RAD_MTP_HD static inline void rad_mtp_tree_cpl_row(
    const RadMtpTreeCfg* cfg, const RadMtpTreeCpl* cpl, const RadMtpCoupleDraw& draw, const int* st,
    const int* didx, const float* dval, long row0, long row_stride, long stride, long seq, int w,
    int* od, float* ov) {
    const int K = RAD_MTP_TREE_K, W = cfg->width;
    const int* nd = st + 4 + W;
    const bool round1 = cfg->round == 1;
    const int f = round1 ? 0 : st[4 + w];
    const int depth = round1 ? 0 : nd[6 * f + 2];
    const long r = row0 + (long)w * stride;
    int32_t ids[8];
    float   vals[8];
    for (int c = 0; c < K; ++c) { ids[c] = didx[r * row_stride + c]; vals[c] = dval[r * row_stride + c]; }
    RadMtpCoupleState s{};
    s.draw = draw;
    s.round = depth + 1;
    const bool ok = rad_mtp_couple_build_law_at(ids, vals, cpl->vocab, draw, depth + 1, &s);
    if (ok) {
        s.status = 1;
        /* the slots: priorities from the descending q, tokens from the draft order */
        float slot[8];
        if (cpl->scheme > 0) {
            /* RheoSampling: slots 0 .. m-1 the top-q tokens, slot m the tail-sampled one, no
             * other slot; every priority is a function of Q alone. */
            rad_mtp_rrs_slot_logq_rheo(s, cpl->scheme - 1, slot);
            rad_mtp_rrs_draw_rheo(&s, (uint64_t)f, 0, cpl->scheme - 1);
            for (int c = 0; c < K; ++c) {
                od[c] = c < s.count && slot[c] != -INFINITY ? s.ids[c] : -1;
                ov[c] = slot[c];
            }
        } else {
            rad_mtp_rrs_slot_logq(s, slot);
            rad_mtp_rrs_draw(&s, (uint64_t)f, 0);
            for (int c = 0; c < K; ++c) {
                od[c] = c < s.count ? s.ids[c] : -1;
                ov[c] = slot[c];
            }
        }
    } else {
        s.status = -1;
        s.count = 0;
        for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
        s.x = -1;
        for (int c = 0; c < K; ++c) { od[c] = ids[c]; ov[c] = vals[c]; }
    }
    if (f < cpl->rows)
        *reinterpret_cast<RadMtpCoupleState*>(cpl->state + (int64_t)f * cpl->st_node +
                                              (int64_t)seq * cpl->st_seq) = s;
}
RAD_MTP_HD static inline void rad_mtp_tree_cpl_post(const RadMtpTreeCfg* cfg, const RadMtpTreeCpl* cpl,
                                                    const int* st, int before, long seq) {
    const int N = cfg->nodes, W = cfg->width;
    const int* nd = st + 4 + W;
    /* The children this pass created, into their parents' states (a dry copy has no probability). */
    const int after = st[1];
    for (int id = before + 1; id <= after && id <= N; ++id) {
        const int* n = nd + 6 * id;
        if (rad_mtp_tree_bits_f(n[3]) == -INFINITY) continue;
        const int p = n[1];
        if (p < 0 || p >= cpl->rows) continue;
        RadMtpCoupleState* ps = reinterpret_cast<RadMtpCoupleState*>(
            cpl->state + (int64_t)p * cpl->st_node + (int64_t)seq * cpl->st_seq);
        if (ps->status == 1 && ps->nkids < ps->count && ps->ids[ps->nkids] == n[0]) ++ps->nkids;
    }
}

RAD_MTP_HD static inline int rad_mtp_tree_grow_seq_cpl(
    const RadMtpTreeCfg* cfg, int* st, const int* didx, const float* dval, long row0,
    long row_stride, long stride, int head_in, const int* table, int* tok, int* par, int* sel,
    int* nsel, long long* vmask, int* rb_out, long seq, int* src_out, int* dst_out,
    const RadMtpTreeCpl* cpl) {
    const int K = RAD_MTP_TREE_K;
    RadMtpCoupleDraw draw;
    RadMtpTreeCfg lcfg;
    if (!rad_mtp_tree_cpl_begin(cfg, cpl, &draw, &lcfg))
        return rad_mtp_tree_grow_seq(cfg, st, didx, dval, row0, row_stride, stride, head_in, table,
                                     tok, par, sel, nsel, vmask, rb_out, seq, src_out, dst_out);
    cfg = &lcfg;
    const int nfront = rad_mtp_tree_cpl_nfront(cfg, st);
    const int before = cfg->round == 1 ? 0 : st[1];
    int   cd[RAD_MTP_RRS_MAX_WIDTH * RAD_MTP_TREE_K];
    float cv[RAD_MTP_RRS_MAX_WIDTH * RAD_MTP_TREE_K];
    for (int w = 0; w < nfront; ++w)
        rad_mtp_tree_cpl_row(cfg, cpl, draw, st, didx, dval, row0, row_stride, stride, seq, w,
                             cd + w * K, cv + w * K);
    const int rc = rad_mtp_tree_grow_seq(cfg, st, cd, cv, 0, K, 1, head_in, table, tok, par, sel,
                                         nsel, vmask, rb_out, seq, src_out, dst_out);
    if (rc < 0) return rc;
    rad_mtp_tree_cpl_post(cfg, cpl, st, before, seq);
    return rc;
}

#endif /* __cplusplus */
#endif /* RAD_MTP_TREE_RRS_H */

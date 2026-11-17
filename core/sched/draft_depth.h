/* draft_depth.h -- same-step adaptive depth of a coupled serial draft chain (--draft-depth adaptive).
 *
 * A fixed chain drafts N tokens every step and its verify pays for N + 1 rows whatever the head
 * thought of them. Here the chain of a step stops as soon as the head's own laws say the next draft
 * is unlikely to be accepted: round j (j >= 2) runs only while the chain's confidence stays at or
 * above the threshold,
 *
 *     round j runs  iff  j <= max  and  ( j <= min  or  conf(j - 1 - lag) >= stop ),
 *     conf(m) = prod_{i = 1..m} s_i,   conf(m <= 0) = 1,
 *
 * where s_i is read from round i's coupled state (abi/rad_spechub.h): the largest mass of the law
 * Q_i the round drew from (`top`), or the mass of the token it drew (`path`, the probability of
 * the drafted path). The first round that does not run ends the chain; its depth D is the number
 * of rounds that ran, and the next verify carries D drafts (D + 1 rows).
 *
 * `lag` is how many rounds the decision trails the chain: with lag 0 round j is decided by every
 * round before it, so the card waits for the host between two rounds; with lag 1 round j is decided
 * when round j - 2 has finished, while round j - 1 runs, so the host's decision (a readback of one
 * state row, then the next round's or the next verify's issue) hides behind a running round.
 *
 * Why this keeps the verify exact. The decision "x_j is drafted" reads Q_1 .. Q_{j-1-lag} and, for
 * `path`, x_1 .. x_{j-1-lag}: draft-side quantities of the chain's ANCESTORS of x_j only, never a
 * target output and never x_j itself. For a row-wise verify any such stopping time is exact (each
 * verified position is exact on its own). For the traversal walk (abi/rad_mtp_traversal.h) the step
 * of its induction at a node v needs v's number of children to be independent of the tokens drawn
 * AT v; the shape of a child's subtree may depend on the child's own token, because the induction
 * hypothesis is applied per child value. Here node v (the prefix x_1 .. x_{j-1}) has a child iff
 * the rule above holds, a function of v's own path. A rule that read x_j's own mass to admit x_j
 * would not be exact (tests/draft_depth_test.cpp shows the statistic that catches it, and one that
 * peeks at the target).
 *
 * Plain code, no engine types beyond the wire row: the engine's driver, the scheduler's tests and
 * the exactness test share it.
 */
#pragma once

#include <cstdint>

#include "rad_spechub.h"

namespace rad {

enum DraftDepthStat : int32_t {
    DRAFT_DEPTH_TOP  = 0,   /* s_i = max_x Q_i(x) */
    DRAFT_DEPTH_PATH = 1,   /* s_i = Q_i(x_i), the drawn token's mass */
};

struct DraftDepthCfg {
    bool   on      = false;
    int    max     = 0;      /* the deepest chain: rounds 1 .. max (the head's declared depth caps it) */
    int    min     = 1;      /* rounds that always run */
    int    lag     = 1;      /* rounds the decision trails the chain, 0 or more */
    int    stat    = DRAFT_DEPTH_TOP;
    double stop    = 0.85;   /* the chain goes on while conf >= stop */
    int    lawless = 3;      /* the depth of a row with no law (a greedy request, a malformed round) */
};

/* One drafting row's chain while its step is drafted. */
struct DraftDepthRow {
    double conf   = 1.0;     /* conf(seen) */
    int    seen   = 0;       /* rounds whose state went into conf */
    int    depth  = 1;       /* rounds decided to run so far (round 1 always does) */
    bool   done   = false;   /* a round was refused: depth is final */
    bool   lawless = false;  /* no law: the fixed fallback depth */
};

/* Round `round`'s statistic from its state, or a negative number when the state carries no law
 * (status != 1: a greedy row or a shortlist the proposer could not turn into Q). */
inline double draft_depth_stat(const RadMtpCoupleState& s, int round, int stat) {
    if (s.status != 1 || s.round != round || s.count < 1 || s.count > 8) return -1.0;
    double top = 0.0, own = -1.0;
    for (int i = 0; i < s.count; ++i) {
        const double q = s.q[i];
        if (!(q > 0.0) || q > 1.0) return -1.0;
        if (q > top) top = q;
        if (s.ids[i] == s.x) own = q;
    }
    return stat == DRAFT_DEPTH_PATH ? own : top;
}

/* The round whose statistic decides round `j` (>= 2); <= 0 when nothing does (conf = 1). */
inline int draft_depth_need(const DraftDepthCfg& c, int j) { return j - 1 - (c.lag > 0 ? c.lag : 0); }

/* The start of a step's chain for one row. `lawless`: the row draws nothing (a greedy request). */
inline void draft_depth_begin(const DraftDepthCfg& c, DraftDepthRow* r, bool lawless) {
    *r = DraftDepthRow{};
    r->lawless = lawless;
    (void)c;
}

/* Feed round `round`'s statistic (rounds in order, each once). A negative one makes the row lawless. */
inline void draft_depth_observe(DraftDepthRow* r, int round, double s) {
    if (round != r->seen + 1) return;
    r->seen = round;
    if (s < 0.0) { r->lawless = true; return; }
    r->conf *= s;
}

/* Whether round `j` (>= 2) runs for this row, given every statistic up to round
 * draft_depth_need(c, j) was observed. Rounds are asked in order; once one is refused the chain has
 * ended. Records the answer in r->depth / r->done. */
inline bool draft_depth_go(const DraftDepthCfg& c, DraftDepthRow* r, int j) {
    if (r->done || j != r->depth + 1) return false;
    bool go;
    if (j > c.max) go = false;
    else if (r->lawless) go = j <= (c.lawless > 0 ? c.lawless : 1);
    else if (j <= c.min) go = true;
    else {
        /* conf(need): the row has seen exactly `need` rounds when asked in order with lag >= 0, or
         * more when the caller observed ahead; recompute is not possible, so the caller observes
         * no further than draft_depth_need before asking (the driver does). */
        go = r->conf >= c.stop;
    }
    if (go) r->depth = j; else r->done = true;
    return go;
}

/* The whole rule at once, for a chain whose statistics are all known (tests, offline): s[i] is
 * round i + 1's, n of them (n >= max - 1 - lag is enough). Returns the depth D in [1, max]. */
inline int draft_depth_of(const DraftDepthCfg& c, const double* s, int n) {
    DraftDepthRow r;
    draft_depth_begin(c, &r, false);
    for (int j = 2; j <= c.max; ++j) {
        const int need = draft_depth_need(c, j);
        while (r.seen < need && r.seen < n) draft_depth_observe(&r, r.seen + 1, s[r.seen]);
        if (!draft_depth_go(c, &r, j)) break;
    }
    return r.depth < 1 ? 1 : r.depth;
}

}  /* namespace rad */

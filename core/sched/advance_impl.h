/* advance_impl.h -- one row of an advance, shared by the kernel (advance.hip) and the host backend
 * (advance.cpp). See advance.h for the modes.
 *
 * Rows are independent: each reads its own slot's state and writes only its own positions, tokens
 * and slots, so the kernel runs one thread per row and the host a loop.
 */
#pragma once
#include "advance.h"

#if defined(__HIPCC__)
#define RAD_ADV_FN __host__ __device__ inline
#else
#define RAD_ADV_FN inline
#endif

namespace rad {

RAD_ADV_FN int32_t* adv_i32(const AdvanceArgs& a, int64_t off) {
    return reinterpret_cast<int32_t*>(a.slab + off);
}

RAD_ADV_FN bool adv_is_eog(const AdvanceArgs& a, int32_t t) {
    int32_t lo = 0, hi = a.n_eog;
    while (lo < hi) {
        const int32_t mid = (lo + hi) / 2;
        if (a.eog[mid] < t) lo = mid + 1; else hi = mid;
    }
    return lo < a.n_eog && a.eog[lo] == t;
}

/* Group `e`'s slot for position `p` of batch row `i`, or -1 past its block table. A runtime
 * window's position at or past ring_pos is in its ring (see adv_groups). */
RAD_ADV_FN int32_t adv_slot(const AdvanceArgs& a, const AdvGroup& e, int i, int32_t p) {
    const int32_t first = adv_i32(a, e.first_off)[i];
    const int32_t* bt = adv_i32(a, e.bt_off) + (int64_t)i * e.pitch;
    int32_t bi = (p - first) / e.block_size;
    if (e.ring && e.ring_blocks > 0 && adv_i32(a, e.tlen_off)[i] > 0) {
        const int32_t rpos = adv_i32(a, e.rpos_off)[i];
        if (p >= rpos) bi = rpos / e.block_size + ((p - rpos) / e.block_size) % e.ring_blocks;
    }
    return (p >= first && bi >= 0 && bi < e.pitch && bt[bi] >= 0)
               ? bt[bi] * e.block_size + p % e.block_size : -1;
}

/* Every group's slots for the row's `n` positions (from the positions just written) and its
 * length from `base`. A position past the block table gets slot -1 ("no slot"); the builder sizes
 * tables for the furthest reachable position, so only a row whose output is unused hits this.
 *
 * A runtime window (e.ring, KVManager::set_window) must be computed exactly as the host builder
 * staged it (BatchBuilder::build_groups), because this runs after the build and overwrites what it
 * wrote:
 *   - `used` stops at the table: the ring reuses its blocks, so the sequence's held span is the
 *     table's, not the context's. Without this the decode walks bt[(used-1)/block] past the end,
 *     reads the -1 padding the builder fills and faults the card on a negative block;
 *   - a position at or past ring_pos lives in the ring, at block
 *     ring_pos/block + ((p - ring_pos)/block) % ring_blocks, not at (p - first)/block, which runs
 *     off the table once the ring wraps.
 *
 * `len`, when given, is the length's count from `base` in place of `n` (ADV_HIST: rows that
 * repeat a position must not lengthen the context). `at`, when given, is where each row stores its
 * K and V instead of its position: a tree verify's rows rotate at their depth but store at their
 * row index, since siblings share a depth (ADV_TRUNK), and a context row the tree rejected stores
 * nowhere (-1, ADV_CTXTREE). */
RAD_ADV_FN void adv_groups(const AdvanceArgs& a, int i, int32_t k, int32_t base, int32_t n,
                           int32_t len = -1, const int32_t* at = nullptr) {
    const AdvGroup* G = reinterpret_cast<const AdvGroup*>(a.slab + a.grp_off);
    const int32_t* pos = adv_i32(a, a.pos_off);
    for (int g = 0; g < a.n_groups; ++g) {
        const AdvGroup& e = G[g];
        const int32_t first = adv_i32(a, e.first_off)[i];
        /* Per sequence: each prompt has its own floor and its own table (advance.h). */
        const int32_t tlen = e.ring ? adv_i32(a, e.tlen_off)[i] : 0;
        const bool ring = e.ring && tlen > 0;
        int64_t used = (int64_t)base + (len >= 0 ? len : n) - first;
        if (ring) {
            const int64_t span = (int64_t)tlen * e.block_size;
            if (used > span) used = span;
        }
        adv_i32(a, e.used_off)[i] = (int32_t)used;
        if (!e.paged) continue;
        int32_t* slot = adv_i32(a, e.slot_off);
        for (int32_t j = 0; j < n; ++j) {
            const int32_t p = at ? at[j] : pos[k + j];
            slot[k + j] = p >= 0 ? adv_slot(a, e, i, p) : -1;
        }
    }
}

/* A tree's node depths from its parents: node 0 (the bonus token) at 0, node j + 1 one below
 * par[j]. A parent outside [0, j] is a drafter bug; it is read as the previous node, so the row
 * still stores and rotates somewhere sane, and the verify rejects whatever it proposed. */
RAD_ADV_FN int32_t adv_parent(const StepSlot& S, int32_t j) {
    const int32_t p = S.par[j - 1];
    return (p < 0 || p >= j) ? j - 1 : p;
}

/* Whether par[0..n) (node j + 1's parent) is a tree no deeper than `depth` (0: any depth): every
 * parent in [0, j] (an earlier node), so parents precede children. */
RAD_ADV_FN bool adv_tree_valid(const int32_t* par, int32_t n, int32_t depth) {
    int32_t dep[1 + RAD_SCHED_MAX_SPEC];
    if (n < 0 || n > RAD_SCHED_MAX_SPEC) return false;
    dep[0] = 0;
    for (int32_t j = 0; j < n; ++j) {
        const int32_t p = par[j];
        if (p < 0 || p > j) return false;
        dep[j + 1] = dep[p] + 1;
        if (depth > 0 && dep[j + 1] > depth) return false;
    }
    return true;
}

/* Writes a row's position and moves its rotary components by the same amount. A row whose
 * position does not change keeps the host's components. */
RAD_ADV_FN void adv_pos(const AdvanceArgs& a, int32_t* pos, int32_t row, int32_t p) {
    const int32_t was = pos[row];
    pos[row] = p;
    if (a.rope_off < 0 || was == p) return;
    int32_t* r = adv_i32(a, a.rope_off);
    const int32_t shift = r[row] - was;
    r[row] = p + shift;
    r[a.rope_pitch + row] = p + shift;
    r[2 * a.rope_pitch + row] = p + shift;
}

/* The node the (t)-th committed position of the last verify came from, in that verify's row order:
 * 0 for the bonus token (t = 0), path[t - 1] for t >= 1 after a tree verify, the row t itself for
 * a chain (npath 0) -- the identity RadBatch::spec_path carries. */
RAD_ADV_FN int32_t adv_path_node(const StepSlot& S, int32_t t) {
    if (t <= 0) return 0;
    /* An npath outside what a verify writes (a slot no acceptance has filled) is a chain. */
    if (S.npath <= 0 || S.npath > RAD_SCHED_MAX_SPEC) return t;
    return t <= S.npath ? S.path[t - 1] : t;
}

/* A serial tree's stage 1: the proposal row's runner-up columns into S.alt, before the next pass
 * overwrites them. Column 0 is the greedy pick (chain node 1), so a leaf equal to it is no
 * sibling; an id out of range would index the embedding out of bounds; a repeat adds nothing.
 * Filtered here so every rank, which holds the same row, keeps the same list. */
RAD_ADV_FN void adv_capture(const AdvanceArgs& a, StepSlot& S, const int32_t* row) {
    S.nalt = 0;
    if (a.leaves <= 0) return;
    const int32_t pick = row[0];
    for (int64_t c = 1; c < a.src_pitch && S.nalt < a.leaves && S.nalt < kAdvMaxLeaves; ++c) {
        const int32_t id = row[c];
        if (id < 0 || (a.vocab > 0 && id >= a.vocab) || id == pick) continue;
        bool dup = false;
        for (int32_t q = 0; q < S.nalt; ++q) dup = dup || S.alt[q] == id;
        if (!dup) S.alt[S.nalt++] = id;
    }
}

RAD_ADV_FN void adv_row(const AdvanceArgs& a, int i) {
    const int32_t* rows = adv_i32(a, a.row_off);
    const int32_t rtok = rows[a.row_pitch + i];
    const int32_t rout = rows[2 * a.row_pitch + i];
    const int32_t aux  = rows[3 * a.row_pitch + i];
    /* Acceptance runs over every row that sampled, whoever staged it; every other mode over the
     * rows the card owns. */
    if (a.mode == ADV_ACCEPT ? rout < 0 : !rows[i]) return;

    int32_t* tok = adv_i32(a, a.tok_off);
    int32_t* pos = adv_i32(a, a.pos_off);
    int32_t* ctx = adv_i32(a, a.ctx_off);
    const int32_t slot = adv_i32(a, a.sid_off)[i];
    const int32_t k = adv_i32(a, a.cu_off)[i];
    const int32_t n = adv_i32(a, a.q_off)[i];
    StepSlot& S = a.st[slot];

    switch (a.mode) {
    case ADV_ACCEPT: {
        const int32_t* y = a.src + rout;
        /* A tree verify, which the trunk's spec_parent row says (its first entry is -1, the root,
         * whoever staged it: ADV_TRUNK for a card row, the builder for a host row): accept_tree's
         * rule. From the bonus token, follow the first child (lowest node index) whose token is
         * the target's pick at its parent's row, as long as one is; then the same
         * end-of-generation cut over the tokens that walk emits. */
        const int32_t* tp = a.parent_off >= 0 ? adv_i32(a, a.parent_off) + rtok : nullptr;
        if (tp && aux > 0 && tp[0] == -1) {
            const int32_t* d = tok + rtok;          /* node j's token is d[j], 1 <= j <= aux */
            const int32_t a0 = rtok - k;            /* parents count rows from the sequence's first */
            int32_t cur = 0, acc = 0;
            for (;;) {
                int32_t nxt = -1;
                for (int32_t j = cur + 1; j <= aux; ++j)
                    if (tp[j] - a0 == cur && d[j] == y[cur]) { nxt = j; break; }
                if (nxt < 0) break;
                S.path[acc++] = nxt;
                cur = nxt;
            }
            for (int32_t t = 0; t <= acc; ++t)
                if (adv_is_eog(a, y[t == 0 ? 0 : S.path[t - 1]])) { acc = t; break; }
            const int32_t head = ctx[i] + (rtok - k) + acc;
            S.acc = acc;
            S.head = head;
            S.ctx = head + 1;
            S.n_acc = acc + 1;
            S.tok[0] = y[acc == 0 ? 0 : S.path[acc - 1]];
            S.hrow = rtok + (acc == 0 ? 0 : S.path[acc - 1]);   /* rows are nodes, in order */
            S.npath = acc;
            return;
        }
        /* accept_greedy's rule, then collect_step's end-of-generation cut: a draft is accepted
         * while it equals the target's pick at its position, and the run stops at an
         * end-of-generation token, not after it. */
        const int32_t* d = tok + rtok + 1;
        int32_t acc = 0;
        while (acc < aux && d[acc] == y[acc]) ++acc;
        for (int32_t j = 0; j <= acc; ++j)
            if (adv_is_eog(a, y[j])) { acc = j; break; }
        const int32_t head = ctx[i] + (rtok - k) + acc;
        S.acc = acc;
        S.head = head;
        S.hrow = rtok + acc;
        S.ctx = head + 1;
        S.n_acc = acc + 1;
        S.tok[0] = y[acc];
        S.npath = 0;
        return;
    }
    case ADV_TRUNK: {
        const int32_t base = S.ctx;
        /* RadBatch::spec_path: the rows of the previous verify that were committed, from the
         * acceptance the card just wrote (the host has not read it yet, the step is issued
         * ahead). The identity past num_accepted and for a chain. */
        if (a.path_off >= 0) {
            int32_t* sp = adv_i32(a, a.path_off) + (int64_t)i * a.path_pitch;
            for (int32_t t = 0; t < a.path_pitch; ++t) sp[t] = t;
            const int32_t np = S.npath < S.acc ? S.npath : S.acc;
            for (int32_t t = 1; t <= np && t < a.path_pitch; ++t) sp[t] = S.path[t - 1];
        }
        if (a.parent_off >= 0) {
            int32_t* par = adv_i32(a, a.parent_off);
            /* A row verifying fewer nodes than the drafter proposed (a window cut it) verifies
             * their prefix, which is a tree too: every parent precedes its child. */
            if (n > 1 && S.ntree >= n - 1) {
                /* A tree: node j stores its K and V at base + j (siblings share a depth, so the
                 * position cannot address them) and rotates at base + its depth, the position the
                 * token would have if its path were committed. Attention reads spec_parent to see
                 * only a row's own ancestors inside the block. */
                int32_t dep[1 + RAD_SCHED_MAX_SPEC], at[1 + RAD_SCHED_MAX_SPEC];
                for (int32_t j = 0; j < n; ++j) {
                    const int32_t p = j == 0 ? -1 : adv_parent(S, j);
                    dep[j] = j == 0 ? 0 : dep[p] + 1;
                    tok[k + j] = S.tok[j];
                    par[k + j] = p;
                    adv_pos(a, pos, k + j, base + dep[j]);
                    at[j] = base + j;
                }
                ctx[i] = base;
                adv_i32(a, a.acc_off)[i] = S.n_acc;
                adv_groups(a, i, k, base, n, -1, at);
                return;
            }
            for (int32_t j = 0; j < n; ++j) par[k + j] = -2;
        }
        for (int32_t j = 0; j < n; ++j) { tok[k + j] = S.tok[j]; adv_pos(a, pos, k + j, base + j); }
        ctx[i] = base;
        adv_i32(a, a.acc_off)[i] = S.n_acc;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_CTXTREE: {
        /* The drafter's context pass over a tree verify's rows (one per node, in node order, as
         * the trunk tapped them), a card row only where the host marked the step's entry as one.
         * Only the accepted path is context: path node t's row writes at the anchor + t + 1,
         * rotated there, which is where its token now sits, and every rejected node's row stores
         * nowhere. Run after ADV_ACCEPT, which wrote the path, and before ADV_BTAKE. A chain
         * verify's path is its first `acc` rows (npath 0). */
        const int32_t base = S.head - S.acc;
        int32_t at[1 + RAD_SCHED_MAX_SPEC];
        at[0] = base;
        adv_pos(a, pos, k, base);
        for (int32_t j = 1; j < n; ++j) { at[j] = -1; adv_pos(a, pos, k + j, base); }
        for (int32_t t = 0; t < S.acc; ++t) {
            const int32_t r = t < S.npath ? S.path[t] : t + 1;
            if (r < 1 || r >= n) continue;
            at[r] = base + t + 1;
            adv_pos(a, pos, k + r, base + t + 1);
        }
        ctx[i] = base;
        adv_groups(a, i, k, base, n, -1, at);
        return;
    }
    case ADV_HIST: {
        if (a.shared_kv) {
            /* A head that reads the trunk's cache and keeps none (rad_drafter_shared_kv). With no
             * history to fill, the pass is round 1 alone: one row at the committed position, with
             * the trunk's row there and the token chosen after it. Its context ends at the
             * committed position, not the verify step's whole run, because the cache past it
             * holds K/V of rejected drafts. The host staged the run's first trunk row; the card
             * adds the accepted drafts. */
            int32_t* out = adv_i32(a, a.out_off);
            const int32_t nd = adv_path_node(S, S.acc);   /* the last committed node's row */
            tok[k] = a.src[rout + nd];
            adv_pos(a, pos, k, S.head + a.rot_shift);
            out[k] += nd;
            ctx[i] = S.head;
            adv_groups(a, i, k, S.head, 1);
            return;
        }
        /* `aux` host-known positions, then the verify step's positions up to the last committed
         * one, each with the token committed after it (the sampler's pick there). The pass has a
         * row for every position the step could have committed; rows past the last committed
         * repeat it, so no rejected draft's hidden state enters the head's cache. The repeat also
         * makes the run's last row draft round 1's input whatever the acceptance was
         * (RadBatch::draft_out_ids). */
        const int32_t base = S.head - S.acc - aux;
        const int32_t last = aux + S.acc;
        int32_t* out = adv_i32(a, a.out_off);
        const int32_t row0 = out[k];
        /* After a tree verify the committed positions are the accepted path's nodes, not the
         * first rows of the block: committed position t (past the `aux` host-known ones) is node
         * path[t - 1]'s, whose hidden row is the node's row and whose next token is the target's
         * pick at that row. A chain (npath 0) is the identity, as before. The run is capped by the
         * deepest path a tree has (the scheduler sizes the pass by it), so a node past `n` rows
         * is never needed. */
        for (int32_t j = 0; j < n; ++j) {
            const int32_t jj = j < last ? j : last;
            const int32_t nd = jj >= aux ? adv_path_node(S, jj - aux) : 0;
            if (jj >= aux) tok[k + j] = a.src[rout + nd];
            adv_pos(a, pos, k + j, base + jj);
            out[k + j] = row0 + (jj >= aux ? aux + nd : jj);
        }
        ctx[i] = base;
        /* The context ends at the last committed position, not at the run's end. Attention places
         * row j of a q_len-row run at seqused - q_len + j and reads keys up to there (attn_paged's
         * contract). With the run's whole length the last row, the only one whose output is used,
         * would also read the n_spec - acc positions after it, which the head has not written for
         * this sequence (stale rounds, or another request's K and V). Earlier rows then count
         * lower than their positions, but nothing reads their output, and every row's K and V
         * still store at its own position's slot. */
        adv_groups(a, i, k, base, n, last + 1);
        return;
    }
    case ADV_ROUND: {
        /* Round 2 onward (round 1 is the history pass's last row); each round embeds the previous
         * round's proposal. A head with no cache of its own stays at the committed position over
         * the committed context; its proposal reaches the next round through its hidden state. */
        const int32_t base = a.shared_kv ? S.head : S.head + a.round - 1;
        const int32_t t = a.src[(int64_t)i * a.src_pitch];
        S.tok[a.round - 1] = t;
        /* A serial tree's leaves: round 1's candidates are in the proposal row right now, and this
         * round's pass overwrites them. */
        if (a.round == 2 && a.tree_nodes > 0) adv_capture(a, S, a.src + (int64_t)i * a.src_pitch);
        tok[k] = t;
        adv_pos(a, pos, k, a.shared_kv ? base + a.rot_shift : base);
        ctx[i] = base;
        adv_groups(a, i, k, base, 1);
        return;
    }
    case ADV_ROUND_TREE: {
        /* A serial head that grows a tree (stage 2): the pass before this one left, per sequence,
         * the children it wants embedded next as tokens (src) with their parents as node ids
         * (par), node 0 the bonus token, at most `width` of them and a negative token ends the
         * list. They become nodes numbered on from the last round's, which the plugin knows by
         * the same rule. Node n's token is embedded at the head index S.head + its depth (the
         * chain's round r sits at head + r - 1: depth r - 1), its K and V stored at the flat slot
         * S.head + n -- siblings share a depth -- and the pass sees the keys [0, head + nodes so
         * far], inside which the plugin's attention keeps a row to its ancestors (spec_parent
         * names the row's parent node). With width 1 this is ADV_ROUND to the bit. A row beyond
         * the list stays inert: token 0, no slot. */
        const int32_t* prow = a.src + (int64_t)i * a.src_pitch;
        const int32_t* qrow = a.par + (int64_t)i * a.src_pitch;
        if (a.round == 2) S.ntree = 0;
        const int32_t first = S.ntree + 1;
        const int32_t room = a.tree_nodes - S.ntree;
        int32_t cnt = 0;
        for (int32_t w = 0; w < a.width && cnt < room && w < n; ++w) {
            if (prow[w] < 0) break;
            int32_t par = qrow[w];
            /* A parent not made by an earlier round (the same pass's rows have no K/V yet): root. */
            if (par < 0 || par >= first) par = 0;
            const int32_t node = S.ntree + 1;
            S.tok[node] = prow[w];
            S.par[node - 1] = par;
            S.ntree = node;
            ++cnt;
        }
        int32_t at[1 + RAD_SCHED_MAX_SPEC];
        int32_t* pp = a.parent_off >= 0 ? adv_i32(a, a.parent_off) : nullptr;
        for (int32_t w = 0; w < n; ++w) {
            if (w < cnt) {
                const int32_t node = first + w;
                int32_t dep = 0;
                for (int32_t x = node; x > 0; x = S.par[x - 1]) ++dep;
                tok[k + w] = S.tok[node];
                adv_pos(a, pos, k + w, S.head + dep);
                at[w] = S.head + node;
                if (pp) pp[k + w] = S.par[node - 1];
            } else {
                tok[k + w] = 0;
                adv_pos(a, pos, k + w, S.head);
                at[w] = -1;
                if (pp) pp[k + w] = -2;
            }
        }
        ctx[i] = S.head + first;
        adv_groups(a, i, k, S.head, n, S.ntree + 1, at);
        return;
    }
    case ADV_TAKE:
        S.tok[a.round] = a.src[(int64_t)i * a.src_pitch];
        if (a.tree_nodes > 0) {
            /* A serial head with one round: its candidates are still in the row (ADV_ROUND would
             * have taken them at round 2). */
            if (a.round == 1) adv_capture(a, S, a.src + (int64_t)i * a.src_pitch);
            /* The chain, node j + 1 under node j, then up to `leaves` siblings of node 1 under the
             * bonus token. The count is fixed by the budget alone (a short list is padded with
             * node 1's token, a sibling the walk never takes: the lowest-index child wins), since
             * the host planned the step issued ahead by it before the card chose. */
            const int32_t done = a.round;
            for (int32_t j = 0; j < done; ++j) S.par[j] = j;
            int32_t nl = a.tree_nodes - done;
            if (nl > a.leaves) nl = a.leaves;
            if (nl < 0) nl = 0;
            for (int32_t q = 0; q < nl; ++q) {
                S.tok[done + 1 + q] = q < S.nalt ? S.alt[q] : S.tok[1];
                S.par[done + q] = 0;
            }
            S.ntree = done + nl;
        }
        return;
    case ADV_BLOCK: {
        /* A block drafter's query row: the anchor (the token the verify just chose) at the
         * committed end, where the next verify starts, then `aux`, the mask id, at every position
         * after it. The host staged the furthest the anchor can be. */
        const int32_t base = S.ctx;
        tok[k] = S.tok[0];
        for (int32_t j = 1; j < n; ++j) tok[k + j] = aux;
        for (int32_t j = 0; j < n; ++j) adv_pos(a, pos, k + j, base + j);
        ctx[i] = base;
        if (a.arow_off >= 0) adv_i32(a, a.arow_off)[i] = S.hrow;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_BTAKE:
        /* Row i is the query pass's i-th block, which is the i-th drafting sequence. A tree
         * drafter's parents come with the tokens, at the same pitch. */
        for (int32_t j = 0; j < a.round; ++j)
            S.tok[1 + j] = a.src[(int64_t)i * a.src_pitch + j];
        if (a.par) {
            for (int32_t j = 0; j < a.round; ++j) S.par[j] = a.par[(int64_t)i * a.src_pitch + j];
            S.ntree = a.round;
            /* A serial tree the plugin grew: the host reserved K/V and history rows for depth
             * <= R, so anything else becomes the chain prefix (nodes 1..R, then siblings of node
             * 1 that the walk never takes), a tree of the same node count. */
            if (a.tree_nodes > 0 && a.depth > 0 &&
                !adv_tree_valid(S.par, S.ntree, a.depth)) {
                for (int32_t j = 0; j < S.ntree; ++j) {
                    if (j < a.depth) S.par[j] = j;
                    else { S.par[j] = 0; S.tok[1 + j] = S.tok[1]; }
                }
            }
        } else {
            S.ntree = 0;
        }
        return;
    default:
        return;
    }
}

/* ADV_COMPACT for one (batch row, pool, layer, head): the moves the row's accepted path asks
 * for, `c0` and every `cstep`-th 16-byte chunk of each K|V row. Node path[t] stored at
 * base + path[t] and belongs at base + t + 1. path[] increases (a parent's index is below its
 * child's) and path[t] >= t + 1, so in t order no move overwrites a node a later move still reads:
 * a destination base + t + 1 is either the node's own place or a place a node already moved out
 * of. Each caller keeps one chunk set across every t, so its reads and writes stay in order. */
RAD_ADV_FN void adv_compact(const AdvanceArgs& a, int i, int p, int l, int h, int c0, int cstep) {
    /* Only a row that sampled in this step: ADV_ACCEPT just wrote its slot. Any other row's slot
     * may hold a path from an earlier request that used the same slot. */
    const int32_t* rows = adv_i32(a, a.row_off);
    if (rows[2 * a.row_pitch + i] < 0) return;
    const StepSlot& S = a.st[adv_i32(a, a.sid_off)[i]];
    const int32_t np = S.npath < S.acc ? S.npath : S.acc;
    if (np <= 0) return;
    const AdvPool& P = a.pool[p];
    if (!P.base || l >= P.n_layers || h >= P.n_head) return;
    const AdvGroup& e = reinterpret_cast<const AdvGroup*>(a.slab + a.grp_off)[P.grp];
    if (!e.paged) return;
    const int32_t base = S.head - S.acc;
    const int32_t bs = e.block_size;
    char* lay = P.base + (int64_t)l * P.layer_stride;
    const int32_t chunks = P.row_bytes / 16;
    for (int32_t t = 0; t < np; ++t) {
        const int32_t src = base + S.path[t], dst = base + t + 1;
        if (src == dst) continue;
        const int32_t ss = adv_slot(a, e, i, src), ds = adv_slot(a, e, i, dst);
        if (ss < 0 || ds < 0) continue;
        const uint32_t* from = reinterpret_cast<const uint32_t*>(
            lay + (((int64_t)(ss / bs) * P.n_head + h) * bs + ss % bs) * P.row_bytes);
        uint32_t* to = reinterpret_cast<uint32_t*>(
            lay + (((int64_t)(ds / bs) * P.n_head + h) * bs + ds % bs) * P.row_bytes);
        for (int32_t c = c0; c < chunks; c += cstep) {
            const uint32_t v0 = from[4 * c], v1 = from[4 * c + 1], v2 = from[4 * c + 2],
                           v3 = from[4 * c + 3];
            to[4 * c] = v0; to[4 * c + 1] = v1; to[4 * c + 2] = v2; to[4 * c + 3] = v3;
        }
    }
}

/* The (pool, layer, head) of ADV_COMPACT's work item `w`, counting every pool's layers x heads in
 * order; false past the last. */
RAD_ADV_FN bool adv_compact_item(const AdvanceArgs& a, int w, int* p, int* l, int* h) {
    for (int q = 0; q < a.n_pool; ++q) {
        const int n = a.pool[q].n_layers * a.pool[q].n_head;
        if (w < n) { *p = q; *l = w / a.pool[q].n_head; *h = w % a.pool[q].n_head; return true; }
        w -= n;
    }
    return false;
}

RAD_ADV_FN int adv_compact_items(const AdvanceArgs& a) {
    int n = 0;
    for (int q = 0; q < a.n_pool; ++q) n += a.pool[q].n_layers * a.pool[q].n_head;
    return n;
}

}  /* namespace rad */

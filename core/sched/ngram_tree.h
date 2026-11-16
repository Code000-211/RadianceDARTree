/* ngram_tree.h -- prompt-lookup branches grafted onto a drafted tree (--draft-ngram-leaves,
 * --draft-ngram-min-match). DESIGN-ngram.md has the proof; the short form:
 *
 * A tree drafter's verify follows, at every row, the first child whose token is the token the
 * sampler emitted at that row, and the emitted token is a draw from the target's law P at the row
 * whatever the tree below it looks like (plain draw, or the coupled recursive-rejection draw over
 * the MODEL's drafted children, which is exact for any prefix-closed child set chosen without
 * reading the drawn values). Extra children therefore cannot change the emitted law; they only give
 * the walk one more successor to follow when the emitted token happens to be theirs. Three rules
 * keep that true under --draft-coupling:
 *
 *   1. the coupled row sees only the MODEL's children: every node past `n_model` is a prompt-lookup
 *      node, and the sampler leaves it out of the row's child list (Request::draft_extra);
 *   2. the model's nodes that are displaced to make room are a SUFFIX of the node order (a suffix
 *      of every parent's draft order, like the dynamic-rows cut), and HOW MANY is a function of the
 *      lookup proposal's length, the budget and the tree's shape only. It must not depend on
 *      whether the proposal happens to agree with the drawn tokens: a cut that is made only when
 *      the drawn child differs from the proposal reads the draw, and the same statistic that
 *      catches a value-dependent prefix cut catches it (tests/mtp_rrs_test.cpp);
 *   3. the proposal is matched against the nodes that SURVIVE the cut only, and what does not
 *      match becomes new lookup nodes; a token that equals a displaced node's is simply grafted
 *      again, uncoupled.
 *
 * The proposal is a root-anchored token string: it predicts the tokens after the root (the last
 * committed token), as NgramIndex::propose does, so it is merged as a path from node 0. Where it
 * agrees with an existing path it follows it (a trie merge), and only the remainder is new, so a
 * proposal that extends a drafted chain is a chain extension and one that diverges at the root is a
 * leaf-first branch. */
#pragma once
#include <algorithm>
#include <cstdint>

namespace rad {

struct NgramTreeCfg {
    int leaves = 0;        /* most nodes one merge may add (and displace); 0: off */
    int min_match = 8;     /* shortest match (tokens) a proposal may have to be grafted */
    int max_nodes = 0;     /* the verify's node budget (<= 64) */
    int max_depth = 0;     /* deepest path a verify may carry (the head's rounds) */
};

struct NgramTreeOut {
    int n = 0;             /* nodes after the merge */
    int n_model = 0;       /* nodes 1..n_model are the model's, n_model + 1..n the lookup's */
    int added = 0;         /* lookup nodes appended */
    int displaced = 0;     /* real model nodes dropped for them (inert duplicates are free) */
};

/* tok[j] / par[j]: node j + 1's token and parent (0 the root), j < n; both must hold max_nodes
 * entries. `n_active` counts the leading nodes that are real (the rest are inert duplicates, which
 * are free room and are overwritten). Nodes are in creation order: parents precede children.
 * With added == 0 and displaced == 0 the arrays and the count are untouched; with displaced > 0 and
 * added == 0 the model's tree was cut and nothing was grafted (the proposal was already in it). */
inline NgramTreeOut ngram_tree_merge(int32_t* tok, int32_t* par, int n, int n_active,
                                     const int32_t* prop, int np, int match,
                                     const NgramTreeCfg& c) {
    NgramTreeOut r;
    r.n = n;
    r.n_model = n;
    if (c.leaves <= 0 || np <= 0 || match < c.min_match || n < 1 || n > c.max_nodes ||
        c.max_nodes > 64 || c.max_depth < 1)
        return r;
    if (n_active < 1) n_active = 1;
    if (n_active > n) n_active = n;
    /* How many nodes the proposal could add if it matched nothing, and so how many real nodes the
     * budget must give up for them: values of the history, the budget and the shape only (rule 2). */
    const int nominal = std::min({ np, c.leaves, c.max_depth });
    const int room = std::max(0, c.max_nodes - n_active);
    const int cut = std::max(0, std::min({ nominal - room, c.leaves, n_active - 1 }));
    const int base = n_active - cut;
    /* walk the proposal down the surviving tree: the first child with the token, in node order */
    int depth[65] = {};
    for (int j = 1; j <= base; ++j) depth[j] = depth[par[j - 1]] + 1;
    int cur = 0, i = 0;
    while (i < np) {
        int hit = -1;
        for (int j = 1; j <= base && hit < 0; ++j)
            if (par[j - 1] == cur && tok[j - 1] == prop[i]) hit = j;
        if (hit < 0) break;
        cur = hit;
        ++i;
    }
    const int need = std::max(0, std::min({ np - i, c.leaves, c.max_depth - depth[cur],
                                            c.max_nodes - base }));
    if (cut == 0 && need == 0) return r;
    int at = cur;
    for (int k = 0; k < need; ++k) {
        tok[base + k] = prop[i + k];
        par[base + k] = at;
        at = base + k + 1;
    }
    r.n = base + need;
    r.n_model = base;
    r.added = need;
    r.displaced = cut;
    return r;
}

}  /* namespace rad */

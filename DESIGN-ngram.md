# Prompt-lookup (n-gram / suffix) branches in the MTP draft tree

Flags: `--draft-ngram-leaves N` (0 = off, default), `--draft-ngram-min-match L` (default 8, gate).
Code: `core/sched/ngram_tree.h` (merge), `Scheduler::graft_ngram` (core/sched/scheduler.cpp),
`Request::draft_extra` (core/rad_core.h), the child list in `Sampler::build_step`.
Tests: `sched_test` (merge + scheduler), `mtp_rrs_test` (statistical exactness incl. negative control),
`config_test` (flags).

## What it does

After the head's tree for a sequence is published (`Scheduler::set_draft`), the sequence's own
lookup index (`NgramIndex`, the `--spec-ngram` matcher, reused as is) proposes up to `tree_rounds`
tokens that followed the latest earlier occurrence of the sequence's last n tokens. If the match runs
at least L tokens back, the proposal is merged into the tree as a path from the root (trie merge):
where it agrees with an existing path it follows it, the remainder is new nodes. So a proposal that
continues the drafted chain is a chain extension, one that leaves at the root is a branch. At most N
nodes are added; room is, in order, the pruned slots (inert duplicates), then the unused node budget
(`draft_cap`), then displacement of up to N of the model's LAST nodes (never the matched path).
Depth stays <= the head's rounds, node count <= the budget, so no reservation, KV or card-state
bound changes.

## Where it applies (important limitation)

Only on HOST-BUILT tree verifies: steps whose tree is published before the step is planned (the first
step after a prefill, a step not issued ahead, the scheduler tests). In the pipelined fast path
(`pipelining_`, the step issued ahead) the tree's tokens are made on the card by ADV_BTAKE while the
host only mirrors them; the lookup needs the sequence's committed end, which the host learns after
the commit that comes AFTER that step was issued, so the lookup cannot be grafted there without a
card-side index (or a stale-suffix variant). That is the follow-up (see open issues); this package
delivers the lossless machinery, the gate, the sampler contract and the proof, and measures nothing
on device. Expect the end-to-end gain to be small until the ahead path is covered.

## Exactness (also under `--draft-coupling`)

Claim: the emitted tokens are distributed as the target law P exactly as without the graft.

1. The verify walk emits at a visited row a token y_row and descends into the first child whose
   token is y_row (ADV_ACCEPT / `accept_tree`). The marginal law of y_row given the committed prefix
   is P_row, whatever the tree: an ordinary draw (u from `(seed, pos)`), or the coupled draw, which is
   exact for a fixed sibling set (recursive rejection sampling, DESIGN-coupling.md, abi/rad_mtp_rrs.h)
   provided that sibling set is chosen without reading the drawn values.
2. A lookup node only adds a successor. Given the row's y_row, the walk enters it iff y_row equals its
   token and no earlier child does; y_row itself was drawn without looking at the node. The next row's
   draw has law P at its own prefix, independent of how the previous row was reached. By induction
   over depth the emitted sequence is a sample of the autoregressive target.
3. The one way this fails: a lookup node entering the COUPLED row's child list. The coupled row
   reads its children `kid[]` (first child = the proposal x, the prefix test `rad_mtp_rrs_prefix`), and
   a lookup node that happens to equal the row's unused draw `x` (a node with nkids = 0 binds as a
   chain-coupled row with proposal x) selects the row's coupling by the value of the draw: given
   x = d the emitted law is the maximal-coupling conditional, not P. Rule: nodes past
   `Request::draft_extra` (the count of model nodes, set by `graft_ngram`, reset by every
   `set_draft`/promote) are skipped by the sampler's child list (tree and chain forms). The negative
   control `a_lookup_node_under_a_coupled_row_stays_exact_only_while_hidden_from_it` (a lookup node
   under a row whose own child was displaced, token = target argmax) is exact hidden (z < 1.3) and
   biased visible (z = 146 / 143, CHAIN / STAGE1).
4. Displacement drops a SUFFIX of the node order, i.e. a suffix of every parent's draft order: the
   same family of cuts as `--draft-tree-rows` (a prefix of each node's draft order, a function of
   shape and history only). HOW MANY nodes go is fixed from the proposal's length, `leaves`, the
   budget and the real-node count BEFORE the proposal is matched against the drawn tokens
   (`nominal = min(np, leaves, max_depth)`, `cut = nominal - free room`). A first version cut by the
   post-match need (fewer when the drawn child equals the proposal): that reads the draw, and the
   statistic caught it (z = 869 on stage 1 + pruning) -- the final rule is the tested one. The
   proposal is then matched against the surviving nodes only; unmatched tokens become grafted nodes
   even when they equal a displaced node's token (uncoupled, so harmless). If the proposal turns out
   to be already in the tree the cut still happens and nothing is grafted (rare, small loss).
5. Dedup: a lookup token equal to an existing child is not added (the merge follows the existing
   node). Siblings stay distinct.

Un-coupled sampled requests and greedy are covered by 1 and 2 directly (a lookup token is followed
iff it equals the row's token).

Because lookup nodes are never coupled, their expected acceptance at a row is P_row(d) (after the
model's children are exhausted: the residual law), not boosted by coupling. At T=0.7/1.0 on text
where the lookup match is long (code edits, quotes, JSON) P_row(d) is large, which is where the gate
L >= 8 selects.

## Measurement plan for the sweep

`SchedMetrics::ngram_graft_{steps,nodes,displaced,verifies,hits,hit_nodes}`: hit rate = hits /
verifies; nodes accepted per graft = hit_nodes / verifies. Sweep L in {4, 6, 8, 12}, N in {1, 2, 3,
4}, with `--draft-coupling on`, T in {0.7, 1.0}, on a code/edit prompt set and a prose set (the gate
must leave prose untouched: steps -> 0). Cost: each grafted node is a verify row (~0.9-1.1 ms), so
the break-even hit rate for N nodes at L is about N * 1.0 ms / (E[extra accepted] * 8.4 ms).

## Open issues / follow-ups

* Ahead path (above): card-side n-gram table (hash of the last k tokens -> position, kept in the
  slot state by ADV_ACCEPT) or a stale-suffix lookup issued with the step ahead.
* The `LookupGate` cost model (lookup.h) is not consulted; the match-length gate is the only gate.
* No server metric export of the graft counters yet.

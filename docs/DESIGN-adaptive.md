# DESIGN-adaptive: cost-aware, calibrated node selection of a serial draft head

Flags: `--draft-adaptive on`, `--draft-cost-table`, `--draft-cost-draft-ms`, `--draft-adaptive-margin`,
`--draft-adaptive-floor`, `--draft-adaptive-cont`, `--draft-calib`. Code: `abi/rad_mtp_adapt.h`
(rule, cost model, calibration, parser), `abi/rad_mtp_tree.h` (grower), `abi/rad_mtp_rrs.h`
(`rad_mtp_rrs_slot_keep_count`, stage-1 leaves under coupling), `core/sched/advance_impl.h`
(`adv_capture`, stage-1 leaves without coupling).

## 1. Why

Decode is device bound: a plain step is ~8.4 ms, every extra verify row costs ~1.0 ms (the MoE expert
union grows). A node that is accepted with probability p buys p tokens for ~1 ms of every step. The tree
shape therefore has to maximise tokens per second, not tokens per step. A fixed tree (chain3, 7 nodes)
over-pays where the head is unsure and under-buys where it is sure.

## 2. The rule (EVICT-style)

With k nodes beside the root the step verifies k+1 rows and is expected to commit

    E(k) = 1 + sum_i exp(cum_i)            (cum_i: the node's cumulative CALIBRATED log probability)

in T(k) = C(k) + D ms (C the measured verify cost of k+1 rows, D the head's rounds). The best-first
grower offers its best pooled candidate (probability p). It is created iff it raises tokens per time:

    (E + p) / T(k+1) > E / T(k)   <=>   p * T(k) > margin * E * (C(k+1) - C(k))

The first `floor` nodes are always created (default 1: the trunk). A refused candidate stays in the pool.
After a refusal the pass ends; later passes can still create better candidates. Nodes never created
leave the verify as inert entries (token -1, parent 0), exactly as `--draft-tree-prune` does, so
`--draft-tree-rows` / `--draft-tree-chain` (dynamic rows, chain fast path) see fewer active nodes and
issue fewer rows with no change.

C(k): `--draft-cost-table a,b,c,...` is the step ms for 0,1,2,... nodes (monotone, last slope extends).
Default: 12.3 ms at 3 nodes, +1.0 ms a node (profile: base ~12.3 ms at 4 rows). D defaults to 0.5 ms a
round. Because the head's rounds are issued unconditionally on the card, D only sets the denominator.

Stage 1 (chain of `rounds` nodes + depth-1 leaves) sees only the first round's row: the chain's tail is
estimated (each level continues with `--draft-adaptive-cont`, default 0.7) and leaf j is kept while it
passes the same test with k = rounds + kept leaves. Stage 2 (`--draft-tree-width >= 2`) applies the exact
rule to the whole best-first tree, including chain depth: use width 2 with `--draft-tree-nodes` = the
budget to get an adaptive chain. The plain chain3 (`--draft-tree off`) is untouched: with the flag
and no serial tree the engine behaves as before ("degrades to chain3").

## 3. Calibration

`--draft-calib T1,T2,T3`: the children of a node at depth d are scored with softmax(v / T_d) over the head's
top-8 (the last T repeats). T > 1 flattens an over-confident head and hence prunes more. The map is applied
to the grower's pool scores, to the relative-confidence test and to stage-1 leaves. `RadMtpCalib`
(abi/rad_mtp_adapt.h) is the host-side online fitter (per depth predicted vs observed acceptance, damped
multiplicative step on T); it is a library plus a unit test. It is NOT wired into the running engine: the
device grower's parameters are graph constants, so the fixed `--draft-calib` is what runs (open item:
a device-resident parameter row the scheduler updates).

## 4. Losslessness (coupling)

Recursive rejection sampling over a node's drafted children is exact iff (a) the children that exist are
a prefix of the draft order and (b) whether a child exists never depends on the value of a token drawn
for it (DESIGN-coupling.md 3.5). Everything the rule reads is the slot priority of the candidate (log of
the j-th largest q of the node's own law plus the parent's score), the cumulative scores of nodes already
created (themselves slot priorities), the node count and the static cost table. Calibration is a monotone
per-depth map of the row's own slot values. The pool always creates its maximum first, so the creation
order within a parent is the slot order: a prefix. `tests/mtp_rrs_test.cpp` adds
`coupling_with_adaptive_selection_emits_the_target_law` (stage 1, 2, 2b, with calibration, beside pruning,
far head), `adaptive_trees_cut_by_dynamic_rows_stay_exact`, `the_stage1_leaf_rule_keeps_a_prefix` and unit
tests of the rule, cost model, calibration and the grower (off = unchanged; monotone in margin; floor;
inert tail). The existing value-dependent negative control stays red as designed.

## 5. Plumbing

`Config` -> `Engine::draft_adapt_` text -> `RadBuildCtx::draft_adapt` (appended ABI field at offset 224)
-> the plugin's `mtp_tree_grow` / `mtp_couple_propose` string param `adapt` (ref, avx, hip). `AdvanceArgs`
carries the parsed rule for non-coupled stage 1. Everything is zero/empty when the flag is off: the off
path is bit-identical (the calibration shortcut `T == 1` skips the float round trip).

## 6. Sweep guidance

margin 0.7 / 1.0 / 1.4 / 2.0; cost tables from a real profile (`--draft-cost-table 9.3,10.3,...`);
calibration `1.3,1.15,1.0` vs off at T=1.0 sampling; width 2..3 with nodes 5..9. Expect the gain where
acceptance is low (T=1.0) and no loss at T=0.7 versus the best fixed tree.

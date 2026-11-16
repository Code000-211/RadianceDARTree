# DESIGN-online-calib: lossless online acceptance calibration and automatic per-depth draft temperature

Flags: `--draft-calib online|static|off[:LIST]`, `--draft-coupling-temp-autofit on|off`,
`--draft-calib-log PATH`, `--draft-calib-ece on|off`, `--draft-calib-interval N`. All need
`--draft-coupling on`. Defaults (`static`, `off`, none, `off`, 256) are the old behaviour: no word in
any draw, no export, no readback, no new op parameter, so outputs and the declared graph are bit for
bit what they were. Code: `abi/rad_mtp_calib.h` (labels, features), `core/sched/mtp_calib.h` (the
controller), `abi/rad_spechub.h` (the draw word, the P export), `abi/rad_mtp_rrs.h`,
`abi/rad_mtp_tree.h`, `abi/rad_mtp_tree_rrs.h`, `plugin/spechub/spechub.{h,hip}` (the device side),
`core/sample/sampler.*` (the readback), `core/engine_draft.cpp` (the hook), `tools/rad_calib_sim.cpp`.

## 1. The label

Under maximal coupling the draft x ~ Q is accepted at a reached row with probability
`alpha = sum_y min(P(y), Q(y))`, averaged over the draw of x, where P is the target's truncated law at
the row and Q the law the draft was actually sampled from (the state stores it). alpha depends on
(P, Q) only, so every reached verified row delivers it with no 0/1 noise. For a tree node with several
children (recursive rejection sampling) child i is judged against the residual P_i = norm(max(P_{i-1}
- q_{i-1}, 0)) and q_i = q_{i-1} without child i-1; its label is `alpha_i = sum min(P_i, q_i)`, i.e.
P(child i accepted | row reached, earlier siblings rejected), averaged over the token the draw put in
slot i. `rad_calib_labels` computes it with the verify's own recursion. Tests: against a brute-force
sweep of the acceptance uniform for chains (agreement 2.5e-3), and against the real recursive decide
over 3e5 seeds for siblings (0.001 to 0.002 in the observed runs).

Raw q(x) of the drawn token is a poor predictor of this: it is a function of the token drawn, while
alpha averages over it. The static slot probability (what `--draft-calib` shapes) is the j-th largest q,
which the exactness argument requires the grower to use (selection reads slot priorities, never the
drawn token); its relation to alpha_j is what is learnt here.

## 2. Getting P without a host sync

The sampler works on the device; its candidate planes never reach the host. The verify op already holds
the FP64 softmax of the candidates for its decision. With the draw's export bit set (bit 31 of
`RadMtpCoupleDraw::dpad_`) the verify, after deciding (the decision code is untouched), rewrites
`state.q[0..count)` with P's mass on the state's own ids and sets `status` to 2
(`rad_mtp_couple_export_p`; the host row, the chain device kernel and the tree path share it). The
sampler enqueues one pinned D2H copy of the verify rows right behind the op on the step's stream (a
ring of four steps, rank 0 only); the step's own waits cover it, so the batch-1 loop gains no
synchronisation. The host has Q already (it staged it), so (Q, P on Q's support) gives every label and
the whole tau grid. A row whose candidate plane is invalid keeps status 1 and yields no label. The
export bit is only set when a mode needs it.

The engine (`Engine::couple_calibrate`) counts a row when it was reached: a chain row k when the
accepted prefix is >= k, a tree row when it is the root or on the accepted path; the judged children are
the rejected ones before the accepted child, or all when none was accepted.

## 3. What the controller learns (core/sched/mtp_calib.h)

Request class = temperature bucket (T x 20). Depth class = 1, 2, 3+. Slot class = 0, 1, 2+.

(a) Calibration map. Binned EMA of alpha against the static slot probability (16 bins, decay 0.9995 per
observation), isotonic by pool-adjacent-violators at query time, prior = the identity on the static
prediction (4 pseudo-observations a bin), so a bin with no data predicts the static value. This is the
map reported as `ece ... map` and used by the replay tool's `online-map` policy.

The device grower can only take a monotone per-depth map of the slot priorities; its parameters are not
graph constants for the coupled grower any more: the draw carries them. A calibration code names
(T, lift) with T in {0.5, 0.75, 1, 1.4, 2} and lift in {0, 0.2, 0.4}: p' = lift + (1 - lift) softmax(log q
/ T). Per depth class the controller keeps the exponentially decayed Brier sum of the 16 hypotheses (the
static temperature and the 15 codes) against the exact labels and, every `--draft-calib-interval` verify
steps, moves to the best one when it beats the live one by more than 1 % (hysteresis); at least 60
effective observations first. The static `--draft-calib` list is the starting point (code 0 = keep the
static temperature) and the prior. The word reaches the adaptive rule through `rad_mtp_calib_apply`
(coupled grower, stage-1 leaf rule); the pool scores, the relative-confidence test inputs, and
`rad_mtp_adapt_admit` all see the replaced temperatures and lifts. Without `--draft-adaptive on` the
calibration is learnt and logged but steers nothing (the engine warns).

(b) Draft temperature autofit. Per depth class and grid point tau in {0.05, 0.6, 0.75, 0.9, 1.0, 1.15} plus
the request's base tau, running sums of `sum min(P, Q_tau)` over the reached rows' first child. Q_tau is
rebuilt from the stored Q (`q^(tau0/tau)` renormalised: Q was `softmax(v / tau0)`), so all grid points are
scored on the SAME rows, no exploration cost. Every interval the live tau_d moves to the argmax if it
beats the live one by > 0.5 % relative. tau = 0.05 is the one-hot / greedy-draft limit, so greedy vs
sampled drafting is decided by data. After a switch the depth's calibration statistics are damped (x0.2):
they were taken against another Q. The state is exact for a draw without top-k/top-p truncation
(`--draft-coupling-truncate off`, the default); with truncation the rebuilt grid ignores the cut, which
only affects which tau is chosen.

## 4. Losslessness

The controller's outputs are words in the next draws. A word picks (i) the Q a depth's children are
drawn from and (ii) the calibrated slot priorities the best-first rule and the leaf rule read. The state
records the Q that was drawn from and the verify reads exactly that Q (draw and residual are the same
law, by construction: `rad_mtp_couple_build_law_at` makes Q once, the state keeps it). (ii) is a monotone
per-depth map of a row's own slot values, so the creation order within a node stays the slot order (a
prefix) and the decisions read slot priorities only: the conditions of DESIGN-coupling.md 3.5 and
docs/DESIGN-adaptive.md section 4. The controller reads only past statistics, and past rows' labels do
not depend on the current step's uniforms. Tests: `a_draft_law_that_changes_every_step_stays_exact`
(tests/mtp_rrs_test.cpp, chi-square over 150k trials per kind with a different word per trial: chain,
stage 1, stage 2, far head, base recipe underneath; z in [-1.1, 0.02]); the existing value-dependent
negative control stays red.

## 5. Replay log and tool

`--draft-calib-log PATH` writes `R step seq T depth slot since acc alpha top1 ent adaedl tau0 n q[8] p[8]`
per judged child of a reached row (q in draft order, p = P on the same ids, `since` = tokens since the
request's last rejection) and `S step ms rows acc` per verify step (ms: time between two step ends, which
includes the head's rounds). `rad-calib-sim LOG [--cost a,b,..] [--draft-ms-round MS] [--static-T ..]`
replays it: fixed chain3/4/K (exact: the same draws, truncated), and the adaptive rule driven by the
static map, by the online temperature, by the online binned map, and by the oracle label. A depth the
log never reached has no law; its probability is the running mean of the policy's predictions there
(stated approximation). It also prints Brier and ECE of the three predictors per depth.

## 6. Cost

Verify: eight exps and a few stores per coupled row, only with the export bit. Readback: n_rows x 176 B
pinned, behind the verify, overlapped by the drafter's passes. Host: about 15 hypotheses x 8 exps per
judged child and one pass over the grid, a few microseconds a step.

## 7. Not done / limits

* The device only applies the 2-parameter family (T, lift) per depth class; the binned map is richer
  (ECE 0.02 against 0.14 in the synthetic test). Feeding the whole map to the grower needs a
  device-resident parameter row.
* Depth classes are 1, 2, 3+ (the draw word has one byte a class). Tokens since the last rejection
  and the AdaEDL proxy are logged and available to the replay tool but are not map features online.
* End-to-end measurements (section 8) were not run here: starting a server was out of bounds in this
  session.

## 8. Measurement plan (for the card)

    radiance ... --draft-coupling on --draft-tree on --draft-tree-nodes 7 --draft-tree-width 2 \
       --draft-adaptive on --draft-cost-table 9.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3 \
       [--draft-calib online:1.0] [--draft-coupling-temp-autofit on] --draft-calib-ece on \
       --draft-calib-log run.log

at T 0.7 and 1.0 (top_p 0.95, top_k 20), prose and code, paired seeds, >= 5k steps, against
`--draft-calib static:1.3,1.1` and fixed chain3. Compare tokens/step and tok/s; the periodic
`MTP online calibration` INFO line gives the chosen tau_d, T_d, lift_d and the ECE per depth
(static / live / map). Replay `run.log` with `rad-calib-sim` to see policies the run did not take.
Keep the flags off if the gain is below 1 %.

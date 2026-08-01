# Technical journey — project 7, PPE compliance

Running record of what was decided, what broke, and why. Metrics land in
`outputs/experiment.json`; this file is the reasoning behind them.

---

## 1. Dataset: what SH17 actually contains

The class index → name map was **not** taken from the paper. It was recovered by aligning YOLO
line order against VOC object order across 400 random images:

| id | name | id | name | id | name |
|---|---|---|---|---|---|
| 0 | person | 6 | foot | 12 | head |
| 1 | ear | 7 | tools | 13 | medical-suit |
| 2 | ear-mufs | 8 | glasses | 14 | shoes |
| 3 | face | 9 | gloves | 15 | safety-suit |
| 4 | face-guard | 10 | helmet | 16 | safety-vest |
| 5 | face-mask-medical | 11 | hands | | |

Purity 1.000 on all 17 classes. Worth the ten minutes: a silently wrong helmet index would have
produced a fully functional pipeline computing a meaningless number.

**Corpus profile** — 8,099 images, 75,994 instances (9.38/image), median max-side 5,866 px.

| persons/image | 0 | 1 | 2 | 3 | 4–6 | 7+ |
|---|---|---|---|---|---|---|
| images | 482 | 4,825 | 1,525 | 580 | 547 | 140 |

## 2. The finding that changed the compliance rule

The brief's verdict was to be helmet **and** vest. Counting the held-out pool killed the vest half:

| | single-person held-out images |
|---|---|
| with head, no helmet (non-compliant source) | 712 |
| with head + helmet (compliant source) | 35 |
| with safety-vest | **14** |

Fourteen source images cannot support a headline metric. **Compliance is head-protection only.**
Vest stays a detector class (it is nearly free and its mAP is reportable) but it is not in the
verdict. This matches the derivation rule already written in `DATASET_DECISION.md`, so it is a
narrowing forced by the data rather than a change of question.

The 35 compliant sources are the binding constraint on the whole project. See §6.

## 3. The baseline was broken, and fixing it is the main methodological call

The original scaffold defined ARM A as *"assign each PPE box to the person box it overlaps most,
above IoU ≥ 0.30"*.

A helmet box is roughly 1–2% of a person box's area, so helmet↔person IoU is about 0.01–0.02. It
**never clears 0.30**. ARM A would assign nothing, mark every person undetermined, and score
zero. ARM B would win by a mile against an arm that was never in the race, and the result would
be worthless.

The fair baseline is the rule practitioners actually use for small-PPE assignment —
**containment**, the fraction of the PPE box inside the person region — with argmax over persons.
That leaves the arms differing in exactly one thing:

| arm | person region | threshold |
|---|---|---|
| A0 | bounding box, scored by IoU | 0.30 |
| A | bounding **box**, scored by containment | 0.50 |
| B | instance **mask**, scored by containment | 0.50 |

A and B **share one threshold** (`config.CONTAINMENT_THRESHOLD`). Tuning each arm separately
would confound the mechanism with threshold search. A0 is kept and reported because its collapse
is the evidence that A is an honest baseline rather than a strawman chosen to lose.

## 4. Ground truth: composites, and two traps avoided

SH17 never says which head belongs to which person — that is why it was chosen, and it means the
corpus contains no association label to score against. Scoring against a box-IoU-derived "truth"
would define ARM A correct by construction.

So: take held-out images with exactly one person (every PPE box provably belongs to them) and
composite *k* of them into one scene, carrying the links through exactly.

**Trap 1 — mask paste would rig the result for ARM B.** Pasting SAM-segmented silhouettes leaves
a sharp figure/ground boundary that SAM can re-segment almost trivially, inflating the arm under
test for a reason unrelated to the mechanism. Rectangular crops carry their own background, so
ARM B's segmentation has to do real work. Less photorealistic, and conservative in the right
direction.

**Trap 2 — a fixed canvas would confound crowding with scale.** The first implementation squeezed
everyone to fit 1280 px, so 9-person scenes had persons at ~half the size of 2-person scenes.
"More crowded" and "smaller objects" would have moved together, and the central claim — that the
ARM B margin *grows with crowding* — would have been untestable. Fixed: person height is held
constant and the **canvas widens with the crowd** (cap 2560 px).

Two smaller fixes from looking at the renders: SH17's zero-person images are object close-ups (a
jar, a hand), which pasted sharp behind workers looked absurd and added structure for the
detector to trip on — they are now heavily blurred; and persons were being clipped by the canvas
bottom, producing unusable GT boxes — placement is now clamped in-frame.

`overlap` is the layout knob; `realized_overlap` (mean per-person max box-IoU) is what actually
occurred and is the axis to plot against, since portrait-heavy source crops make the two diverge.

**Built:** 700 composites, 3,029 persons, mean 4.33 persons/scene, max 9.

## 5. Splits — the authors' boundary is never crossed

```
SH17 train (6,479) ─┬─ det_train 5,831   detector fitting
                    └─ det_val     648   checkpoint selection
SH17 val   (1,620) ─── HELD OUT          composite sources + test mAP
```

Every person in the evaluation set comes from an image the detector has never seen.

## 6. Known limitations, stated before the numbers

- **The headline comes from synthetic composites.** Rectangular seams are not photorealistic.
  Both arms consume identical detections, so seams cannot manufacture a margin between them — but
  they do make absolute recall pessimistic versus real footage.
- **35 distinct compliant persons.** 700 composites re-arrange them; the count of *scenes* is not
  the count of *people*. Confidence intervals resample composites, and the paired design means
  the A→B delta is unaffected by reuse, but absolute values are.
- **Portrait skew.** Single-person SH17 images lean toward portrait framing, so the composite
  population is not a random sample of workplace scenes.
- **Derived labels.** "Head with no helmet on the same person ⇒ non-compliant" is an assumption,
  and it is precisely the assumption the two arms disagree about.

## 7. Environment notes

- `uv` venv, Python 3.11.14, ultralytics 8.4.114, torch 2.13.0, MPS available. Torch-only
  process (no xgboost/lightgbm — that pairing segfaults on this Mac).
- Corpus prep: 13 GB → 1.8 GB in 376 s (8 workers). **macOS spawn-start requires a `__main__`
  guard** in any script using `ProcessPoolExecutor`, or every worker re-runs the script body and
  the pool dies with `BrokenProcessPool`. Cost this run one failed job.
- MobileSAM is reached through `ultralytics.SAM`, so no extra dependency. Verified on MPS that
  box-prompted masks for two *overlapping* boxes come back **disjoint** — the property ARM B
  depends on. Worth checking before building on it; if masks had overlapped, the mechanism would
  have had nothing to exploit.

---

## 8. The result: the hypothesis is refuted, and the mechanism is visible

The oracle ablation ran first, on all 700 composites (3,029 persons, 1,981 of them
non-compliant), because it is the decisive test and needs no detector — ground-truth boxes,
with MobileSAM still segmenting the real composite image.

| arm | nc-recall (strict) | helmet assoc. acc. | undetermined | compliant F1 |
|---|---|---|---|---|
| A0 box IoU @0.30 | 0.0227 | 0.0000 | **0.9848** | 0.0000 |
| A box containment @0.50 | **0.9329** | **0.8986** | 0.0409 | **0.9332** |
| B mask containment @0.50 | 0.7991 | 0.7181 | 0.1799 | 0.8561 |

**B − A = −0.1338**, 95% CI [−0.1521, −0.1157], P(Δ>0) = 0.000.

Two things settle it. First, ARM B is *worse*, not better, with detection removed entirely —
and `src/oracle.py` committed in advance to the reading that if ARM B shows no advantage under
perfect detections it cannot show one with a real detector. Second, the margin does **not** grow
with crowding: −0.019 / −0.147 / −0.145 / −0.122 across the 1, 2–3, 4–6, 7+ strata. That is the
directional claim, and it is flat.

A0 collapsing to 0.0227 recall with a 98.5% undetermined rate is the check that ARM A was an
honest baseline rather than a strawman: the original IoU rule really does assign almost nothing,
exactly as the arithmetic in §3 predicted.

## 9. Why masks lose — measured, not inferred

Before accepting the refutation, one objection had to be answered: a person's segmentation mask
is very nearly a *subset* of their bounding box, so `mask_containment <= box_containment` always.
The two arms were never being scored on the same scale, and a shared threshold of 0.50 is
systematically harsher on ARM B. "B loses at 0.50" could have meant "masks are worse" or "0.50 is
the wrong operating point for a compressed range" — opposite conclusions.

So `src/sweep.py` scores every ground-truth (PPE box, its known owner) pair under both rules:

| | mean | median | p05 | fraction < 0.50 | n |
|---|---|---|---|---|---|
| helmet → owner **box** | 0.954 | 1.000 | 0.920 | 0.041 | 1,114 |
| helmet → owner **mask** | 0.643 | 0.767 | **0.000** | **0.265** | 1,114 |
| head → owner **box** | 0.968 | 1.000 | 0.934 | 0.028 | 3,125 |
| head → owner **mask** | 0.709 | 0.823 | 0.000 | 0.206 | 3,125 |

That is the whole story in one table. **A helmet is not part of a person's silhouette.** SAM,
prompted with the person's box, segments the person — and a hard hat is an object worn *on top of*
them, so the mask frequently excludes it. For 26.5% of helmets the person's own mask covers less
than half the helmet box, and the 5th percentile is 0.000, meaning a substantial tail of helmets
falls entirely outside the mask of the person wearing it.

The bounding box has no such problem: it is a crude region, and crudeness is exactly what this
task wants, because the PPE item is spatially adjacent to the person rather than inside them.

## 10. Robustness: it is not a threshold artifact

Both arms were swept over t ∈ [0.05, 0.95] and compared at each arm's own best operating point
(best = highest non-compliance recall, ties broken on compliant F1 so a degenerate
"call-everyone-non-compliant" point cannot win):

| | best threshold | nc-recall | compliant F1 |
|---|---|---|---|
| ARM A (box) | 0.60 | **0.9334** | 0.9341 |
| ARM B (mask) | 0.05 | 0.8915 | 0.9216 |

**Best-vs-best B − A = −0.0419**, 95% CI [−0.0567, −0.0283], P(Δ>0) = 0.000. By crowding:
0.000 / −0.042 / −0.048 / −0.038 — flat again.

ARM A is essentially threshold-independent (0.932–0.933 across the whole sweep) because box
containment for a true pair is ~1.0. ARM B improves monotonically as the threshold falls and its
best point is t=0.05 — i.e. ARM B is at its strongest when the containment test is switched off
almost entirely, and it *still* loses. Note this comparison selects each arm's threshold on the
same data it is scored on, which is optimistic for both arms and therefore generous to ARM B; the
refutation survives that generosity.

## 10a. Stopping the detector run early, and why it was safe

Detector training was stopped at **epoch 12 of the configured 20**. The reasoning was fixed by
what the experiment needs, not by convenience:

- The headline result is the oracle ablation, which uses ground-truth boxes. No detector appears
  in it at all.
- `src/oracle.py` had committed in advance to treating the oracle as decisive in one direction:
  no advantage under perfect detections means no advantage with a real one.
- Both arms consume identical detections, so detector quality enters the A→B difference as a
  common term. A weaker detector compresses the delta; it cannot flip its sign.

The full-pipeline run confirmed all three: delta −0.0167 [−0.0244, −0.0088] versus the oracle's
−0.1338, same sign, same flat crowding profile, with the compression explained by the detector
matching only 2,219 of 3,029 persons.

The cost is stated rather than hidden: mAP50-95 = 0.432 is lower than the configured schedule
would reach, because `close_mosaic` was due to activate at epoch 13 and that phase usually adds a
few points. Detector quality is a reported secondary number in this project, and 12 epochs is
enough for it to play its only role — showing that the association conclusion survives imperfect
perception.

## 11. What the finding actually is

The project set out to test whether instance segmentation fixes person↔PPE association in crowded
scenes. It does not, and the reason is geometric rather than a matter of model quality:
segmentation answers "which pixels are this person", while PPE association needs "which region is
*associated with* this person" — and worn equipment sits outside the first region by construction.
Masks are the sharper tool and the task wanted the blunter one.

This is a negative result, reported as a finding rather than a failure. It is a more
useful one than a marginal win would have been, because it is diagnosed: the containment
distribution in §9 says precisely which assumption failed, and it generalises to any
"associate small object to its wearer/holder" problem, not just PPE.

**What would have made segmentation win**, and is the honest next step rather than a saved result:
dilating the person mask by roughly the scale of the worn item before testing containment, or
segmenting the *helmet* and testing mask-to-mask adjacency rather than containment-in-person. Both
change the mechanism under test, so neither was folded in after the fact.

## Candidate skill

**When two arms are scored by a quantity with different ranges, a shared threshold is not a fair
control — it is a hidden confound.** The shared-threshold rule was adopted for exactly the right
reason (tuning each arm separately confounds the mechanism with threshold search), but
`mask_containment <= box_containment` holds by construction, so the "fair" shared threshold was
systematically harsher on the arm under test. The resolution that keeps both properties is a
sweep: report the full curve and compare each arm at its own optimum. If the loser still loses
there, the conclusion is robust; if it wins, the single-threshold result was an artifact. Check
whether the two arms' score distributions are commensurable *before* deciding a shared threshold
is the neutral choice.

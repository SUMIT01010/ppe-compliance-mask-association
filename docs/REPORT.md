# Instance segmentation for person↔PPE association — report

Every number in this document was produced by running the code in this repository and is
reproduced in `outputs/experiment_oracle.json`, `outputs/threshold_sweep.json` and
`outputs/experiment.json`.

---

## 1. Problem, hypothesis, and decision rule

A workplace-safety monitor must answer a per-person question: *is this worker wearing head
protection?* Detection alone does not answer it. A frame containing one `person` box, one `head`
box and one `helmet` box is ambiguous the moment a second person enters: the helmet has to be
**assigned** to somebody, and the assignment rule is what decides the verdict.

The costly error is asymmetric. Scoring a non-compliant worker as compliant leaves someone
unprotected; the reverse merely triggers a check. The headline metric is therefore **per-person
non-compliance recall**, with "undetermined" counted as a **miss** — "we could not tell" protects
nobody.

**Hypothesis, fixed before any training run:**

> Mask-containment association yields higher per-person non-compliance recall than box-containment
> association, and the margin **increases** with the number of persons per frame.

**Decision rule, also fixed in advance:** the paired difference B − A in non-compliance recall,
with a 95% confidence interval from a bootstrap that resamples **composites** (not persons), plus
the per-stratum profile across crowding bands 1 / 2–3 / 4–6 / 7+. A flat profile refutes the
second limb even if the first holds.

## 2. The arms

All three arms consume **one** detection pass per image; only the association step differs.

Let $p$ index persons and $j$ index PPE boxes. Every arm is the same argmax rule under a different
region and score:

$$\text{owner}(j) \;=\; \arg\max_{p} \; s(j, p), \qquad \text{assign iff } \max_p s(j,p) \ge \tau$$

| Arm | Region | Score $s(j,p)$ | $\tau$ |
|---|---|---|---|
| A0 | person **box** | $\mathrm{IoU}(b_j, b_p)$ | 0.30 |
| A | person **box** | $\dfrac{\lvert b_j \cap b_p\rvert}{\lvert b_j \rvert}$ | 0.50 |
| B | person **mask** | $\dfrac{\lvert b_j \cap M_p\rvert}{\lvert b_j \rvert}$ | 0.50 |

with $b$ a box, $M_p$ the instance mask of person $p$, and $\lvert\cdot\rvert$ area in pixels.

The compliance rule, applied identically to every arm's assignment:

$$
v(p) = \begin{cases}
\text{non-compliant} & \text{head assigned to } p,\ \text{no helmet assigned to } p\\
\text{compliant} & \text{helmet assigned to } p\\
\text{undetermined} & \text{neither assigned}
\end{cases}
$$

That derivation is an **assumption**, not a dataset fact, and it is exactly the assumption the two
arms disagree about — so it is stated here rather than buried.

### Why A0 exists and why A is the real baseline

A0 is the rule the original scaffold specified: assign a PPE box to the person box it overlaps
most, above IoU ≥ 0.30. It is broken, and not interestingly. A helmet box is roughly 1–2% of a
person box's area, so

$$\mathrm{IoU}(b_{\text{helmet}}, b_{\text{person}}) \;\approx\; \frac{|b_\text{helmet}|}{|b_\text{person}|} \;\approx\; 0.015,$$

a factor of twenty below the threshold. It never fires. Measured: **0.0227** non-compliance recall
at a **98.5%** undetermined rate.

Had it been left as the baseline, ARM B would have "won" by roughly sixty points against an arm
that never assigned anything, and nothing would have errored. A0 is retained and reported because
its collapse is the evidence that ARM A was chosen to be fair rather than chosen to lose.

## 3. Ground truth, and the traps in generating it

SH17 annotates `person`, `head`, `helmet`, `safety-vest` as **independent** boxes and never records
which head belongs to which person. That absence is why the dataset was chosen — the association is
not pre-solved — and it cuts both ways: there is no association label to score against.

Deriving ground-truth links from box IoU would define ARM A correct by construction and the
experiment would measure nothing. Restricting to single-person images gives unambiguous links and
zero crowding, which is the variable under test.

**Adopted: synthetic composites.** Take held-out images containing exactly one `person` — every
PPE box in them provably belongs to that person — and composite $k$ of them into one scene,
carrying the links through exactly. Exact ground truth at any crowding level, and crowding becomes
a *swept* variable rather than whatever the corpus happens to contain.

Two generator choices would have decided the result in advance:

1. **Mask paste would have rigged it for ARM B.** Pasting SAM-segmented silhouettes onto a new
   background leaves a sharp figure/ground boundary that SAM can re-segment almost trivially at
   evaluation time — the generator would have drawn the answer into the image. Rectangular crops
   carry their own background, so segmentation must do real work. Less photorealistic, and
   conservative in the direction that matters.
2. **A fixed canvas would have confounded the independent variable.** The first implementation
   squeezed everyone into 1280 px, so 9-person scenes had persons at roughly half the size of
   2-person scenes. "More crowded" and "smaller objects" would have moved together and the
   directional claim would have been untestable. Fixed: person height is held constant and the
   canvas widens with the crowd.

Built: **700 composites, 3,029 persons** (1,981 non-compliant), mean 4.33 persons/scene, max 9.

### Splits

```
SH17 train (6,479) ─┬─ det_train 5,831   detector fitting
                    └─ det_val     648   checkpoint selection
SH17 val   (1,620) ─── HELD OUT          composite sources + test mAP
```

The authors' boundary is never crossed. Every person in the evaluation set comes from an image the
detector has never seen.

### A narrowing forced by the data

The brief's verdict was helmet **and** vest. Counting the held-out single-person pool: 712 images
with head-and-no-helmet, 35 with head-and-helmet, and **14** with a safety vest. Fourteen source
images cannot support a headline metric, so **compliance is head-protection only**. Vest remains a
detector class — it is nearly free and its mAP is reportable — but it is not in the verdict. The
count was made before any arm was scored.

## 4. Estimators

**Non-compliance recall (strict).** Over all ground-truth non-compliant persons, including those
the detector never found:

$$R = \frac{\#\{p : v_{\text{gt}}(p) = \text{NC} \ \wedge\ v_{\text{pred}}(p) = \text{NC}\}}{\#\{p : v_{\text{gt}}(p) = \text{NC}\}}$$

Undetermined counts as a miss. A *conditioned* variant restricted to detected persons is also
reported; it isolates the association step, and since both arms consume identical detections the
detector's contribution cancels from the A→B difference either way.

**Compliant F1** guards against a degenerate policy: an arm that calls everyone non-compliant
scores recall 1.0 and is useless.

**Paired bootstrap over composites.** Persons in a scene are correlated through the scene they
share, so resampling persons independently would understate variance. $B = 2000$ resamples of
composites with replacement; both arms see the same resampled scenes, so the detector's
contribution cancels within each replicate:

$$\widehat{\Delta} = R_B - R_A, \qquad \text{CI}_{95} = \left[q_{2.5}(\Delta^*),\ q_{97.5}(\Delta^*)\right]$$

## 5. The oracle ablation — the decisive test

The full pipeline feeds both arms the same detections, which controls for detector quality but
does not remove it: a missed helmet hurts both arms and compresses the measurable gap. Substituting
ground-truth boxes removes detection from the loop and asks the mechanism question in its purest
form. MobileSAM still segments the real composite image and has no knowledge of how it was built.

This was committed to in advance (`src/oracle.py` docstring) as decisive in one direction: *if ARM
B shows no advantage here, it cannot show one with a real detector either.*

**700 composites, 3,029 persons:**

| Arm | nc-recall | helmet assoc. acc. | undetermined | compliant F1 |
|---|---|---|---|---|
| A0 box IoU @0.30 | 0.0227 | 0.0000 | 0.9848 | 0.0000 |
| **A box containment @0.50** | **0.9329** | **0.8986** | 0.0409 | **0.9332** |
| B mask containment @0.50 | 0.7991 | 0.7181 | 0.1799 | 0.8561 |

$$\widehat{\Delta} = -0.1338, \quad \text{CI}_{95} = [-0.1521,\ -0.1157], \quad P(\Delta > 0) = 0.000$$

By crowding: **−0.019 / −0.147 / −0.145 / −0.122** for 1 / 2–3 / 4–6 / 7+ persons.

Both limbs of the hypothesis fail. ARM B is worse, and the margin does not grow with crowding.

## 6. Why — the containment distribution

A person's mask is very nearly a subset of their box, so $\text{mask containment} \le \text{box
containment}$ pointwise. Scoring every ground-truth (PPE box, **known** owner) pair under both
rules:

| | mean | median | p05 | fraction < 0.50 | n |
|---|---|---|---|---|---|
| helmet → owner **box** | 0.954 | 1.000 | 0.920 | 0.041 | 1,114 |
| helmet → owner **mask** | 0.643 | 0.767 | **0.000** | **0.265** | 1,114 |
| head → owner **box** | 0.968 | 1.000 | 0.934 | 0.028 | 3,125 |
| head → owner **mask** | 0.709 | 0.823 | 0.000 | 0.206 | 3,125 |

For **26.5%** of helmets the wearer's own mask covers less than half the helmet box; the 5th
percentile is **0.000**, so a real tail of helmets falls entirely outside the mask of the person
wearing them. ARM B's elevated undetermined rate (0.1799 vs 0.0409) is precisely this: the true
link exists and the mask test rejects it.

The reason is not a SAM failure. SAM is doing its job — it segments *the person*. A hard hat is an
object worn **on top of** a person and is not part of their silhouette. Segmentation answers
"which pixels are this person"; association needs "which region is associated with this person",
and worn equipment falls in the gap. The bounding box succeeds *because* it is crude: the PPE item
is spatially adjacent to the person rather than inside them, and a crude region captures adjacency.

## 7. Robustness — the threshold sweep

The shared threshold was adopted so the arms differ in exactly one thing, which is the right
instinct and contains a hidden confound: because mask containment is bounded above by box
containment, the two arms are not scored on the same scale, and a shared $\tau = 0.50$ is
systematically harsher on ARM B. "B loses at 0.50" could mean "masks are worse" or "0.50 is the
wrong operating point for a compressed range" — opposite conclusions.

Both arms were swept over $\tau \in [0.05, 0.95]$ and compared at each arm's own optimum (highest
non-compliance recall, ties broken on compliant F1 so a degenerate all-non-compliant point cannot
win):

| | best $\tau$ | nc-recall | compliant F1 |
|---|---|---|---|
| ARM A (box) | 0.60 | **0.9334** | 0.9341 |
| ARM B (mask) | 0.05 | 0.8915 | 0.9216 |

$$\widehat{\Delta}_{\text{best vs best}} = -0.0419, \quad \text{CI}_{95} = [-0.0567,\ -0.0283], \quad P(\Delta>0) = 0.000$$

By crowding: 0.000 / −0.042 / −0.048 / −0.038 — flat again.

ARM A is effectively threshold-independent (0.932–0.933 across the whole sweep) because box
containment for a true pair is ≈1.0. ARM B improves monotonically as $\tau$ falls and peaks at
$\tau = 0.05$ — at its strongest when the containment test is switched off almost entirely — and
still loses. This procedure selects each arm's threshold on the data it is scored on, which is
optimistic for both and therefore generous to ARM B. The refutation survives that generosity.

## 6a. The same comparison with a real detector

The oracle settles the mechanism; this run says what survives contact with imperfect perception.
YOLOv8n trained on SH17 (`SCALE=demo`, 640 px, 12 of 20 configured epochs), one detection pass per
image shared by every arm, same 700 composites.

**Detector quality**, held-out SH17 val (1,620 images, 5,412 instances): mAP50-95 **0.4321**,
mAP50 0.630. Per class: person 0.634, head 0.642, helmet 0.289, safety-vest 0.164. The small
classes are the hard ones, which is the expected profile for this corpus and the reason absolute
recall below is well under the oracle's.

| Arm | nc-recall strict | nc-recall conditioned | helmet assoc. | undetermined | compliant F1 |
|---|---|---|---|---|---|
| A0 box IoU | 0.0288 | 0.0391 | 0.0014 | 0.9776 | 0.0019 |
| **A box containment** | **0.6224** | **0.8463** | **0.8555** | 0.3371 | **0.6718** |
| B mask containment | 0.6058 | 0.8236 | 0.8289 | 0.3569 | 0.6554 |

$$\widehat{\Delta}_{\text{strict}} = -0.0167,\ \text{CI}_{95}=[-0.0244,\ -0.0088];\qquad
\widehat{\Delta}_{\text{cond}} = -0.0226,\ \text{CI}_{95}=[-0.0332,\ -0.0119]$$

both with $P(\Delta>0) = 0.000$. By crowding: −0.037 / −0.015 / −0.013 / −0.020 — flat.

The detector matched 2,219 of 3,029 ground-truth persons, so absolute recall falls from 0.93 to
0.62 and the A→B gap compresses from 13.4 points to 1.7. Both effects are expected and neither
changes the sign: detector misses are identical across arms by construction, so they add a common
loss that dilutes the difference without biasing it. This is precisely why the oracle ablation,
not this run, is the decisive test — and why the two agreeing matters.

**On the truncated schedule.** Training was stopped at epoch 12 of the configured 20. The
headline finding uses no detector, and the paired design means detector quality cannot flip the
sign of the delta — it can only compress it, which it did. The honest cost is that mAP50-95 =
0.432 understates what the configured schedule would produce (`close_mosaic` activates at epoch
13). Detector quality is reported here as context, and it is not the claim.

## 7a. The cost side

Latency was measured on the serving path (`app/main.py`), one composite scene of two persons,
after warm-up:

| Arm | per-frame latency |
|---|---|
| A — box containment | 37 ms |
| B — mask containment (MobileSAM pass) | 611 / 714 / 727 ms |

Roughly **17x** the cost. These absolute numbers were taken while detector training occupied the
same GPU, so the ratio is the reportable quantity, not the milliseconds. It matters for the
conclusion: the arm under test is not merely less accurate, it is substantially more expensive,
so there is no operating regime in this evaluation where it is the right choice.

## 8. Inference

**The hypothesis is refuted on both limbs**, with the mechanism identified rather than merely
observed:

1. Mask-containment association does not beat box-containment association. It is 13.4 points worse
   at the shared threshold and 4.2 points worse at each arm's own optimum, both with confidence
   intervals excluding zero and $P(\Delta>0) = 0.000$.
2. The margin does not grow with crowding. It is flat across all four strata under both threshold
   regimes.
3. The cause is geometric and measured: worn PPE lies outside the wearer's segmentation mask for a
   quarter of helmets.

**Assumptions this inference depends on**, stated explicitly:

- The derived compliance rule (head-without-helmet ⇒ non-compliant) is correct. It is the arms'
  point of disagreement, and it is an assumption.
- Composited scenes are representative of crowding as a *geometric* phenomenon. They are not
  photorealistic, but both arms see identical pixels and identical detections, so a compositing
  artifact cannot manufacture a difference *between* arms.
- MobileSAM prompted with the person box is representative of instance segmentation for this task.
  A model trained to segment person-plus-equipment would change the containment distribution and
  would deserve its own run.

**What this does not establish.** It does not show segmentation is useless for safety monitoring —
only that *mask containment of a PPE box within a person mask* is a worse association rule than box
containment. Two variants are untested and are the honest next step: dilating the person mask by
the scale of the worn item before testing containment, and segmenting the helmet to test
mask-to-mask adjacency rather than containment. Both change the mechanism, so neither was folded in
after seeing the result.

**Generalisation.** The failure is not specific to helmets. Any "associate a small worn or carried
object with its owner" problem — hi-vis vests, tools, badges, handheld devices — has the same
geometry: the object is adjacent to the person, not inside them.

## 9. Reproduction

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt
uv run python scripts/01_prepare_data.py
uv run python scripts/03_build_composites.py 700
uv run python scripts/04b_oracle_experiment.py     # the decisive result, no detector needed
uv run python scripts/04c_threshold_sweep.py       # robustness + the diagnostic table
SCALE=demo uv run python scripts/02_train_detector.py
uv run python scripts/04_run_experiment.py
```

Oracle and sweep each take roughly 12 minutes on Apple Silicon (MPS); detector training at
`SCALE=demo` (YOLOv8n, 640 px, 20 epochs) takes a few hours on the same hardware and is not
required to reproduce the headline finding.

## 10. Research anchors

- **Kirillov et al., 2023 — Segment Anything.** The promptable-segmentation formulation ARM B
  depends on; box-prompted masks for overlapping persons come back disjoint, which is the property
  the hypothesis was built on and which does hold.
- **Zhang et al., 2023 — MobileSAM.** The distilled variant used, chosen so the segmentation pass
  is affordable in a per-frame safety monitor; the latency cost is reported rather than assumed.
- **Ahmad et al., 2024 — SH17 dataset.** The corpus, chosen specifically because it annotates
  person and PPE as *independent* boxes and records no association, so the question is not
  pre-solved by the labels.
- **Redmon et al. / Jocher et al. — YOLO family.** The shared detector; used identically by every
  arm so that detection quality cannot enter the comparison.

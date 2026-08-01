# Does instance segmentation fix person↔PPE association?

A controlled test of whether **instance masks** beat **bounding boxes** at deciding *whose helmet
is whose* in crowded workplace-safety footage — and a measured explanation of why they do not.

**Headline result — the hypothesis is refuted.** With detection removed from the loop entirely
(ground-truth boxes, real segmentation), mask-containment association scores **13.4 points lower**
non-compliance recall than box-containment: 0.7991 vs 0.9329, 95% CI on the difference
**[−0.152, −0.116]**, P(Δ>0) = 0.000 over 700 crowded scenes and 3,029 persons. Giving each arm
its own best threshold narrows it to **−4.2 points** [−0.057, −0.028] and does not flip it. The
margin does **not** grow with crowding, which was the directional claim.

**Root cause, measured:** for **26.5%** of helmets, the wearer's own segmentation mask covers less
than half the helmet box — 5th percentile 0.000. A helmet is not part of a person's silhouette.

**And it is not free.** Measured on the serving path, the segmentation arm costs ~610–730 ms per
frame against ~37 ms for the box arm — roughly **17× the latency for worse accuracy**.

---

## The question

In real workplace-safety monitoring the dangerous error is a missed non-compliant worker. When two
workers overlap in frame, a bounding-box overlap test cannot tell whose helmet is whose, so a
bare-headed worker standing behind a helmeted one can be scored compliant. Instance segmentation
looks like the obvious fix: two overlapping person *boxes* both contain the same helmet, but their
*masks* are disjoint.

> **Hypothesis (fixed before any training run):** mask-containment association yields higher
> per-person non-compliance recall than box-containment association, and the margin **increases**
> with the number of persons per frame.

Both limbs are falsifiable, and both are false.

## Result

Oracle ablation — ground-truth person/head/helmet boxes, MobileSAM segmenting the real composite
image. Detection quality cannot influence this comparison because there is no detector in it.
700 composites, 3,029 persons, 1,981 non-compliant.

| Arm | Non-compliance recall | Helmet assoc. accuracy | Undetermined | Compliant F1 |
|---|---|---|---|---|
| A0 — box IoU @ 0.30 | 0.0227 | 0.0000 | 0.9848 | 0.0000 |
| **A — box containment @ 0.50** | **0.9329** | **0.8986** | 0.0409 | **0.9332** |
| B — mask containment @ 0.50 | 0.7991 | 0.7181 | 0.1799 | 0.8561 |

**B − A = −0.1338**, 95% CI [−0.1521, −0.1157], P(Δ>0) = 0.000.

By crowding stratum (persons per frame) — the directional claim was that this grows:

| 1 | 2–3 | 4–6 | 7+ |
|---|---|---|---|
| −0.019 | −0.147 | −0.145 | −0.122 |

It is flat and negative. A0's collapse to 0.0227 recall at a 98.5% undetermined rate is the
evidence that ARM A is an honest baseline rather than a strawman picked to lose — see
[Baseline](#the-baseline-had-to-be-fixed-first).

### The same experiment with a real detector

Full pipeline, YOLOv8n trained on SH17, same 700 composites:

| Arm | nc-recall (strict) | nc-recall (detected persons only) | assoc. acc. | undetermined |
|---|---|---|---|---|
| A0 — box IoU | 0.0288 | 0.0391 | 0.0014 | 0.9776 |
| **A — box containment** | **0.6224** | **0.8463** | **0.8555** | 0.3371 |
| B — mask containment | 0.6058 | 0.8236 | 0.8289 | 0.3569 |

**B − A = −0.0167**, CI [−0.0244, −0.0088], P(Δ>0) = 0.000 — and −0.0226 [−0.0332, −0.0119]
restricted to persons the detector actually found. Same sign, same flat crowding profile
(−0.037 / −0.015 / −0.013 / −0.020).

Absolute recall drops from 0.93 to 0.62 because the detector finds only 2,219 of 3,029 persons,
and the A→B gap compresses from 13.4 to 1.7 points because detector misses hurt both arms
identically and dominate the residual. That is the expected direction and the reason the oracle
ablation is the decisive test rather than this one.

**Detector quality** (held-out SH17 val, 1,620 images / 5,412 instances): mAP50-95 **0.432**,
mAP50 0.630 — person 0.634, head 0.642, helmet 0.289, safety-vest 0.164. Reported so association
results are not confused with detection results. Trained 12 epochs of the configured 20 at
`SCALE=demo`; training was stopped early because the finding does not depend on it — see
[Limitations](#honest-limitations).

## Why masks lose

A person's mask is very nearly a subset of their box, so `mask_containment ≤ box_containment`
always. Scoring every ground-truth (PPE box, known owner) pair under both rules:

| | mean | median | p05 | **fraction < 0.50** | n |
|---|---|---|---|---|---|
| helmet → owner **box** | 0.954 | 1.000 | 0.920 | 0.041 | 1,114 |
| helmet → owner **mask** | 0.643 | 0.767 | **0.000** | **0.265** | 1,114 |
| head → owner **box** | 0.968 | 1.000 | 0.934 | 0.028 | 3,125 |
| head → owner **mask** | 0.709 | 0.823 | 0.000 | 0.206 | 3,125 |

SAM, prompted with the person's box, segments *the person*. A hard hat is worn **on top of** them,
so the mask routinely excludes it — for a quarter of helmets by more than half their area, and for
a tail of cases entirely. The bounding box has no such problem precisely because it is crude, and
crudeness is what this task wants: the PPE item is spatially *adjacent* to the person, not inside
them.

Segmentation answers "which pixels are this person". Association needs "which region is associated
with this person". Those are different questions, and worn equipment falls in the gap.

## It is not a threshold artifact

The shared threshold was chosen so the arms differ in exactly one thing. But since mask
containment is bounded above by box containment, a shared threshold is systematically harsher on
ARM B — so "B loses at 0.50" could have meant either "masks are worse" or "0.50 is the wrong
operating point for a compressed range". Both arms were therefore swept over t ∈ [0.05, 0.95] and
compared at each arm's **own best** point:

| | best t | nc-recall | compliant F1 |
|---|---|---|---|
| ARM A (box) | 0.60 | **0.9334** | 0.9341 |
| ARM B (mask) | 0.05 | 0.8915 | 0.9216 |

**Best-vs-best B − A = −0.0419**, 95% CI [−0.0567, −0.0283], P(Δ>0) = 0.000. Still flat across
crowding (0.000 / −0.042 / −0.048 / −0.038).

ARM A is essentially threshold-independent (0.932–0.933 across the entire sweep). ARM B is at its
strongest when the containment test is switched off almost entirely (t = 0.05) and still loses.
This selects each arm's threshold on the data it is scored on, which is optimistic for both and
therefore generous to ARM B — the refutation survives that generosity.

## Design

```
composite scene ──► detections (shared by every arm)
                          ├── ARM A0  box IoU @0.30          → per-person verdict
                          ├── ARM A   box containment @0.50  → per-person verdict
                          └── ARM B   MobileSAM masks → mask containment @0.50 → verdict
                                              ↓
                             crowding-stratified paired comparison
```

Both arms consume the **same** detections from one pass. Only the association step differs, so no
result can be attributed to detector noise.

### Ground truth: synthetic composites

SH17 annotates `person`, `head`, `helmet` as independent boxes and never records which head
belongs to which person. That absence is *why* the dataset was chosen — the association is not
pre-solved — and it means there is no association label to score against either. Scoring against a
box-IoU-derived "truth" would define ARM A correct by construction.

So: take held-out images containing exactly **one** person (every PPE box provably belongs to
them) and composite *k* of them into one crowded scene, carrying the known links through. Exact
ground truth at any crowding level.

Two generator choices that would have decided the result before it was measured, and were avoided:

- **Rectangular crops, not SAM-segmented silhouettes.** A pasted silhouette leaves a sharp
  figure/ground boundary that SAM can re-segment trivially, inflating the arm under test for a
  reason unrelated to the mechanism. Rectangular crops carry their own background, so segmentation
  has to do real work.
- **The canvas widens with the crowd.** A fixed canvas shrinks persons as *k* rises, making object
  scale co-vary with crowding — which would have made the central claim untestable.

### The baseline had to be fixed first

The original scaffold defined ARM A as box IoU ≥ 0.30. A helmet box is ~1–2% of a person box's
area, so helmet↔person IoU is ~0.01–0.02 and **never** clears 0.30. That baseline assigns nothing,
marks every person undetermined, and scores 0 — ARM B would have "won" by ~60 points against an
arm that was never in the race.

The fair baseline is the rule practitioners actually use for small-object-to-container assignment:
**containment**, with argmax over persons. The broken IoU rule is retained and reported as **A0**,
because its collapse is the evidence that ARM A was chosen to be fair rather than chosen to lose.

### Splits

```
SH17 train (6,479) ─┬─ det_train 5,831   detector fitting
                    └─ det_val     648   checkpoint selection
SH17 val   (1,620) ─── HELD OUT          composite sources + test mAP
```

Every person in the evaluation set comes from an image the detector has never seen.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt
```

```bash
uv run python scripts/01_prepare_data.py
```

```bash
uv run python scripts/03_build_composites.py 700
```

The decisive experiment needs no detector — ground-truth boxes, real segmentation:

```bash
uv run python scripts/04b_oracle_experiment.py
```

```bash
uv run python scripts/04c_threshold_sweep.py
```

Train the shared detector and run the full pipeline:

```bash
SCALE=demo uv run python scripts/02_train_detector.py
```

```bash
uv run python scripts/04_run_experiment.py
```

Serve it:

```bash
uv run uvicorn app.main:app --port 8000
```

```bash
curl -s -F 'file=@scene.jpg' 'localhost:8000/analyze?arm=mask'
```

`arm=box` runs the baseline against the same detections, so the deployed service can be A/B'd
against the arm this project actually validated rather than the one it set out to.

Measured warm on this machine: `arm=mask` 611/714/727 ms, `arm=box` 37 ms. Absolute figures were
taken while detector training occupied the same GPU, so treat the **ratio** (~17×) as the finding
rather than the absolute values.

## Layout

| Path | Contents | Ownership |
|---|---|---|
| `src/association.py` | The three arms: IoU, box containment, mask containment | core |
| `src/composites.py` | Synthetic crowded-scene generator with exact person↔PPE links | core |
| `src/sweep.py` | Threshold sweep + containment diagnostics | core |
| `src/oracle.py` | Oracle-detector ablation | core |
| `src/metrics.py` | Recall definitions, crowding strata, composite-level paired bootstrap | core |
| `src/perception.py` | YOLOv8 + MobileSAM wrapper (one detection pass, shared by all arms) | scaffolding |
| `src/prepare.py`, `src/sh17.py` | Dataset prep, class map, split protocol | scaffolding |
| `app/` | FastAPI service with label-free drift monitoring + Dockerfile | scaffolding |
| `outputs/*.json` | All computed results | output |

## Honest limitations

- **The evaluation set is synthetic.** Rectangular paste seams are not photorealistic. Both arms
  consume identical detections, so seams cannot manufacture a margin *between* arms — but absolute
  recall is pessimistic versus real footage.
- **35 distinct compliant source persons.** 700 composites re-arrange them. The paired design
  means the A→B delta is unaffected by reuse and confidence intervals resample composites, but
  absolute values rest on a narrow base.
- **Compliance is head-protection only.** The brief called for helmet *and* vest; counting the
  held-out pool found only 14 single-person images containing a safety vest, against 35 compliant
  and 712 non-compliant helmet sources. Vest stays a detector class (its mAP is reported) but is
  not in the verdict. A narrowing forced by the data, made before any result was seen.
- **Derived labels.** "Head assigned to a person with no helmet assigned to that same person ⇒
  non-compliant" is an assumption — and it is precisely the assumption the two arms disagree about.
- **Portrait skew.** Single-person SH17 images lean toward portrait framing, so the composite
  population is not a random sample of workplace scenes.
- **No hand-labelled real-image check.** The composite arm is the primary result and is complete;
  labelling ~200 genuinely crowded real SH17 images for person↔PPE links is the external-validity
  step and has not been done. Until it exists the claim is "holds on controlled synthetic
  crowding", not "holds in the field".
- **One segmentation model.** MobileSAM prompted with the person box. A model that segments
  person-plus-worn-equipment would change the containment distribution and is untested here.
- **The detector ran 12 of 20 configured epochs.** Training was stopped early and deliberately.
  The headline finding comes from the oracle ablation, which uses ground-truth boxes and no
  detector at all, and both arms consume identical detections, so a weaker detector compresses
  absolute recall equally and leaves the paired delta intact. The cost is that the reported
  mAP50-95 (0.432) is lower than the configured schedule would reach — `close_mosaic` was due to
  activate at epoch 13 and typically adds a few points. Detector quality is a reported secondary
  number here, not the result.

## What would make segmentation win

Stated as future work rather than folded in after the fact, because each changes the mechanism
under test: dilate the person mask by roughly the scale of the worn item before testing
containment, or segment the *helmet* and test mask-to-mask adjacency instead of
containment-within-person. The diagnostic table above predicts both would help; neither is
evidenced here.

## Further reading

- **[`docs/p7_ppe_association_book.pdf`](docs/p7_ppe_association_book.pdf)** — the technical book
  (17 pages): full derivations, model deep-dives, every design decision with the measurement
  behind it. Regenerate with `uv run python scripts/05_build_book.py`; every number in it is read
  from `outputs/*.json` at build time, so it cannot drift from the results.
- [`docs/REPORT.md`](docs/REPORT.md) — full write-up with the estimators and every design decision
- [`docs/TECHNICAL_JOURNEY.md`](docs/TECHNICAL_JOURNEY.md) — build order, the three traps avoided,
  and the bug that would have produced a 60-point fake win
- [`docs/EVAL_DESIGN.md`](docs/EVAL_DESIGN.md) — why synthetic composites, and the options rejected
- [`docs/DATASET_DECISION.md`](docs/DATASET_DECISION.md) — dataset licensing and selection

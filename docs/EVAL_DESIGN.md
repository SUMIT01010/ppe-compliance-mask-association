# Evaluation design — and the ground-truth problem

## The problem

SH17 annotates `person`, `head`, `helmet`, `safety-vest` as **independent boxes**. It does **not**
annotate which head belongs to which person.

That is exactly why it was chosen — the association is not pre-solved. But it cuts both ways:
**there is no ground-truth association to score against either.** Both arms produce person↔PPE
assignments and nothing in the dataset says which assignment is correct.

Scoring the arms against a rule derived from the boxes themselves (e.g. "the GT link is whichever
person box has highest IoU with the head") would define ARM A to be correct by construction. The
experiment would measure nothing.

## Options considered

| Option | Ground truth | Verdict |
|---|---|---|
| Score against box-IoU-derived links | Circular — defines ARM A as correct | **Reject** |
| Evaluate only on single-person images | Unambiguous, but contains zero crowding | **Reject** — crowding is the claim |
| Hand-label a crowded subset with person↔PPE links | Real, but slow and small | Keep as external-validity check |
| **Synthetic composites from single-person images** | **Exact, by construction** | **Primary** |

## Recommended: synthetic composites as the primary test, hand-labelled real subset as the check

**Primary — controlled experiment.** Take SH17 images containing exactly one `person`. In those,
every PPE box provably belongs to that person, so the link is known for free. Composite *k* such
images into a single crowded scene with controlled overlap, carrying the known links through.
This yields:

- Exact ground-truth association at any crowding level, with no labelling cost
- A **directly swept independent variable** — overlap fraction between person boxes — instead of
  relying on whatever crowding distribution the corpus happens to contain
- The ability to isolate the mechanism: if ARM B's margin over ARM A does not grow with overlap
  on data where the answer is known exactly, the hypothesis is refuted cleanly

**Secondary — external validity.** Hand-label person↔PPE links on ~200 genuinely crowded real
SH17 images. Report both. If the composite result holds and the real-image result does not, the
honest conclusion is that the effect is a compositing artifact — and that is a finding worth
reporting, not a failure to hide.

## Consequences to state in the write-up

- Composites are not photorealistic. Lighting and scale discontinuities at paste boundaries could
  affect the *detector*, which is why **both arms share the same detections** — any compositing
  artifact hits both arms identically and cannot manufacture a margin between them.
- The headline number comes from synthetic data. That must be said plainly in the abstract of the
  report, with the real-image subset as the corroborating evidence, not buried.
- Single-person source images are not a random sample of SH17 — they skew toward portrait-style
  framing. Note it as a limitation on generalisation.

## Decision status

**LOCKED 2026-08-01**, on Sumit's instruction to proceed and "assume whatever is necessary".
Implemented in `src/composites.py` + `src/evaluate.py`. Three things changed between this
document and the built version; all three are recorded in `TECHNICAL_JOURNEY.md` §3–§4:

1. **Rectangular paste, not mask paste.** A SAM-segmented silhouette pasted on a new background
   is trivially re-segmentable, which would inflate ARM B for a reason unrelated to the
   mechanism. Rectangular crops carry their own background and make ARM B work for it.
2. **Canvas width grows with the crowd.** A fixed canvas shrinks persons as *k* rises, which
   would make object scale co-vary with crowding and render the central claim untestable.
3. **The baseline changed** from box-IoU@0.30 to box-*containment*@0.50, because helmet↔person
   IoU is ~0.01 and never clears 0.30 — the original ARM A would have scored zero by
   construction. The IoU rule survives as ARM A0 and is reported.

**Deferred: the hand-labelled real-image check.** The composite arm is the primary result and is
complete. Hand-labelling ~200 crowded real SH17 images for person↔PPE links is the external
validity check and has **not** been done — it is the honest next step, and until it exists the
claim is "holds on controlled synthetic crowding", not "holds in the field".

# Dataset decision + licensing check

Required by `PROJECT_METHODOLOGY.md` §1 before any download. Checked 2026-08-01.

## Candidates

| | **SH17** | **Roboflow Construction Site Safety** |
|---|---|---|
| Images | 8,099 (75,994 instances, ~9.38/image) | 2,801 |
| Classes | 17: Person, Head, Face, Glasses, Face-mask-medical, Face-guard, Ear, Earmuffs, Hands, Gloves, Foot, Shoes, **Safety-vest**, Tools, **Helmet**, Medical-suit, Safety-suit | 10: Hardhat, Mask, **NO-Hardhat**, **NO-Mask**, **NO-Safety Vest**, Person, Safety Cone, Safety Vest, machinery, vehicle |
| Negative classes | **None** — positive annotations only | **Yes** — explicit NO-* classes |
| License | **CC BY-NC-SA 4.0** (non-commercial, share-alike) | **CC BY 4.0** (attribution only) |
| Source | Scraped from Pexels; educational/research use stated | Roboflow Universe |
| Small objects | 39,764 annotations < 1% image area | not stated |

## The trade-off — cleaner license, weaker experiment

The Roboflow set has the friendlier licence, and it is the one to reach for by reflex. It is also
the one that **destroys the hypothesis**.

Its `NO-Hardhat` / `NO-Safety Vest` boxes already encode the per-person compliance verdict. If a
label says "this person lacks a hardhat", association has been done by the annotator, and there
is nothing left for mask-containment to disambiguate. Both arms would score near-identically,
and the result would be an artifact of the labelling scheme rather than a finding.

SH17 annotates `Person`, `Head`, `Helmet` and `Safety-vest` as **separate objects with no verdict
attached**. A per-person compliance call can only be produced by deciding *whose* head and *whose*
helmet each box is — which is precisely the mechanism under test. Its ~9.38 instances per image
also supplies the crowded frames the stratified ablation needs; the Roboflow set is sparser.

## Decision: SH17

**Licence obligations, and how they are met:**
- **BY** — attribute the SH17 authors (Ahmad, M. et al., *SH17: A Dataset for Human Safety and
  Personal Protective Equipment Detection in Manufacturing Industry*, arXiv:2407.04590) in the
  repo README and any write-up.
- **NC** — portfolio/research use only. No commercial deployment, no paid product built on it.
  Not a constraint for this project's purpose.
- **SA** — applies to adaptations of the dataset. Mitigated by not redistributing images or
  derived annotation files at all: `.gitignore` already excludes `datasets/` and `data/`, so the
  repo ships code and metrics, never the corpus.

**Consequence to state plainly in the write-up:** compliance labels are *derived*, not given. The
derivation rule (a `Head` associated to a person with no `Helmet` associated to that same person
⇒ non-compliant for head protection) is itself an assumption, and it is the assumption the
box-IoU baseline and the mask-containment arm disagree about. That is the experiment.

## Rejected

**Roboflow Construction Site Safety** — reject for the primary experiment. Worth keeping as an
optional external validity check: run the finished detector on it and report whether conclusions
transfer to a differently-annotated corpus.

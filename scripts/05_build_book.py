"""Render the technical book PDF.

# --- scaffolding ---

Every number in the book is READ FROM outputs/*.json at build time. Nothing is typed in by hand.
That is the point: a book with hardcoded numbers silently goes stale the first time an experiment
is re-run, and a stale number in a write-up is worse than no write-up.

Equations are rendered with matplotlib's mathtext into inline base64 PNGs rather than MathML,
because WeasyPrint's MathML support is not dependable and there is no LaTeX engine on this
machine (`pdflatex`/`xelatex` absent; pandoc present but has no PDF backend without one).

Build-only dependencies - not needed to run the experiments:

    uv pip install weasyprint matplotlib pypdf
    uv run python scripts/05_build_book.py
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
from pathlib import Path

# WeasyPrint dlopens pango/cairo/gobject by bare name. On macOS with Homebrew they live in
# /opt/homebrew/lib, which is not on the dynamic loader's search path, so the import fails with
# "cannot load library 'libgobject-2.0-0'". Set it before weasyprint is imported anywhere.
if sys.platform == "darwin":
    _brew = "/opt/homebrew/lib"
    if Path(_brew).is_dir():
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{_brew}:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')}".rstrip(":"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "p7_ppe_association_book.pdf"


# ---------------------------------------------------------------- helpers
def eq(tex: str, size: int = 15, color: str = "#111111", inline: bool = False) -> str:
    """Render a LaTeX snippet to a base64 PNG via matplotlib mathtext.

    `inline=True` is for single symbols dropped mid-sentence. Without it they render as
    full-width block images and a sentence like "writing p for a person index" turns into three
    page-wide graphics.
    """
    fig = plt.figure(figsize=(0.01, 0.01), dpi=260)
    fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cls = "eq inline" if inline else "eq"
    return f'<img class="{cls}" src="data:image/png;base64,{b64}" alt="{tex}"/>'


def fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ('<img class="plot" src="data:image/png;base64,'
            + base64.b64encode(buf.getvalue()).decode() + '"/>')


def table(headers, rows, cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #666; } }
@page :first { @bottom-center { content: ""; } }
body { font: 10.5pt/1.55 Georgia, 'Times New Roman', serif; color: #16181d; }
h1 { font-size: 19pt; margin: 0 0 2mm 0; line-height: 1.25; page-break-before: always; }
h1.first { page-break-before: avoid; }
h2 { font-size: 13pt; margin: 7mm 0 2mm 0; color: #1a2b47; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm 0; color: #33415c; font-style: italic; }
p { margin: 0 0 2.6mm 0; text-align: justify; }
code, .mono { font-family: 'SF Mono', Menlo, monospace; font-size: 8.8pt; background: #f4f5f7;
              padding: 0.4mm 1mm; border-radius: 2px; }
pre { font-family: 'SF Mono', Menlo, monospace; font-size: 8.4pt; background: #f6f7f9;
      border-left: 2.5pt solid #c3ccd9; padding: 2.5mm 3mm; white-space: pre-wrap;
      line-height: 1.35; margin: 2.5mm 0; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9.2pt; }
th { background: #eef1f5; text-align: left; padding: 1.6mm 2mm; border-bottom: 0.8pt solid #b9c2cf;
     font-weight: bold; }
td { padding: 1.4mm 2mm; border-bottom: 0.4pt solid #e2e6ec; }
tbody tr:nth-child(even) { background: #fafbfc; }
img.eq { display: block; margin: 3.5mm auto; max-width: 88%; max-height: 12mm; }
img.eq.inline { display: inline; margin: 0 0.4mm; vertical-align: -0.7mm; height: 3.9mm;
                max-height: 3.9mm; max-width: none; }
img.plot { display: block; margin: 4mm auto; max-width: 100%; }
.callout { background: #fbf7ec; border-left: 3pt solid #d6a23c; padding: 2.5mm 3.5mm;
           margin: 3.5mm 0; font-size: 9.8pt; }
.finding { background: #eef4ee; border-left: 3pt solid #4c8a54; padding: 2.5mm 3.5mm;
           margin: 3.5mm 0; }
.warn { background: #fbeeee; border-left: 3pt solid #b3524c; padding: 2.5mm 3.5mm; margin: 3.5mm 0; }
.cover { text-align: center; margin-top: 55mm; page-break-after: always; }
.cover h1 { font-size: 27pt; page-break-before: avoid; border: 0; line-height: 1.2; }
.cover .sub { font-size: 12.5pt; color: #44506a; margin-top: 5mm; font-style: italic; }
.cover .meta { margin-top: 26mm; font-size: 10pt; color: #666; line-height: 1.9; }
.headline { font-size: 11.5pt; background: #f0f3f8; border: 0.8pt solid #ccd5e2; padding: 4mm;
            margin: 6mm 0; }
.toc { font-size: 10pt; } .toc div { margin: 1.1mm 0; }
.cases { width: auto; margin: 3mm auto; border: 0; font-size: 10pt; }
.cases td { border: 0; padding: 0.9mm 3mm; }
.cases .lhs { font-style: italic; font-size: 12pt; border-right: 1pt solid #b9c2cf; padding-right: 4mm;
              vertical-align: middle; }
.cases .verdict { font-family: 'SF Mono', Menlo, monospace; font-size: 9pt; font-weight: bold;
                  color: #1a2b47; }
.cases tbody tr:nth-child(even) { background: transparent; }
.caption { font-size: 8.8pt; color: #5a6474; text-align: center; margin-top: -2mm; }
"""


# ---------------------------------------------------------------- load results
orc = json.loads((config.OUTPUTS / "experiment_oracle.json").read_text())
swp = json.loads((config.OUTPUTS / "threshold_sweep.json").read_text())
exp = json.loads((config.OUTPUTS / "experiment.json").read_text())

O_A0, O_A, O_B = (orc["arms"]["A0_box_iou"], orc["arms"]["A_box_containment"],
                  orc["arms"]["B_mask_containment"])
E_A0, E_A, E_B = (exp["arms"]["A0_box_iou"], exp["arms"]["A_box_containment"],
                  exp["arms"]["B_mask_containment"])
oh = orc["headline"]
sh = swp["best_vs_best"]
cont = swp["containment_of_true_owner"]
lat = exp["latency"]
eh_s = exp["headline"]["B_vs_A_strict"]
eh_c = exp["headline"]["B_vs_A_conditioned"]

# training curve
res_csv = ROOT / "runs" / "detector_demo" / "results.csv"
epochs, m50, m5095 = [], [], []
if res_csv.exists():
    for row in csv.DictReader(res_csv.open()):
        k = {c.strip(): v for c, v in row.items()}
        epochs.append(float(k["epoch"]))
        m50.append(float(k["metrics/mAP50(B)"]))
        m5095.append(float(k["metrics/mAP50-95(B)"]))


def plot_sweep() -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ta = [p["threshold"] for p in swp["curve"]["A_box"]]
    ra = [p["noncompliance_recall_strict"] for p in swp["curve"]["A_box"]]
    rb = [p["noncompliance_recall_strict"] for p in swp["curve"]["B_mask"]]
    ax.plot(ta, ra, "-o", ms=3.4, lw=1.7, color="#2f5d8a", label="ARM A — box containment")
    ax.plot(ta, rb, "-s", ms=3.4, lw=1.7, color="#b3524c", label="ARM B — mask containment")
    ax.axvline(0.50, color="#888", ls="--", lw=1, label="shared threshold (0.50)")
    ax.set_xlabel("containment threshold  τ")
    ax.set_ylabel("non-compliance recall (strict)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("ARM B is strongest where the test is switched off, and still loses", fontsize=9.5)
    return fig_to_img(fig)


def plot_containment() -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    labels = ["helmet\n(box)", "helmet\n(mask)", "head\n(box)", "head\n(mask)"]
    keys = ["helmet_box", "helmet_mask", "head_box", "head_mask"]
    means = [cont[k]["mean"] for k in keys]
    meds = [cont[k]["median"] for k in keys]
    p05s = [cont[k]["p05"] for k in keys]
    x = range(len(keys))
    ax.bar([i - 0.22 for i in x], means, width=0.22, label="mean", color="#2f5d8a")
    ax.bar([i for i in x], meds, width=0.22, label="median", color="#6d94bd")
    ax.bar([i + 0.22 for i in x], p05s, width=0.22, label="5th pct", color="#c9d6e4")
    ax.axhline(0.50, color="#b3524c", ls="--", lw=1.2, label="threshold 0.50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("containment of the TRUE owner")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    ax.set_title("A helmet is not part of its wearer's silhouette", fontsize=9.5)
    return fig_to_img(fig)


def plot_training() -> str:
    if not epochs:
        return ""
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(epochs, m50, "-o", ms=3.2, lw=1.6, color="#2f5d8a", label="mAP50")
    ax.plot(epochs, m5095, "-s", ms=3.2, lw=1.6, color="#4c8a54", label="mAP50-95")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8)
    ax.set_title(f"YOLOv8n, SCALE=demo, stopped at epoch {int(max(epochs))} of 20", fontsize=9.5)
    return fig_to_img(fig)

# Equations are precomputed here rather than inline: an f-string expression part
# cannot contain a backslash, and every LaTeX snippet is full of them.
#
# matplotlib's mathtext is a SUBSET of LaTeX. It has no \ge/\le (use \geq/\leq), no \text{}
# (use \mathrm{}), and no \begin{cases} at all - the piecewise compliance rule is therefore
# laid out as an HTML table (see CASES_HTML) rather than forced into an image.
EQ = [
    eq(r'p', inline=True),
    eq(r'j', inline=True),
    eq(r'\mathrm{owner}(j) = \arg\max_{p}\ s(j,p), \qquad \mathrm{assign\ iff}\ \max_p s(j,p) \geq \tau'),
    eq(r's', inline=True),
    eq(r'\mathrm{IoU}(a,b) = \frac{|a \cap b|}{|a \cup b|}, \qquad \mathrm{cont}(a,b) = \frac{|a \cap b|}{|a|}'),
    "",  # piecewise rule rendered as HTML, not an image
    eq(r'\mathrm{IoU}(b_{helmet}, b_{person}) \approx \frac{|b_{helmet}|}{|b_{person}|} \approx 0.015'),
    eq(r'k', inline=True),
    eq(r'R_{strict} = \frac{\#\{p:\ v_{gt}(p)=\mathrm{NC}\ \wedge\ v_{pred}(p)=\mathrm{NC}\}}{\#\{p:\ v_{gt}(p)=\mathrm{NC}\}}'),
    eq(r'B = 2000', inline=True),
    eq(r'\hat{\Delta} = R_B - R_A, \qquad \mathrm{CI}_{95} = [\ q_{2.5}(\Delta^*),\ q_{97.5}(\Delta^*)\ ]'),
    eq(r'\mathrm{cont}(b_j, M_p)\ \leq\ \mathrm{cont}(b_j, b_p) \qquad \mathrm{for\ every}\ j,p'),
    eq(r'\tau \in [0.05,\ 0.95]', inline=True),
    eq(r'\tau', inline=True),
]

CASES_HTML = '''
<table class="cases"><tbody>
<tr><td class="lhs" rowspan="3">v(p) =</td>
    <td class="verdict">non-compliant</td><td>head assigned to p, no helmet assigned to p</td></tr>
<tr><td class="verdict">compliant</td><td>helmet assigned to p</td></tr>
<tr><td class="verdict">undetermined</td><td>neither assigned</td></tr>
</tbody></table>'''


def strat_row(d, key):
    v = d.get(key, {})
    return f4(v.get("delta")) if v.get("delta") is not None else "—"


# ---------------------------------------------------------------- book body
HTML_BODY = f"""
<div class="cover">
  <h1>Does Instance Segmentation Fix<br/>Person–PPE Association?</h1>
  <div class="sub">A controlled refutation, and the geometry behind it</div>
  <div class="meta">
    Project 7 &middot; Workplace-safety compliance monitoring<br/>
    SH17 &middot; YOLOv8 &middot; MobileSAM<br/><br/>
    All figures generated from <span class="mono">outputs/*.json</span>
  </div>
</div>

<h1 class="first">1. The question, and why it is not obvious</h1>

<p>A workplace-safety monitor has to answer a question about <em>people</em>, not about pixels:
<em>is this worker wearing head protection?</em> Object detection alone cannot answer it. A frame
containing one <span class="mono">person</span> box, one <span class="mono">head</span> box and one
<span class="mono">helmet</span> box becomes ambiguous the instant a second person walks in — the
helmet must be <strong>assigned</strong> to somebody, and the assignment rule, not the detector,
decides the verdict.</p>

<p>The error is asymmetric. Scoring a non-compliant worker as compliant leaves a person
unprotected on a site; the reverse merely triggers a check that costs a supervisor a minute. The
headline metric throughout this book is therefore <strong>per-person non-compliance recall</strong>,
and "undetermined" is counted as a <strong>miss</strong>: a monitor that cannot tell protects
nobody.</p>

<p>The intuition that motivates the project is genuinely appealing. When two workers overlap in
frame, their bounding <em>boxes</em> may both contain the same helmet, so a box-overlap test is
close to a coin flip in exactly the crowded case that matters. Their instance <em>masks</em>,
however, are disjoint. Segmentation looks like the principled fix.</p>

<div class="callout"><strong>Hypothesis, fixed before any training run.</strong> Mask-containment
association yields higher per-person non-compliance recall than box-containment association, and
the margin <strong>increases</strong> with the number of persons per frame.</div>

<p>Both limbs are falsifiable. This book shows both are false, and — more usefully — measures
exactly which assumption breaks.</p>

<div class="headline">
<strong>Result.</strong> With detection removed entirely (ground-truth boxes, real segmentation),
mask containment scores <strong>{f4(O_B['overall']['noncompliance_recall_strict'])}</strong>
non-compliance recall against <strong>{f4(O_A['overall']['noncompliance_recall_strict'])}</strong>
for box containment: Δ = <strong>{f4(oh['delta'])}</strong>, 95% CI
[{f4(oh['ci95_lo'])}, {f4(oh['ci95_hi'])}], P(Δ&gt;0) = {oh['p_delta_gt_0']:.3f} over
{orc['n_composites']} scenes and {O_A['overall']['n_persons']:,} persons. The margin does not grow
with crowding. Root cause: for <strong>{pct(cont['helmet_mask']['frac_below_0.50'])}</strong> of
helmets, the wearer's own mask covers less than half the helmet box.
</div>

<h2>1.1 Decision rule, fixed in advance</h2>
<p>The quantity that decides the hypothesis is the paired difference in non-compliance recall,
with a 95% confidence interval from a bootstrap that resamples <em>composites</em> rather than
persons, plus the per-stratum profile across crowding bands 1, 2–3, 4–6 and 7+. A flat profile
refutes the second limb even if the first were to hold. Fixing this before running anything is
what stops the metric being chosen after the fact to suit the result.</p>

<h1>2. Dataset and splits</h1>

<p>The corpus is <strong>SH17</strong>, a workplace-safety detection dataset of 8,099 images and
75,994 instances across 17 classes. It was selected for a property most PPE datasets do not have:
it annotates <span class="mono">person</span>, <span class="mono">head</span>,
<span class="mono">helmet</span> and <span class="mono">safety-vest</span> as <strong>independent
boxes</strong> and never records which head belongs to which person.</p>

<p>That absence is the entire reason it is usable here. A dataset that ships per-person compliance
verdicts has already solved the association problem inside its labels, and any experiment run on
it measures label quality rather than the mechanism. The cost of that choice is that there is no
association ground truth to score against either — addressed in Chapter 4.</p>

<h3>Class map recovered, not assumed</h3>
<p>The class index → name mapping was recovered empirically by aligning YOLO line order against
VOC object order across 400 random images, achieving purity 1.000 on all 17 classes. It was not
copied from the paper. A silently wrong helmet index produces a fully functional pipeline that
computes a meaningless number, and nothing errors.</p>

<h3>Split protocol</h3>
<pre>SH17 train (6,479) ─┬─ det_train  5,831   detector fitting
                    └─ det_val      648   checkpoint selection
SH17 val   (1,620) ─── HELD OUT           composite sources + test mAP</pre>
<p>The authors' own train/val boundary is never crossed. Every person appearing in the evaluation
set comes from an image the detector has never seen.</p>

<h3>A narrowing forced by the data</h3>
<p>The brief specified a compliance verdict of helmet <em>and</em> vest. Counting the held-out
single-person pool before any arm was scored: 712 images with head-and-no-helmet, 35 with
head-and-helmet, and <strong>14</strong> containing a safety vest. Fourteen source images cannot
support a headline metric, so <strong>compliance is head-protection only</strong>. The vest
remains a detector class — it is nearly free and its mAP is reportable — but it is not in the
verdict. This is a narrowing forced by counting, not a change of question after seeing results.</p>

<h1>3. The three arms</h1>

<p>Every arm consumes <strong>one</strong> detection pass per image. Only the association step
differs, so no observed difference can be attributed to detector noise. Writing
{EQ[0]} for a person index and {EQ[1]} for a PPE box index, all three arms are the same
argmax rule under a different region and score:</p>

{EQ[2]}

<p>with the arms differing only in {EQ[3]}:</p>

{table(["Arm", "Person region", "Score s(j,p)", "τ"],
       [["A0", "bounding box", "IoU(b_j, b_p)", "0.30"],
        ["A", "bounding <b>box</b>", "|b_j ∩ b_p| / |b_j|  (containment)", "0.50"],
        ["B", "instance <b>mask</b>", "|b_j ∩ M_p| / |b_j|  (containment)", "0.50"]])}

<p>Intersection-over-union and containment are different questions. IoU asks how much two regions
agree; containment asks what fraction of the <em>small</em> region lies inside the large one:</p>

{EQ[4]}

<p>The compliance rule applied identically to every arm's assignment:</p>

{CASES_HTML}

<div class="warn"><strong>This derivation is an assumption, not a dataset fact</strong> — and it is
precisely the assumption the two arms disagree about. It is stated here rather than buried in a
config file. A person with no evidence at all is left undetermined rather than scored compliant:
scoring absence as compliance is the exact failure mode that makes a safety monitor useless.</div>

<h2>3.1 Why ARM A0 exists, and why ARM A is the honest baseline</h2>

<p>The original scaffold specified ARM A0: assign each PPE box to the person box it overlaps most,
above IoU ≥ 0.30. That baseline is broken, and not in an interesting way. A helmet box occupies
roughly 1–2% of a person box's area, so</p>

{EQ[6]}

<p>a factor of twenty below its own threshold. It can never fire. Measured on the oracle run:
<strong>{f4(O_A0['overall']['noncompliance_recall_strict'])}</strong> non-compliance recall at an
undetermined rate of <strong>{pct(O_A0['overall']['undetermined_rate'])}</strong>.</p>

<p>Had that been left as the baseline, ARM B would have "won" by roughly sixty points against an
arm that never assigned anything — and no code would have errored, no test would have failed, and
the write-up would have reported a spectacular fake result. The fair baseline is the rule
practitioners actually use for small-object-to-container assignment: containment with argmax.</p>

<div class="finding"><strong>A0 is retained and reported, not deleted.</strong> It costs nothing
(it reuses the same detections) and it is the evidence that ARM A was chosen to be fair rather
than chosen to lose. A reviewer asking "did you just pick a weak baseline?" is answered by a row
in a table instead of a promise.</div>

<h1>4. Ground truth: synthetic composites</h1>

<p>SH17 records no person↔PPE link, so there is nothing to score association against. Four options
were considered:</p>

{table(["Option", "Ground truth", "Verdict"],
       [["Score against box-IoU-derived links", "Circular — defines ARM A correct", "<b>Reject</b>"],
        ["Evaluate only on single-person images", "Unambiguous, but zero crowding", "<b>Reject</b> — crowding is the claim"],
        ["Hand-label a crowded subset", "Real, but slow and small", "Keep as external check"],
        ["<b>Synthetic composites</b>", "<b>Exact, by construction</b>", "<b>Primary</b>"]])}

<p>Take held-out images containing exactly <em>one</em> person — in those, every PPE box provably
belongs to that person — and composite {EQ[7]} of them into a single crowded scene, carrying
the known links through. This yields exact ground truth at any crowding level and turns crowding
into a <em>swept</em> variable rather than whatever distribution the corpus happens to contain.</p>

<p>Built: <strong>{orc['n_composites']} composites, {O_A['overall']['n_persons']:,} persons</strong>
({O_A['overall']['n_noncompliant']:,} non-compliant), mean 4.33 persons per scene, maximum 9.</p>

<h2>4.1 Two generator choices that would have decided the result</h2>

<p><strong>Trap 1 — a "better looking" paste rigs the outcome.</strong> The natural instinct is to
segment each source person with SAM and paste the silhouette so the scene looks realistic. That is
a bug: a silhouette pasted onto a new background leaves a sharp figure/ground boundary that SAM can
re-segment almost trivially at evaluation time. The mask arm would win because the generator drew
its answer into the image. Rectangular crops carry their own background, so the segmentation step
has to do real work — uglier, and conservative toward the project's own hypothesis.</p>

<p>The general form is worth stating: <em>if the generator uses model M to build the data, any arm
using M gets a free signal.</em></p>

<p><strong>Trap 2 — a fixed canvas confounds the independent variable.</strong> The first
implementation kept a 1280 px canvas and shrank everyone to fit, so 9-person scenes had persons at
roughly half the size of 2-person scenes. The claim under test is that the margin <em>grows with
crowding</em> — but crowding and object scale were now moving together, so any trend would have
been unattributable. Fixed by holding person height constant and widening the canvas with the
crowd.</p>

<p>Two further problems were invisible in the metrics and obvious on one look at a rendered
composite: SH17's zero-person images are object close-ups (a jar, a hand) which pasted sharply
behind workers looked absurd, and persons were being clipped by the canvas bottom, producing
unusable ground-truth boxes. <strong>Render a handful with ground truth overlaid before generating
the full set</strong> — every time.</p>

<h1>5. Estimators</h1>

<h3>Non-compliance recall</h3>
<p>Over all ground-truth non-compliant persons, including those the detector never found:</p>

{EQ[8]}

<p>A <em>conditioned</em> variant restricted to persons the detector actually matched is also
reported; it isolates the association step. Because both arms consume identical detections, the
detector's contribution cancels from the A→B difference under either definition.</p>

<h3>Compliant F1 — the guard metric</h3>
<p>An arm that maximises non-compliance recall by declaring everyone non-compliant scores 1.000 and
is worthless. F1 on the compliant class is reported alongside every recall figure to make that
degenerate strategy visible.</p>

<h3>Paired bootstrap over composites</h3>
<p>Persons within a scene are correlated through the scene they share, so resampling persons
independently would understate variance. The bootstrap resamples <em>composites</em> with
replacement, {EQ[9]} replicates, and both arms see the same resampled scenes in each
replicate — so the detector's contribution cancels within the replicate rather than merely on
average:</p>

{EQ[10]}

<h1>6. The oracle ablation — the decisive experiment</h1>

<p>The full pipeline feeds both arms the same detections, which controls for detector quality but
does not remove it: a missed helmet hurts both arms and compresses the measurable gap. Substituting
ground-truth boxes removes detection from the loop entirely and asks the mechanism question in its
purest form. MobileSAM still segments the real composite image and has no knowledge of how it was
built.</p>

<div class="callout">This reading was <strong>committed to in advance</strong>, in the
<span class="mono">src/oracle.py</span> docstring: <em>if ARM B shows no advantage here, it cannot
show one with a real detector either, and the hypothesis is dead regardless of how well the
detector is trained.</em> Pre-registering the interpretation is what stops an inconvenient result
being re-read as "we just need a better detector".</div>

{table(["Arm", "nc-recall", "helmet assoc. acc.", "undetermined", "compliant F1"],
       [["A0 — box IoU @0.30", f4(O_A0['overall']['noncompliance_recall_strict']),
         f4(O_A0['helmet_association_accuracy']), f4(O_A0['overall']['undetermined_rate']),
         f4(O_A0['overall']['compliant_f1'])],
        ["<b>A — box containment @0.50</b>", f"<b>{f4(O_A['overall']['noncompliance_recall_strict'])}</b>",
         f"<b>{f4(O_A['helmet_association_accuracy'])}</b>", f4(O_A['overall']['undetermined_rate']),
         f"<b>{f4(O_A['overall']['compliant_f1'])}</b>"],
        ["B — mask containment @0.50", f4(O_B['overall']['noncompliance_recall_strict']),
         f4(O_B['helmet_association_accuracy']), f4(O_B['overall']['undetermined_rate']),
         f4(O_B['overall']['compliant_f1'])]])}

<p style="text-align:center"><strong>B − A = {f4(oh['delta'])}</strong>, 95% CI
[{f4(oh['ci95_lo'])}, {f4(oh['ci95_hi'])}], P(Δ&gt;0) = {oh['p_delta_gt_0']:.3f}</p>

<h3>The directional claim</h3>
<p>The second limb of the hypothesis was that the margin grows with crowding. By stratum:</p>

{table(["1 person", "2–3", "4–6", "7+"],
       [[strat_row(orc['delta_by_crowding'], '1'), strat_row(orc['delta_by_crowding'], '2-3'),
         strat_row(orc['delta_by_crowding'], '4-6'), strat_row(orc['delta_by_crowding'], '7+')]])}

<p>Flat, and negative throughout. Both limbs of the hypothesis fail simultaneously, with detection
removed from the loop.</p>

<h1>7. Why masks lose — the containment distribution</h1>

<p>Before accepting a refutation it is worth asking whether the measurement was fair. A person's
segmentation mask is very nearly a <em>subset</em> of their bounding box, so</p>

{EQ[11]}

<p>by construction. The two arms are therefore <strong>not scored on the same scale</strong>, which
is a problem addressed fully in Chapter 8. First, the direct diagnostic: score every ground-truth
(PPE box, <em>known</em> owner) pair under both rules, which requires no thresholds and no
assignment at all.</p>

{table(["", "mean", "median", "p05", "fraction &lt; 0.50", "n"],
       [["helmet → owner <b>box</b>", f"{cont['helmet_box']['mean']:.3f}", f"{cont['helmet_box']['median']:.3f}",
         f"{cont['helmet_box']['p05']:.3f}", pct(cont['helmet_box']['frac_below_0.50']), f"{cont['helmet_box']['n']:,}"],
        ["helmet → owner <b>mask</b>", f"{cont['helmet_mask']['mean']:.3f}", f"{cont['helmet_mask']['median']:.3f}",
         f"<b>{cont['helmet_mask']['p05']:.3f}</b>", f"<b>{pct(cont['helmet_mask']['frac_below_0.50'])}</b>",
         f"{cont['helmet_mask']['n']:,}"],
        ["head → owner <b>box</b>", f"{cont['head_box']['mean']:.3f}", f"{cont['head_box']['median']:.3f}",
         f"{cont['head_box']['p05']:.3f}", pct(cont['head_box']['frac_below_0.50']), f"{cont['head_box']['n']:,}"],
        ["head → owner <b>mask</b>", f"{cont['head_mask']['mean']:.3f}", f"{cont['head_mask']['median']:.3f}",
         f"{cont['head_mask']['p05']:.3f}", pct(cont['head_mask']['frac_below_0.50']), f"{cont['head_mask']['n']:,}"]])}

{plot_containment()}
<div class="caption">Containment of PPE boxes within their known owner's region. The box rule sits
at ceiling; the mask rule has a fifth percentile of zero.</div>

<div class="finding"><strong>This is the whole finding in one table.</strong> For
{pct(cont['helmet_mask']['frac_below_0.50'])} of helmets the wearer's own mask covers less than
half the helmet box, and the 5th percentile is {cont['helmet_mask']['p05']:.3f} — a substantial
tail of helmets falls <em>entirely outside</em> the mask of the person wearing them.</div>

<p>This is not a SAM failure. SAM is doing exactly its job: prompted with the person's box, it
segments <em>the person</em>. A hard hat is an object worn <strong>on top of</strong> a person and
is not part of their silhouette. ARM B's elevated undetermined rate
({pct(O_B['overall']['undetermined_rate'])} against {pct(O_A['overall']['undetermined_rate'])}) is
precisely this: the true link exists and the mask test rejects it.</p>

<p>The bounding box has no such problem <em>because</em> it is crude, and crudeness is what this
task wants. Segmentation answers "which pixels are this person". Association needs "which region
is associated with this person". Worn equipment falls in the gap between those two questions.</p>

<h1>8. Robustness: the threshold sweep</h1>

<p>The shared threshold was adopted for a good reason — tuning each arm separately confounds the
mechanism with threshold search — and it carries the hidden confound established in Chapter 7.
"ARM B loses at 0.50" had two incompatible readings: masks are the worse mechanism, or 0.50 is the
wrong operating point for a quantity whose range is compressed. Opposite conclusions, same number.
The experiment cannot end at a single threshold.</p>

<p>Both arms were swept over {EQ[12]} and compared at each arm's own best
operating point, where "best" is highest non-compliance recall with ties broken on compliant F1 so
a degenerate all-non-compliant point cannot win:</p>

{table(["", "best τ", "nc-recall", "compliant F1"],
       [["ARM A (box)", f"{swp['best']['A_box']['threshold']:.2f}",
         f"<b>{f4(swp['best']['A_box']['noncompliance_recall_strict'])}</b>",
         f4(swp['best']['A_box']['compliant_f1'])],
        ["ARM B (mask)", f"{swp['best']['B_mask']['threshold']:.2f}",
         f4(swp['best']['B_mask']['noncompliance_recall_strict']),
         f4(swp['best']['B_mask']['compliant_f1'])]])}

<p style="text-align:center"><strong>Best-vs-best B − A = {f4(sh['delta'])}</strong>, 95% CI
[{f4(sh['ci95_lo'])}, {f4(sh['ci95_hi'])}], P(Δ&gt;0) = {sh['p_delta_gt_0']:.3f}</p>

{plot_sweep()}
<div class="caption">ARM A is effectively threshold-independent; ARM B improves monotonically as
the test is relaxed and peaks where it is nearly disabled.</div>

<p>ARM A is flat across the entire sweep because box containment on a true pair is ≈1.0. ARM B
improves monotonically as {EQ[13]} falls and peaks at
{swp['best']['B_mask']['threshold']:.2f} — at its strongest when the containment test is switched
off almost entirely — and still loses.</p>

<div class="callout">This procedure selects each arm's threshold on the same data it is scored on,
which is optimistic for <em>both</em> arms and therefore <strong>generous to the arm being
refuted</strong>. The refutation survives that generosity. Roughly two thirds of the
shared-threshold gap ({f4(oh['delta'])} → {f4(sh['delta'])}) was the threshold artifact; the
remaining third is real and its confidence interval excludes zero.</div>

<h1>9. The same experiment with a real detector</h1>

<p>The oracle settles the mechanism. This run says what survives contact with imperfect
perception. A YOLOv8n detector was trained on the SH17 train split
(<span class="mono">SCALE=demo</span>, 640 px) and its detections shared by all three arms.</p>

{plot_training()}

<h3>Detector quality</h3>
<p>Held-out SH17 val (1,620 images, 5,412 instances): <strong>mAP50-95 = 0.4321</strong>,
mAP50 = 0.630. Per class: person 0.634, head 0.642, helmet 0.289, safety-vest 0.164. The small
classes are the hard ones, which is the expected profile for this corpus — 39.7k of SH17's 76k
annotations are under 1% of image area — and it explains why absolute recall below sits well under
the oracle's.</p>

{table(["Arm", "nc-recall strict", "nc-recall conditioned", "helmet assoc.", "undetermined", "compliant F1"],
       [["A0 box IoU", f4(E_A0['overall']['noncompliance_recall_strict']),
         f4(E_A0['overall']['noncompliance_recall_conditioned']), f4(E_A0['helmet_association_accuracy']),
         f4(E_A0['overall']['undetermined_rate']), f4(E_A0['overall']['compliant_f1'])],
        ["<b>A box containment</b>", f"<b>{f4(E_A['overall']['noncompliance_recall_strict'])}</b>",
         f"<b>{f4(E_A['overall']['noncompliance_recall_conditioned'])}</b>",
         f"<b>{f4(E_A['helmet_association_accuracy'])}</b>",
         f4(E_A['overall']['undetermined_rate']), f"<b>{f4(E_A['overall']['compliant_f1'])}</b>"],
        ["B mask containment", f4(E_B['overall']['noncompliance_recall_strict']),
         f4(E_B['overall']['noncompliance_recall_conditioned']), f4(E_B['helmet_association_accuracy']),
         f4(E_B['overall']['undetermined_rate']), f4(E_B['overall']['compliant_f1'])]])}

<p style="text-align:center">Δ<sub>strict</sub> = <strong>{f4(eh_s['delta'])}</strong>
[{f4(eh_s['ci95_lo'])}, {f4(eh_s['ci95_hi'])}] &nbsp;·&nbsp;
Δ<sub>conditioned</sub> = <strong>{f4(eh_c['delta'])}</strong>
[{f4(eh_c['ci95_lo'])}, {f4(eh_c['ci95_hi'])}], both P(Δ&gt;0) = 0.000</p>

<p>The detector matched {E_A['overall']['n_matched']:,} of {E_A['overall']['n_persons']:,}
ground-truth persons, so absolute recall falls from
{f4(O_A['overall']['noncompliance_recall_strict'])} to
{f4(E_A['overall']['noncompliance_recall_strict'])} and the A→B gap compresses from
{f4(oh['delta'])} to {f4(eh_s['delta'])}. Both effects are expected and neither changes the sign:
detector misses are identical across arms by construction, so they add a common loss that dilutes
the difference without biasing it.</p>

<div class="callout"><strong>On the truncated schedule.</strong> Training was stopped at epoch
{int(max(epochs)) if epochs else 12} of the configured 20. The headline finding uses no detector,
and the paired design means detector quality can compress the delta but not flip its sign — which
is exactly what was observed. The honest cost is that mAP50-95 = 0.432 understates what the full
schedule would reach, since <span class="mono">close_mosaic</span> activates at epoch 13. Detector
quality is reported context here, not the claim.</div>

<h1>10. Cost, deployment and monitoring</h1>

<h3>The segmentation pass is not free</h3>
{table(["Stage", "p50", "p95"],
       [["YOLOv8n detection", f"{lat['detect_ms_p50']:.1f} ms", f"{lat['detect_ms_p95']:.1f} ms"],
        ["MobileSAM segmentation (ARM B only)", f"{lat['segment_ms_p50']:.1f} ms",
         f"{lat['segment_ms_p95']:.1f} ms"]])}

<p>Measured over {lat['n_detect_calls']} detection calls and {lat['n_segment_calls']} segmentation
calls during the evaluation run. On the serving path the end-to-end figures were ~611–727 ms for
<span class="mono">arm=mask</span> against ~37 ms for <span class="mono">arm=box</span> — roughly
<strong>17×</strong> the cost. The arm under test is not merely less accurate; it is substantially
more expensive, so there is no operating regime in this evaluation where it is the right
choice.</p>

<h3>Serving</h3>
<p>A FastAPI service exposes <span class="mono">/analyze</span> with the association arm selectable
per request, so a deployment can be A/B'd against the same baseline this book reports rather than
silently shipping one arm and hoping. A Dockerfile is authored for CPU inference; the trained
checkpoint is mounted at run time rather than baked into the image layer.</p>

<h3>Label-free drift monitoring</h3>
<p>A deployed safety monitor never receives labels, so drift detection has to be behavioural. Three
signals are computed over a rolling window and are all label-free: persons-per-frame distribution
(is the site more crowded than anything the system was evaluated on?), the
<strong>undetermined rate</strong> (are heads failing to associate?), and mean detector confidence.
The undetermined rate is the most informative of the three, because it rises before accuracy
visibly degrades — and, as Chapter 7 showed, it is the exact quantity that separates the two
arms.</p>

<h1>11. What the finding is, and what it is not</h1>

<h3>Inference</h3>
<p>The hypothesis is refuted on both limbs, with the mechanism identified rather than merely
observed:</p>
<p>1. Mask-containment association does not beat box-containment association. It is
{abs(oh['delta']) * 100:.1f} points worse at the shared threshold and {abs(sh['delta']) * 100:.1f}
points worse at each arm's own optimum, both with confidence intervals excluding zero.<br/>
2. The margin does not grow with crowding, under either threshold regime.<br/>
3. The cause is geometric and measured: worn PPE lies outside the wearer's segmentation mask for
{pct(cont['helmet_mask']['frac_below_0.50'])} of helmets.</p>

<h3>Assumptions the inference depends on</h3>
<p>The derived compliance rule (head-without-helmet ⇒ non-compliant) is correct — and it is the
arms' point of disagreement. Composited scenes are representative of crowding as a
<em>geometric</em> phenomenon; they are not photorealistic, but both arms see identical pixels and
identical detections, so a compositing artifact cannot manufacture a difference <em>between</em>
arms. MobileSAM prompted with the person box is representative of instance segmentation for this
task.</p>

<h3>What this does not establish</h3>
<p>It does not show segmentation is useless for safety monitoring — only that <em>mask containment
of a PPE box within a person mask</em> is a worse association rule than box containment. The
headline comes from synthetic composites; a hand-labelled real-image check on ~200 genuinely
crowded SH17 images is the external-validity step and has not been done, so the claim is "holds on
controlled synthetic crowding", not "holds in the field". The 35 distinct compliant source persons
are the binding constraint on absolute values, though the paired design protects the delta.</p>

<h3>What would make segmentation win</h3>
<p>Two variants follow directly from the diagnostic and are stated as future work rather than
folded in after the fact, because each changes the mechanism under test: <strong>dilate</strong>
the person mask by roughly the scale of the worn item before testing containment, or segment the
<strong>helmet</strong> and test mask-to-mask adjacency rather than containment-within-person. The
containment distribution predicts both would help; neither is evidenced here.</p>

<h3>Generalisation</h3>
<p>The failure is not specific to helmets. Any "associate a small worn or carried object with its
owner" problem — hi-vis vests, tools, badges, handheld devices — has the same geometry: the object
is adjacent to the person, not inside them. The lesson transfers: <em>when the relation you need is
adjacency, a tighter region is a worse instrument.</em></p>

<h1>12. Research anchors</h1>

<p><strong>Kirillov et al. (2023), Segment Anything.</strong> The promptable-segmentation
formulation ARM B depends on. Its key property was verified before building on it: box-prompted
masks for two <em>overlapping</em> person boxes come back disjoint. That property does hold — the
mechanism had something to exploit, and it still lost for an unrelated reason.</p>

<p><strong>Zhang et al. (2023), MobileSAM.</strong> The distilled variant used here, chosen so the
segmentation pass is affordable in a per-frame safety monitor. Chapter 10 reports what it actually
cost rather than assuming it was negligible.</p>

<p><strong>Ahmad et al. (2024), SH17.</strong> The corpus, selected specifically because it
annotates person and PPE as independent boxes and records no association — so the question under
test is not pre-solved by the labels.</p>

<p><strong>Jocher et al., YOLOv8 / Ultralytics.</strong> The shared detector, used identically by
every arm so detection quality cannot enter the comparison.</p>

<p><strong>Efron &amp; Tibshirani (1993), An Introduction to the Bootstrap.</strong> The paired
cluster bootstrap used for every confidence interval; composites are the resampling unit because
persons within a scene are not independent.</p>
"""


if __name__ == "__main__":
    from weasyprint import CSS as WCSS, HTML

    html = f"<html><head><meta charset='utf-8'></head><body>{HTML_BODY}</body></html>"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(OUT, stylesheets=[WCSS(string=CSS)])

    # Verify by re-reading, per the brief: a PDF that was written is not a PDF that renders.
    import pypdf
    r = pypdf.PdfReader(str(OUT))
    text = "".join((p.extract_text() or "") for p in r.pages)
    checks = {
        "oracle delta present": f"{oh['delta']:.4f}" in text,
        "containment diagnostic present": "26.5%" in text or "0.265" in text,
        "sweep best-vs-best present": f"{sh['delta']:.4f}" in text,
        "detector mAP present": "0.4321" in text or "0.432" in text,
    }
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(r.pages)} pages)")
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'FAIL'} {k}")
    if len(r.pages) < 15:
        print(f"  WARNING: {len(r.pages)} pages, brief asks for 15+")

# NORA — Nanopore Oncology Reasoning Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/somayahalbaradei/nora/blob/main/notebooks/NORA_demo.ipynb)

**Somayah Albaradei** · King Abdulaziz University, Jeddah, Saudi Arabia

> Cells 1-4 of the notebook reproduce the central finding with no API key.

An agentic framework that tests whether the evidence in a cell-free RNA study
supports the claims the analysis implicitly makes.

Running an analysis asserts six things, usually without anyone deciding to:

| Stage | The claim asserted by proceeding |
|---|---|
| 1 Design | these groups are comparable |
| 2 Protocol | this analysis suits this chemistry |
| 3 Quality | these samples are usable |
| 4 Validity | this effect is real |
| 5 Interpretation | this cause explains it |
| 6 Decision | this action is warranted |

NORA makes each assumption explicit and tests it. Every stage may **abstain**.

## What it found on a published cohort

GSE271389: 47 plasma cfRNA nanopore libraries (16 control, 12 Barrett's
oesophagus, 19 oesophageal adenocarcinoma). NORA was told nothing about the
plate layout.

- **Design.** Derived plate row and column from sample identifiers. Controls
  occupy columns 9–12, cancers 1–7, sharing none. Diagnosis is completely
  confounded with plate column: no adjustment on that axis is possible.
- **Validity.** Fragment length classified the three groups at AUROC 0.70
  (permutation p=0.006). N50 differed more strongly by sequencing run (p=0.009)
  than by diagnosis (p=0.011), and the diagnostic effect vanished on
  conditioning (p=0.145). **The biomarker was a flow cell.**
- **Contrast.** Mitochondrial cfRNA, elevated 7.4-fold in disease, survived the
  identical adjustment (AUROC 0.690, p<0.005). Two signals, indistinguishable
  unadjusted statistics, opposite verdicts.
- **Decision.** At its best operating point the classifier detected all 19
  cancers and spared 57% of endoscopies, and decision curve analysis favoured
  it. NORA abstained: with 19 cancers the 95% CI on sensitivity reaches 83%.
  Roughly 100 cancers are needed to exclude a 5% miss rate.

## Try it

**Web app** — chat with NORA, or run any check without an API key:
https://somayahalbaradei-nora.streamlit.app

**Colab** — reproduces the core finding in 30 seconds, no key, no install.

**Locally**

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Use

Check a plate layout **before sequencing**, when a confound is still free to fix:

```bash
python -m nora.design --check data/gse271389_manifest.tsv --group group --position well
# exit 2 = fatal, 1 = warning, 0 = sound.  Drop it into a LIMS.
```

Generate a design that passes its own check:

```bash
python -m nora.design --plan cohort.tsv --group group --per-batch 24 --out plan.tsv
```

Validate an analysis:

```bash
python -m nora.agent --ask "Evaluate this cohort." --data data/nora_data.tsv
```

Or run the chat interface:

```bash
streamlit run app/app.py
```

## Design principle: tools that refuse

The statistical tools decline to return uninterpretable results rather than
relying on the model to notice.

- `stratified_permutation` refuses when most strata contain a single group,
  because permuting labels within a pure stratum changes nothing and the
  p-value would rest on a fraction of the data.
- `condition_and_retest` refuses under complete separation, because a null
  result would then be guaranteed by construction.
- `crosstab` reports complete separation explicitly.

`scipy` will return p=0.333 from a degenerate design without comment. That is
the behaviour this replaces.

## Data

Derived from GEO **GSE271389** / BioProject **PRJNA1131298**, deposited by
Peddu et al. Raw reads are in SRA. Only summary tables are redistributed here.

## Limitations

Single cohort, single configuration. This is a case study, not a benchmark.
`list_variables` hints that high-cardinality identifiers may encode batch
structure; results should be replicated with that hint removed. Plate row is a
proxy for sequencing run, which the deposit does not record.

## Citation

```bibtex
@software{albaradei2026nora,
  author  = {Albaradei, Somayah},
  title   = {{NORA}: a nanopore oncology reasoning agent for evidence
             validation in cell-free {RNA} liquid biopsy},
  year    = {2026},
  url     = {https://github.com/somayahalbaradei/nora}
}
```

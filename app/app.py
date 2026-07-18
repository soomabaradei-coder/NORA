"""
NORA — evidence validation for nanopore sequencing.

    streamlit run app/app.py
"""
import os, sys, json, importlib.util
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("nora_agent", ROOT / "nora" / "agent.py")
na = importlib.util.module_from_spec(spec)
sys.modules["nora_agent"] = na
spec.loader.exec_module(na)

st.set_page_config(page_title="NORA", page_icon="🔬", layout="wide")
st.markdown("""<style>
.stApp header {visibility:hidden}
.verdict-no {background:#fff5f5;border-left:4px solid #b30000;padding:14px 18px;margin:10px 0}
.verdict-yes{background:#f2f9f2;border-left:4px solid #1a7f37;padding:14px 18px;margin:10px 0}
.verdict-na {background:#fffaf0;border-left:4px solid #9a6700;padding:14px 18px;margin:10px 0}
.step {font-family:ui-monospace,monospace;font-size:12px;color:#57606a;margin:2px 0}
.refbox {background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px 14px;font-size:13px}
</style>""", unsafe_allow_html=True)

st.title("NORA")
st.markdown("##### Determines whether Nanopore sequencing evidence is sufficient "
            "to support a scientific or clinical claim.")

# ---------------------------------------------------------------- reference
DEFAULT_PROV = {"platform": "PromethION", "protocol": "cDNA",
                "library_kit": "SQK-NBD-114.24", "flowcell": "R10.4",
                "basecaller": "dorado 0.7", "reference_genome": "GRCh38",
                "annotation": "GENCODE 39", "aligner": "minimap2 2.31"}

with st.sidebar:
    st.subheader("Evidence Reference")
    if "ref" not in st.session_state:
        st.session_state.ref = None

    ref = st.session_state.ref
    if ref:
        p = ref["provenance"]
        st.markdown(f"""<div class="refbox">
<b>{st.session_state.get('ref_name','Reference v1')}</b><br>
<span style="color:#57606a">Built from</span><br>
&nbsp;&nbsp;{ref['n_cohort']} samples<br>
<span style="color:#57606a">Protocol</span><br>
&nbsp;&nbsp;✓ {p.get('protocol')}<br>
&nbsp;&nbsp;✓ {p.get('library_kit')}<br>
&nbsp;&nbsp;✓ {p.get('flowcell')}<br>
&nbsp;&nbsp;✓ {p.get('basecaller')}<br>
<span style="color:#1a7f37"><b>Status: Ready</b></span>
</div>""", unsafe_allow_html=True)
        st.caption("A reference is valid only for the conditions that built it. "
                   "Every sample is checked for compatibility first.")
    else:
        st.info("No reference model. Load a cohort and build one below.")

    st.divider()
    key = st.text_input("Anthropic API key", type="password",
                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Needed only for the reasoning agent. Every check runs without it.")

# ---------------------------------------------------------------- mode
mode = st.radio("What are you validating?",
                ["A cohort", "A single sample", "A published claim"],
                horizontal=True, label_visibility="visible")
st.divider()

def ledger(rows):
    """The evidence ledger. One row per check, with the number that decided it."""
    df = pd.DataFrame(rows, columns=["Check", "Status", "Evidence"])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Check": st.column_config.TextColumn(width="medium"),
                                "Status": st.column_config.TextColumn(width="small"),
                                "Evidence": st.column_config.TextColumn(width="large")})

def verdict(kind, title, body):
    cls = {"no": "verdict-no", "yes": "verdict-yes", "na": "verdict-na"}[kind]
    st.markdown(f'<div class="{cls}"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

# ================================================================ COHORT
if mode == "A cohort":
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Cohort table", type=["tsv", "csv", "txt"],
                              label_visibility="collapsed")
        demo = st.checkbox("Use GSE271389 (47 plasma cfRNA libraries)", value=True)
    if demo and not up:
        na.DATA = pd.read_csv(ROOT / "data" / "nora_data.tsv", sep="\t")
    elif up:
        na.DATA = pd.read_csv(up, sep=None, engine="python")
    with c2:
        if na.DATA is not None:
            st.metric("Samples", len(na.DATA)); st.metric("Variables", len(na.DATA.columns))

    if na.DATA is not None:
        q = st.text_input("Claim to validate",
                          "Fragment length is a valid biomarker for disease group")
        if st.button("Validate", type="primary"):
            rows = []
            prog = st.empty()
            def step(t): prog.markdown(f'<div class="step">✓ {t}</div>', unsafe_allow_html=True)

            step("Reading metadata…")
            cats = [c for c in na.DATA.columns if not pd.api.types.is_numeric_dtype(na.DATA[c])]
            ids = [c for c in cats if na.DATA[c].nunique() > len(na.DATA) * .5]
            rows.append(["Metadata read", "✅",
                         f"{len(na.DATA)} samples, {len(na.DATA.columns)} variables"])

            step("Decomposing identifiers…")
            derived = []
            for c in ids:
                s = na.DATA[c].astype(str)
                if s.str.match(r"^[A-Za-z]\d+$").all():
                    na.derive_variable("_row", c, "first_char")
                    na.derive_variable("_col", c, "numeric_part")
                    derived = ["_row", "_col"]
                    rows.append(["Technical axes found", "⚠️",
                                 f"'{c}' encodes position: derived _row, _col"])
            if not derived:
                rows.append(["Technical axes found", "—", "no structured identifier detected"])

            grp = st.session_state.get("grp") or ("group" if "group" in na.DATA.columns else cats[0])
            step("Searching for hidden confounders…")
            fatal = None
            for ax in derived:
                ct = na.crosstab(ax, grp)
                if ct.get("COMPLETE_SEPARATION"):
                    fatal = ax
                    rows.append(["Design confounding", "❌",
                                 f"{grp} completely confounded with {ax}: "
                                 + ct["COMPLETE_SEPARATION"][0]])
                else:
                    rows.append([f"{ax} balanced across groups", "✅",
                                 f"{ct['levels_with_only_one_group']} levels hold one group"])

            step("Testing the claim…")
            num = [c for c in na.DATA.columns if pd.api.types.is_numeric_dtype(na.DATA[c])]
            feat = "n50" if "n50" in num else num[0]
            raw = na.test_association(feat, grp, nperm=2000)
            rows.append([f"{feat} differs by {grp}", "✅" if raw["p_permutation"] < .05 else "—",
                         f"p = {raw['p_permutation']:.3f} (unadjusted)"])

            adjustable = [a for a in derived if a != fatal]
            if adjustable:
                ax = adjustable[0]
                tech = na.test_association(feat, ax, nperm=2000)
                rows.append([f"{feat} differs by {ax} (technical)", 
                             "❌" if tech["p_permutation"] < .05 else "✅",
                             f"p = {tech['p_permutation']:.3f}"])
                step("Conditioning on technical structure…")
                cond = na.condition_and_retest(feat, grp, ax, nperm=2000)
                survived = cond["after"]["p"] < .05
                rows.append(["Effect survives conditioning", "✅" if survived else "❌",
                             f"p = {cond['before']['p']:.3f} → {cond['after']['p']:.3f}"])
                step("Checking effect size…")
                es = na.effect_size(feat, grp)
                worst = max(es["comparisons"], key=lambda o: o["overlap_pct"])
                rows.append(["Biologically meaningful separation",
                             "✅" if worst["overlap_pct"] < 60 else "❌",
                             f"{worst['comparison']}: {worst['overlap_pct']}% overlap, "
                             f"d = {worst['cohens_d']}"])
                step("Testing reproducibility…")
                rep = na.check_reproducibility(feat, grp, stratum=ax, n_splits=40)
                pct = int(rep["replicated_in_both_halves"].split("=")[1].strip().rstrip("%"))
                rows.append(["Reproducible across split halves", "✅" if pct > 50 else "❌",
                             rep["replicated_in_both_halves"]])
            prog.empty()

            st.subheader("Evidence ledger")
            ledger(rows)
            fails = sum(1 for r in rows if r[1] == "❌")
            if fails:
                verdict("no", "Evidence does not support the claim.",
                        f"{fails} of {len(rows)} checks failed. See the ledger. "
                        "A statistically significant result is not a finding until it "
                        "survives the technical structure of the study that produced it.")
            else:
                verdict("yes", "Evidence supports the claim.",
                        "All checks passed. This licenses the claim at this sample size; "
                        "it does not establish clinical utility.")

# ================================================================ SAMPLE
elif mode == "A single sample":
    if st.session_state.ref is None:
        st.warning("A single sample cannot be judged without a reference model. "
                   "Build one from a cohort first.")
        if st.button("Build reference from GSE271389"):
            na.DATA = pd.read_csv(ROOT / "data" / "nora_data.tsv", sep="\t")
            r = na.fit_reference(technical_features=["n_reads", "mean_len", "n50"],
                                 provenance=DEFAULT_PROV)
            st.session_state.ref = r
            st.session_state.ref_name = "EAC cfRNA cDNA v1"
            st.rerun()
    else:
        st.markdown("**Step 1 · Protocol compatibility**")
        st.caption("Checked before any number is compared.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Reference")
            for k in ["platform", "flowcell", "protocol", "basecaller"]:
                st.text(f"{k:14s} {st.session_state.ref['provenance'].get(k)}")
        with c2:
            st.markdown("Uploaded sample")
            sp = {}
            sp["platform"]   = st.selectbox("platform", ["PromethION","MinION","GridION"])
            sp["flowcell"]   = st.selectbox("flowcell", ["R10.4","R10.4.1","R9.4.1"])
            sp["protocol"]   = st.selectbox("protocol", ["cDNA","direct RNA"])
            sp["basecaller"] = st.text_input("basecaller", "dorado 0.7")

        comp = na.check_provenance(sp, st.session_state.ref)
        if not comp["applicable"]:
            verdict("na", "❌ Incompatible",
                    "<br>".join(f"<b>{f['field']}</b>: reference {f['reference']}, "
                                f"sample {f['sample']}<br><small>{f['why']}</small>"
                                for f in comp["incompatible"])
                    + "<br><br>Evidence model cannot be applied.")
            st.stop()
        st.success("✅ Compatible. " + (f"{len(comp['differences'])} non-fatal difference(s)."
                                        if comp["differences"] else ""))

        st.markdown("**Step 2 · Sample values**")
        c1, c2, c3 = st.columns(3)
        vals = {}
        for col, f in zip((c1, c2, c3), list(st.session_state.ref["technical"])[:3]):
            with col:
                r = st.session_state.ref["technical"][f]
                vals[f] = st.number_input(f, value=float(r["median"]), format="%.1f")
                st.caption(f"reference {r['p05']:.0f} – {r['p95']:.0f}")

        if st.button("Evaluate evidence", type="primary"):
            res = na.check_sample(vals, sample_provenance=sp, reference=st.session_state.ref)
            rows = [["Protocol compatible", "✅", comp["verdict"][:70]]]
            for f, v in vals.items():
                r = st.session_state.ref["technical"][f]
                bad = v < r["p05"]
                rows.append([f, "❌" if bad else "✅",
                             f"{v:.0f}  (reference {r['p05']:.0f} – {r['p95']:.0f})"])
            st.subheader("Evidence ledger"); ledger(rows)
            if res["verdict"] == "INVALID":
                verdict("no", "Evidence insufficient. Do not report a result.",
                        res["failures"][0]["reason"] + "<br><br>" + res["action"])
            else:
                verdict("yes", "Evidence sufficient to report.",
                        res["action"] + f"<br><small>{res['caveat']}</small>")

# ================================================================ CLAIM
else:
    st.markdown("Paste a claim from a paper or an analysis. NORA tests whether the "
                "evidence given supports it.")
    claim = st.text_area("Claim", "Fragment length classifies disease at AUROC 0.70 "
                                  "(permutation p=0.006) in a cohort of 47 plasma cfRNA "
                                  "libraries sequenced on PromethION R10.4.", height=90)
    c1, c2 = st.columns(2)
    with c1: k = st.number_input("Critical cases detected", 0, 10000, 19)
    with c2: n = st.number_input("Critical cases total", 1, 10000, 19)
    if st.button("Validate claim", type="primary"):
        rows = []
        suff = na.evidence_sufficiency(int(k), int(n), target=0.95)
        rows.append(["Sample size supports the claim", "✅" if suff["supported"] else "❌",
                     f"{suff['observed']}, 95% CI lower bound {suff['ci_lower']:.1%}"])
        if not suff["supported"]:
            rows.append(["Required evidence", "—",
                         f"~{suff['n_required_if_all_correct']} cases, all correct, "
                         f"to exclude a {100*(1-0.95):.0f}% failure rate"])
        sim = na.simulate_null([16, 12, 19], leaky=True, nsim=12)
        rows.append(["Chance performance at this design", "⚠️",
                     f"selection outside CV gives AUROC {sim['mean_AUROC_on_noise']} "
                     f"on data with no signal"])
        st.subheader("Evidence ledger"); ledger(rows)
        if not suff["supported"]:
            verdict("no", "Evidence does not support the claim.", suff["verdict"])
        else:
            verdict("yes", "Sample size supports the claim.", suff["verdict"])

# ---------------------------------------------------------------- advanced
st.divider()
with st.expander("Advanced tools — run any check directly"):
    st.caption("Implementation detail. NORA orchestrates these automatically above.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Evidence sufficiency**")
        k2 = st.number_input("correct", 0, 10000, 19, key="a1")
        n2 = st.number_input("total", 1, 10000, 19, key="a2")
        if st.button("Run", key="b1"): st.json(na.evidence_sufficiency(int(k2), int(n2)))
    with c2:
        st.markdown("**Association**")
        if na.DATA is not None:
            cols = list(na.DATA.columns)
            v = st.selectbox("measure", [c for c in cols if pd.api.types.is_numeric_dtype(na.DATA[c])], key="a3")
            g = st.selectbox("grouping", [c for c in cols if not pd.api.types.is_numeric_dtype(na.DATA[c])], key="a4")
            if st.button("Run", key="b2"): st.json(na.test_association(v, g, nperm=2000))
    with c3:
        st.markdown("**Null calibration**")
        lk = st.checkbox("selection outside CV", True, key="a5")
        if st.button("Run", key="b3"):
            st.json(na.simulate_null([16, 12, 19], leaky=lk, nsim=12))

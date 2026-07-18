#!/usr/bin/env python3
"""
NORA — an agent that RUNS the checks, not one that names them.

THE TEST
  Give it your cohort and ask: "Is fragment length a valid biomarker here?"
  It is NOT told about plate position. It must discover it.

GROUND TRUTH (established by execution, not opinion)
  n50 ~ diagnosis            p = 0.011   looks real
  n50 ~ plate row            p = 0.009   stronger
  n50 ~ diagnosis | row      p = 0.145   gone
  -> fragment length is a flow-cell artefact. PASS = the agent finds this.

USAGE
  pip install anthropic --break-system-packages
  export ANTHROPIC_API_KEY=sk-...
  python3 nora_agent.py --build          # merge your files into nora_data.tsv
  python3 nora_agent.py --ask "Is fragment length a valid biomarker for disease group?"
"""
import json, argparse, re, sys
import numpy as np, pandas as pd

DATA = None
MODEL = "claude-sonnet-5"

# ============================== statistics =================================

def _kruskal(groups):
    a = np.concatenate(groups); r = pd.Series(a).rank().values
    n, i, R = len(a), 0, []
    for g in groups:
        R.append(r[i:i + len(g)].sum()); i += len(g)
    return 12 / (n * (n + 1)) * sum(x ** 2 / len(g) for x, g in zip(R, groups)) - 3 * (n + 1)


def _perm(df, col, by, nperm=10000, seed=0):
    d = df[[col, by]].dropna()
    cats = sorted(d[by].unique())
    if len(cats) < 2: return None, None
    obs = _kruskal([d.loc[d[by] == k, col].values for k in cats])
    v, lab = d[col].values, d[by].values
    rng = np.random.default_rng(seed)
    null = np.array([_kruskal([v[p == k] for k in cats])
                     for p in (rng.permutation(lab) for _ in range(nperm))])
    return float(obs), float((null >= obs).mean())

# ================================ tools ====================================

def list_variables():
    """List every variable in the cohort, with example values for high-cardinality fields."""
    num = DATA.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in DATA.columns if c not in num]
    out = {"n_samples": len(DATA), "numeric_variables": num, "categorical_variables": {}}
    for c in cat:
        u = sorted(map(str, DATA[c].dropna().unique()))
        e = {"n_levels": int(len(u))}
        if len(u) <= 12:
            e["levels"] = u
            e["counts"] = DATA[c].value_counts().to_dict()
        else:
            e["example_values"] = u[:6] + ["..."] + u[-3:]
        out["categorical_variables"][c] = e
    return out


def derive_variable(new_name, from_column, method, regex=None):
    """Create a new variable from an existing one, so it can be tested.
    method: 'first_char' | 'last_chars' | 'regex' | 'prefix_alpha' | 'numeric_part'
    Use this when an identifier looks like it encodes structure (e.g. A01, B07)."""
    global DATA
    if from_column not in DATA.columns:
        return {"error": f"no such column: {from_column}"}
    s = DATA[from_column].astype(str)
    if method == "first_char":        v = s.str[0]
    elif method == "last_chars":      v = s.str[1:]
    elif method == "prefix_alpha":    v = s.str.extract(r"^([A-Za-z]+)")[0]
    elif method == "numeric_part":    v = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    elif method == "regex":
        if not regex: return {"error": "regex method needs a regex with one capture group"}
        v = s.str.extract(regex)[0]
    else:
        return {"error": "method must be first_char|last_chars|regex|prefix_alpha|numeric_part"}
    DATA[new_name] = v
    u = sorted(map(str, pd.Series(v).dropna().unique()))
    return {"created": new_name, "from": from_column, "method": method,
            "n_levels": len(u), "levels": u[:12],
            "counts": pd.Series(v).value_counts().to_dict() if len(u) <= 12 else "many"}


def describe(variable, by=None):
    """Summary of a numeric variable, optionally split by a categorical one."""
    if variable not in DATA.columns: return {"error": f"no such variable: {variable}"}
    if by is None:
        s = DATA[variable]
        return {"n": int(s.count()), "median": float(s.median()), "mean": float(s.mean()),
                "min": float(s.min()), "max": float(s.max())}
    g = DATA.groupby(by)[variable].agg(["count", "median", "mean", "min", "max"])
    return {"by": by, "table": json.loads(g.round(3).to_json(orient="index"))}


def crosstab(var1, var2):
    """Cross-tabulate two categorical variables. Detects complete separation."""
    ct = pd.crosstab(DATA[var1], DATA[var2])
    out = {"table": json.loads(ct.to_json(orient="index"))}
    pure = [str(i) for i in ct.index if (ct.loc[i] > 0).sum() == 1]
    lv = list(ct.columns); sep = []
    for i, a in enumerate(lv):
        for b in lv[i + 1:]:
            if not [x for x in ct.index if ct.loc[x, a] > 0 and ct.loc[x, b] > 0]:
                sep.append(f"{a} and {b} share NO level of {var1}")
    out["levels_with_only_one_group"] = f"{len(pure)}/{len(ct.index)}"
    if sep:
        out["COMPLETE_SEPARATION"] = sep
        out["consequence"] = (f"{var1} and {var2} are completely confounded for these pairs. "
                              "No adjustment can separate them: conditioning on "
                              f"{var1} removes the {var2} effect BY CONSTRUCTION. "
                              "This is a design flaw, not an analysis choice. Do not "
                              "attempt condition_and_retest or stratified_permutation "
                              "on this pair and report the p-value as evidence.")
    return out


def test_association(variable, grouping, nperm=10000):
    """Kruskal-Wallis with permutation null. Does `variable` differ by `grouping`?"""
    if variable not in DATA.columns: return {"error": f"no such variable: {variable}"}
    if grouping not in DATA.columns: return {"error": f"no such grouping: {grouping}"}
    H, p = _perm(DATA, variable, grouping, nperm)
    if H is None: return {"error": "grouping has <2 levels"}
    return {"variable": variable, "grouping": grouping, "H": round(H, 3),
            "p_permutation": round(p, 4), "n_permutations": nperm,
            "note": f"p<{1/nperm}" if p == 0 else ""}


def condition_and_retest(variable, grouping, covariate, nperm=10000):
    """Remove the effect of `covariate` from `variable`, then retest against `grouping`.
    This is how you tell a real effect from a confounded one."""
    for c in (variable, grouping, covariate):
        if c not in DATA.columns: return {"error": f"no such column: {c}"}
    d = DATA[[variable, grouping, covariate]].dropna().copy()

    # VALIDITY GUARD: if grouping is nested in covariate, conditioning removes
    # the effect by construction and the "after" p-value is meaningless.
    if d[covariate].dtype == object or d[covariate].nunique() < 15:
        ct = pd.crosstab(d[covariate], d[grouping]); lv = list(ct.columns)
        sep = [f"{a}/{b}" for i, a in enumerate(lv) for b in lv[i+1:]
               if not [x for x in ct.index if ct.loc[x, a] > 0 and ct.loc[x, b] > 0]]
        if sep:
            return {"variable": variable, "grouping": grouping,
                    "conditioned_on": covariate, "VALID": False,
                    "error": "CANNOT CONDITION - complete separation",
                    "separated_pairs": sep,
                    "why": (f"These levels of '{grouping}' share no level of "
                            f"'{covariate}'. Conditioning removes the effect by "
                            "construction, so a null result would prove nothing."),
                    "what_this_means": "Confounded by design. No adjustment can fix it."}

    if d[covariate].dtype == object:
        C = pd.get_dummies(d[covariate], drop_first=True).values.astype(float)
    else:
        C = d[[covariate]].values.astype(float)
    X = np.column_stack([np.ones(len(d)), C])
    beta, *_ = np.linalg.lstsq(X, d[variable].values.astype(float), rcond=None)
    d["_resid"] = d[variable].values - X @ beta
    H0, p0 = _perm(d, variable, grouping, nperm)
    H1, p1 = _perm(d, "_resid", grouping, nperm)
    return {"variable": variable, "grouping": grouping, "conditioned_on": covariate,
            "before": {"H": round(H0, 3), "p": round(p0, 4)},
            "after": {"H": round(H1, 3), "p": round(p1, 4)},
            "interpretation": ("effect SURVIVES conditioning" if p1 < 0.05
                               else "effect DISAPPEARS after conditioning - likely confounded")}


def stratified_permutation(variable, grouping, stratum, nperm=10000):
    """Permute labels WITHIN each level of `stratum`. The correct test when
    batch structure exists."""
    for c in (variable, grouping, stratum):
        if c not in DATA.columns: return {"error": f"no such column: {c}"}
    d = DATA[[variable, grouping, stratum]].dropna().copy()

    # VALIDITY GUARD: a stratum containing one group contributes NOTHING to a
    # within-stratum permutation. If most strata are pure the statistic rests
    # on a fraction of the data and the p-value is not interpretable.
    npg = d.groupby(stratum)[grouping].nunique()
    pure = npg[npg <= 1].index.tolist()
    mixed = npg[npg > 1].index.tolist()
    n_eff = int(d[d[stratum].isin(mixed)].shape[0])
    if len(mixed) == 0 or n_eff < 0.5 * len(d):
        return {"variable": variable, "grouping": grouping, "stratum": stratum,
                "VALID": False,
                "error": "DEGENERATE TEST - refusing to return a p-value",
                "pure_strata": f"{len(pure)}/{len(npg)} levels of "
                               f"'{stratum}' contain only one '{grouping}'",
                "effective_n": f"{n_eff}/{len(d)} samples in mixed strata",
                "why": ("Permuting labels within a pure stratum changes nothing. "
                        "Any p-value here would rest only on the mixed strata. "
                        f"With {n_eff}/{len(d)} samples this is not interpretable."),
                "what_this_means": (f"'{grouping}' is largely nested within '{stratum}'. "
                                    "That is complete or near-complete confounding: a "
                                    "design flaw no test can resolve. Report the "
                                    "confounding, not a p-value.")}

    cats = sorted(d[grouping].unique())
    d["_r"] = d[variable] - d.groupby(stratum)[variable].transform("median")
    obs = _kruskal([d.loc[d[grouping] == k, "_r"].values for k in cats])
    rng = np.random.default_rng(0); null = []
    for _ in range(nperm):
        lab = d.groupby(stratum, group_keys=False)[grouping].apply(
            lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        null.append(_kruskal([d.loc[lab == k, "_r"].values for k in cats]))
    p = float((np.array(null) >= obs).mean())
    return {"variable": variable, "grouping": grouping, "stratum": stratum,
            "VALID": True, "H": round(obs, 3), "p_stratified": round(p, 4),
            "pure_strata": f"{len(pure)}/{len(npg)}",
            "effective_n": f"{n_eff}/{len(d)}",
            "note": "labels shuffled within strata, so batch structure is preserved"}


def simulate_null(n_per_group, n_features=3000, k_selected=50, leaky=True, nsim=100):
    """What AUROC does PURE NOISE give at this design? Set leaky=True to put
    feature selection outside the CV loop (the common mistake)."""
    ns = [int(x) for x in n_per_group]
    y = np.concatenate([np.full(n, i) for i, n in enumerate(ns)])
    def F(X, y):
        n, p = X.shape; gm = X.mean(0); ssb = np.zeros(p); ssw = np.zeros(p)
        for c in np.unique(y):
            Xc = X[y == c]; m = Xc.mean(0)
            ssb += len(Xc) * (m - gm) ** 2; ssw += ((Xc - m) ** 2).sum(0)
        return (ssb / (len(ns) - 1)) / (ssw / (n - len(ns)) + 1e-12)
    def sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    def fit(X, y, it=250):
        n, p = X.shape; W = np.zeros((p, len(ns))); b = np.zeros(len(ns))
        Y = np.zeros((n, len(ns))); Y[np.arange(n), y] = 1
        for _ in range(it):
            P = sm(X @ W + b); W -= .5 * (X.T @ (P - Y) / n + W / n); b -= .5 * (P - Y).mean(0)
        return W, b
    def auc1(s, l):
        pos = l == 1; a, b = pos.sum(), (~pos).sum()
        o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        return (r[pos].sum() - a * (a + 1) / 2) / (a * b)
    res = []
    for s in range(nsim):
        rng = np.random.default_rng(s)
        X = rng.standard_normal((len(y), n_features))
        P = np.zeros((len(y), len(ns)))
        sel_all = np.argsort(F(X, y))[::-1][:k_selected] if leaky else None
        idx = rng.permutation(len(y))
        for k in range(5):
            te = idx[k::5]; tr = np.setdiff1d(idx, te)
            sel = sel_all if leaky else np.argsort(F(X[tr], y[tr]))[::-1][:k_selected]
            mu, sd = X[np.ix_(tr, sel)].mean(0), X[np.ix_(tr, sel)].std(0) + 1e-9
            W, b = fit((X[np.ix_(tr, sel)] - mu) / sd, y[tr])
            P[te] = sm(((X[np.ix_(te, sel)] - mu) / sd) @ W + b)
        res.append(np.mean([auc1(P[:, c], (y == c).astype(int)) for c in range(len(ns))]))
    return {"design": f"n={len(y)} {ns}, {n_features} PURE NOISE features, top {k_selected}",
            "selection": "OUTSIDE cv (leaky)" if leaky else "inside cv (correct)",
            "mean_AUROC_on_noise": round(float(np.mean(res)), 3),
            "sd": round(float(np.std(res)), 3),
            "pct_above_0.90": round(float((np.array(res) > .9).mean() * 100), 1),
            "nsim": nsim}



def evidence_sufficiency(k, n, target=0.95, conf=0.95):
    """Is n large enough to support a claim? Wilson interval on k/n, and the n
    required for the lower bound to exceed `target`. Use before accepting any
    performance claim, especially a perfect one."""
    from math import sqrt
    z = 1.959963984540054 if conf == 0.95 else 2.5758293035489004
    def wilson(k, n):
        if n == 0: return (0.0, 1.0)
        p = k / n; d = 1 + z*z/n
        c = (p + z*z/(2*n)) / d
        h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
        return (max(0.0, c-h), min(1.0, c+h))
    lo, hi = wilson(k, n)
    need = None
    for m in range(n, 20001):
        if wilson(m, m)[0] >= target: need = m; break
    out = {"observed": f"{k}/{n} = {k/n:.1%}",
           "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
           "conf": conf, "target_lower_bound": target,
           "supported": bool(lo >= target)}
    if lo >= target:
        out["verdict"] = f"n={n} supports a claim of >={target:.0%}: lower bound {lo:.1%}"
    else:
        out["worst_case_failure_rate"] = round(1-lo, 4)
        out["n_required_if_all_correct"] = need
        out["verdict"] = (f"n={n} does NOT support >={target:.0%}. Point estimate {k/n:.0%} "
                          f"but lower bound {lo:.1%}, admitting a failure rate of "
                          f"{100*(1-lo):.0f} per 100. Approximately {need} observations, "
                          f"all correct, would be required.")
    return out


def operating_points(prob_col, label_col, positive_labels, critical_label=None):
    """Sweep decision thresholds. For each: how many flagged, how many of the
    critical class are missed, how many negatives spared. `critical_label` is
    the class a miss is unacceptable for (e.g. cancer)."""
    import numpy as np
    for c in (prob_col, label_col):
        if c not in DATA.columns: return {"error": f"no such column: {c}"}
    p = DATA[prob_col].values.astype(float)
    pos = DATA[label_col].isin(positive_labels).values
    crit = (DATA[label_col] == critical_label).values if critical_label else pos
    rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        s = p >= t
        rows.append({"threshold": round(float(t), 2),
                     "flagged": int(s.sum()),
                     "critical_missed": int((~s & crit).sum()),
                     "negatives_spared": int((~s & ~pos).sum())})
    zero = [r for r in rows if r["critical_missed"] == 0]
    best = max(zero, key=lambda r: r["negatives_spared"]) if zero else None
    return {"n": int(len(p)), "n_critical": int(crit.sum()), "n_negative": int((~pos).sum()),
            "table": rows,
            "best_zero_miss_point": best,
            "note": ("At least one threshold detects every critical case. Its point estimate "
                     "is favourable; check evidence_sufficiency before accepting it."
                     if best else "Every threshold misses critical cases.")}


def decision_curve(prob_col, label_col, positive_labels):
    """Net benefit (Vickers & Elkin) versus treat-all and treat-none.
    WARNING returned: net benefit assumes the threshold encodes an acceptable
    exchange rate between a missed case and an unnecessary procedure. That
    assumption fails where a miss is fatal and the procedure is minor."""
    import numpy as np
    p = DATA[prob_col].values.astype(float)
    pos = DATA[label_col].isin(positive_labels).values
    N = len(p); rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        s = p >= t
        tp = int((s & pos).sum()); fp = int((s & ~pos).sum())
        nb = tp/N - fp/N * (t/(1-t))
        nb_all = pos.sum()/N - (~pos).sum()/N * (t/(1-t))
        rows.append({"threshold": round(float(t),2), "net_benefit": round(nb,3),
                     "treat_all": round(nb_all,3), "treat_none": 0.0,
                     "model_best": bool(nb > max(nb_all, 0))})
    wins = [r["threshold"] for r in rows if r["model_best"]]
    return {"table": rows,
            "model_superior_between": (min(wins), max(wins)) if wins else None,
            "WARNING": ("Net benefit treats a missed case and an unnecessary procedure as "
                        "exchangeable at the odds implied by the threshold. Where a miss is "
                        "fatal and the procedure is minor, no exchange rate is acceptable and "
                        "a favourable curve does not license deployment. Check "
                        "evidence_sufficiency on the critical class before concluding.")}


def check_protocol(library_selection=None, kit=None, flowcell=None, molecule=None,
                   data_available=None, intended_analysis=None):
    """Given library metadata, what is measurable? Returns constraints implied by
    the chemistry, independent of any analysis already run."""
    ls = (library_selection or "").lower(); mol = (molecule or "").lower()
    da = (data_available or "").lower(); ia = (intended_analysis or "").lower()
    c = []
    if "cdna" in ls:
        c.append({"claim": "base modifications (m6A, pseudouridine)", "possible": False,
                  "why": "reverse transcription copies a modified base as an ordinary one; "
                         "the modification never reaches the pore. Requires direct RNA."})
        c.append({"claim": "read strand orientation", "possible": "conditional",
                  "why": "cDNA is unoriented unless a primer-aware step (e.g. Pychopper) "
                         "orients it. Probe reads for primer motifs before choosing "
                         "minimap2 -uf (oriented) versus -ub (unoriented). -uf on "
                         "unoriented reads mis-strands roughly half of them, silently."})
    if "direct rna" in ls or "direct-rna" in ls:
        c.append({"claim": "base modifications", "possible": True,
                  "why": "native RNA translocates; modifications perturb the signal. "
                         "Requires signal-level data (POD5/FAST5), not basecalled reads."})
    if da and ("fastq" in da or "basecall" in da):
        c.append({"claim": "any signal-level analysis", "possible": False,
                  "why": "only basecalled reads are available; the raw signal is not "
                         "recoverable from FASTQ."})
    if "polya" in mol:
        c.append({"claim": "non-polyadenylated RNA (most lncRNA, histone mRNA)", "possible": False,
                  "why": "polyA selection depletes it."})
    if flowcell and "r9" in str(flowcell).lower():
        c.append({"note": "R9 chemistry: expect ~Q10-12; thresholds tuned for R10 will "
                          "discard most reads."})
    if flowcell and "r10" in str(flowcell).lower():
        c.append({"note": "R10 chemistry: expect ~Q15-20; a Q7 threshold filters almost nothing."})
    blocked = [x for x in c if x.get("possible") is False]
    out = {"constraints": c, "n_blocked": len(blocked)}
    if ia:
        hit = [x for x in blocked if any(w in ia for w in x["claim"].lower().split()[:2])]
        out["intended_analysis"] = intended_analysis
        out["intended_analysis_supported"] = not hit
        if hit: out["verdict"] = f"NOT SUPPORTED: {hit[0]['why']}"
    return out


def detection_vs_technical(detection_col, technical_col, biological_col, nperm=10000):
    """Does feature detection track a technical axis rather than biology? Gene
    detection scales with depth; if depth is run-determined, apparent complexity
    differences are technical."""
    a = test_association(detection_col, technical_col, nperm)
    b = test_association(detection_col, biological_col, nperm)
    if "error" in a: return a
    if "error" in b: return b
    tech_wins = a["p_permutation"] < b["p_permutation"]
    return {"detection_by_technical": a, "detection_by_biological": b,
            "verdict": ("Detection tracks the technical axis more strongly than the "
                        "biological one. Any difference in apparent complexity must be "
                        "modelled with the technical axis, or tested after rarefying to "
                        "equal depth." if tech_wins else
                        "Detection tracks the biological axis more strongly, but the "
                        "technical axis should still enter the model.")}


# --- Stage 5: alternative causes ------------------------------------------
# Known artefact signatures in plasma cfRNA. Each names a molecular proxy that
# should co-vary with the signal IF the artefact explains it. Absence of
# co-variation is evidence against that cause; it does not establish the
# intended one.
ARTEFACT_SIGNATURES = {
  "mitochondrial": {
    "genes": ["ENSG00000198695","ENSG00000198840","ENSG00000198938","ENSG00000210082",
              "ENSG00000198786","ENSG00000198712","ENSG00000198727","ENSG00000198888",
              "ENSG00000198886","ENSG00000198763","ENSG00000211459","ENSG00000198804",
              "ENSG00000228253","ENSG00000198899","ENSG00000212907"],
    "alternatives": [
      {"cause": "haemolysis / red-cell lysis",
       "proxy_genes": ["ENSG00000244734","ENSG00000206172","ENSG00000188536",
                       "ENSG00000223609","ENSG00000213934","ENSG00000196565",
                       "ENSG00000086506","ENSG00000130656","ENSG00000229988"],
       "proxy_name": "haemoglobin transcripts (HBB, HBA1/2, HBD, HBG1/2, HBQ1, HBZ, HBM)",
       "logic": "lysed erythrocytes release both mitochondrial and haemoglobin RNA; "
                "if haemolysis drives the signal the two must co-vary"},
      {"cause": "platelet activation",
       "proxy_genes": ["ENSG00000005961","ENSG00000102804","ENSG00000163736"],
       "proxy_name": "platelet transcripts (ITGA2B, TSC22D1, PPBP)",
       "logic": "platelets are mitochondria-rich and lyse readily in plasma"},
      {"cause": "leukocyte lysis",
       "proxy_genes": ["ENSG00000010610","ENSG00000081237","ENSG00000170458"],
       "proxy_name": "leukocyte transcripts (CD4, PTPRC, CD14)",
       "logic": "buffy-coat contamination introduces nucleated-cell RNA"},
    ],
    "not_testable_here": ["delayed plasma separation (needs processing time)",
                          "freeze-thaw cycles (needs handling log)"],
  },
  "intergenic": {
    "genes": [],
    "alternatives": [{"cause": "genomic DNA carryover", "proxy_genes": [],
                      "proxy_name": "intron/exon ratio; monoexonic fraction",
                      "logic": "gDNA maps uniformly, without splice structure"}],
    "not_testable_here": ["DNase treatment efficiency (needs protocol detail)"],
  },
}


def check_alternative_causes(signal, group_col, signal_genes=None, nperm=10000):
    """Stage 5. Given a signal, enumerate known alternative causes and test each
    against a molecular proxy in the data. `signal` is a key of
    ARTEFACT_SIGNATURES ("mitochondrial", "intergenic") or "custom" with
    signal_genes supplied. Requires a gene x sample matrix loaded via
    load_expression(). Absence of co-variation argues against a cause; it does
    not establish the intended one."""
    import numpy as np
    if EXPR is None:
        return {"error": "no expression matrix loaded; call load_expression(path) first"}
    sig = ARTEFACT_SIGNATURES.get(signal)
    if sig is None and signal_genes is None:
        return {"error": f"unknown signal '{signal}'; known: {list(ARTEFACT_SIGNATURES)}, "
                         "or pass signal_genes"}
    base = {g.split(".")[0]: g for g in EXPR.index}
    want = signal_genes or sig["genes"]
    have = [base[g] for g in want if g in base]
    if len(have) < 2:
        return {"error": f"only {len(have)} of {len(want)} signal genes present"}
    cpm = np.log2(EXPR / EXPR.sum(0) * 1e6 + 1)
    sigvec = cpm.loc[have].mean(0)
    d = DATA.set_index(DATA.columns[0]).loc[cpm.columns] if DATA.columns[0] in ("srr","sample") else DATA
    out = {"signal": signal, "signal_genes_found": f"{len(have)}/{len(want)}", "tested": []}
    work = DATA.copy()
    work["_sig"] = [float(sigvec[s]) for s in work[work.columns[0]]]
    H0, p0 = _perm(work, "_sig", group_col, nperm)
    out["signal_by_group"] = {"H": round(H0,2), "p": round(p0,4)}
    for alt in (sig["alternatives"] if sig else []):
        ph = [base[g] for g in alt["proxy_genes"] if g in base]
        if len(ph) < 1:
            out["tested"].append({"cause": alt["cause"], "testable": False,
                                  "why": "proxy genes not detected in this matrix"})
            continue
        pv = cpm.loc[ph].mean(0)
        work["_proxy"] = [float(pv[s]) for s in work[work.columns[0]]]
        r = float(np.corrcoef(work["_sig"], work["_proxy"])[0,1])
        X = np.column_stack([np.ones(len(work)), work["_proxy"].values])
        b, *_ = np.linalg.lstsq(X, work["_sig"].values, rcond=None)
        work["_resid"] = work["_sig"].values - X @ b
        H1, p1 = _perm(work, "_resid", group_col, nperm)
        Hp, pp = _perm(work, "_proxy", group_col, nperm)
        excluded = abs(r) < 0.3 and p1 < 0.05
        out["tested"].append({
            "cause": alt["cause"], "proxy": alt["proxy_name"],
            "proxy_genes_found": f"{len(ph)}/{len(alt['proxy_genes'])}",
            "logic": alt["logic"],
            "corr_signal_proxy": round(r, 3),
            "proxy_by_group_p": round(pp, 4),
            "signal_by_group_after_conditioning_p": round(p1, 4),
            "excluded": bool(excluded),
            "verdict": (f"EXCLUDED: correlation {r:+.3f} with the proxy, and the signal "
                        f"survives conditioning on it (p={p1:.4f})." if excluded else
                        f"NOT EXCLUDED: correlation {r:+.3f}; signal after conditioning "
                        f"p={p1:.4f}. This cause may contribute.")})
    out["not_testable_with_available_data"] = sig["not_testable_here"] if sig else []
    ex = [t for t in out["tested"] if t.get("excluded")]
    ne = [t for t in out["tested"] if t.get("testable") is not False and not t.get("excluded")]
    out["summary"] = (f"{len(ex)} of {len(out['tested'])} testable alternatives excluded. "
                      + (f"Still live: {[t['cause'] for t in ne]}. " if ne else "")
                      + (f"Not testable without additional metadata: "
                         f"{out['not_testable_with_available_data']}. " if sig and sig["not_testable_here"] else "")
                      + "Excluding an alternative does not establish the intended cause.")
    return out


EXPR = None

def load_expression(path):
    """Load a gene x sample expression matrix (rows genes, columns sample IDs)
    so alternative causes can be tested against molecular proxies."""
    global EXPR
    import pandas as pd
    EXPR = pd.read_csv(path, sep=None, engine="python", index_col=0)
    return {"loaded": path, "genes": int(EXPR.shape[0]), "samples": int(EXPR.shape[1]),
            "note": "check_alternative_causes is now available"}


# --- literature: PubMed E-utilities, no key required -----------------------
_PM = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_LAST = [0.0]

def _throttle():
    import time
    gap = time.time() - _LAST[0]
    if gap < 0.35: time.sleep(0.35 - gap)   # NCBI: <=3 req/s without a key
    _LAST[0] = time.time()


def search_literature(query, n=5, years=None, abstracts=True):
    """Search PubMed. Use to check whether a claim is already established, or
    whether a known alternative cause has been reported. Returns PMIDs so every
    claim you make can be attributed. Do not assert a biological fact you have
    not retrieved."""
    import urllib.request, urllib.parse, json, re
    try:
        term = query if not years else f"{query} AND {years}[dp]"
        _throttle()
        u = _PM + "esearch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "term": term, "retmax": int(n), "retmode": "json",
             "sort": "relevance"})
        ids = json.load(urllib.request.urlopen(u, timeout=25))["esearchresult"]["idlist"]
        if not ids:
            return {"query": term, "n_hits": 0, "results": [],
                    "note": "no hits; the claim may be novel, or the query too narrow"}
        _throttle()
        u = _PM + "esummary.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        s = json.load(urllib.request.urlopen(u, timeout=25))["result"]
        out = []
        for i in ids:
            r = s.get(i, {})
            au = r.get("authors", [])
            out.append({"pmid": i,
                        "title": r.get("title", "").rstrip("."),
                        "first_author": au[0]["name"] if au else "",
                        "journal": r.get("source", ""),
                        "year": (r.get("pubdate", "") or "")[:4],
                        "doi": next((x["value"] for x in r.get("articleids", [])
                                     if x.get("idtype") == "doi"), None),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/"})
        if abstracts:
            _throttle()
            u = _PM + "efetch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"})
            xml = urllib.request.urlopen(u, timeout=25).read().decode("utf-8", "ignore")
            for rec, o in zip(re.split(r"<PubmedArticle>", xml)[1:], out):
                txt = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", rec, re.S))
                o["abstract"] = re.sub(r"<[^>]+>", "", txt)[:900] or None
        return {"query": term, "n_hits": len(out), "results": out,
                "note": "cite by PMID; do not state a biological claim you did not retrieve"}
    except Exception as e:
        return {"error": f"PubMed unreachable: {e}",
                "advice": "state the claim as unverified rather than asserting it"}


def check_claim_against_literature(claim, search_terms, n=5):
    """Test whether a biological claim has support. Returns the evidence found
    and an explicit statement of what was NOT found, so an absence of support is
    reported rather than silently ignored."""
    hits = []
    for t in (search_terms if isinstance(search_terms, list) else [search_terms]):
        r = search_literature(t, n=n)
        if "error" in r: return r
        hits.append({"term": t, "n_hits": r["n_hits"], "results": r["results"]})
    total = sum(h["n_hits"] for h in hits)
    empty = [h["term"] for h in hits if h["n_hits"] == 0]
    return {"claim": claim, "searches": hits, "total_hits": total,
            "terms_with_no_support": empty,
            "verdict": ("No literature retrieved for: " + ", ".join(empty) +
                        ". Report the claim as unsupported by retrieved evidence."
                        if empty and total == 0 else
                        f"{total} records retrieved across {len(hits)} searches. "
                        "Read them before asserting support; relevance is not "
                        "established by retrieval alone.")}


def effect_size(variable, grouping):
    """Statistical significance is not biological significance. Returns effect
    magnitude, not just a p-value: Cohen's d, fold change, and the overlap
    between groups. A tiny effect in a large sample is significant and may be
    meaningless; check this before calling anything a biomarker."""
    import numpy as np
    if variable not in DATA.columns: return {"error": f"no such variable: {variable}"}
    if grouping not in DATA.columns: return {"error": f"no such grouping: {grouping}"}
    d = DATA[[variable, grouping]].dropna()
    cats = sorted(d[grouping].unique())
    if len(cats) < 2: return {"error": "grouping has <2 levels"}
    def auc1(a, b):
        s = np.concatenate([a, b]); l = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
        o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
        return (r[l == 1].sum() - len(a)*(len(a)+1)/2) / (len(a)*len(b))
    out = []
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            x = d.loc[d[grouping] == a, variable].values.astype(float)
            y = d.loc[d[grouping] == b, variable].values.astype(float)
            sp = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1))
                         / max(len(x)+len(y)-2, 1))
            cd = (y.mean() - x.mean()) / (sp + 1e-12)
            a_ = auc1(y, x)
            mag = ("negligible" if abs(cd) < 0.2 else "small" if abs(cd) < 0.5
                   else "medium" if abs(cd) < 0.8 else "large")
            out.append({"comparison": f"{b} vs {a}",
                        "cohens_d": round(float(cd), 3), "magnitude": mag,
                        "median_ratio": round(float(np.median(y) / (np.median(x) + 1e-12)), 3),
                        "prob_superiority": round(float(a_), 3),
                        "overlap_pct": round(float(100 * (1 - abs(2*a_ - 1))), 1),
                        "n": [int(len(x)), int(len(y))]})
    worst = max(out, key=lambda o: o["overlap_pct"])
    best = min(out, key=lambda o: o["overlap_pct"])
    return {"variable": variable, "grouping": grouping, "comparisons": out,
            "note": ("prob_superiority is the chance a random member of one group exceeds "
                     "a random member of the other; 0.5 is no separation. overlap_pct is "
                     "the proportion of the distributions that coincide."),
            "verdict": (f"Largest separation: {best['comparison']} (d={best['cohens_d']}, "
                        f"{best['overlap_pct']}% overlap, {best['magnitude']}). "
                        f"Weakest: {worst['comparison']} ({worst['overlap_pct']}% overlap). "
                        + ("Groups overlap substantially; a significant p-value here does not "
                           "imply a usable biomarker." if worst["overlap_pct"] > 60 else
                           "Separation is appreciable."))}


def check_reproducibility(variable, grouping, stratum=None, n_splits=200, seed=0):
    """Is the finding reproducible, or an artefact of this particular sample?
    Repeatedly splits the cohort in half, tests the effect in each half, and
    reports how often it replicates. If `stratum` is given, splits respect it so
    halves are balanced on the technical axis."""
    import numpy as np
    for c in [variable, grouping] + ([stratum] if stratum else []):
        if c not in DATA.columns: return {"error": f"no such column: {c}"}
    d = DATA[[variable, grouping] + ([stratum] if stratum else [])].dropna().reset_index(drop=True)
    cats = sorted(d[grouping].unique())
    rng = np.random.default_rng(seed)
    both = one = neither = 0
    ps = []
    for _ in range(n_splits):
        if stratum:
            idx = []
            for _s, sub in d.groupby(stratum):
                i = sub.index.values.copy(); rng.shuffle(i); idx.append(i[:len(i)//2])
            h1 = np.concatenate(idx)
        else:
            i = d.index.values.copy(); rng.shuffle(i); h1 = i[:len(i)//2]
        h2 = np.setdiff1d(d.index.values, h1)
        res = []
        for h in (h1, h2):
            s = d.loc[h]
            if s[grouping].nunique() < 2: res.append(1.0); continue
            _, p = _perm(s, variable, grouping, nperm=400, seed=int(rng.integers(1e6)))
            res.append(p)
        ps.append(res)
        sig = sum(1 for p in res if p < 0.05)
        both += sig == 2; one += sig == 1; neither += sig == 0
    ps = np.array(ps)
    return {"variable": variable, "grouping": grouping, "stratified_by": stratum,
            "n_splits": n_splits,
            "replicated_in_both_halves": f"{both}/{n_splits} = {both/n_splits:.0%}",
            "one_half_only": f"{one}/{n_splits} = {one/n_splits:.0%}",
            "neither_half": f"{neither}/{n_splits} = {neither/n_splits:.0%}",
            "median_p_across_halves": round(float(np.median(ps)), 4),
            "verdict": (f"Replicates in both halves {both/n_splits:.0%} of splits. "
                        + ("Reproducible within this cohort." if both/n_splits > .5 else
                           "NOT reliably reproducible even within this cohort: the effect "
                           "depends on which samples are included. Within-cohort split-half "
                           "is a weak test; an independent cohort remains necessary."))}


REFERENCE = None

def fit_reference(technical_features, provenance, biological_features=None,
                  healthy_group=None, group_col=None, out=None):
    """Build an Evidence Reference Model from a cohort.

    A reference is only valid for the experimental conditions that produced it,
    exactly as a clinical reference interval is valid only for the assay that
    established it. `provenance` is therefore REQUIRED and is recorded with the
    distributions:

        {"platform": "PromethION", "protocol": "cDNA",
         "library_kit": "SQK-NBD-114.24", "flowcell": "R10.4",
         "basecaller": "dorado 0.7", "reference_genome": "GRCh38",
         "annotation": "GENCODE 39", "aligner": "minimap2 2.31"}

    Technical features take their reference from ALL samples: adequacy is a
    property of the assay. Biological features take theirs from the healthy
    group only: elevation is defined against health."""
    import numpy as np, json as _j
    global REFERENCE
    if DATA is None: return {"error": "no cohort loaded"}
    need = ["protocol", "flowcell"]
    missing = [k for k in need if not provenance.get(k)]
    if missing:
        return {"error": f"provenance must record {missing}. A reference without "
                         "provenance cannot be applied to any sample, because "
                         "there is no way to know whether it applies."}
    ref = {"provenance": dict(provenance), "n_cohort": int(len(DATA)),
           "technical": {}, "biological": {}}
    for f in technical_features:
        if f not in DATA.columns: return {"error": f"no such column: {f}"}
        v = DATA[f].dropna().values.astype(float)
        ref["technical"][f] = {"source": "all samples", "n": int(len(v)),
                               "median": float(np.median(v)),
                               "p05": float(np.percentile(v, 5)),
                               "p95": float(np.percentile(v, 95)),
                               "min_observed": float(v.min()), "max_observed": float(v.max())}
    if biological_features:
        if not (healthy_group and group_col):
            return {"error": "biological_features require healthy_group and group_col"}
        sub = DATA[DATA[group_col] == healthy_group]
        if len(sub) < 5:
            return {"error": f"only {len(sub)} healthy samples; too few for a reference"}
        for f in biological_features:
            v = sub[f].dropna().values.astype(float)
            ref["biological"][f] = {"source": f"{group_col}=={healthy_group}", "n": int(len(v)),
                                    "median": float(np.median(v)),
                                    "p05": float(np.percentile(v, 5)),
                                    "p95": float(np.percentile(v, 95)),
                                    "sd": float(v.std(ddof=1))}
    REFERENCE = ref
    if out: _j.dump(ref, open(out, "w"), indent=1); ref["saved_to"] = out
    ref["note"] = (f"Evidence Reference Model built from n={ref['n_cohort']} under "
                   f"{provenance.get('protocol')} / {provenance.get('flowcell')}. "
                   "It applies ONLY to samples produced under compatible conditions. "
                   "check_sample enforces this.")
    return ref


# Which provenance mismatches invalidate a reference, and which merely warn.
_FATAL = {
  "protocol":  "direct RNA and cDNA measure different molecules; base modifications, "
               "strand orientation and length distributions are not comparable",
  "flowcell":  "chemistry generations differ in accuracy and length; a Q or length "
               "reference from one does not describe the other",
}
_WARN = {
  "library_kit":      "kit affects yield and fragment recovery",
  "basecaller":       "basecaller version shifts the quality distribution",
  "platform":         "throughput differs; per-sample yield references may not transfer",
  "annotation":       "annotation version changes which features are detectable",
  "reference_genome": "assembly differences affect mapping",
  "aligner":          "aligner and parameters affect mapping rate",
}


def _norm(v):
    s = str(v or "").lower().strip()
    if "direct" in s and "rna" in s: return "direct_rna"
    if "cdna" in s: return "cdna"
    if s.startswith("r9"):  return "r9"
    if s.startswith("r10"): return "r10"
    return s


def check_provenance(sample_provenance, reference=None):
    """Does this reference apply to this sample at all? Runs before any value is
    compared. A reference built under different conditions cannot judge a
    sample, however good the sample is."""
    import json as _j
    ref = reference or REFERENCE
    if isinstance(ref, str): ref = _j.load(open(ref))
    if not ref: return {"error": "no reference; call fit_reference first"}
    rp = ref.get("provenance", {})
    fatal, warn, unknown = [], [], []
    for k, why in _FATAL.items():
        a, b = _norm(rp.get(k)), _norm(sample_provenance.get(k))
        if not b: unknown.append(f"{k} not stated for the sample")
        elif a and a != b:
            fatal.append({"field": k, "reference": rp.get(k), "sample": sample_provenance.get(k),
                          "why": why})
    for k, why in _WARN.items():
        a, b = _norm(rp.get(k)), _norm(sample_provenance.get(k))
        if a and b and a != b:
            warn.append({"field": k, "reference": rp.get(k), "sample": sample_provenance.get(k),
                         "why": why})
    ok = not fatal
    return {"applicable": ok,
            "reference_conditions": rp, "sample_conditions": dict(sample_provenance),
            "incompatible": fatal, "differences": warn, "unstated": unknown,
            "verdict": ("Reference does not apply to this sample. "
                        + "; ".join(f"{f['field']}: reference {f['reference']}, sample "
                                    f"{f['sample']} ({f['why']})" for f in fatal)
                        + ". Build a reference under matching conditions, or state that "
                          "no reference exists for this assay."
                        if fatal else
                        "Reference applies." + (f" {len(warn)} non-fatal difference(s); "
                        "interpret with caution." if warn else "")
                        + (f" Unstated: {unknown}. Absent provenance is assumed compatible, "
                           "which is an assumption, not a finding." if unknown else ""))}


def check_sample(values, sample_provenance=None, reference=None):
    """Judge ONE sample against the Evidence Reference Model. Clinical mode: not
    whether a finding is real, but whether THIS result is trustworthy.

    Compatibility is checked FIRST. A reference built under different conditions
    cannot judge this sample, however good the sample is."""
    import numpy as np, json as _j
    ref = reference or REFERENCE
    if isinstance(ref, str): ref = _j.load(open(ref))
    if not ref: return {"error": "no reference; call fit_reference first"}
    if sample_provenance is not None:
        comp = check_provenance(sample_provenance, ref)
        if not comp.get("applicable"):
            return {"verdict": "NOT_APPLICABLE", "provenance_check": comp,
                    "action": "DO NOT COMPARE. " + comp["verdict"],
                    "why": "The reference describes a different assay. Judging this "
                           "sample against it would produce a confident verdict about "
                           "the wrong experiment."}
    else:
        comp = {"applicable": None,
                "verdict": "No sample provenance supplied. Applicability is ASSUMED, "
                           "not established."}
    flags, notes = [], []
    for f, v in values.items():
        v = float(v)
        if f in ref["technical"]:
            r = ref["technical"][f]
            if v < r["p05"]:
                flags.append({"feature": f, "value": v, "severity": "INVALID",
                              "reason": f"below the 5th percentile of the cohort "
                                        f"({v:.0f} vs p05 {r['p05']:.0f}); the assay "
                                        f"did not work on this sample"})
            elif v > r["p95"]:
                notes.append(f"{f} above the 95th percentile ({v:.0f} vs {r['p95']:.0f}); "
                             "unusually high, not disqualifying")
        elif f in ref["biological"]:
            r = ref["biological"][f]
            z = (v - r["median"]) / (r["sd"] + 1e-9)
            pos = ("elevated" if v > r["p95"] else "reduced" if v < r["p05"] else "within")
            notes.append(f"{f} = {v:.2f}, {pos} the healthy reference "
                         f"({r['p05']:.2f}-{r['p95']:.2f}), z={z:+.2f}")
        else:
            notes.append(f"{f}: no reference; cannot be judged")
    bad = [f for f in flags if f["severity"] == "INVALID"]
    verdict = "INVALID" if bad else ("BORDERLINE" if flags else "VALID")
    return {"verdict": verdict, "reference_n": ref["n_cohort"],
            "reference_conditions": ref.get("provenance", {}),
            "provenance_check": comp.get("verdict"),
            "failures": flags, "observations": notes,
            "action": ("DO NOT REPORT A RESULT. Repeat the draw. A low-yield sample "
                       "produces a confident answer that reflects the assay, not the "
                       "patient." if verdict == "INVALID" else
                       "Sample is within reference. This licenses reporting a result; "
                       "it does not validate the model that produces it."),
            "caveat": (f"The reference derives from n={ref['n_cohort']}. Percentiles "
                       "from a small cohort are themselves uncertain, and a sample may "
                       "be unusual for reasons the reference never observed.")}

TOOLS = {"list_variables": list_variables, "derive_variable": derive_variable,
         "describe": describe, "crosstab": crosstab,
         "test_association": test_association, "condition_and_retest": condition_and_retest,
         "stratified_permutation": stratified_permutation, "simulate_null": simulate_null,
         "evidence_sufficiency": evidence_sufficiency, "operating_points": operating_points,
         "decision_curve": decision_curve, "check_protocol": check_protocol,
         "detection_vs_technical": detection_vs_technical,
         "load_expression": load_expression,
         "check_alternative_causes": check_alternative_causes,
         "search_literature": search_literature,
         "check_claim_against_literature": check_claim_against_literature,
         "effect_size": effect_size,
         "check_reproducibility": check_reproducibility,
         "fit_reference": fit_reference,
         "check_sample": check_sample,
         "check_provenance": check_provenance}

SCHEMA = [
 {"name": "list_variables", "description": "List every variable in the cohort with types, level counts, and example values. Call this first.",
  "input_schema": {"type": "object", "properties": {}}},
 {"name": "derive_variable", "description": "Create a new variable from an existing column so it can be tested. Use when an identifier appears to encode structure (e.g. sample names like A01, B07 may encode plate row and column).",
  "input_schema": {"type": "object", "properties": {"new_name": {"type": "string"}, "from_column": {"type": "string"},
    "method": {"type": "string", "enum": ["first_char", "last_chars", "regex", "prefix_alpha", "numeric_part"]},
    "regex": {"type": "string"}}, "required": ["new_name", "from_column", "method"]}},
 {"name": "describe", "description": "Summarise a numeric variable, optionally split by a categorical one.",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "by": {"type": "string"}}, "required": ["variable"]}},
 {"name": "crosstab", "description": "Cross-tabulate two categorical variables. Use to check whether they are confounded.",
  "input_schema": {"type": "object", "properties": {"var1": {"type": "string"}, "var2": {"type": "string"}}, "required": ["var1", "var2"]}},
 {"name": "test_association", "description": "Kruskal-Wallis with a permutation null. Does a numeric variable differ across levels of a categorical one?",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "grouping": {"type": "string"}}, "required": ["variable", "grouping"]}},
 {"name": "condition_and_retest", "description": "Remove a covariate's effect from a variable, then retest against a grouping. This distinguishes a real effect from a confounded one.",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "grouping": {"type": "string"}, "covariate": {"type": "string"}}, "required": ["variable", "grouping", "covariate"]}},
 {"name": "stratified_permutation", "description": "Permute group labels WITHIN levels of a stratum, preserving batch structure. The correct test when batch structure exists.",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "grouping": {"type": "string"}, "stratum": {"type": "string"}}, "required": ["variable", "grouping", "stratum"]}},
 {"name": "evidence_sufficiency", "description": "Is n large enough to support a performance claim? Wilson confidence interval on k/n and the sample size required. Call this before accepting ANY performance claim, especially a perfect one.",
  "input_schema": {"type": "object", "properties": {"k": {"type": "integer"}, "n": {"type": "integer"}, "target": {"type": "number"}}, "required": ["k", "n"]}},
 {"name": "operating_points", "description": "Sweep decision thresholds: how many flagged, how many critical cases missed, how many negatives spared at each.",
  "input_schema": {"type": "object", "properties": {"prob_col": {"type": "string"}, "label_col": {"type": "string"}, "positive_labels": {"type": "array", "items": {"type": "string"}}, "critical_label": {"type": "string"}}, "required": ["prob_col", "label_col", "positive_labels"]}},
 {"name": "decision_curve", "description": "Net benefit versus treat-all and treat-none across thresholds. Returns a warning about the exchange-rate assumption net benefit makes.",
  "input_schema": {"type": "object", "properties": {"prob_col": {"type": "string"}, "label_col": {"type": "string"}, "positive_labels": {"type": "array", "items": {"type": "string"}}}, "required": ["prob_col", "label_col", "positive_labels"]}},
 {"name": "check_protocol", "description": "Given library metadata, what is measurable? Returns constraints implied by the chemistry, independent of any analysis already run.",
  "input_schema": {"type": "object", "properties": {"library_selection": {"type": "string"}, "kit": {"type": "string"}, "flowcell": {"type": "string"}, "molecule": {"type": "string"}, "data_available": {"type": "string"}, "intended_analysis": {"type": "string"}}}},
 {"name": "detection_vs_technical", "description": "Does feature detection track a technical axis rather than biology?",
  "input_schema": {"type": "object", "properties": {"detection_col": {"type": "string"}, "technical_col": {"type": "string"}, "biological_col": {"type": "string"}}, "required": ["detection_col", "technical_col", "biological_col"]}},
 {"name": "load_expression", "description": "Load a gene x sample expression matrix so alternative causes can be tested against molecular proxies.",
  "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
 {"name": "check_alternative_causes", "description": "Stage 5. Given a signal, enumerate known alternative causes (haemolysis, platelet activation, leukocyte lysis, gDNA carryover) and test each against a molecular proxy in the data. Excluding a cause does not establish the intended one.",
  "input_schema": {"type": "object", "properties": {"signal": {"type": "string"}, "group_col": {"type": "string"}, "signal_genes": {"type": "array", "items": {"type": "string"}}}, "required": ["signal", "group_col"]}},
 {"name": "search_literature", "description": "Search PubMed and return titles, journals, years, DOIs, PMIDs and abstracts. Use to check whether a claim is established or an alternative cause has been reported. Never assert a biological fact you have not retrieved.",
  "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "n": {"type": "integer"}, "years": {"type": "string"}}, "required": ["query"]}},
 {"name": "check_claim_against_literature", "description": "Test whether a biological claim has published support. Reports explicitly what was NOT found, so absence of support is stated rather than ignored.",
  "input_schema": {"type": "object", "properties": {"claim": {"type": "string"}, "search_terms": {"type": "array", "items": {"type": "string"}}}, "required": ["claim", "search_terms"]}},
 {"name": "effect_size", "description": "Statistical significance is not biological significance. Returns Cohen's d, fold change, and the overlap between groups. Call this before describing any significant result as a biomarker.",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "grouping": {"type": "string"}}, "required": ["variable", "grouping"]}},
 {"name": "check_reproducibility", "description": "Is the finding reproducible or an artefact of this sample? Repeated split-half testing. Pass `stratum` to keep halves balanced on a technical axis.",
  "input_schema": {"type": "object", "properties": {"variable": {"type": "string"}, "grouping": {"type": "string"}, "stratum": {"type": "string"}}, "required": ["variable", "grouping"]}},
 {"name": "fit_reference", "description": "Build an Evidence Reference Model from a cohort. Provenance (protocol, flowcell, kit, basecaller) is REQUIRED and is recorded with the distributions, because a reference is only valid for the conditions that produced it.",
  "input_schema": {"type": "object", "properties": {"technical_features": {"type": "array", "items": {"type": "string"}}, "provenance": {"type": "object"}, "biological_features": {"type": "array", "items": {"type": "string"}}, "healthy_group": {"type": "string"}, "group_col": {"type": "string"}}, "required": ["technical_features", "provenance"]}},
 {"name": "check_provenance", "description": "Does this reference apply to this sample at all? Runs before any value is compared. Protocol and flowcell mismatches are disqualifying.",
  "input_schema": {"type": "object", "properties": {"sample_provenance": {"type": "object"}}, "required": ["sample_provenance"]}},
 {"name": "check_sample", "description": "Judge ONE sample against the Evidence Reference Model. Compatibility is checked first. Returns NOT_APPLICABLE, INVALID, BORDERLINE or VALID.",
  "input_schema": {"type": "object", "properties": {"values": {"type": "object"}, "sample_provenance": {"type": "object"}}, "required": ["values"]}},
 {"name": "simulate_null", "description": "Simulate AUROC on PURE NOISE at a given design, with feature selection inside or outside the CV loop. Tells you what chance looks like.",
  "input_schema": {"type": "object", "properties": {"n_per_group": {"type": "array", "items": {"type": "integer"}}, "n_features": {"type": "integer"}, "k_selected": {"type": "integer"}, "leaky": {"type": "boolean"}}, "required": ["n_per_group"]}},
]

SYSTEM = """You are reviewing a genomics cohort before publication.

You have tools that RUN statistical tests. Do not speculate about what should
be checked - check it. Call tools until you can answer with evidence.

You are not asked a question. You are given an analysis and its evidence, and
must test the claims the analysis implicitly makes by proceeding:
  the groups are comparable · the analysis suits the chemistry · the samples are
  usable · the effect is real · this cause explains it · this action is warranted
Each may be false. Test each. Abstain where the evidence does not support it.

Principles:
- Any metadata field could encode a technical batch (plate, run, date, kit).
  If a variable's level structure looks like processing order or position,
  test it against the signal before believing the biology.
- A biological effect that vanishes when you condition on a technical variable
  was never biological.
- Report actual numbers from the tools, never remembered ones.
- A favourable point estimate is not sufficiency. Call evidence_sufficiency
  before accepting any performance claim, especially a perfect one.
- Never assert a biological fact from memory. Retrieve it with
  search_literature and cite the PMID, or state that it is unverified.
- If the evidence says the finding is an artefact, say so plainly.

Finish with a verdict: is the claimed biomarker trustworthy? Cite the numbers
your tools returned."""

# ================================ agent ====================================

def build():
    m = pd.read_csv("metadata/full.tsv", sep="\t")
    f = pd.read_csv("lengths/fragmentomics_summary.tsv", sep="\t")
    d = m.merge(f, on="srr", how="inner", suffixes=("", "_y"))
    d = d[[c for c in d.columns if not c.endswith("_y")]]
    keep = ["srr", "title", "group", "n_reads", "mean_len", "median_len", "n50",
            "frac_lt_100", "frac_gt_500", "iqr", "mean_qual"]
    d = d[[c for c in keep if c in d.columns]]
    d.to_csv("nora_data.tsv", sep="\t", index=False)
    print(f"wrote nora_data.tsv  {d.shape}")
    print("NOTE: 'title' holds plate positions (A01..D12). The agent is NOT told this.")
    print(d.head().to_string(index=False))


def ask(question, data_path, model, max_turns=25):
    global DATA
    import anthropic
    DATA = pd.read_csv(data_path, sep="\t")
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content":
             f"Cohort loaded: {len(DATA)} samples, columns: {list(DATA.columns)}\n\n{question}"}]
    for turn in range(max_turns):
        r = client.messages.create(model=model, max_tokens=2000, system=SYSTEM,
                                   tools=SCHEMA, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            print("\n" + "=" * 70 + "\nVERDICT\n" + "=" * 70)
            for b in r.content:
                if b.type == "text": print(b.text)
            return
        results = []
        for b in r.content:
            if b.type == "text" and b.text.strip():
                print(f"\n[think] {b.text.strip()[:300]}")
            if b.type == "tool_use":
                print(f"[tool ] {b.name}({json.dumps(b.input)})")
                try:
                    out = TOOLS[b.name](**b.input)
                except Exception as e:
                    out = {"error": str(e)}
                print(f"[  ->  ] {json.dumps(out)[:300]}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out)})
        msgs.append({"role": "user", "content": results})
    print("\n!! hit max turns")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--ask")
    p.add_argument("--data", default="nora_data.tsv")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    if a.build: build()
    elif a.ask: ask(a.ask, a.data, a.model)
    else: p.print_help()

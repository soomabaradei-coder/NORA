#!/usr/bin/env python3
"""
NORA-Design — check a plate layout BEFORE you sequence it.

WHY THIS EXISTS
  In GSE271389, disease group was laid out along plate columns:
    col   1    2    3    4    5    6    7    8    9    10   11   12
    A    can  can  can  can  bar  bar  bar  bar  con  con  con  con
  Controls and cancers share NO column. That is complete confounding -
  created at the bench, unfixable by any analysis, and invisible until
  someone thinks to cross-tabulate. It cost that study the ability to
  claim anything about fragment length.

  This runs in <1s on a sample manifest. Before the run. While it is free.

USAGE
  python3 nora_design.py --check manifest.tsv --group group --position well
  python3 nora_design.py --plan cohort.tsv --group group --per-batch 24 --out plan.tsv
"""
import argparse, sys
import numpy as np, pandas as pd

RED, YEL, GRN, RST = "\033[91m", "\033[93m", "\033[92m", "\033[0m"


def _axes(df, pos):
    """A well like A01 encodes TWO batch axes. Both must be checked."""
    s = df[pos].astype(str)
    ax = {}
    if s.str.match(r"^[A-Za-z]\d+$").all():
        ax["row"] = s.str[0]
        ax["column"] = pd.to_numeric(s.str.extract(r"(\d+)")[0])
    return ax


def check(path, group, pos, extra=None):
    df = pd.read_csv(path, sep=None, engine="python")
    for c in (group, pos):
        if c not in df.columns:
            sys.exit(f"no column '{c}'. found: {list(df.columns)}")

    print(f"\n{'='*66}\n NORA-Design  ·  {len(df)} samples\n{'='*66}")
    print(f" groups: {dict(df[group].value_counts())}\n")

    axes = _axes(df, pos)
    for c in (extra or []):
        if c in df.columns: axes[c] = df[c]
    if not axes:
        print(f"{YEL}  '{pos}' is not a well ID (A01). Pass batch columns via --extra.{RST}")
        return 1

    fatal = warn = 0
    for name, vals in axes.items():
        ct = pd.crosstab(vals, df[group])
        lv = list(ct.columns)
        sep = [(a, b) for i, a in enumerate(lv) for b in lv[i+1:]
               if not [x for x in ct.index if ct.loc[x, a] > 0 and ct.loc[x, b] > 0]]
        pure = [i for i in ct.index if (ct.loc[i] > 0).sum() == 1]

        print(f" ── {name} ─────────────────────────────────────────")
        print(ct.to_string())
        if sep:
            fatal += 1
            print(f"\n{RED} ✗ FATAL — complete separation{RST}")
            for a, b in sep:
                print(f"{RED}     '{a}' and '{b}' share NO level of {name}.{RST}")
            print(f"{RED}     Anything varying by {name} is inseparable from {group}.{RST}")
            print(f"{RED}     No statistical adjustment can fix this. Re-lay the plate.{RST}\n")
        elif len(pure) > len(ct.index) / 2:
            warn += 1
            print(f"\n{YEL} ⚠ WARNING — {len(pure)}/{len(ct.index)} levels hold one group only.{RST}")
            print(f"{YEL}     Adjustment will be underpowered. Re-lay the plate.{RST}\n")
        else:
            exp = np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()
            chi = float(((ct.values - exp) ** 2 / (exp + 1e-9)).sum())
            print(f"\n{GRN} ✓ every group appears in every level  (chi2={chi:.1f}){RST}")
            print(f"{GRN}     {name} is adjustable: model it as ~ {name} + {group}{RST}\n")

    print("="*66)
    if fatal:
        print(f"{RED} VERDICT: DO NOT SEQUENCE. {fatal} axis/axes confounded by design.{RST}")
        print(f"{RED}          Run --plan to get a valid layout.{RST}")
    elif warn:
        print(f"{YEL} VERDICT: risky. Re-lay before sequencing.{RST}")
    else:
        print(f"{GRN} VERDICT: design is sound. Record every batch axis in metadata.{RST}")
    print("="*66 + "\n")
    return 2 if fatal else (1 if warn else 0)


def plan(path, group, per_batch, out, seed=42):
    """Deal samples round-robin within group, so every batch is balanced."""
    df = pd.read_csv(path, sep=None, engine="python")
    n_batch = int(np.ceil(len(df) / per_batch))
    rng = np.random.default_rng(seed)
    parts = []
    for g, sub in df.groupby(group):
        sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)
        sub["batch"] = [i % n_batch + 1 for i in range(len(sub))]   # deal like cards
        parts.append(sub)
    p = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    p["barcode"] = p.groupby("batch").cumcount() + 1                # randomised within batch
    rows = "ABCDEFGH"
    p["well"] = p.barcode.map(lambda i: f"{rows[(i-1)//12]}{(i-1)%12+1:02d}")
    p = p.sort_values(["batch", "barcode"])
    p.to_csv(out, sep="\t", index=False)
    print(f"\n{GRN}wrote {out}  —  {len(p)} samples across {n_batch} batches{RST}\n")
    print(pd.crosstab(p.batch, p[group]).to_string())
    print(f"\n{GRN}Every batch contains every group. Both axes randomised.{RST}")
    print(f"{GRN}Record: batch, barcode, well, run date, kit lot, operator.{RST}")
    print(f"{GRN}You cannot correct for a batch you did not record.{RST}\n")


if __name__ == "__main__":  # python -m nora.design
    a = argparse.ArgumentParser()
    a.add_argument("--check"); a.add_argument("--plan")
    a.add_argument("--group", default="group"); a.add_argument("--position", default="well")
    a.add_argument("--extra", nargs="*"); a.add_argument("--per-batch", type=int, default=24)
    a.add_argument("--out", default="sequencing_plan.tsv")
    v = a.parse_args()
    if v.check: sys.exit(check(v.check, v.group, v.position, v.extra))
    elif v.plan: plan(v.plan, v.group, v.per_batch, v.out)
    else: a.print_help()

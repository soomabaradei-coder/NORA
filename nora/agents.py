"""
NORA — six sub-agents, one orchestrator.

Each sub-agent guards one stage at which a cfRNA analysis can fail. It sees only
the tools relevant to its stage, answers one question, and returns a verdict it
must justify with numbers its tools returned. Findings pass forward, so a later
stage knows what an earlier one established.

    python -m nora.agents --data data/nora_data.tsv --expr counts.tsv
    python -m nora.agents --data data/nora_data.tsv --only design validity
"""
import json, re, sys, argparse, importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_na", Path(__file__).parent / "agent.py")
na = importlib.util.module_from_spec(_spec)
sys.modules["_na"] = na
_spec.loader.exec_module(na)

MODEL = "claude-sonnet-5"

BASE = """You are one stage of a validation framework. You do not perform the
analysis; you test whether the evidence supports a claim that has already been
made by proceeding with it.

You share a workspace with the other stages. Variables they derived are present
in the data and you may use them directly. Their numbers are given to you; do
not recompute what is already established, but do re-test anything you doubt.

Rules:
- Use your tools. Do not speculate about what should be checked; check it.
- Report only numbers your tools returned. Never a remembered one.
- If a tool refuses to return a result, that refusal IS the finding.
- You may abstain. Abstention is correct when the evidence does not support the
  claim, and it is not a failure.

Finish with a JSON block, nothing after it:
{"verdict": "SUPPORTED" | "ABSTAIN" | "REJECT",
 "claim_tested": "<the claim in one line>",
 "evidence": ["<number and what it means>", ...],
 "reasoning": "<2-3 sentences>",
 "findings": {"technical_axes": ["<column names that encode batch>"],
              "confounded_with_group": ["<axis completely confounded, if any>"],
              "adjustable_axes": ["<axis balanced enough to condition on>"],
              "key_numbers": {"<name>": "<value>"},
              "warnings": ["<what a later stage must not assume>"]},
 "pass_forward": "<one line the next stage must know, or empty>"}

Omit any key of `findings` you did not establish. Do not invent entries."""

STAGES = [
 dict(id="design", n="1 DESIGN",
      claim="the groups are comparable",
      q=("Is this study answerable? Any metadata field may encode a technical batch. "
         "Identifiers may be structured; decompose them and test each axis against the "
         "biological grouping before anything else is believed."),
      tools=["list_variables","derive_variable","crosstab","describe","test_association"]),
 dict(id="protocol", n="2 PROTOCOL",
      claim="the analysis suits the chemistry",
      q=("Given the library metadata, what is measurable? State what the chemistry "
         "forecloses, independent of any analysis already run."),
      tools=["check_protocol","search_literature","list_variables"]),
 dict(id="quality", n="3 QUALITY",
      claim="the samples are usable",
      q=("Are these samples usable, and does technical structure drive detection? "
         "Distinguish a yield problem from a quality problem: they have different causes."),
      tools=["describe","test_association","detection_vs_technical","list_variables"]),
 dict(id="validity", n="4 VALIDITY",
      claim="the effect is real",
      q=("Does any apparent biological effect survive conditioning on the technical "
         "structure the Design stage identified? Calibrate what chance looks like at "
         "this design before interpreting any performance figure."),
      tools=["test_association","condition_and_retest","stratified_permutation",
             "simulate_null","derive_variable","crosstab"]),
 dict(id="interpretation", n="5 INTERPRETATION",
      claim="this cause explains the signal",
      q=("What else could produce this signal? Enumerate known alternative causes and "
         "test each against a molecular proxy in the data. Excluding a cause does not "
         "establish the intended one. Retrieve any biological claim you make."),
      tools=["load_expression","check_alternative_causes","search_literature",
             "check_claim_against_literature","test_association","effect_size",
             "check_reproducibility"]),
 dict(id="decision", n="6 DECISION",
      claim="this action is warranted",
      q=("Does the evidence support clinical action? A favourable point estimate is not "
         "sufficiency, and a favourable decision curve is not permission: net benefit "
         "assumes a missed case and an unnecessary procedure are exchangeable. Check "
         "whether n supports the claim before accepting any performance figure."),
      tools=["evidence_sufficiency","operating_points","decision_curve","describe"]),
]


def _schema(names):
    return [s for s in na.SCHEMA if s["name"] in names]


def _render(bb, original_cols):
    """The shared workspace, rendered for a sub-agent. Structured, not prose."""
    if not bb: return ""
    L = ["\nSHARED WORKSPACE (established by earlier stages)"]
    derived = [c for c in na.DATA.columns if c not in original_cols]
    if derived:
        L.append(f"  Variables derived by earlier stages and PRESENT IN THE DATA NOW:")
        for c in derived:
            u = na.DATA[c].dropna().unique()
            L.append(f"    {c}  ({len(u)} levels: {sorted(map(str,u))[:8]})"
                     f"{'  <- you may test against this directly' if len(u)<=12 else ''}")
    if bb.get("technical_axes"):
        L.append(f"  Technical axes identified: {sorted(set(bb['technical_axes']))}")
    if bb.get("confounded_with_group"):
        L.append(f"  COMPLETELY CONFOUNDED with the grouping: {sorted(set(bb['confounded_with_group']))}")
        L.append(f"    -> conditioning on these is meaningless; do not attempt it")
    if bb.get("adjustable_axes"):
        L.append(f"  Adjustable (balanced) axes: {sorted(set(bb['adjustable_axes']))}")
        L.append(f"    -> condition on these when testing whether an effect is real")
    if bb.get("key_numbers"):
        L.append("  Numbers already established:")
        for k, v in bb["key_numbers"].items(): L.append(f"    {k} = {v}")
    if bb.get("warnings"):
        L.append("  Warnings from earlier stages:")
        for w in sorted(set(bb["warnings"])): L.append(f"    ! {w}")
    if bb.get("verdicts"):
        L.append("  Verdicts so far:")
        for k, v in bb["verdicts"].items(): L.append(f"    {k}: {v}")
    return "\n".join(L) + "\n"


def run_stage(client, st, bb, original_cols, max_turns=14, verbose=True):
    sysmsg = (f"{BASE}\n\nYOUR STAGE: {st['n']}\n"
              f"THE CLAIM YOU MUST TEST: \"{st['claim']}\"\n"
              f"YOUR QUESTION: {st['q']}")
    user = f"Cohort: {len(na.DATA)} samples. Columns: {list(na.DATA.columns)}\n"
    user += _render(bb, original_cols)
    user += "\nTest your claim."
    msgs = [{"role": "user", "content": user}]
    calls = []
    for _ in range(max_turns):
        r = client.messages.create(model=MODEL, max_tokens=2000, system=sysmsg,
                                   tools=_schema(st["tools"]), messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            txt = "".join(b.text for b in r.content if b.type == "text")
            m = re.search(r"\{[\s\S]*\}", txt)
            try: v = json.loads(m.group(0))
            except Exception:
                v = {"verdict": "ERROR", "reasoning": txt[:400],
                     "evidence": [], "pass_forward": "", "claim_tested": st["claim"]}
            v["_tool_calls"] = calls
            return v
        res = []
        for b in r.content:
            if b.type == "tool_use":
                try: out = na.TOOLS[b.name](**b.input)
                except Exception as e: out = {"error": str(e)}
                calls.append(b.name)
                if verbose:
                    print(f"      {b.name}({json.dumps(b.input)[:90]})")
                res.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out)})
        msgs.append({"role": "user", "content": res})
    return {"verdict": "ERROR", "reasoning": "max turns", "evidence": [],
            "pass_forward": "", "claim_tested": st["claim"], "_tool_calls": calls}


def _merge(bb, st, v):
    """Fold a stage's structured findings into the shared workspace."""
    f = v.get("findings") or {}
    for k in ("technical_axes", "confounded_with_group", "adjustable_axes", "warnings"):
        if f.get(k): bb.setdefault(k, []).extend(
            f[k] if isinstance(f[k], list) else [f[k]])
    if f.get("key_numbers"):
        bb.setdefault("key_numbers", {}).update(
            {f"{st['id']}.{k}": val for k, val in f["key_numbers"].items()})
    if v.get("pass_forward"):
        bb.setdefault("warnings", []).append(f"[{st['n']}] {v['pass_forward']}")
    bb.setdefault("verdicts", {})[st["n"]] = v["verdict"]
    return bb


def orchestrate(data_path, expr_path=None, only=None, verbose=True):
    import pandas as pd, anthropic
    na.DATA = pd.read_csv(data_path, sep=None, engine="python")
    original_cols = list(na.DATA.columns)
    if expr_path: na.load_expression(expr_path)
    client = anthropic.Anthropic()
    stages = [s for s in STAGES if not only or s["id"] in only]
    bb, report = {}, []
    for st in stages:
        if verbose:
            print(f"\n{'='*72}\n  {st['n']}  ·  claim: \"{st['claim']}\"\n{'='*72}")
        v = run_stage(client, st, bb, original_cols, verbose=verbose)
        v["stage"] = st["n"]; v["stage_id"] = st["id"]
        report.append(v)
        if verbose:
            print(f"\n   -> {v['verdict']}")
            for e in v.get("evidence", [])[:4]: print(f"      · {e}")
            print(f"      {v.get('reasoning','')[:300]}")
            nd = [c for c in na.DATA.columns if c not in original_cols]
            if nd: print(f"      workspace now carries: {nd}")
        bb = _merge(bb, st, v)
    if verbose:
        print(f"\n{'='*72}\n  VALIDITY REPORT\n{'='*72}")
        for v in report:
            print(f"  {v['stage']:20s} {v['verdict']:10s} {v.get('claim_tested','')[:44]}")
        ab = [v for v in report if v["verdict"] in ("ABSTAIN", "REJECT")]
        print(f"\n  {len(ab)}/{len(report)} stages did not license their claim.")
        if any(v["stage_id"] == "decision" and v["verdict"] != "SUPPORTED" for v in report):
            print("  No clinical claim is licensed by this evidence.")
        derived = [c for c in na.DATA.columns if c not in original_cols]
        if derived: print(f"\n  variables derived during the run: {derived}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nora_data.tsv")
    p.add_argument("--expr")
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", default="nora_report.json")
    a = p.parse_args()
    rep = orchestrate(a.data, a.expr, a.only)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")

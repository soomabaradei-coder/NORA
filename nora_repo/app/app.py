"""
NORA - chat interface.

    pip install streamlit anthropic pandas numpy
    streamlit run app/app.py

Deploy free at share.streamlit.io: point it at this repo, set
ANTHROPIC_API_KEY in Secrets, and you have a public URL.
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

st.markdown("""
<style>
.stApp header {visibility:hidden}
.tool {background:#f6f8fa;border-left:3px solid #6e7781;padding:8px 12px;
       margin:6px 0;font-family:ui-monospace,monospace;font-size:12px}
.abstain {background:#fff5f5;border-left:3px solid #b30000;padding:8px 12px;
          margin:6px 0;font-size:13px;color:#7f0000}
</style>""", unsafe_allow_html=True)

st.title("NORA")
st.caption("Nanopore Oncology Reasoning Agent. It does not produce answers. "
           "It tests whether the evidence supports the claims your analysis already makes.")

with st.sidebar:
    st.subheader("Evidence")
    key = st.text_input("Anthropic API key", type="password",
                        value=os.environ.get("ANTHROPIC_API_KEY", ""))
    up = st.file_uploader("Cohort table (TSV/CSV)", type=["tsv", "csv", "txt"])
    demo = st.checkbox("Use GSE271389 demo cohort", value=True)
    if demo and not up:
        na.DATA = pd.read_csv(ROOT / "data" / "nora_data.tsv", sep="\t")
    elif up:
        na.DATA = pd.read_csv(up, sep=None, engine="python")
    if na.DATA is not None:
        st.success(f"{len(na.DATA)} samples · {len(na.DATA.columns)} variables")
        st.dataframe(na.DATA.head(6), height=180, use_container_width=True)

    st.divider()
    st.subheader("The six claims")
    st.markdown("""
Running an analysis asserts these, usually without anyone deciding to:

1. the groups are comparable
2. the analysis suits the chemistry
3. the samples are usable
4. the effect is real
5. this cause explains it
6. this action is warranted

Each may be false. NORA tests each.
""")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.msgs = []; st.session_state.log = []; st.rerun()

if "msgs" not in st.session_state: st.session_state.msgs = []
if "log" not in st.session_state: st.session_state.log = []

for entry in st.session_state.log:
    with st.chat_message(entry["role"]):
        for b in entry["blocks"]:
            if b["t"] == "text":
                st.markdown(b["v"])
            elif b["t"] == "tool":
                st.markdown(f'<div class="tool"><b>{b["name"]}</b>({b["args"]})<br>'
                            f'→ {b["out"]}</div>', unsafe_allow_html=True)

if not st.session_state.log:
    st.info("Try: **Evaluate this cohort.** NORA is told nothing about the plate layout. "
            "Or paste a claim: *fragment length classifies disease at AUROC 0.70, p=0.006*.")

q = st.chat_input("Give NORA an analysis or a claim to validate")

if q:
    if not key:
        st.error("An Anthropic API key is required for the agent. "
                 "The tools below the chat run without one."); st.stop()
    if na.DATA is None:
        st.error("Load a cohort first."); st.stop()

    os.environ["ANTHROPIC_API_KEY"] = key
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    st.session_state.log.append({"role": "user", "blocks": [{"t": "text", "v": q}]})
    with st.chat_message("user"): st.markdown(q)

    st.session_state.msgs.append({"role": "user", "content":
        f"Cohort loaded: {len(na.DATA)} samples, columns: {list(na.DATA.columns)}\n\n{q}"})

    with st.chat_message("assistant"):
        blocks = []
        for turn in range(25):
            with st.spinner(f"checking… ({turn+1})"):
                r = client.messages.create(model=na.MODEL, max_tokens=2000,
                                           system=na.SYSTEM, tools=na.SCHEMA,
                                           messages=st.session_state.msgs)
            st.session_state.msgs.append({"role": "assistant", "content": r.content})
            if r.stop_reason != "tool_use":
                for b in r.content:
                    if b.type == "text":
                        st.markdown(b.text); blocks.append({"t": "text", "v": b.text})
                break
            results = []
            for b in r.content:
                if b.type == "text" and b.text.strip():
                    st.markdown(b.text); blocks.append({"t": "text", "v": b.text})
                if b.type == "tool_use":
                    try: out = na.TOOLS[b.name](**b.input)
                    except Exception as e: out = {"error": str(e)}
                    a = json.dumps(b.input); o = json.dumps(out)
                    st.markdown(f'<div class="tool"><b>{b.name}</b>({a[:120]})<br>'
                                f'→ {o[:400]}</div>', unsafe_allow_html=True)
                    blocks.append({"t": "tool", "name": b.name, "args": a[:120], "out": o[:400]})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out)})
            st.session_state.msgs.append({"role": "user", "content": results})
        st.session_state.log.append({"role": "assistant", "blocks": blocks})

st.divider()
st.subheader("Tools, without the model")
st.caption("Every check runs standalone. No API key needed.")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Is n enough?**")
    k = st.number_input("correct", 0, 10000, 19)
    n = st.number_input("total", 1, 10000, 19)
    tgt = st.slider("required lower bound", .50, .99, .95)
    if st.button("Check sufficiency"):
        st.json(na.evidence_sufficiency(int(k), int(n), target=tgt))
with c2:
    st.markdown("**Is it batch?**")
    if na.DATA is not None:
        cols = list(na.DATA.columns)
        v = st.selectbox("measure", [c for c in cols if pd.api.types.is_numeric_dtype(na.DATA[c])])
        g = st.selectbox("grouping", [c for c in cols if not pd.api.types.is_numeric_dtype(na.DATA[c])])
        if st.button("Test association"):
            st.json(na.test_association(v, g, nperm=2000))
with c3:
    st.markdown("**What does chance look like?**")
    leaky = st.checkbox("selection outside CV (the common error)", True)
    if st.button("Simulate null"):
        st.json(na.simulate_null([16, 12, 19], n_features=3000, k_selected=50,
                                 leaky=leaky, nsim=15))

# NORA web interface

## Local
```bash
pip install streamlit anthropic pandas numpy
streamlit run app/app.py
```

## Public URL, free
1. Push this repo to GitHub
2. share.streamlit.io → New app → pick the repo → main file `app/app.py`
3. Settings → Secrets → `ANTHROPIC_API_KEY = "sk-..."`
4. Deploy. You get `https://nora-nanoporeoncologyreasoningagent.streamlit.app`

The three panels below the chat run every check **without an API key**, so a
reviewer can reproduce the finding in a browser with no credentials.

# 🧱 AI-Powered SCC Mix Design Studio

A Streamlit web app that uses machine learning to predict Self-Compacting
Concrete (SCC) compressive strength and to reverse-engineer candidate mix
designs for a target strength — plus an admin retraining workflow and an
LLM-backed chatbot for SCC/concrete questions.

> **Live link:** this project is delivered as source code for you to run or
> deploy yourself (see [Deploying](#deploying-to-get-a-live-link) below) — I
> can't host a persistent public URL on your behalf. Streamlit Community
> Cloud (free) gets you a shareable `*.streamlit.app` link in a few minutes.

## Features

- **ML prediction** — XGBoost regressor (auto-falls back to scikit-learn's
  GradientBoostingRegressor if `xgboost` isn't installed) trained on
  Cement, Water, Fine Aggregate, Coarse Aggregate, Superplasticizer, Age.
- **Reverse mix design** — enter a target compressive strength and get 3
  diverse candidate mixes (Economical / Balanced / High-Performance) found
  via a constrained Monte-Carlo search over the trained model.
- **Admin Mode** — password-protected dataset upload, retraining, model
  save/load, and performance/diagnostic charts.
- **Interactive Plotly charts** everywhere (mix comparison, predicted vs.
  target, actual vs. predicted, feature importance, dataset distributions).
- **Downloads** — model metrics (CSV), predictions/mix designs (CSV), and
  every chart (PNG).
- **AI chatbot** — Anthropic Claude or OpenAI backed, with optional live
  DuckDuckGo web search grounding, chat history, and an offline FAQ fallback
  when no API key is configured.

## Project structure

```
scc_app/
├── app.py                        # Main Streamlit app (layout + tabs)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml.example      # copy to secrets.toml and fill in
├── data/
│   ├── sample_scc_data.csv       # bundled synthetic demo dataset
│   └── generate_sample_data.py   # script that generated it (reproducible)
├── models/                       # trained model saved here at runtime
└── src/
    ├── config.py                 # paths, schema, SCC domain bounds
    ├── data_utils.py             # load / validate / clean datasets
    ├── model_utils.py            # train / save / load / predict
    ├── optimization.py           # reverse mix-design search
    ├── visualization.py          # all Plotly chart builders
    └── chatbot.py                # LLM + web search + offline FAQ
```

## Quickstart

```bash
git clone <this-repo>        # or unzip the download
cd scc_app
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

**First run:** since no model has been saved yet, the app automatically
trains once on the bundled sample dataset and saves it to
`models/scc_model.joblib` — this takes a few seconds. Every run after that
loads the saved model instantly.

## Configuring secrets

Copy the template and fill in real values:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
ADMIN_PASSWORD = "your-own-password"     # default demo password is admin123
ANTHROPIC_API_KEY = "sk-ant-..."         # optional, pre-fills the chatbot
OPENAI_API_KEY = "sk-..."                # optional, pre-fills the chatbot
```

You can also just paste an API key directly into the Chatbot tab's
"⚙️ AI Configuration" panel each session — secrets.toml is only a convenience
so you don't have to re-enter it. **Without any key**, the chatbot still
works using a small built-in FAQ.

## Bringing your own dataset

Admin Mode → Upload Dataset accepts any CSV with these columns (extra
columns are ignored):

| Column               | Meaning                        | Typical units |
|----------------------|---------------------------------|----------------|
| `Cement`              | Cement content                  | kg/m³ |
| `Water`                | Water content                    | kg/m³ |
| `Fine_Aggregate`       | Fine aggregate content           | kg/m³ |
| `Coarse_Aggregate`     | Coarse aggregate content         | kg/m³ |
| `Superplasticizer`     | Superplasticizer content         | kg/m³ |
| `Age`                   | Curing/testing age                | days |
| `Compressive_Strength` | Measured compressive strength (target) | MPa |

Missing values are median-imputed automatically; rows with negative or
zero/negative Age are dropped. A short report is shown after upload.

> ⚠️ **The bundled `sample_scc_data.csv` is synthetic demo data**, generated
> from domain-informed formulas (see `data/generate_sample_data.py`) so the
> app is usable immediately — it is *not* real laboratory data. Replace it
> with your own experimental mix database for real use, and re-check the SCC
> proportioning bounds in `src/config.py` (`FEATURE_BOUNDS`,
> `WC_RATIO_BOUNDS`, etc.) still make sense for your data before trusting
> the reverse-design search.

## How reverse mix design works

For a target strength, the optimizer samples tens of thousands of candidate
mixes constrained to realistic SCC proportioning ranges (water/cement ratio,
total aggregate content, fine/coarse split, superplasticizer dosage as % of
cement), predicts each candidate's strength with the trained model in one
batch call, keeps the closest matches, and picks 3 diverse options (grouped
by cement content) rather than three near-identical mixes. If even the
closest match misses the target by more than 15%, that design is flagged as
not reliably achievable within the search's domain constraints.

## Deploying to get a live link

**Streamlit Community Cloud (free, easiest):**
1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → *New app* →
   point it at your repo, branch, and `app.py`.
3. In the app's **Settings → Secrets**, paste the contents of your
   `secrets.toml` (admin password, API keys).
4. Deploy — you'll get a `https://<something>.streamlit.app` URL.

Any other host that runs a long-lived Python process works too (Render,
Railway, Fly.io, an EC2/VM with `streamlit run app.py --server.port 80`,
etc.) — just make sure `requirements.txt` installs and secrets are set the
same way.

## Notes & limitations

- **PNG chart downloads** need the `kaleido` package (pinned to `0.2.1` in
  `requirements.txt` specifically because it bundles its own headless
  Chrome — newer Kaleido v1 requires a separately installed Chrome). If
  export ever fails in your environment, every chart is still fully
  interactive on-screen; the download button just won't appear.
- **XGBoost fallback** — if `xgboost` can't be installed in your
  environment, the app automatically trains a scikit-learn
  `GradientBoostingRegressor` instead and labels it clearly in the UI.
- **Chatbot API keys** are only used for that session's API calls and are
  never written to disk by the app itself.
- This is an educational/demo tool. Always validate final SCC mix designs
  with laboratory trial batches (slump-flow, L-box, V-funnel) and applicable
  standards (EFNARC, ACI 237R, EN 206, etc.) before real-world use.

## Tech stack

Python · Streamlit · XGBoost / scikit-learn · Pandas · Plotly · Anthropic /
OpenAI SDKs · `ddgs` (DuckDuckGo search) · joblib

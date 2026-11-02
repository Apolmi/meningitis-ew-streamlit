---
title: Meningitis Environmental Risk Early Warning
emoji: 🌡️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# Meningitis Environmental Risk Early Warning — research prototype

A Gradio demo of the stacked heterogeneous ensemble (ARNN + CatBoost, with ARNN +
CatBoost + SVM-RBF reported alongside it) from the thesis *A Climate-Based Machine
Learning Framework for Forecasting Meningitis Environmental Risk in Northeast
Nigeria: A Stacked Heterogeneous Ensemble Learning Approach* (Andrew Apolmi, AUN).

It forecasts, two weeks ahead and year-round, whether environmental conditions in
Adamawa, Borno or Yobe fall in the elevated MEWS-derived classification (Orange/Red).
**It does not forecast meningitis cases or outbreaks** — the target is a rule-based
function of climate and dust, not disease incidence.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (default `http://127.0.0.1:7860`).

## Data and model files (`data/`)

Copied from the final (v6) pipeline run's app bundle:

- `final_models.pkl` — the fitted base learners, scalers and per-stack meta-learners,
  thresholds and feature lists (13 model predictors; `epi_week` is display-only).
- `climate_weekly.csv` — weekly NASA POWER / MERRA-2 climate history by state
  (2018–2024), used by the "Historical week" input mode.
- `background.csv` — a background sample for the SHAP `KernelExplainer`.
- `shap_importance.csv` — the primary configuration's global SHAP ranking
  (sealed test).
- `stage20_tier_predictions.csv`, `stage20_tier_decision_map.csv` — sealed-test
  forecast history and the tier-to-decision-guidance mapping, used by the
  "History" tab.

**Do not retrain or re-pickle these files with a different scikit-learn/CatBoost
version** — `requirements.txt` is pinned to the exact versions the pipeline used
(see `run_manifest.json` in the pipeline output), or the pickled models may fail to
load or load with inconsistent internals.

## Live data mode

The "Live (NASA POWER)" input on the Prediction tab calls the public
[NASA POWER](https://power.larc.nasa.gov/) daily point API for temperature,
humidity, precipitation and wind, at an approximate centroid for the selected
state. Dust (`DUSMASS`) is a MERRA-2 product with a release delay unsuitable for a
low-latency public pull, so this mode substitutes the historical climatological
dust level for the state and calendar week — this is clearly labelled in the app,
not presented as a live observation. This mirrors a limitation already noted in the
thesis (Section 5.9 / Major 8): a genuinely live MERRA-2 feed would need a different,
faster-release data source.

This call needs outbound internet access, so it only works where the app is
actually deployed — it cannot be exercised from a network-restricted sandbox.

## Deployment (Hugging Face Spaces)

Hugging Face Spaces is the standard free host for Gradio apps (the Gradio
equivalent of Streamlit Community Cloud):

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   choosing the **Gradio** SDK.
2. Push this repo's contents to the Space's git remote (Spaces are themselves git
   repos), or connect the Space to this GitHub repo for automatic sync.
3. The `packages.txt` file installs `libgomp1` (needed by CatBoost); no other
   secrets or API keys are required — the NASA POWER API used here needs none.

The app can also run on any host that supports a `requirements.txt` + `app.py`
Python app (e.g. a plain VM or container running `python app.py`), since it no
longer depends on Streamlit Cloud specifically.

## Limits

The sealed test covers about two annual cycles; a weekly climatology baseline is
reported in the thesis for comparison. SHAP explanations describe the fitted model
and are not causal. This is a research prototype, not a production surveillance
tool, and does not replace professional epidemiological judgement.

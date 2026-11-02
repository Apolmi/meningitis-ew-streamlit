# Meningitis Environmental Risk Early Warning — research prototype

A Streamlit demo of the stacked heterogeneous ensemble (ARNN + CatBoost, with ARNN +
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
streamlit run app.py
```

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
actually deployed (e.g. Streamlit Community Cloud) — it cannot be exercised from a
network-restricted sandbox.

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (see below).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   that repo, branch `main`, and `app.py` as the entry point.
3. No secrets are required — the NASA POWER API used here needs no API key.

## Limits

The sealed test covers about two annual cycles; a weekly climatology baseline is
reported in the thesis for comparison. SHAP explanations describe the fitted model
and are not causal. This is a research prototype, not a production surveillance
tool, and does not replace professional epidemiological judgement.

# Processed data

Outputs of the v6 pipeline (`../pipeline/workflow_v6_clean_13predictors.py`),
used by the thesis analysis and by the deployed prototype:

- `climate_weekly.csv` — the state-week panel: NASA POWER / MERRA-2 derived
  climate variables, engineered predictors and the MEWS-derived target, for
  Borno, Yobe and Adamawa.
- `stage20_tier_predictions.csv`, `stage20_tier_decision_map.csv` — the
  four-tier probability-band outputs used for the dashboard.
- `stage13_shap_importance.csv` — the ranked SHAP values for the selected
  configuration (C1), reported in the thesis's explainability chapter.

No raw NASA POWER/MERRA-2 downloads or the fitted model binary are included
here; both can be regenerated from the pipeline script.

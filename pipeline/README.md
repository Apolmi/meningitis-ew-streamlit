# Pipeline

`workflow_v6_clean_13predictors.py` is the final (v6) training pipeline used
to produce the results reported in the thesis: thirteen model predictors
(epi_week is display-only), expanding-window `TimeSeriesSplit` validation
(7 outer splits, gap=6 rows), nested hyperparameter tuning of both the base
learners and the meta-learner within each outer fold, and the single
prespecified selection rule used to choose the primary configuration (C1).

Running it end to end (Stages 1-14) regenerates the app bundle used by the
deployed prototype at
[meningitis-ew.onrender.com](https://meningitis-ew.onrender.com/), including
the processed data and model artefacts in `../data_processed/`.

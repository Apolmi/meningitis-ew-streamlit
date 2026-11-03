"""
================================================================================
MENINGITIS ENVIRONMENTAL-RISK FORECASTING PIPELINE  --  version 6 (clean run)
Stacked heterogeneous ensemble: CatBoost + ARNN + SVM-RBF -> Elastic Net meta-learner
Target: MEWS-derived elevated environmental vigilance, two weeks ahead (Adamawa, Borno, Yobe)
================================================================================
Run in Google Colab (or locally) top to bottom:
    !pip install -q catboost optuna shap
Upload NASA_MERRA2_weekly_all_states.csv next to this script (columns: state, T2M_mean,
T2M_max, T2M_min, RH2M, PREC, WS10M, WD10M, date, DUSMASS, iso_week). Outputs go to ./outputs/.
Quick smoke test (a few minutes, NOT for reporting):  QUICK=1 python workflow_v6_clean_13predictors.py

WHAT CHANGED RELATIVE TO VERSION 5
  1. 13 model predictors. epi_week is NOT a model input. It is kept in the data as a display
     field (and for the calendar baselines) and is shown by the app only. The calendar enters the
     model through week_sin and week_cos.
  2. Hyperparameters are chosen ONLY on training rows, at every level:
       - base learners: tuned inside each outer fold on that fold's training rows (inner expanding
         windows, purged), as in v5;
       - meta-learner (Elastic Net l1_ratio, C): NOW ALSO NESTED, and EVERY evaluation fold has its own
         validation fold. For evaluation fold k the meta-learner is tuned on out-of-fold predictions of
         earlier folds only (training on folds < j, validating on fold j, for 2 <= j < k) and then fitted on
         all folds < k and scored on fold k. Fold 1 (training of the meta-learner only) and fold 2 (its
         first validation block) are therefore not scored, and stack evaluation covers folds 3 to N_OUTER.
         With N_OUTER = 7 this gives FIVE evaluation folds. (With N_OUTER = 5 it would give only three.)
       - final deployed models: tuned on the whole development set only. The sealed test is never used.
  3. Overfitting evidence is computed WITHOUT resubstitution optimism: train and validation AUC per
     outer fold use that fold's nested hyperparameters (Stage 9).
  4. Overfitting and robustness tests (Stage 10) run on DEVELOPMENT folds only:
       learning curve, predictor ablation, shifted-label and permuted-label controls.
  5. ARNN class balance: class-weighted loss through sample_weight (positive class weighted by n_neg/n_pos).
     No oversampling, so no duplicated rows reach MLPClassifier's internal early-stopping split.
     This needs scikit-learn >= 1.7 (checked below); the version is written to run_manifest.json.
  6. Stages 4 (EDA), 7 (fold-count sensitivity) and 22 (earlier sensitivity analyses) are removed.

LOCKED DESIGN (unchanged from v5)
  - Harmattan_Wind_Flag: WD10M 30-100 deg, WS10M > 3.0, RH2M < 40.
  - epi_week = ISO 8601 week of the Monday date (the raw iso_week column is not used).
  - Lags and rolling features are built within state (groupby state).
  - Rows are sorted by (date, state) for splitting, so every fold mixes the three states.
  - Development = weeks before 2024-01-01; sealed test = 2024-01-01 onward (opened once, Stage 12).
  - Forecast horizon 2 weeks -> purge of 6 rows (2 weeks x 3 states) between training and
    validation in every split, and removal of the last 6 development rows (their labels fall in
    the sealed-test period).
  - Selection rule (development data only): highest mean outer-fold PR-AUC; among stacks within
    one standard error of the best, the fewest base learners; ties by higher mean.
  - Cost-sensitive threshold: 5 x FN + 1 x FP minimised on the stack's out-of-fold predictions.
================================================================================
"""
import os, json, time, pickle, warnings, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import sklearn
assert tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 7), "scikit-learn >= 1.7 is required (MLPClassifier sample_weight). Run: pip install -U scikit-learn"
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, fbeta_score, matthews_corrcoef)
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------ configuration
QUICK = os.environ.get("QUICK", "0") == "1"
SEED = 42
DATA_FILE = os.environ.get("DATA_FILE", "NASA_MERRA2_weekly_all_states.csv")
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
SPLIT_DATE = "2024-01-01"
GAP_WEEKS = 2; N_STATES = 3; GAP_ROWS = GAP_WEEKS * N_STATES
N_OUTER = int(os.environ.get("N_OUTER", "7")); N_INNER = 3   # 7 outer splits -> 5 stack-evaluation folds, each with its own validation fold (see Stage 8)
TRIALS_FINAL = 2 if QUICK else 40        # whole-development tuning (final refit)
TRIALS_NESTED = 2 if QUICK else 20       # per learner per outer fold
TRIALS_META_INNER = 3 if QUICK else 25   # meta-learner, inside each evaluation fold
TRIALS_META_FINAL = 3 if QUICK else 40
N_BOOT = 100 if QUICK else 2000
BOOT_LENGTHS = [1, 3, 5, 8, 13]
COST_FN, COST_FP = 5, 1
MODEL_FEATURES = ["T2M_mean", "RH2M", "PREC", "WS10M", "DUSMASS_ugm3", "DTR", "Harmattan_Wind_Flag",
                  "week_sin", "week_cos", "DUSMASS_lag1", "RH2M_lag1", "DUSMASS_2wk_rolling_mean",
                  "PREC_2wk_rolling_sum"]
DISPLAY_FIELDS = ["epi_week"]                # shown in the app, never passed to a model
assert len(MODEL_FEATURES) == 13 and "epi_week" not in MODEL_FEATURES
BASE_COLS = {"C1_ARNN_CatBoost": ["arnn_proba", "catboost_proba"],
             "C2_ARNN_SVM": ["arnn_proba", "svm_proba"],
             "C3_CatBoost_SVM": ["catboost_proba", "svm_proba"],
             "C4_Full_Ensemble": ["catboost_proba", "arnn_proba", "svm_proba"]}
np.random.seed(SEED)
T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def jdump(obj, name): json.dump(obj, open(f"{OUT}/{name}", "w"), indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

# ------------------------------------------------------------------ metrics
def ece(y, p, n_bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float); edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def full_metrics(y, p, thr):
    yp = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier_score": brier_score_loss(y, p),
            "ece": ece(y, p), "f1": fbeta_score(y, yp, beta=1, zero_division=0), "f2": fbeta_score(y, yp, beta=2, zero_division=0),
            "mcc": float(matthews_corrcoef(y, yp)), "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0, "precision": precision_score(y, yp, zero_division=0),
            "false_negatives": int(fn), "false_positives": int(fp), "cost_5to1": float(COST_FN * fn + COST_FP * fp), "threshold": float(thr),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}

def threshold_sweep(y, p):
    """cost, sensitivity, specificity and Youden's J at every candidate threshold from 0.05 to 0.95"""
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        yp = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        rows.append({"threshold": float(t), "cost_5to1": float(COST_FN * fn + COST_FP * fp), "sensitivity": sens, "specificity": spec, "youden_j": sens + spec - 1})
    return pd.DataFrame(rows)
def best_threshold(y, p):
    sw = threshold_sweep(y, p); return float(sw.loc[sw["cost_5to1"].idxmin(), "threshold"])

# ==============================================================================
# STAGE 1-3  Data, features, MEWS target
# ==============================================================================
log("STAGE 1-3: data, features, MEWS target")
df = pd.read_csv(DATA_FILE)
try: df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
except Exception: df["date"] = pd.to_datetime(df["date"])
for s, sub in df.sort_values(["state", "date"]).groupby("state"):
    assert (sub["date"].dt.dayofweek == 0).all() and (sub["date"].diff().dropna() == pd.Timedelta(days=7)).all(), f"{s}: weekly continuity"
df["DUSMASS_ugm3"] = df["DUSMASS"] * 1e9
df = df.sort_values(["state", "date"]).reset_index(drop=True)
df["epi_week"] = df["date"].dt.isocalendar().week.astype(int)                  # display field
df["week_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52); df["week_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)
df["DTR"] = df["T2M_max"] - df["T2M_min"]
df["Harmattan_Wind_Flag"] = ((df["WD10M"] >= 30) & (df["WD10M"] <= 100) & (df["WS10M"] > 3.0) & (df["RH2M"] < 40)).astype(int)
g = df.groupby("state")
df["DUSMASS_lag1"] = g["DUSMASS_ugm3"].shift(1); df["RH2M_lag1"] = g["RH2M"].shift(1)
df["DUSMASS_2wk_rolling_mean"] = g["DUSMASS_ugm3"].transform(lambda x: x.rolling(2).mean())
df["PREC_2wk_rolling_sum"] = g["PREC"].transform(lambda x: x.rolling(2).sum())

def classify_mews(T, RH, sdc):
    """MEWS four-tier classification (Dione et al., 2022), as adapted in Section 3.8 and Appendix B."""
    if T >= 30 and RH <= 20 and sdc >= 400: return "Red"
    if 27 <= T < 30 and RH <= 20 and sdc >= 400: return "Orange"
    if T >= 30 and RH <= 20 and 150 < sdc < 400: return "Orange"
    if T >= 30 and 40 < RH <= 60 and sdc >= 400: return "Orange"
    if 27 <= T < 30 and 20 < RH <= 40 and 150 < sdc < 400: return "Orange"
    if 27 <= T < 30 and 20 < RH <= 40 and sdc >= 400: return "Orange"
    if 27 <= T < 30 and 40 < RH <= 60 and sdc >= 400: return "Orange"
    if 27 <= T < 30 and RH <= 20 and 150 < sdc < 400: return "Orange"
    if T > 30 and 20 < RH <= 40 and 150 < sdc < 400: return "Orange"
    if T > 30 and 20 < RH <= 40 and sdc >= 400: return "Orange"
    if T >= 27 and RH <= 60 and sdc <= 150: return "Yellow"
    if T >= 27 and 40 < RH <= 60 and 150 < sdc < 400: return "Yellow"
    if RH > 60: return "White"
    if T < 27: return "White"
    raise ValueError(f"Unclassified: T={T}, RH={RH}, sdc={sdc}")
df["MEWS_level"] = [classify_mews(t, r, s) for t, r, s in zip(df["T2M_mean"], df["RH2M"], df["DUSMASS_ugm3"])]
df["elevated_risk"] = df["MEWS_level"].map({"White": 0, "Yellow": 0, "Orange": 1, "Red": 1})
g = df.groupby("state")
df["elevated_risk_2wk"] = g["elevated_risk"].shift(-2)
df["_lag2"] = g["elevated_risk"].shift(2)                    # only used to drop the first two weeks of each state, so that the
df = df.dropna(subset=["_lag2", "elevated_risk_2wk"]).drop(columns="_lag2").reset_index(drop=True)   # sample matches version 5 (last two weeks have no label)
df["elevated_risk_2wk"] = df["elevated_risk_2wk"].astype(int)
log(f"modelling table: {len(df)} state-weeks; positive rate {df['elevated_risk_2wk'].mean():.3f}")

# ==============================================================================
# STAGE EDA  Descriptive exploratory analysis of the full modelling table
#   Purely descriptive: computed on the whole table (development + sealed test) because it
#   describes the raw input data, not a modelling choice. No tuning, selection, calibration or
#   threshold decision reads any output of this stage, so it introduces no leakage.
# ==============================================================================
log("STAGE EDA: descriptive exploratory analysis")
EDA_NUMERIC = ["T2M_mean", "T2M_max", "T2M_min", "RH2M", "PREC", "WS10M", "WD10M", "DUSMASS_ugm3", "DTR",
               "DUSMASS_lag1", "RH2M_lag1", "DUSMASS_2wk_rolling_mean", "PREC_2wk_rolling_sum"]
# 1. Summary statistics per state
df.groupby("state")[EDA_NUMERIC].describe().T.to_csv(f"{OUT}/stage_eda_summary_by_state.csv")
# 2. Missingness (post feature engineering; lag/rolling columns are expected to be NaN only at each state's series start, already dropped above)
pd.DataFrame({"column": df.columns, "n_missing": df.isna().sum().values, "pct_missing": (df.isna().mean() * 100).round(3).values}).to_csv(f"{OUT}/stage_eda_missingness.csv", index=False)
# 3. Target balance: overall, by state, by MEWS level, by calendar week
df["MEWS_level"].value_counts().to_csv(f"{OUT}/stage_eda_mews_level_counts.csv")
df.groupby("state")["elevated_risk_2wk"].agg(["count", "sum", "mean"]).to_csv(f"{OUT}/stage_eda_target_balance_by_state.csv")
df.groupby("epi_week")["elevated_risk_2wk"].agg(["count", "sum", "mean"]).to_csv(f"{OUT}/stage_eda_target_balance_by_week.csv")
# 4. Correlation matrix among numeric predictors (Pearson)
corr = df[EDA_NUMERIC].corr()
corr.to_csv(f"{OUT}/stage_eda_correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(EDA_NUMERIC))); ax.set_xticklabels(EDA_NUMERIC, rotation=90, fontsize=7)
ax.set_yticks(range(len(EDA_NUMERIC))); ax.set_yticklabels(EDA_NUMERIC, fontsize=7); fig.colorbar(im, shrink=0.8)
plt.tight_layout(); plt.savefig(f"{OUT}/stage_eda_correlation_heatmap.png", dpi=150); plt.close()
# 5. Weekly time series of the key predictors, by state
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, col in zip(axes, ["T2M_mean", "RH2M", "DUSMASS_ugm3"]):
    for s_ in df["state"].unique(): d_ = df[df["state"] == s_]; ax.plot(d_["date"], d_[col], label=s_, linewidth=0.8)
    ax.set_ylabel(col); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/stage_eda_timeseries.png", dpi=150); plt.close()
# 6. Distribution (boxplot) of each numeric predictor by state
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for ax, col in zip(axes.flat, EDA_NUMERIC):
    df.boxplot(column=col, by="state", ax=ax); ax.set_title(col, fontsize=8); ax.set_xlabel("")
for ax in axes.flat[len(EDA_NUMERIC):]: ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/stage_eda_boxplots_by_state.png", dpi=150); plt.close()
log("  EDA outputs written: stage_eda_summary_by_state.csv, stage_eda_missingness.csv, stage_eda_mews_level_counts.csv, "
    "stage_eda_target_balance_by_state.csv, stage_eda_target_balance_by_week.csv, stage_eda_correlation_matrix.csv, "
    "stage_eda_correlation_heatmap.png, stage_eda_timeseries.png, stage_eda_boxplots_by_state.png")

# ==============================================================================
# STAGE 4  Chronological split, boundary purge, outer folds
# ==============================================================================
dev = df[df["date"] < SPLIT_DATE].sort_values(["date", "state"]).reset_index(drop=True)
test = df[df["date"] >= SPLIT_DATE].sort_values(["date", "state"]).reset_index(drop=True)
assert dev["date"].max() < test["date"].min()
dev = dev.iloc[:-GAP_ROWS].reset_index(drop=True)      # boundary purge: labels of the last 2 dev weeks lie in the test period
dev.to_csv(f"{OUT}/dev.csv", index=False)               # sealed test is NOT written to disk before Stage 12
y = dev["elevated_risk_2wk"].values; X = dev[MODEL_FEATURES].values
outer = list(TimeSeriesSplit(n_splits=N_OUTER, gap=GAP_ROWS).split(np.arange(len(dev))))
log(f"development {len(dev)} rows ({dev['date'].min().date()} to {dev['date'].max().date()}); sealed test {len(test)} rows held back")
log("outer folds (n_train, n_val): " + ", ".join(f"({len(a)},{len(b)})" for a, b in outer))

# ==============================================================================
# STAGE 5  Model constructors and the single hyperparameter-tuning routine
# ==============================================================================
def pos_weight(yt): return (yt == 0).sum() / max((yt == 1).sum(), 1)
def class_weight_dict(yt):
    n = len(yt); return {0: n / (2 * (yt == 0).sum()), 1: n / (2 * max((yt == 1).sum(), 1))}

def fit_catboost(Xt, yt, p, seed=SEED):
    n_mon = max(1, int(0.2 * len(Xt)))                   # monitoring block = latest 20% of the TRAINING rows
    m = CatBoostClassifier(depth=p["depth"], learning_rate=p["learning_rate"], l2_leaf_reg=p["l2_leaf_reg"],
                           bagging_temperature=p["bagging_temperature"], n_estimators=p["n_estimators"],
                           scale_pos_weight=pos_weight(yt), loss_function="Logloss", eval_metric="AUC",
                           random_state=seed, early_stopping_rounds=15, verbose=False)
    m.fit(Xt[:-n_mon], yt[:-n_mon], eval_set=(Xt[-n_mon:], yt[-n_mon:]), use_best_model=True)
    return lambda Z: m.predict_proba(Z)[:, 1], m

def fit_arnn(Xt, yt, p, seed=SEED):
    """ARNN with class-weighted loss through sample_weight (no oversampling, no duplicated rows). Requires scikit-learn >= 1.7."""
    sc = StandardScaler().fit(Xt); w = np.where(yt == 1, pos_weight(yt), 1.0)
    m = MLPClassifier(hidden_layer_sizes=(int(p["hidden_units"]),), alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                      batch_size=32, activation="relu", solver="adam", max_iter=300, early_stopping=True,
                      validation_fraction=0.25, n_iter_no_change=5, random_state=seed)
    m.fit(sc.transform(Xt), yt, sample_weight=w)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], (m, sc)

def fit_svm(Xt, yt, p, seed=SEED):
    sc = StandardScaler().fit(Xt)
    m = SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight=class_weight_dict(yt), probability=True, random_state=seed).fit(sc.transform(Xt), yt)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], (m, sc)

FITTERS = {"catboost": fit_catboost, "arnn": fit_arnn, "svm": fit_svm}
def suggest(name, trial):
    if name == "catboost":
        return {"depth": trial.suggest_int("depth", 3, 6), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 25.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50)}
    if name == "arnn":
        return {"hidden_units": trial.suggest_categorical("hidden_units", [8, 16, 32]),
                "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)}
    return {"C": trial.suggest_float("C", 1e-2, 2.0, log=True), "gamma": trial.suggest_float("gamma", 1e-4, 3e-1, log=True)}

def tune(name, Xt, yt, splits, n_trials):
    """Choose hyperparameters by mean ROC-AUC over `splits`, which index ONLY the rows (Xt, yt) passed in.
    Used (a) inside each outer fold, on that fold's training rows, with inner purged splits and
    (b) once on the whole development set for the final refit. It never sees any row outside Xt."""
    def obj(trial):
        p = suggest(name, trial); aucs = []
        for a, b in splits:
            if len(np.unique(yt[a])) < 2 or len(np.unique(yt[b])) < 2: continue
            f, _ = FITTERS[name](Xt[a], yt[a], p); aucs.append(roc_auc_score(yt[b], f(Xt[b])))
        return float(np.mean(aucs)) if aucs else 0.5
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params), float(st.best_value)

def inner_splits(n): return list(TimeSeriesSplit(n_splits=N_INNER, gap=GAP_ROWS).split(np.arange(n)))

# ==============================================================================
# STAGE 6  Whole-development tuning (used ONLY for the final refit, Stage 11)
# ==============================================================================
log("STAGE 6: whole-development tuning for the final refit")
FINAL_PARAMS = {}
for nm in ["catboost", "arnn", "svm"]:
    FINAL_PARAMS[nm], auc = tune(nm, X, y, outer, TRIALS_FINAL); log(f"  {nm}: mean outer-fold AUC {auc:.4f}  {FINAL_PARAMS[nm]}")
jdump(FINAL_PARAMS, "stage6_final_hyperparameters.json")

# ==============================================================================
# STAGE 7  Nested out-of-fold predictions (base learners)
# ==============================================================================
log("STAGE 7: nested out-of-fold base-learner predictions")
oof = pd.DataFrame({"date": dev["date"], "state": dev["state"], "y": y, "fold": -1,
                    "catboost_proba": np.nan, "arnn_proba": np.nan, "svm_proba": np.nan})
nested_params, gap_rows = {}, []
for k, (tr, va) in enumerate(outer, 1):
    oof.loc[va, "fold"] = k; Xtr, ytr = X[tr], y[tr]; isp = inner_splits(len(tr)); nested_params[k] = {}
    for nm in ["catboost", "arnn", "svm"]:
        p, _ = tune(nm, Xtr, ytr, isp, TRIALS_NESTED); nested_params[k][nm] = p
        f, _ = FITTERS[nm](Xtr, ytr, p)
        oof.loc[va, f"{nm}_proba"] = f(X[va])
        gap_rows.append({"fold": k, "learner": nm, "n_train": len(tr), "train_auc": roc_auc_score(ytr, f(Xtr)),
                         "val_auc": roc_auc_score(y[va], f(X[va]))})
    log(f"  outer fold {k}: train {len(tr)}, validation {len(va)}")
oof.to_csv(f"{OUT}/stage7_oof_predictions.csv", index=False); jdump(nested_params, "stage7_nested_hyperparameters.json")

# ==============================================================================
# STAGE 8  Nested meta-learner and stack comparison (C1-C4)
# ==============================================================================
log("STAGE 8: nested meta-learner, stack comparison and selection rule")
cov = oof[oof["fold"] > 0].reset_index(drop=True)          # all outer validation folds (1..N_OUTER): fold 1 only trains the meta-learner, fold 2 is its first validation block
yc = cov["y"].values; fc = cov["fold"].values; EVAL_FOLDS = [int(k) for k in sorted(set(fc)) if k >= 3]   # >= 2 earlier folds: meta trains on <j, validates on j

def make_meta(l1, C):
    try: return LogisticRegression(l1_ratio=l1, C=C, penalty="elasticnet", solver="saga", max_iter=5000, random_state=SEED)
    except TypeError: return LogisticRegression(l1_ratio=l1, C=C, solver="saga", max_iter=5000, random_state=SEED)

def tune_meta(Xm, ym, fm, train_folds, n_trials):
    """Tune (l1_ratio, C) using ONLY rows of `train_folds`: expanding pairs (folds < j) -> fold j."""
    pairs = [(np.where((fm < j) & np.isin(fm, train_folds))[0], np.where(fm == j)[0]) for j in train_folds if j > min(train_folds)]
    assert pairs, "every evaluation fold needs at least one earlier validation fold"
    def obj(trial):
        l1 = trial.suggest_float("l1_ratio", 0.0, 1.0); C = trial.suggest_float("C", 1e-3, 1e2, log=True); s = []
        for a, b in pairs:
            if len(np.unique(ym[b])) < 2: continue
            s.append(roc_auc_score(ym[b], make_meta(l1, C).fit(Xm[a], ym[a]).predict_proba(Xm[b])[:, 1]))
        return float(np.mean(s)) if s else 0.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return dict(st.best_params), float(st.best_value)

def nested_meta_oof(cols):
    Xm = cov[cols].values; out = np.full(len(Xm), np.nan); chosen = {}
    for k in EVAL_FOLDS:
        earlier = [j for j in sorted(set(fc)) if j < k]
        prm, _ = tune_meta(Xm, yc, fc, earlier, TRIALS_META_INNER); chosen[k] = prm
        a, b = np.where(fc < k)[0], np.where(fc == k)[0]
        out[b] = make_meta(prm["l1_ratio"], prm["C"]).fit(Xm[a], yc[a]).predict_proba(Xm[b])[:, 1]
    return out, chosen

meta_oof, meta_chosen = {}, {}
for cfg, cols in BASE_COLS.items(): meta_oof[cfg], meta_chosen[cfg] = nested_meta_oof(cols)
jdump(meta_chosen, "stage8_nested_meta_hyperparameters.json")
ev = ~np.isnan(meta_oof["C4_Full_Ensemble"]); y_ev = yc[ev]; f_ev = fc[ev]
log(f"  stack evaluation rows: {int(ev.sum())} (folds {EVAL_FOLDS})")

# calibration policy (decided once, on the full stack, by leave-one-fold-out Brier; applied to every stack)
def loo_calibrate(s, method):
    outc = np.full(len(s), np.nan); br = []; au = []
    for h in EVAL_FOLDS:
        trm = ev & (fc != h); tem = ev & (fc == h)
        if method == "raw": c = s[tem]
        elif method == "platt": c = LogisticRegression().fit(s[trm].reshape(-1, 1), yc[trm]).predict_proba(s[tem].reshape(-1, 1))[:, 1]
        else: c = IsotonicRegression(out_of_bounds="clip").fit(s[trm], yc[trm]).predict(s[tem])
        outc[tem] = c; br.append(brier_score_loss(yc[tem], c)); au.append(roc_auc_score(yc[tem], c))
    return float(np.mean(br)), float(np.mean(au)), outc
_cal_outc = {}; cal_res = {}
for m in ["raw", "platt", "isotonic"]:
    _br, _au, _cal_outc[m] = loo_calibrate(meta_oof["C4_Full_Ensemble"], m); cal_res[m] = {"brier": _br, "auc": _au}
best_m = min(cal_res, key=lambda m: cal_res[m]["brier"])
CAL_POLICY = best_m if (best_m != "raw" and cal_res["raw"]["brier"] - cal_res[best_m]["brier"] > 0.002 and cal_res["raw"]["auc"] - cal_res[best_m]["auc"] < 0.01) else "raw"
jdump({"policy": CAL_POLICY, "results": cal_res}, "stage8_calibration.json"); log(f"  calibration policy: {CAL_POLICY}")

# Figure 4.14: calibration (reliability) curves for raw vs Platt vs isotonic, leave-one-fold-out, on C4's out-of-fold predictions
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
for m, color in [("raw", "#2980b9"), ("platt", "#27ae60"), ("isotonic", "#c0392b")]:
    frac_pos, mean_pred = calibration_curve(yc[ev], _cal_outc[m][ev], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", color=color, label=f"{m} (Brier={cal_res[m]['brier']:.4f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency of elevated risk")
ax.set_title("Calibration curves: C4 out-of-fold predictions (leave-one-fold-out)"); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/stage8_calibration_curves.png", dpi=150); plt.close()
log("  saved outputs/stage8_calibration_curves.png (Figure 4.14)")

def calibrated(cfg):
    s = meta_oof[cfg]
    if CAL_POLICY == "raw": return s
    outc = np.full(len(s), np.nan)
    for h in EVAL_FOLDS:
        trm = ev & (fc != h); tem = ev & (fc == h)
        outc[tem] = (LogisticRegression().fit(s[trm].reshape(-1, 1), yc[trm]).predict_proba(s[tem].reshape(-1, 1))[:, 1] if CAL_POLICY == "platt"
                     else IsotonicRegression(out_of_bounds="clip").fit(s[trm], yc[trm]).predict(s[tem]))
    return outc

rows = []; STACK_OOF = {}
for cfg, cols in BASE_COLS.items():
    p_ = calibrated(cfg)[ev]; thr = best_threshold(y_ev, p_); m = full_metrics(y_ev, p_, thr)
    fold_pr = [average_precision_score(y_ev[f_ev == k], p_[f_ev == k]) for k in EVAL_FOLDS if len(np.unique(y_ev[f_ev == k])) > 1]
    m.update(configuration=cfg, n_base_learners=len(cols), mean_fold_pr_auc=float(np.mean(fold_pr)),
             se_fold_pr_auc=float(np.std(fold_pr, ddof=1) / np.sqrt(len(fold_pr))), fold_pr_aucs=json.dumps([round(float(v), 4) for v in fold_pr]))
    rows.append(m); STACK_OOF[cfg] = (p_, thr)
abl = pd.DataFrame(rows).sort_values("pr_auc", ascending=False); abl.to_csv(f"{OUT}/stage8_stack_comparison.csv", index=False)
log("\n" + abl[["configuration", "n_base_learners", "mean_fold_pr_auc", "se_fold_pr_auc", "pr_auc", "roc_auc", "brier_score", "ece", "cost_5to1"]].round(4).to_string(index=False))

# Cost-sensitive threshold sweep (preliminary sweep on C4, as in the thesis text): cost and Youden's J vs threshold,
# plus the per-configuration thresholds actually used (each on its own out-of-fold predictions).
_sw = threshold_sweep(y_ev, calibrated("C4_Full_Ensemble")[ev]); _sw.to_csv(f"{OUT}/stage8_threshold_sweep_C4.csv", index=False)
_t_cost = float(_sw.loc[_sw["cost_5to1"].idxmin(), "threshold"]); _t_j = float(_sw.loc[_sw["youden_j"].idxmax(), "threshold"])
_cost_default = float(_sw.loc[_sw["threshold"] == 0.5, "cost_5to1"].iloc[0]) if (_sw["threshold"] == 0.5).any() else np.nan
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(_sw["threshold"], _sw["cost_5to1"], color="#c0392b", label="Cost (5×FN + FP)")
ax1.axvline(_t_cost, color="#c0392b", ls="--", lw=1, label=f"Cost-minimising t*={_t_cost:.2f}")
ax1.set_xlabel("Decision threshold"); ax1.set_ylabel("Cost (5×FN + FP)", color="#c0392b"); ax1.tick_params(axis="y", labelcolor="#c0392b")
ax2 = ax1.twinx(); ax2.plot(_sw["threshold"], _sw["youden_j"], color="#2980b9", label="Youden's J")
ax2.axvline(_t_j, color="#2980b9", ls=":", lw=1, label=f"Youden-maximising t={_t_j:.2f}")
ax2.set_ylabel("Youden's J (sensitivity + specificity - 1)", color="#2980b9"); ax2.tick_params(axis="y", labelcolor="#2980b9")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.5)
ax1.set_title("Cost-sensitive threshold sweep, full-ensemble (C4) out-of-fold predictions")
plt.tight_layout(); plt.savefig(f"{OUT}/stage8_threshold_sweep.png", dpi=150); plt.close()
log(f"  cost-sensitive threshold sweep: t*={_t_cost:.2f} (cost={_sw['cost_5to1'].min():.0f}), Youden-maximising t={_t_j:.2f}, "
    f"t=0.50 cost={_cost_default:.0f} (saved outputs/stage8_threshold_sweep.png and stage8_threshold_sweep_C4.csv)")

best = abl.loc[abl["mean_fold_pr_auc"].idxmax()]
within = abl[abl["mean_fold_pr_auc"] >= best["mean_fold_pr_auc"] - best["se_fold_pr_auc"]].sort_values(["n_base_learners", "mean_fold_pr_auc"], ascending=[True, False])
PRIMARY = within.iloc[0]["configuration"]; SECOND = "C4_Full_Ensemble" if PRIMARY != "C4_Full_Ensemble" else abl[abl["configuration"] != PRIMARY].iloc[0]["configuration"]
jdump({"primary": PRIMARY, "comparison_stack": SECOND, "best_by_mean_fold_pr_auc": best["configuration"], "within_one_se": list(within["configuration"]),
       "table": abl[["configuration", "n_base_learners", "mean_fold_pr_auc", "se_fold_pr_auc", "fold_pr_aucs"]].to_dict("records")}, "stage8_selection.json")
log(f"  PRIMARY (rule) = {PRIMARY}; comparison stack = {SECOND}; both fixed BEFORE the sealed test is opened")

# ==============================================================================
# STAGE 9  Overfitting diagnostics from the NESTED fits (no resubstitution optimism)
# ==============================================================================
gap = pd.DataFrame(gap_rows); gap["gap"] = gap["train_auc"] - gap["val_auc"]; gap.to_csv(f"{OUT}/stage9_overfitting_nested_by_fold.csv", index=False)
gsum = gap.groupby("learner").agg(mean_train_auc=("train_auc", "mean"), mean_val_auc=("val_auc", "mean"), mean_gap=("gap", "mean"), max_gap=("gap", "max")).reset_index()
gsum["verdict"] = np.where(gsum["mean_gap"] <= 0.05, "LOW", np.where(gsum["mean_gap"] <= 0.10, "MODERATE", "HIGH"))
gsum.to_csv(f"{OUT}/stage9_overfitting_nested_summary.csv", index=False); log("STAGE 9: nested overfitting summary\n" + gsum.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4))
for nm, c in zip(["catboost", "arnn", "svm"], ["#c0392b", "#2980b9", "#27ae60"]):
    s = gap[gap["learner"] == nm]; ax.plot(s["fold"], s["gap"], marker="o", label=nm, color=c)
ax.axhline(0.05, ls="--", c="grey"); ax.set_xlabel("Outer fold"); ax.set_ylabel("Train AUC minus validation AUC"); ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/stage9_gap_by_fold.png", dpi=150); plt.close()

# ==============================================================================
# STAGE 10  Robustness tests on DEVELOPMENT folds only (hyperparameters = that fold's nested choice)
# ==============================================================================
log("STAGE 10: overfitting and robustness tests (the evaluation folds, nested hyperparameters)")
SEEDS = [0] if QUICK else [0, 1, 2]
def fold_eval(cols, learners=("catboost", "arnn"), frac=1.0, ytrain_fn=None, seeds=SEEDS):
    idx = [MODEL_FEATURES.index(c) for c in cols]; res = []
    for k in EVAL_FOLDS:
        tr, va = outer[k - 1]; n = max(30, int(len(tr) * frac)); trs = tr[:n]
        for nm in learners:
            for sd in seeds:
                yt = y[trs] if ytrain_fn is None else ytrain_fn(trs, sd)
                if len(np.unique(yt)) < 2: continue
                f, _ = FITTERS[nm](X[trs][:, idx], yt, nested_params[k][nm], seed=sd)
                res.append({"fold": k, "learner": nm, "seed": sd, "train_auc": roc_auc_score(yt, f(X[trs][:, idx])),
                            "val_auc": roc_auc_score(y[va], f(X[va][:, idx])), "val_pr": average_precision_score(y[va], f(X[va][:, idx]))})
    return pd.DataFrame(res)
def summarise(r):
    """mean of each metric over (fold x seed) fits, plus the SD of validation AUC and the number of fits (label controls are noisy)"""
    a = r.groupby("learner")[["train_auc", "val_auc", "val_pr"]].mean(); a["val_auc_sd"] = r.groupby("learner")["val_auc"].std(); a["n_fits"] = r.groupby("learner").size()
    return a.round(4)

rob = []
# (a) learning curve: expanding fractions of each fold's training rows
for fr in [0.25, 0.5, 0.75, 1.0]:
    s = summarise(fold_eval(MODEL_FEATURES, frac=fr)).reset_index(); s.insert(0, "test", f"learning curve, {int(fr*100)}% of training rows"); rob.append(s)
# (b) predictor ablation
LABEL_INPUTS = ["T2M_mean", "RH2M", "DUSMASS_ugm3", "DTR", "Harmattan_Wind_Flag", "DUSMASS_lag1", "RH2M_lag1", "DUSMASS_2wk_rolling_mean"]
sets = {"all 13 predictors": MODEL_FEATURES,
        "without current-week T2M_mean, RH2M, DUSMASS_ugm3": [c for c in MODEL_FEATURES if c not in ("T2M_mean", "RH2M", "DUSMASS_ugm3")],
        "without all 8 temperature, humidity and dust inputs": [c for c in MODEL_FEATURES if c not in LABEL_INPUTS],
        "without WS10M (flag kept)": [c for c in MODEL_FEATURES if c != "WS10M"],
        "calendar only (week_sin, week_cos)": ["week_sin", "week_cos"]}
for nm_, cols in sets.items():
    s = summarise(fold_eval(cols)).reset_index(); s.insert(0, "test", f"ablation: {nm_}"); rob.append(s)
# (c) label controls: training labels destroyed, validation labels true -> AUC should be near 0.5
def shift_labels(trs, sd):                       # circular shift within state: keeps seasonality and autocorrelation, breaks the link to the inputs
    rng = np.random.RandomState(100 + sd); yy = y[trs].copy(); st_ = dev["state"].values[trs]
    for s_ in np.unique(st_):
        ii = np.where(st_ == s_)[0]
        if len(ii) > 40: yy[ii] = np.roll(yy[ii], rng.randint(20, len(ii) - 20))
    return yy
def permute_labels(trs, sd): return np.random.RandomState(200 + sd).permutation(y[trs])
for nm_, fn in [("label control: circular shift within state", shift_labels), ("label control: random permutation", permute_labels)]:
    s = summarise(fold_eval(MODEL_FEATURES, ytrain_fn=fn, seeds=SEEDS if QUICK else list(range(10)))).reset_index(); s.insert(0, "test", nm_); rob.append(s)
ROB = pd.concat(rob, ignore_index=True); ROB.to_csv(f"{OUT}/stage10_robustness_development.csv", index=False); print(ROB.to_string(index=False))

# ==============================================================================
# STAGE 11  Reference baselines (development, same rows as the stacks) and final refit
# ==============================================================================
log("STAGE 11: baselines on the same development rows, then final refit on the whole development set")
STATES = sorted(dev["state"].unique()); wk_dev = dev["epi_week"].values; st_dev = dev["state"].values
def clim_fit(wk, yy, st, win=1):
    tab = {}
    for s_ in STATES:
        ms = st == s_; glob = float(yy[ms].mean()) if ms.any() else float(yy.mean())
        for w in range(1, 54):
            nb = [((w - 1 + d) % 53) + 1 for d in range(-win, win + 1)]; m = ms & np.isin(wk, nb)
            tab[(s_, w)] = float(yy[m].mean()) if m.any() else glob
    return tab
def clim_pred(tab, wk, st): return np.array([tab[(s_, int(w))] for w, s_ in zip(wk, st)])
def cal_design(d): return np.column_stack([d["week_sin"].values, d["week_cos"].values] + [(d["state"].values == s_).astype(float) for s_ in STATES[1:]])
oc, ol = np.full(len(dev), np.nan), np.full(len(dev), np.nan)
for k, (tr, va) in enumerate(outer, 1):
    oc[va] = clim_pred(clim_fit(wk_dev[tr], y[tr], st_dev[tr]), wk_dev[va], st_dev[va])
    ol[va] = LogisticRegression(C=1.0, max_iter=5000).fit(cal_design(dev.iloc[tr]), y[tr]).predict_proba(cal_design(dev.iloc[va]))[:, 1]
# map the evaluation rows of `cov` back to dev rows (cov = dev rows with fold >= 1; ev keeps only the evaluation folds)
dev_idx_ev = np.where(oof["fold"].values > 0)[0][ev]
# POLICY: the thesis compares the stacks against only the three agreed reference baselines
# (naive persistence, per-state weekly climatology, calendar-only logistic regression). A shallow
# decision tree and the single base learners (CatBoost/ARNN/SVM alone) are NOT scored as competitors
# of the stacks anywhere in development or sealed-test reporting (see thesis Section 5.9.1).
BL = {"clim": oc[dev_idx_ev], "lr": ol[dev_idx_ev]}
BLM = {}
for key, nm_ in [("clim", "Weekly climatology by state"), ("lr", "Calendar-only logistic regression")]:
    t_ = best_threshold(y_ev, BL[key]); BLM[key] = {"name": nm_, "threshold": t_, **full_metrics(y_ev, BL[key], t_)}
pd.DataFrame([{"model": v["name"], **{k: v[k] for k in ["roc_auc", "pr_auc", "brier_score", "ece", "sensitivity", "false_negatives", "false_positives", "cost_5to1", "threshold"]}} for v in BLM.values()]
             + [{"model": c, **{k: abl.set_index("configuration").loc[c, k] for k in ["roc_auc", "pr_auc", "brier_score", "ece", "sensitivity", "false_negatives", "false_positives", "cost_5to1"]},
                 "threshold": STACK_OOF[c][1]} for c in BASE_COLS]).to_csv(f"{OUT}/stage11_development_comparison_same_rows.csv", index=False)
# diagnostic only: base-learner out-of-fold performance on ALL outer folds (not used for any selection)
_diag = []
for _k in sorted(set(oof.loc[oof["fold"] > 0, "fold"])):
    _m = (oof["fold"] == _k).values
    if len(np.unique(y[_m])) < 2: continue
    for _nm in ["catboost", "arnn", "svm"]:
        _p = oof[f"{_nm}_proba"].values[_m]; _diag.append({"fold": int(_k), "learner": _nm, "n": int(_m.sum()), "roc_auc": roc_auc_score(y[_m], _p), "pr_auc": average_precision_score(y[_m], _p), "scored_for_stacks": int(_k in EVAL_FOLDS)})
pd.DataFrame(_diag).to_csv(f"{OUT}/stage11_base_learners_all_folds_diagnostic.csv", index=False)
clim_tab = clim_fit(wk_dev, y, st_dev); lr_final = LogisticRegression(C=1.0, max_iter=5000).fit(cal_design(dev), y)

FINAL = {nm: FITTERS[nm](X, y, FINAL_PARAMS[nm]) for nm in ["catboost", "arnn", "svm"]}
def base_proba(Z): return {f"{nm}_proba": FINAL[nm][0](Z) for nm in FINAL}
STACKS = {}
for cfg in BASE_COLS:   # refit and calibrate ALL FOUR stack configurations (C1-C4), so Stage 12 scores all of them on the sealed test; PRIMARY/SECOND remain the ones actually deployed
    cols = BASE_COLS[cfg]; Xm = cov[cols].values
    prm, _ = tune_meta(Xm, yc, fc, sorted(set(fc)), TRIALS_META_FINAL)         # whole-OOF tuning for the deployed meta-learner
    meta = make_meta(prm["l1_ratio"], prm["C"]).fit(Xm, yc)
    s_ev = meta_oof[cfg]; cal = None
    if CAL_POLICY == "platt": cal = LogisticRegression().fit(s_ev[ev].reshape(-1, 1), yc[ev])
    elif CAL_POLICY == "isotonic": cal = IsotonicRegression(out_of_bounds="clip").fit(s_ev[ev], yc[ev])
    STACKS[cfg] = {"base_cols": cols, "meta_learner": meta, "calibrator": cal, "threshold": STACK_OOF[cfg][1], "l1_ratio": prm["l1_ratio"], "C": prm["C"]}
    log(f"  {cfg}: meta l1_ratio={prm['l1_ratio']:.3f}, C={prm['C']:.3f}, threshold={STACK_OOF[cfg][1]:.2f}, coefficients={meta.coef_[0].round(3).tolist()}")
# Figure 4.13: Elastic Net meta-learner coefficients, PRIMARY configuration
_coef = STACKS[PRIMARY]["meta_learner"].coef_[0]; _cols = STACKS[PRIMARY]["base_cols"]; _names = {"catboost_proba": "CatBoost", "arnn_proba": "ARNN", "svm_proba": "SVM-RBF"}
fig, ax = plt.subplots(figsize=(6, 4)); ax.barh([_names.get(c, c) for c in _cols], _coef, color=["#c0392b" if v == 0 else "#2980b9" for v in _coef])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Elastic Net coefficient"); ax.set_title(f"Meta-learner coefficients, {PRIMARY} (selected primary configuration)")
plt.tight_layout(); plt.savefig(f"{OUT}/stage11_meta_coefficients.png", dpi=150); plt.close()

def stack_predict(cfg, Z):
    st = STACKS[cfg]; bp = base_proba(Z); raw = st["meta_learner"].predict_proba(np.column_stack([bp[c] for c in st["base_cols"]]))[:, 1]
    if CAL_POLICY == "platt": return st["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    if CAL_POLICY == "isotonic": return st["calibrator"].predict(raw)
    return raw
# deployment bundle written NOW, before the sealed test is read
pickle.dump({"catboost": FINAL["catboost"][1], "arnn": FINAL["arnn"][1][0], "arnn_scaler": FINAL["arnn"][1][1], "svm": FINAL["svm"][1][0], "svm_scaler": FINAL["svm"][1][1],
             "stacks": STACKS, "deployment_config": PRIMARY, "second_config": SECOND, "calibration_policy": CAL_POLICY,
             "feature_lists": {"full": MODEL_FEATURES, "svm": MODEL_FEATURES}, "display_fields": DISPLAY_FIELDS, "threshold": STACKS[PRIMARY]["threshold"],
             "winning_config": PRIMARY, "versions": {"python": platform.python_version(), "numpy": np.__version__}}, open(f"{OUT}/stage11_final_models.pkl", "wb"))
log("  saved outputs/stage11_final_models.pkl (frozen before the sealed test)")

# ==============================================================================
# STAGE 12  Sealed test: opened ONCE; stacks, baselines, block bootstrap
# ==============================================================================
def tier(p): return "White" if p < 0.20 else "Yellow" if p < 0.40 else "Orange" if p < 0.60 else "Red"
log("STAGE 12: sealed test (single look)")
yt_ = test["elevated_risk_2wk"].values; Xt_ = test[MODEL_FEATURES].values
P = {c: stack_predict(c, Xt_) for c in STACKS}
P["clim"] = clim_pred(clim_tab, test["epi_week"].values, test["state"].values); P["lr"] = lr_final.predict_proba(cal_design(test))[:, 1]
THR = {**{c: STACKS[c]["threshold"] for c in STACKS}, **{k: BLM[k]["threshold"] for k in BLM}}      # thresholds frozen on development data
NAMES = {**{c: c for c in STACKS}, **{k: BLM[k]["name"] for k in BLM}}
sealed = {k: full_metrics(yt_, P[k], THR[k]) for k in P}
sealed["naive_persistence"] = full_metrics(yt_, test["elevated_risk"].values.astype(float), 0.5)
pd.DataFrame([{"model": NAMES.get(k, k), **v} for k, v in sealed.items()]).to_csv(f"{OUT}/stage12_sealed_test_results.csv", index=False)
jdump({"primary": PRIMARY, "comparison": SECOND, "n_test": len(test), "results": sealed}, "stage12_sealed_test_results.json")
for k, v in sealed.items(): print(f"  {k:28s} ROC {v['roc_auc']:.4f}  PR {v['pr_auc']:.4f}  Brier {v['brier_score']:.4f}  ECE {v['ece']:.4f}  sens {v['sensitivity']:.3f}  FN {v['false_negatives']}  FP {v['false_positives']}  cost {v['cost_5to1']:.0f}")
pd.DataFrame({"state": test["state"], "date": test["date"], "y": yt_, **{f"p_{k}": P[k] for k in P}}).to_csv(f"{OUT}/stage12_sealed_test_predictions.csv", index=False)

# ------------------------------------------------------------------------------
# Misclassification analysis (sealed test): every false negative and false positive
# for every stack configuration (C1-C4), at each stack's own development-frozen threshold.
# ------------------------------------------------------------------------------
mis_rows = []
for cfg in STACKS:
    yp = (P[cfg] >= THR[cfg]).astype(int)
    for i in np.where(yp != yt_)[0]:
        mis_rows.append({"configuration": cfg, "error_type": "false_negative" if yt_[i] == 1 else "false_positive",
                          "state": test["state"].iloc[i], "date": str(test["date"].iloc[i])[:10], "epi_week": int(test["epi_week"].iloc[i]),
                          "true_label": int(yt_[i]), "predicted_probability": float(P[cfg][i]), "threshold": float(THR[cfg]), "tier": tier(P[cfg][i])})
mis_df = pd.DataFrame(mis_rows); mis_df.to_csv(f"{OUT}/stage12_misclassification_cases.csv", index=False)
mis_summary = (mis_df.groupby(["configuration", "error_type"]).agg(n=("state", "size"), states=("state", lambda x: x.value_counts().to_dict()),
               weeks=("epi_week", lambda x: sorted(x.tolist()))).reset_index() if len(mis_df) else pd.DataFrame())
mis_summary.to_csv(f"{OUT}/stage12_misclassification_summary.csv", index=False)

# Discussion Figure 2: summary of sealed-test misclassifications by type, all four stack configurations
if len(mis_df):
    _ct = mis_df.groupby(["configuration", "error_type"]).size().unstack(fill_value=0).reindex(columns=["false_negative", "false_positive"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4)); xw = np.arange(len(_ct)); w = 0.35
    ax.bar(xw - w / 2, _ct["false_negative"], width=w, label="False negative", color="#c0392b")
    ax.bar(xw + w / 2, _ct["false_positive"], width=w, label="False positive", color="#e67e22")
    ax.set_xticks(xw); ax.set_xticklabels(_ct.index); ax.set_ylabel("Count"); ax.set_title("Summary of sealed-test misclassifications by type"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/discussion_fig2_misclassification_summary.png", dpi=150); plt.close()
_yp_p = (P[PRIMARY] >= THR[PRIMARY]).astype(int)
log(f"  misclassification cases (PRIMARY={PRIMARY}): {int(((yt_==1)&(_yp_p==0)).sum())} FN, {int(((yt_==0)&(_yp_p==1)).sum())} FP "
    f"(full detail in stage12_misclassification_cases.csv)")

# Figure 4.15: confusion matrix, PRIMARY configuration, sealed test
_cm = confusion_matrix(yt_, _yp_p, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.5, 4)); im = ax.imshow(_cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, str(_cm[i, j]), ha="center", va="center", fontsize=14, color="white" if _cm[i, j] > _cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-elevated", "Elevated"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-elevated", "Elevated"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(f"Confusion matrix, {PRIMARY} (sealed test, t={THR[PRIMARY]:.2f})")
plt.tight_layout(); plt.savefig(f"{OUT}/stage12_confusion_matrix.png", dpi=150); plt.close()

# Discussion Figure 1: elevated-risk positive rate by state, development vs sealed test
_pos_dev = dev.groupby("state")["elevated_risk_2wk"].mean(); _pos_test = test.groupby("state")["elevated_risk_2wk"].mean()
fig, ax = plt.subplots(figsize=(6, 4)); xw = np.arange(len(STATES))
ax.bar(xw - 0.18, [_pos_dev[s_] for s_ in STATES], width=0.36, label="Development", color="#2980b9")
ax.bar(xw + 0.18, [_pos_test[s_] for s_ in STATES], width=0.36, label="Sealed test", color="#c0392b")
ax.set_xticks(xw); ax.set_xticklabels(STATES); ax.set_ylabel("Positive rate (elevated_risk_2wk)"); ax.set_title("Elevated-risk positive rate by state, development vs. sealed test"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/discussion_fig1_positive_rate_by_state.png", dpi=150); plt.close()

# reviewer (Major 2): performance restricted to epidemiological weeks 1-26 (the weeks for which MEWS is issued) and weeks 27-53
def subset_table(y_, P_, THR_, wk_, label):
    out = []
    for nm_, m_ in [("weeks 1-26", wk_ <= 26), ("weeks 27-53", wk_ > 26)]:
        for k in [PRIMARY, SECOND, "clim"]:
            yy, pp = y_[m_], P_[k][m_]
            if len(np.unique(yy)) < 2: continue
            yp = (pp >= THR_[k]).astype(int); tn, fp, fn, tp = confusion_matrix(yy, yp, labels=[0, 1]).ravel()
            out.append({"data": label, "subset": nm_, "model": NAMES.get(k, k), "n": int(m_.sum()), "elevated": int(yy.sum()), "roc_auc": roc_auc_score(yy, pp), "pr_auc": average_precision_score(yy, pp),
                        "sensitivity": tp / max(tp + fn, 1), "false_negatives": int(fn), "false_positives": int(fp), "cost_5to1": float(COST_FN * fn + COST_FP * fp)})
    return out
_dev_P = {PRIMARY: STACK_OOF[PRIMARY][0], SECOND: STACK_OOF[SECOND][0], "clim": BL["clim"]}; _dev_T = {PRIMARY: STACK_OOF[PRIMARY][1], SECOND: STACK_OOF[SECOND][1], "clim": BLM["clim"]["threshold"]}
pd.DataFrame(subset_table(y_ev, _dev_P, _dev_T, dev["epi_week"].values[dev_idx_ev], f"development ({len(y_ev)} rows, folds {EVAL_FOLDS[0]}-{EVAL_FOLDS[-1]})") + subset_table(yt_, P, THR, test["epi_week"].values, "sealed test")).to_csv(f"{OUT}/stage12_weeks_1_26_sensitivity.csv", index=False)
# reviewer (Major 5): serial dependence and effective sample size of the target, per state
ess_rows = []
for nm_, d_ in [("development", dev), ("sealed test", test)]:
    for s_ in STATES:
        v_ = d_.loc[d_["state"] == s_].sort_values("date")["elevated_risk_2wk"].values.astype(float); r1 = float(np.corrcoef(v_[:-1], v_[1:])[0, 1])
        ess_rows.append({"data": nm_, "state": s_, "n_weeks": len(v_), "elevated": int(v_.sum()), "lag1_autocorrelation": r1, "effective_n": len(v_) * (1 - r1) / (1 + r1)})
pd.DataFrame(ess_rows).to_csv(f"{OUT}/stage12_effective_sample_size.csv", index=False)

# circular moving-block bootstrap over whole weeks (3 states per week), several block lengths
weeks = np.sort(test["date"].unique()); wk_idx = {w: np.where(test["date"].values == w)[0] for w in weeks}; W = len(weeks)
def boot_stats(rows, k):
    yy, pp = yt_[rows], P[k][rows]
    if len(np.unique(yy)) < 2: return None
    yp = (pp >= THR[k]).astype(int); fn = ((yy == 1) & (yp == 0)).sum(); fp = ((yy == 0) & (yp == 1)).sum(); tp = ((yy == 1) & (yp == 1)).sum()
    return {"roc_auc": roc_auc_score(yy, pp), "pr_auc": average_precision_score(yy, pp), "brier_score": brier_score_loss(yy, pp), "sensitivity": tp / max(tp + fn, 1), "cost_5to1": COST_FN * fn + COST_FP * fp}
boot_rows = []; paired_rows = []; rng = np.random.RandomState(SEED)
for L in BOOT_LENGTHS:
    reps = {k: [] for k in [PRIMARY, SECOND, "clim", "lr"]}
    for _ in range(N_BOOT):
        nb = int(np.ceil(W / L)); starts = rng.randint(0, W, nb); wsel = np.concatenate([(np.arange(s, s + L) % W) for s in starts])[:W]
        rows = np.concatenate([wk_idx[weeks[i]] for i in wsel]); r = {k: boot_stats(rows, k) for k in reps}
        if any(v is None for v in r.values()): continue
        for k in reps: reps[k].append(r[k])
    for k, lst in reps.items():
        d = pd.DataFrame(lst)
        for m in d.columns: boot_rows.append({"block_length": L, "model": NAMES.get(k, k), "metric": m, "estimate": sealed[k][m] if m in sealed[k] else np.nan, "ci_low": d[m].quantile(.025), "ci_high": d[m].quantile(.975)})
    for a, b in [(PRIMARY, SECOND), (PRIMARY, "clim")]:
        da, db = pd.DataFrame(reps[a]), pd.DataFrame(reps[b])
        for m in da.columns: dd = da[m].values - db[m].values; paired_rows.append({"block_length": L, "difference": f"{a} minus {b}", "metric": m, "mean": dd.mean(), "ci_low": np.quantile(dd, .025), "ci_high": np.quantile(dd, .975)})
pd.DataFrame(boot_rows).to_csv(f"{OUT}/stage12_block_bootstrap.csv", index=False); pd.DataFrame(paired_rows).to_csv(f"{OUT}/stage12_block_bootstrap_paired.csv", index=False)

# ==============================================================================
# STAGE 13  SHAP on the complete decision function of the primary stack
# ==============================================================================
import shap
log("STAGE 13: SHAP (KernelExplainer on the complete primary-stack function)")
rng = np.random.RandomState(SEED); bg = X[rng.choice(len(X), 50, replace=False)]; ei = rng.choice(len(Xt_), 60, replace=False)
fn_ = lambda Z: stack_predict(PRIMARY, Z)
ke = shap.KernelExplainer(fn_, bg); sv = ke.shap_values(Xt_[ei], silent=True)
imp = pd.Series(np.abs(sv).mean(axis=0), index=MODEL_FEATURES).sort_values(ascending=False); imp.to_csv(f"{OUT}/stage13_shap_importance.csv")
plt.figure(figsize=(9, 6)); shap.summary_plot(sv, Xt_[ei], feature_names=MODEL_FEATURES, show=False); plt.tight_layout(); plt.savefig(f"{OUT}/stage13_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(figsize=(8, 5)); imp.sort_values().plot(kind="barh", ax=ax, color="#2980b9"); ax.set_xlabel("Mean |SHAP value|"); plt.tight_layout(); plt.savefig(f"{OUT}/stage13_shap_bar.png", dpi=150); plt.close()
cases = {"most_confident_elevated": int(np.argmax(P[PRIMARY])), "most_confident_non_elevated": int(np.argmin(P[PRIMARY])), "borderline_near_threshold": int(np.argmin(np.abs(P[PRIMARY] - THR[PRIMARY])))}
_yp_primary = (P[PRIMARY] >= THR[PRIMARY]).astype(int)
_fn_idx = np.where((yt_ == 1) & (_yp_primary == 0))[0]; _fp_idx = np.where((yt_ == 0) & (_yp_primary == 1))[0]
if len(_fn_idx): cases["worst_false_negative"] = int(_fn_idx[np.argmin(P[PRIMARY][_fn_idx])])    # true positive given the lowest probability
if len(_fp_idx): cases["worst_false_positive"] = int(_fp_idx[np.argmax(P[PRIMARY][_fp_idx])])    # true negative given the highest probability
info = {}
for nm_, i in cases.items():
    s1 = ke.shap_values(Xt_[i:i + 1], nsamples=200, silent=True)[0]; base = float(np.ravel(ke.expected_value)[0]); o = np.argsort(np.abs(s1))[::-1]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([MODEL_FEATURES[j] for j in o][::-1], s1[o][::-1], color=["#c0392b" if v > 0 else "#2980b9" for v in s1[o][::-1]]); ax.axvline(0, c="k", lw=.8)
    ax.set_title(f"Local explanation: {nm_} (base {base:.3f}, prediction {P[PRIMARY][i]:.3f})"); plt.tight_layout(); plt.savefig(f"{OUT}/stage13_local_{nm_}.png", dpi=150, bbox_inches="tight"); plt.close()
    info[nm_] = {"state": test.state[i], "date": str(test.date[i])[:10], "p": float(P[PRIMARY][i]), "y": int(yt_[i]), "base": base, "top": [(MODEL_FEATURES[j], float(s1[j])) for j in o[:3]]}
jdump(info, "stage13_local_cases.json")

# ==============================================================================
# STAGE 14  Four-tier probability bands and deployment bundle for the app
# ==============================================================================
tp_ = test[["state", "date", "elevated_risk_2wk"]].copy(); tp_["forecast_tier"] = [tier(p) for p in P[PRIMARY]]; tp_.to_csv(f"{OUT}/stage14_tier_predictions.csv", index=False)
print(pd.crosstab(tp_["forecast_tier"], tp_["elevated_risk_2wk"]))
pd.DataFrame({"tier": ["White", "Yellow", "Orange", "Red"], "decision": ["Routine surveillance", "Increased vigilance", "Preparedness and investigation", "Intensified response and resource mobilisation"]}).to_csv(f"{OUT}/stage14_tier_decision_map.csv", index=False)
pd.DataFrame(bg, columns=MODEL_FEATURES).to_csv(f"{OUT}/stage14_shap_background.csv", index=False)

def predict_risk(state, climate_lookup, week_start=None, stack=None):
    """Forecast for the week that starts on `week_start` (a Monday). If `week_start` is omitted, the forecast
    origin is the current date at the moment the prediction is made, rounded back to that week's Monday -- the
    deployed app always forecasts from "now", never from a date the user must supply.
    climate_lookup(state, date_str) -> dict with T2M_mean, T2M_max, T2M_min, RH2M, PREC, WS10M, WD10M, DUSMASS (kg/m3).
    epi_week is returned for DISPLAY only; it is never a model input. The model predicts conditions TWO WEEKS AFTER
    week_start (the forecast horizon baked into training), so forecast_target_date = week_start + 14 days."""
    if week_start is None:
        now = pd.Timestamp.now().normalize(); d = now - pd.Timedelta(days=int(now.dayofweek))   # most recent Monday on or before "now"
    else:
        d = pd.Timestamp(week_start)
    cur = climate_lookup(state, str(d.date())); pri = climate_lookup(state, str((d - pd.Timedelta(days=7)).date()))
    wk = int(d.isocalendar()[1]); dust, dust_p = cur["DUSMASS"] * 1e9, pri["DUSMASS"] * 1e9
    f = {"T2M_mean": cur["T2M_mean"], "RH2M": cur["RH2M"], "PREC": cur["PREC"], "WS10M": cur["WS10M"], "DUSMASS_ugm3": dust, "DTR": cur["T2M_max"] - cur["T2M_min"],
         "Harmattan_Wind_Flag": int(30 <= cur["WD10M"] <= 100 and cur["WS10M"] > 3.0 and cur["RH2M"] < 40), "week_sin": np.sin(2 * np.pi * wk / 52), "week_cos": np.cos(2 * np.pi * wk / 52),
         "DUSMASS_lag1": dust_p, "RH2M_lag1": pri["RH2M"], "DUSMASS_2wk_rolling_mean": (dust + dust_p) / 2, "PREC_2wk_rolling_sum": cur["PREC"] + pri["PREC"]}
    cfg = stack or PRIMARY; p = float(stack_predict(cfg, np.array([[f[c] for c in MODEL_FEATURES]]))[0]); target = d + pd.Timedelta(days=14)
    return {"state": state, "prediction_made_at": str(pd.Timestamp.now()), "forecast_origin_week": str(d.date()), "forecast_target_date": str(target.date()),
            "epi_week_display_only": wk, "forecast_target_epi_week_display_only": int(target.isocalendar()[1]), "probability": p, "tier": tier(p),
            "binary": "Elevated" if p >= STACKS[cfg]["threshold"] else "Non-elevated", "stack": cfg}


# ------------------------------------------------------------------ app bundle (copy these files into the app's data/ folder)
import shutil
os.makedirs(f"{OUT}/app_bundle", exist_ok=True)
for src, dst in [("stage11_final_models.pkl", "final_models.pkl"), ("stage13_shap_importance.csv", "shap_importance.csv"), ("stage14_shap_background.csv", "background.csv"),
                 ("stage14_tier_predictions.csv", "stage20_tier_predictions.csv"), ("stage14_tier_decision_map.csv", "stage20_tier_decision_map.csv")]:
    shutil.copy(f"{OUT}/{src}", f"{OUT}/app_bundle/{dst}")
shutil.copy(DATA_FILE, f"{OUT}/app_bundle/climate_weekly.csv")

jdump({"model_predictors": MODEL_FEATURES, "display_fields": DISPLAY_FIELDS, "primary": PRIMARY, "comparison": SECOND, "calibration_policy": CAL_POLICY,
       "quick_mode": QUICK, "trials": {"final": TRIALS_FINAL, "nested": TRIALS_NESTED, "meta_inner": TRIALS_META_INNER, "meta_final": TRIALS_META_FINAL},
       "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": __import__("sklearn").__version__, "catboost": __import__("catboost").__version__, "shap": shap.__version__},
       "note": "pin scikit-learn and catboost in the app requirements.txt to these exact versions, or the saved models will not load correctly", "runtime_minutes": (time.time() - T0) / 60}, "run_manifest.json")
log("DONE. Results are in ./outputs/. A QUICK run is a smoke test and must not be reported.")

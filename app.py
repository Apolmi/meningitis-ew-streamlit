"""Meningitis Environmental Risk Early Warning - research prototype (Gradio).

Two-week-ahead forecast of the MEWS-derived environmental classification (Orange/Red
vs White/Yellow) for Adamawa, Borno and Yobe states, Northeast Nigeria. This does NOT
forecast meningitis cases or outbreaks: the target is a deterministic, rule-based
function of temperature, relative humidity and dust, and the model forecasts that
rule-based classification ahead of time, year-round (not restricted to weeks 1-26 as
in Dione et al., 2022).

Model artifacts are the final (v6) pipeline run: thirteen model predictors (epi_week
is display-only, not a model input), base learners (ARNN, CatBoost, SVM-RBF) and
Elastic-Net-logistic meta-learners tuned in a nested loop within each outer
cross-validation fold, raw (uncalibrated) probabilities. The selected primary
configuration is C1 (ARNN + CatBoost); C4 (ARNN + CatBoost + SVM-RBF) is reported
alongside it as the comparison stack fixed by the thesis's prespecified selection
rule.
"""
import os, pickle, warnings, datetime as dt
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import gradio as gr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(HERE, "data", f)

TIER_COLOR = {"White": "#9aa0a6", "Yellow": "#f2c200", "Orange": "#f28c28", "Red": "#d93025"}
STATES = ["Adamawa", "Borno", "Yobe"]
# Approximate state-centroid coordinates, used for the live NASA POWER pull.
# The thesis aggregates gridded data to one weekly value per state by spatially
# averaging grid cells within each state's rectangular bounding box (Section 3.5/3.6);
# the live demo below uses a single centroid point per state as a lightweight
# approximation of that same procedure for the API call.
STATE_COORDS = {"Adamawa": (9.33, 12.43), "Borno": (11.83, 13.15), "Yobe": (11.75, 11.96)}
NAMES = {"arnn_proba": "ARNN", "catboost_proba": "CatBoost", "svm_proba": "SVM-RBF"}
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS10M,WD10M"


def load():
    M = pickle.load(open(D("final_models.pkl"), "rb"))
    clim = pd.read_csv(D("climate_weekly.csv"))
    clim["date"] = pd.to_datetime(clim["date"], format="%m/%d/%Y")
    bg = pd.read_csv(D("background.csv"))
    hist = pd.read_csv(D("stage20_tier_predictions.csv"))
    tiermap = pd.read_csv(D("stage20_tier_decision_map.csv")).set_index("tier")["decision"].to_dict()
    shap_imp = pd.read_csv(D("shap_importance.csv"), index_col=0).iloc[:, 0].sort_values()
    return M, clim, bg, hist, tiermap, shap_imp


M, CLIM, BG, HIST, TIER_MAP, GLOBAL = load()
FEATURES = M["feature_lists"]["full"]        # the 13 real model predictors (no epi_week)
STACKS = M["stacks"]
PRIMARY, SECOND = M["deployment_config"], M["second_config"]


def _label(c, role):
    return f"{c.split('_')[0]}: {' + '.join(NAMES[b] for b in STACKS[c]['base_cols'])} ({role})"


STACK_LABEL = {PRIMARY: _label(PRIMARY, "primary"), **({SECOND: _label(SECOND, "comparison stack")} if SECOND in STACKS and SECOND != PRIMARY else {})}
STACK_CHOICES = [(v, k) for k, v in STACK_LABEL.items()]


def map_to_tier(p):
    return "White" if p < 0.20 else "Yellow" if p < 0.40 else "Orange" if p < 0.60 else "Red"


def build_features(week_start, cur, prior):
    """cur / prior: dicts of raw weekly climate values for the forecast-origin week
    and the week before. DUSMASS is in kg/m^3, converted here to ug/m^3 (x1e9),
    matching the thesis's unit handling."""
    wk = int(pd.Timestamp(week_start).isocalendar()[1])
    dust, dust_p = cur["DUSMASS"] * 1e9, prior["DUSMASS"] * 1e9
    feats = {
        "T2M_mean": cur["T2M_mean"], "RH2M": cur["RH2M"], "PREC": cur["PREC"], "WS10M": cur["WS10M"],
        "DUSMASS_ugm3": dust, "DTR": cur["T2M_max"] - cur["T2M_min"],
        "Harmattan_Wind_Flag": int(30 <= cur["WD10M"] <= 100 and cur["WS10M"] > 3.0 and cur["RH2M"] < 40),
        "week_sin": np.sin(2 * np.pi * wk / 52), "week_cos": np.cos(2 * np.pi * wk / 52),
        "DUSMASS_lag1": dust_p, "RH2M_lag1": prior["RH2M"],
        "DUSMASS_2wk_rolling_mean": (dust + dust_p) / 2, "PREC_2wk_rolling_sum": cur["PREC"] + prior["PREC"],
    }
    feats["epi_week"] = wk  # display only; never passed to the models
    return feats


def stack_proba(X, stack):
    """Raw (uncalibrated) stack probability. X columns in FEATURES order (13 model predictors)."""
    X = np.asarray(X, dtype=float).reshape(-1, len(FEATURES))
    cols = STACKS[stack]["base_cols"]
    base = {}
    if "catboost_proba" in cols: base["catboost_proba"] = M["catboost"].predict_proba(X)[:, 1]
    if "arnn_proba" in cols: base["arnn_proba"] = M["arnn"].predict_proba(M["arnn_scaler"].transform(X))[:, 1]
    if "svm_proba" in cols: base["svm_proba"] = M["svm"].predict_proba(M["svm_scaler"].transform(X))[:, 1]
    return STACKS[stack]["meta_learner"].predict_proba(np.column_stack([base[c] for c in cols]))[:, 1]


def lookup(state, date):
    r = CLIM[(CLIM.state == state) & (CLIM.date == pd.Timestamp(date))]
    return None if r.empty else r.iloc[0].to_dict()


def climatological_dust(state, iso_week):
    """Fallback dust value for the live-pull mode: the historical median DUSMASS for
    this state in this ISO week. NASA POWER does not serve MERRA-2 aerosol mass
    concentration, and a genuinely live MERRA-2 pull is not available at low latency
    (see the thesis's own limitation on MERRA-2 release delay); this climatological
    substitute is used instead and is clearly labelled wherever it appears."""
    sub = CLIM[(CLIM.state == state) & (CLIM.iso_week == iso_week)]
    return float(sub["DUSMASS"].median()) if not sub.empty else float(CLIM["DUSMASS"].median())


def fetch_power_week(state, week_start_str):
    """Live pull from the NASA POWER daily point API for the 7 days of week_start,
    aggregated to a single weekly record in the same units as climate_weekly.csv."""
    lat, lon = STATE_COORDS[state]
    start = pd.Timestamp(week_start_str)
    end = start + pd.Timedelta(days=6)
    params = {
        "parameters": POWER_PARAMS, "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=20)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    df = pd.DataFrame(p)
    return {
        "T2M_mean": float(df["T2M"].mean()), "T2M_max": float(df["T2M_MAX"].max()),
        "T2M_min": float(df["T2M_MIN"].min()), "RH2M": float(df["RH2M"].mean()),
        "PREC": float(df["PRECTOTCORR"].mean()), "WS10M": float(df["WS10M"].mean()),
        "WD10M": float(df["WD10M"].mean()),
    }


def verdict_html(state, stack, feats, week_start, source):
    p = float(stack_proba([feats[f] for f in FEATURES], stack)[0])
    thr = float(STACKS[stack]["threshold"])
    tier = map_to_tier(p)
    target = (pd.Timestamp(week_start) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    binary = "Elevated environmental vigilance" if p >= thr else "Non-elevated"
    html = (
        f"<div style='border-left:10px solid {TIER_COLOR[tier]};padding:12px 16px;background:#f7f7f8;border-radius:6px;color:#222'>"
        f"<div style='font-size:13px;color:#555'>{state}: forecast for the week starting {target} (issued from the week of {week_start}; {source})</div>"
        f"<div style='font-size:28px;font-weight:700;margin:4px 0'>{tier} tier &nbsp;<span style='font-size:18px;font-weight:500'>probability {p:.3f}</span></div>"
        f"<div><b>Decision guidance (prototype):</b> {TIER_MAP[tier]}</div>"
        f"<div><b>Epidemiological week of the issue date:</b> {int(feats['epi_week'])} (display only; not a model input)</div>"
        f"<div><b>Binary classification:</b> {binary} (threshold {thr:.2f}; {STACK_LABEL[stack]})</div>"
        f"<div style='font-size:12px;color:#666;margin-top:6px'>Research prototype. The output is a MEWS-derived environmental classification, not a forecast of meningitis cases or outbreaks.</div></div>"
    )
    return html, feats


def local_shap_fig(feats, stack):
    """KernelExplainer on the complete stack function (all base learners and the meta-learner)."""
    import shap
    f = lambda Z: stack_proba(Z, stack)
    ex = shap.KernelExplainer(f, BG[FEATURES].values)
    x = np.array([[feats[k] for k in FEATURES]], dtype=float)
    sv = np.asarray(ex.shap_values(x, nsamples=200, silent=True)).reshape(-1)
    o = np.argsort(np.abs(sv))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([FEATURES[i] for i in o], sv[o], color=["#d93025" if sv[i] > 0 else "#1a73e8" for i in o])
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("SHAP value (impact on predicted probability)")
    ax.set_title(f"Local explanation (base value {float(np.ravel(ex.expected_value)[0]):.3f}); descriptive, not causal")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Callback functions for the Gradio UI
# ---------------------------------------------------------------------------

def default_live_week():
    today = pd.Timestamp(dt.date.today())
    default_start = (today - pd.Timedelta(days=today.weekday()) - pd.Timedelta(weeks=2))
    return default_start.strftime("%Y-%m-%d")


def run_live(state, stack, week_start):
    try:
        ws = pd.Timestamp(week_start)
    except Exception:
        return "<div style='color:#d93025'>Enter the date as YYYY-MM-DD.</div>", None
    if ws.weekday() != 0:
        return "<div style='color:#d93025'>The week start must be a Monday, consistent with the weekly training structure.</div>", None
    prior_start = ws - pd.Timedelta(days=7)
    try:
        cur_live = fetch_power_week(state, ws.strftime("%Y-%m-%d"))
        prior_live = fetch_power_week(state, prior_start.strftime("%Y-%m-%d"))
    except Exception as e:
        return f"<div style='color:#d93025'>Could not reach the NASA POWER API ({e}). Try the Historical or Manual modes instead.</div>", None
    wk_cur = int(ws.isocalendar()[1])
    wk_prior = int(prior_start.isocalendar()[1])
    dust_cur = climatological_dust(state, wk_cur) * 1e-9   # back to kg/m^3 for build_features
    dust_prior = climatological_dust(state, wk_prior) * 1e-9
    cur = {**cur_live, "DUSMASS": dust_cur}
    prior = {**prior_live, "DUSMASS": dust_prior}
    feats = build_features(ws, cur, prior)
    note = f"<div style='font-size:12px;color:#555;margin-bottom:6px'>Live POWER data for {state}; dust is a climatological substitute for ISO week {wk_cur} (DUSMASS ≈ {dust_cur*1e9:.0f} µg/m³).</div>"
    html, feats = verdict_html(state, stack, feats, ws.strftime("%Y-%m-%d"), "live NASA POWER pull + climatological dust")
    return note + html, feats


def historical_choices(state):
    dates = CLIM[CLIM.state == state].sort_values("date")["date"].iloc[1:]
    opts = [d.strftime("%Y-%m-%d") for d in dates][::-1]
    return gr.update(choices=opts, value=(opts[0] if opts else None))


def run_historical(state, stack, date):
    if not date:
        return "<div style='color:#d93025'>Select a week.</div>", None
    cur, prior = lookup(state, date), lookup(state, pd.Timestamp(date) - pd.Timedelta(days=7))
    if cur is None or prior is None:
        return "<div style='color:#d93025'>No recorded climate data for that week or its prior week.</div>", None
    feats = build_features(date, cur, prior)
    return verdict_html(state, stack, feats, date, "recorded climate data")


def run_manual(state, stack, date, t2m, tmax, tmin, rh, prec, ws_, wd, dust, rh_p, dust_p, prec_p):
    try:
        date_ts = pd.Timestamp(date)
    except Exception:
        return "<div style='color:#d93025'>Enter the date as YYYY-MM-DD.</div>", None
    if date_ts.weekday() != 0:
        return "<div style='color:#d93025'>The week start must be a Monday, consistent with the weekly training structure.</div>", None
    cur = dict(T2M_mean=t2m, T2M_max=tmax, T2M_min=tmin, RH2M=rh, PREC=prec, WS10M=ws_, WD10M=wd, DUSMASS=dust / 1e9)
    prior = dict(RH2M=rh_p, DUSMASS=dust_p / 1e9, PREC=prec_p)
    feats = build_features(date, cur, prior)
    return verdict_html(state, stack, feats, date, "manual entry")


def run_explain(feats, stack):
    if feats is None:
        return None
    return local_shap_fig(feats, stack)


def set_mode(mode):
    return (gr.update(visible=mode == "Live (NASA POWER)"),
            gr.update(visible=mode == "Historical week"),
            gr.update(visible=mode == "Manual entry (new week)"))


def global_shap_fig():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(GLOBAL.index, GLOBAL.values, color="#1a73e8")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    return fig


def history_fig_and_table(state):
    h = HIST[HIST.state == state].copy()
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ys = h["forecast_tier"].map({"White": 0, "Yellow": 1, "Orange": 2, "Red": 3})
    ax.scatter(pd.to_datetime(h["date"]), ys, c=h["forecast_tier"].map(TIER_COLOR), s=24)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["White", "Yellow", "Orange", "Red"])
    fig.autofmt_xdate()
    fig.tight_layout()
    h["elevated_risk_2wk"] = h["elevated_risk_2wk"].map({1.0: "Elevated", 0.0: "Non-elevated"})
    h = h.rename(columns={
        "state": "State", "date": "Week start",
        "elevated_risk_2wk": "Observed MEWS-derived class (2 weeks ahead)",
        "forecast_tier": "Forecast tier",
    })
    return fig, h


ABOUT_MD = """### About
**Purpose.** A research prototype that forecasts, two weeks ahead and year-round, whether environmental
conditions in Adamawa, Borno or Yobe will fall in the elevated MEWS-derived classification (Orange or
Red) of the Meningitis Environmental Risk Early Warning System (MEWS, after Dione et al., 2022). Unlike
the published MEWS, which is scoped to epidemiological weeks 1-26, this prototype forecasts year-round
by design, since outbreaks in Nigeria have also occurred outside the dry season.

**Model.** Stacked heterogeneous ensemble (ARNN, CatBoost, SVM-RBF base learners) with an Elastic Net
logistic meta-learner. Both the base learners and the meta-learner are tuned in a nested loop within
each outer cross-validation fold. The primary configuration, C1 (ARNN + CatBoost), places genuine
positive weight on both base learners and is the configuration fixed by the thesis's prespecified
selection rule; C4 (ARNN + CatBoost + SVM-RBF) is reported alongside it as the comparison stack.
Thirteen model predictors from NASA POWER and MERRA-2 data; the epidemiological week is displayed for
reference only and is not a model input. Probability tiers: White < 0.20, Yellow 0.20-0.40,
Orange 0.40-0.60, Red >= 0.60 (probability bands that reuse the MEWS colours for a different quantity).

**Live data.** The "Live (NASA POWER)" mode pulls temperature, humidity, precipitation and wind from
the public NASA POWER API for the state's approximate centroid. Dust (DUSMASS) is a MERRA-2 product
without a low-latency public feed, so this mode substitutes the historical climatological dust level
for the state and calendar week, clearly labelled wherever it is used.

**Limits.** The target is derived from MEWS rules that share inputs with the predictors, so skill
reflects anticipation of a MEWS-derived classification and is not evidence about meningitis incidence
or outbreaks. The sealed test covers about two annual cycles; a weekly climatology baseline is reported
in the thesis for comparison. The tiers and guidance have not been evaluated against surveillance
outcomes or with end users. SHAP explanations describe the fitted model and are not causal. This is not
a production surveillance tool and does not replace professional epidemiological judgement.
"""

EXPLAIN_CAPTION = (
    "SHAP values taken from the pipeline's own stage13_shap_importance.csv for the selected "
    "configuration C1, on the sealed-test rows. The ranking is led by week_sin, a calendar/seasonal "
    "term, followed by RH2M_lag1 and T2M_mean; DUSMASS_lag1 ranks 11th of 13 predictors. "
    "Attributions describe the fitted model and are not causal. Temperature, humidity and dust are "
    "also inputs to the MEWS rules that define the target. The local explanation for a single "
    "forecast is shown on the Prediction tab."
)

with gr.Blocks(title="Meningitis Environmental Risk Early Warning - Prototype") as demo:
    gr.Markdown("# Meningitis Environmental Risk Early Warning (research prototype)")
    gr.Markdown(
        "Two-week-ahead forecast of the MEWS-derived environmental classification for Adamawa, "
        "Borno and Yobe. Forecasts year-round, not restricted to epidemiological weeks 1-26."
    )

    last_feats = gr.State(None)
    last_stack = gr.State(PRIMARY)

    with gr.Tabs():
        with gr.Tab("Prediction"):
            with gr.Row():
                state_dd = gr.Dropdown(STATES, value="Borno", label="State")
                stack_dd = gr.Dropdown(STACK_CHOICES, value=PRIMARY, label="Stack")
            mode_radio = gr.Radio(
                ["Live (NASA POWER)", "Historical week", "Manual entry (new week)"],
                value="Live (NASA POWER)", label="Input",
            )

            with gr.Group(visible=True) as live_group:
                gr.Markdown(
                    "Pulls the most recently complete week of temperature, humidity, precipitation and wind "
                    "from the NASA POWER API for the state's approximate centroid. Dust (DUSMASS) is a MERRA-2 "
                    "product with a release delay unsuitable for low-latency use (noted as a limitation in the "
                    "thesis); this prototype substitutes the historical median dust level for that state and "
                    "ISO week, and marks it below as a climatological substitute, not a live observation."
                )
                live_week = gr.Textbox(label="Forecast-origin week (Monday, YYYY-MM-DD)", value=default_live_week())
                live_btn = gr.Button("Pull live data and forecast", variant="primary")

            with gr.Group(visible=False) as hist_group:
                hist_date = gr.Dropdown(choices=[], label="Week start (Monday)")
                hist_btn = gr.Button("Generate forecast", variant="primary")

            with gr.Group(visible=False) as manual_group:
                manual_date = gr.Textbox(label="Week start (Monday, YYYY-MM-DD)", value="2026-01-05")
                gr.Markdown("**Forecast-origin week**")
                with gr.Row():
                    t2m = gr.Number(label="Mean temperature T2M_mean (°C)", value=24.0)
                    tmax = gr.Number(label="Maximum temperature T2M_max (°C)", value=32.0)
                    tmin = gr.Number(label="Minimum temperature T2M_min (°C)", value=14.0)
                    rh = gr.Number(label="Relative humidity RH2M (%)", value=25.0, minimum=0.0, maximum=100.0)
                with gr.Row():
                    prec = gr.Number(label="Precipitation PREC (mm/day)", value=0.0, minimum=0.0)
                    ws_ = gr.Number(label="Wind speed WS10M (m/s)", value=4.0, minimum=0.0)
                    wd = gr.Number(label="Wind direction WD10M (°)", value=50.0, minimum=0.0, maximum=360.0)
                    dust = gr.Number(label="Dust surface mass (µg/m³)", value=500.0, minimum=0.0)
                gr.Markdown("**Prior week**")
                with gr.Row():
                    rh_p = gr.Number(label="Prior-week relative humidity (%)", value=25.0, minimum=0.0, maximum=100.0)
                    dust_p = gr.Number(label="Prior-week dust (µg/m³)", value=500.0, minimum=0.0)
                    prec_p = gr.Number(label="Prior-week precipitation (mm/day)", value=0.0, minimum=0.0)
                manual_btn = gr.Button("Generate forecast", variant="primary")

            verdict_out = gr.HTML()
            explain_btn = gr.Button("Compute local SHAP explanation")
            explain_plot = gr.Plot(label="Local explanation")

            mode_radio.change(set_mode, mode_radio, [live_group, hist_group, manual_group])
            state_dd.change(historical_choices, state_dd, hist_date)
            demo.load(historical_choices, state_dd, hist_date)

            live_btn.click(run_live, [state_dd, stack_dd, live_week], [verdict_out, last_feats])
            hist_btn.click(run_historical, [state_dd, stack_dd, hist_date], [verdict_out, last_feats])
            manual_btn.click(
                run_manual,
                [state_dd, stack_dd, manual_date, t2m, tmax, tmin, rh, prec, ws_, wd, dust, rh_p, dust_p, prec_p],
                [verdict_out, last_feats],
            )
            for btn in (live_btn, hist_btn, manual_btn):
                btn.click(lambda s: s, stack_dd, last_stack)
            explain_btn.click(run_explain, [last_feats, last_stack], explain_plot)

        with gr.Tab("Explainability"):
            gr.Markdown("### Global feature importance (primary configuration, sealed test)")
            gr.Plot(global_shap_fig(), label="Global SHAP importance")
            gr.Markdown(EXPLAIN_CAPTION)

        with gr.Tab("History"):
            hist_state_dd = gr.Dropdown(STATES, value="Borno", label="State")
            hist_plot = gr.Plot(label="Forecast tier over time")
            hist_table = gr.Dataframe(label="Sealed-test forecasts")

            def _history_update(state):
                fig, h = history_fig_and_table(state)
                return fig, h

            hist_state_dd.change(_history_update, hist_state_dd, [hist_plot, hist_table])
            demo.load(_history_update, hist_state_dd, [hist_plot, hist_table])
            gr.Markdown("Sealed-test forecasts of the primary configuration (309 state-weeks, 2024-2025).")

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

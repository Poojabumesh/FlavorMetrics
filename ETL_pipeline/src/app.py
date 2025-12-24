# app.py — Beer production live KPIs
import glob
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh

# Prediction helpers
from predict_next_step import (
    build_transition_rows,
    load_step_stats,
    train_transition_model,
)

RAW_ROOT = Path("data/raw")
MART_ROOT = Path("data/marts")
PREDICTIONS_ROOT = MART_ROOT / "predictions"
ENGINE = "fastparquet"
RAW_ROOT.mkdir(parents=True, exist_ok=True)
MART_ROOT.mkdir(parents=True, exist_ok=True)
PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)

# Spec limits to visualize boundaries on charts (aligned with consumer).
SPECS_LIMITS = {
    ("mashing", "temp"): (62, 68),
    ("boiling", "temp"): (98, 101),
    ("fermentation", "temp"): (18, 22),
    ("fermentation", "ph"): (3.8, 4.6),
    ("fermentation", "gravity_degP"): (10.0, 13.5),
    ("fermentation", "co2_pressure"): (0.6, 1.2),
    ("fermentation", "gravity"): (1.010, 1.030),
    ("packaging", "count"): (80, 120),
    ("packaging", "fill_level"): (330, 340),
    ("packaging", "cap_torque"): (12, 20),
    ("packaging", "line_speed"): (90, 150),
    ("packaging", "reject_rate"): (0.0, 4.0),
}

st.set_page_config(page_title="Beer Production KPIs", layout="wide")

# --- custom styling (sky blue + pastel palette) ---
PALETTE = {
    "sky": "#4CB5F5",
    "mint": "#A3E4D7",
    "peach": "#FFC97B",
    "lavender": "#C7A4FF",
    "ink": "#0B1224",
    "deep_blue": "#0B3C5D",
}

st.markdown(
    f"""
    <style>
        /* Background + container */
        div[data-testid="stAppViewContainer"] {{
            background: radial-gradient(circle at 10% 20%, {PALETTE["mint"]} 0%, #d6f5ff 25%, #b8e4ff 60%);
            color: {PALETTE["ink"]};
        }}
        div.block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PALETTE["sky"]} 0%, {PALETTE["lavender"]} 100%);
            color: {PALETTE["ink"]};
        }}
        section[data-testid="stSidebar"] .stSelectbox > label,
        section[data-testid="stSidebar"] .stRadio > label {{
            color: {PALETTE["ink"]};
        }}
        /* Metrics + cards */
        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, {PALETTE["sky"]}, {PALETTE["peach"]});
            color: {PALETTE["ink"]};
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 14px;
            padding: 12px;
            box-shadow: 0 12px 24px rgba(12, 60, 93, 0.25);
        }}
        /* Dataframes */
        div[data-testid="stDataFrame"] table thead tr th {{
            background: {PALETTE["deep_blue"]} !important;
            color: #F6FAFF !important;
        }}
        div[data-testid="stDataFrame"] table tbody tr:nth-child(even) {{
            background: rgba(255,255,255,0.08) !important;
        }}
        div[data-testid="stDataFrame"] table tbody tr:nth-child(odd) {{
            background: rgba(255,255,255,0.14) !important;
        }}
        /* Headers */
        h1, h2, h3 {{
            color: {PALETTE["ink"]};
        }}
        .stAlert {{
            border-radius: 12px;
        }}
        /* Charts */
        div[data-testid="stVegaLiteChart"] canvas {{
            background: linear-gradient(135deg, #ecf7ff, #f7ecff) !important;
            border-radius: 12px;
        }}
        div[data-testid="stAltairChart"] canvas {{
            background: linear-gradient(135deg, #ecf7ff, #f7ecff) !important;
            border-radius: 12px;
        }}
        div[data-testid="stPlotlyChart"] {{
            background: linear-gradient(135deg, #ecf7ff, #f7ecff) !important;
            border-radius: 12px;
        }}
        /* Buttons */
        button[kind="secondary"] {{
            background: linear-gradient(135deg, {PALETTE["sky"]}, {PALETTE["lavender"]});
            color: {PALETTE["ink"]} !important;
            border: none;
        }}
        button[kind="secondary"]:hover {{
            filter: brightness(1.05);
        }}
        /* QA / testing cards */
        .qa-card {{
            background: #f8fafc;
            border: 1px solid #e5e9f2;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }}
        .qa-card.compact {{
            padding: 10px 12px;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        .metric-block {{
            min-height: 180px;
            display: flex;
            align-items: stretch;
        }}
        .metric-block .qa-card {{
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin: 0;
        }}
        .qa-title {{
            font-size: 0.9rem;
            color: #4a5568;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .qa-value {{
            font-size: 1.8rem;
            color: #1f2937;
            font-weight: 700;
        }}
        .qa-card.compact .qa-value {{
            font-size: 1.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- sidebar controls ---
st.sidebar.title("Filters")
page = st.sidebar.radio("Page", ["Overview", "Predictions", "Testing"], index=0)
sel_date = st.sidebar.date_input("Date", value=date.today())
auto_refresh = st.sidebar.selectbox("Auto-refresh", ["5s", "10s", "30s", "Off"], index=1)
refresh_ms = {"5s": 5000, "10s": 10000, "30s": 30000, "Off": 0}[auto_refresh]
if refresh_ms:
    st_autorefresh(interval=refresh_ms, key="autorefresh")

view_mode = st.sidebar.radio(
    "Show",
    ["All data", "Only anomalies"],
    index=0,
)

# --- data loaders ---
@st.cache_data(ttl=10, show_spinner=False)
def load_raw_for_date(d: date) -> pd.DataFrame | None:
    # our consumer_beer_parquet.py writes: data/raw/date=YYYY-MM-DD/beer-*.parquet
    part_dir = RAW_ROOT / f"date={d.isoformat()}"
    parts = sorted(glob.glob(str(part_dir / "*.parquet")))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p, engine=ENGINE) for p in parts], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    return df

@st.cache_data(ttl=10, show_spinner=False)
def load_kpi_for_date(d: date) -> pd.DataFrame | None:
    # kpi_beer_minute.py writes: data/marts/beer_kpi_date=YYYY-MM-DD.parquet
    f = MART_ROOT / f"beer_kpi_date={d.isoformat()}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, engine=ENGINE)
    # Normalize column names so charts downstream are consistent.
    if "avg_value" in df.columns and "mean_value" not in df.columns:
        df["mean_value"] = df["avg_value"]
    if "in_spec_rate" in df.columns and "oos_rate" not in df.columns:
        df["oos_rate"] = 1 - df["in_spec_rate"]
    # minute should already be datetime, but make sure
    if "minute" in df.columns:
        df["minute"] = pd.to_datetime(df["minute"], utc=True, errors="coerce")
    return df


def mean_chart_with_limits(df: pd.DataFrame, step: str, sensor: str, title: str) -> None:
    """Render mean value line with spec limit rules if known."""
    if df.empty or "minute" not in df.columns or "mean_value" not in df.columns:
        st.info("No KPI rows yet for this filter.")
        return

    chart_df = df[["minute", "mean_value"]].dropna()
    if chart_df.empty:
        st.info("No KPI rows yet for this filter.")
        return

    base = (
        alt.Chart(chart_df)
        .mark_line(color=PALETTE["deep_blue"], strokeWidth=2)
        .encode(
            x=alt.X("minute:T", title="Time"),
            y=alt.Y("mean_value:Q", title="Mean value"),
            tooltip=[
                alt.Tooltip("minute:T", title="Time"),
                alt.Tooltip("mean_value:Q", title="Mean"),
            ],
        )
    )

    overlays = []
    key = (step, sensor)
    if key in SPECS_LIMITS:
        low, high = SPECS_LIMITS[key]
        limits_df = pd.DataFrame({"value": [low, high], "Limit": ["Lower", "Upper"]})
        band_df = pd.DataFrame({"low": [low], "high": [high]})
        band = (
            alt.Chart(band_df)
            .mark_rect(opacity=0.18, color=PALETTE["mint"])
            .encode(y="low:Q", y2="high:Q")
        )
        rules = (
            alt.Chart(limits_df)
            .mark_rule(strokeDash=[6, 4], color=PALETTE["peach"])
            .encode(y="value:Q", tooltip=["Limit:N", "value:Q"])
        )
        overlays.extend([band, rules])

    chart = alt.layer(*(overlays + [base])).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- main ---
st.title("🍺 Beer Production — Live KPIs")

raw_df = load_raw_for_date(sel_date)
kpi_df = load_kpi_for_date(sel_date)
kpi_sel = pd.DataFrame()

if page == "Overview":
    if raw_df is None or raw_df.empty:
        st.info(
            f"No raw parts found in {RAW_ROOT}/date={sel_date.isoformat()}/ yet. "
            "Keep the beer simulator & consumer running to generate data."
        )
    else:
        # dynamic filters
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            plant = st.selectbox("Plant", sorted(raw_df["plant_id"].dropna().unique()), key="plant_live")
        with c2:
            line = st.selectbox("Line", sorted(raw_df["line_id"].dropna().unique()), key="line_live")
        with c3:
            step = st.selectbox("Step", sorted(raw_df["step"].dropna().unique()), key="step_live")
        with c4:
            step_sensors = sorted(raw_df[raw_df["step"] == step]["sensor"].dropna().unique())
            if not step_sensors:
                step_sensors = sorted(raw_df["sensor"].dropna().unique())
            sensor = st.selectbox("Sensor", step_sensors, key="sensor_live")

        df_sel = raw_df.query(
            "plant_id == @plant and line_id == @line and step == @step and sensor == @sensor"
        ).copy()

        df_sel["ts"] = pd.to_datetime(df_sel["ts"], utc=True, errors="coerce")
        df_sel = df_sel.dropna(subset=["ts"])
        
        # --- anomaly detection (last 5 minutes) ---
        if view_mode == "Only anomalies":

            now_utc = datetime.now(timezone.utc)
            window_start = now_utc - timedelta(minutes=5)

        
            recent = df_sel[df_sel["ts"] >= window_start]
            recent_oos = recent[recent["in_spec"] == False]

            if not recent_oos.empty:
                st.error(f"⚠️ {len(recent_oos)} out-of-spec readings in the last 5 minutes.")
            else:
                st.success("✅ No out-of-spec readings in the last 5 minutes.")
        
            st.subheader("Recent anomalies")
            if not recent_oos.empty:
                show_cols = [
                    "ts",
                    "value",
                    "unit",
                    "step",
                    "sensor",
                    "plant_id",
                    "line_id",
                    "batch_id",
                ]
                st.dataframe(
                    recent_oos.sort_values("ts", ascending=False)[show_cols].head(30),
                    use_container_width=True,
                    height=250,
                )
            st.stop()

        if df_sel.empty:
            st.warning("No data after applying filters.")
        else:
            # if KPI file exists, use it (it already has per-minute aggregates)
            if kpi_df is not None and not kpi_df.empty:
                kpi_sel = kpi_df.query(
                    "step == @step and sensor == @sensor"
                ).sort_values("minute")

            # fallback to ad-hoc aggregation if no precomputed KPIs for this filter
            if kpi_df is None or kpi_df.empty or kpi_sel.empty:
                df_sel["value"] = pd.to_numeric(df_sel["value"], errors="coerce")
                df_sel = df_sel.dropna(subset=["value"])
                df_sel["minute"] = df_sel["ts"].dt.floor("min")
                kpi_sel = (
                    df_sel.groupby(["plant_id", "line_id", "step", "sensor", "minute"])
                    .agg(
                        readings=("value", "count"),
                        mean_value=("value", "mean"),
                        oos_rate=("in_spec", lambda x: 1 - x.mean()),
                    )
                    .reset_index()
                    .sort_values("minute")
                )

            latest_row = kpi_sel.iloc[-1] if not kpi_sel.empty else None
            latest_readings = int(latest_row["readings"]) if latest_row is not None else 0
            latest_mean = (
                f"{latest_row['mean_value']:.2f}"
                if latest_row is not None and "mean_value" in latest_row
                else "—"
            )
            latest_oos = (
                f"{(latest_row['oos_rate']*100):.1f}%"
                if latest_row is not None and "oos_rate" in latest_row
                else "—"
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Readings (last minute)",
                    latest_readings,
                )
            with m2:
                st.metric(
                    "Mean value (last minute)",
                    latest_mean,
                )
            with m3:
                st.metric(
                    "OOS rate (last minute)",
                    latest_oos,
                )


    #Helper func for Historical trends
    @st.cache_data(ttl=10, show_spinner=False)
    def load_parts_for(day_iso: str):
        import glob
        parts = sorted((RAW_ROOT / f"date={day_iso}").glob("*.parquet"))
        if not parts:
            return None
        df = pd.concat([pd.read_parquet(p, engine=ENGINE) for p in parts], ignore_index=True)
        # normalize
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        if "sensor" not in df.columns and "param" in df.columns:
            df = df.rename(columns={"param": "sensor"})
        # types
        df["in_spec"] = df["in_spec"].astype(bool)
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df


    # ---- Historical Trends ----
    st.header("📈 Historical Trends")

    end_date = st.date_input("End date", value=date.today(), key="trend_end")
    days_back = st.slider("Days back", 1, 14, 7, key="trend_window")

    frames = []
    for i in range(days_back):
        d = (pd.to_datetime(end_date) - pd.Timedelta(days=i)).date().isoformat()
        df = load_parts_for(d)
        if df is not None and not df.empty:
            df["day"] = pd.to_datetime(d).date()
            frames.append(df)

    if not frames:
        st.info("No data found in the selected window.")
    else:
        hist = pd.concat(frames, ignore_index=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            plant_t = st.selectbox("Plant", sorted(hist["plant_id"].dropna().unique()), key="plant_hist")
        with c2:
            line_t = st.selectbox("Line", sorted(hist["line_id"].dropna().unique()), key="line_hist")
        with c3:
            step_t = st.selectbox("Step", sorted(hist["step"].dropna().unique()), key="step_hist")
        with c4:
            step_sensors_t = sorted(hist[hist["step"] == step_t]["sensor"].dropna().unique())
            if not step_sensors_t:
                step_sensors_t = sorted(hist["sensor"].dropna().unique())
            sensor_t = st.selectbox("Sensor", step_sensors_t, key="sensor_hist")

        hist_sel = hist.query(
            "plant_id == @plant_t and line_id == @line_t and step == @step_t and sensor == @sensor_t"
        ).copy()

        if hist_sel.empty:
            st.warning("No rows after applying filters.")
        else:
            daily = (hist_sel.groupby(["day","plant_id","line_id","step","sensor"])
                     .agg(readings=("value","count"),
                          in_spec_rate=("in_spec","mean"),
                          avg_value=("value","mean"))
                     .reset_index()
                     .sort_values("day"))

            m1, m2 = st.columns(2)
            with m1:
                st.subheader("In-spec rate by day")
                st.line_chart(daily.set_index("day")[["in_spec_rate"]])
            with m2:
                st.subheader("Average value by day")
                st.line_chart(daily.set_index("day")[["avg_value"]])

            st.subheader("Daily breakdown")
            st.dataframe(daily, use_container_width=True, height=320)# ---- Summary KPIs (today) ----

    st.header("🧪 Process Summary KPIs (today)")

    today_df = load_parts_for(date.today().isoformat())
    if today_df is None or today_df.empty:
        st.info("No data for today yet.")
    else:
        df = today_df.copy()

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # Define your critical steps (edit to match your simulator)
        CRITICAL = ["mashing","boiling","fermentation"]

        critical = df[df["step"].isin(CRITICAL)]
        yield_rate = critical["in_spec"].mean() if not critical.empty else None

        # Batch success: min in-spec across critical steps >= 95%
        STEP_TARGET = 0.95
        if not critical.empty:
            per_batch_step = (critical.groupby(["batch_id","step"])
                              .agg(in_spec_rate=("in_spec","mean"))
                              .reset_index())
            ok_by_batch = per_batch_step.groupby("batch_id")["in_spec_rate"].min()
            batch_success_rate = (ok_by_batch >= STEP_TARGET).mean()
        else:
            batch_success_rate = None

        # Throughput from packaging step (sum of units if you emit it there)
        packaging = df[df["step"]=="packaging"]
        throughput = int(packaging["value"].sum()) if not packaging.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Yield (critical steps)",
                  f"{(yield_rate*100):.1f}%" if yield_rate is not None else "—")
        c2.metric("Batch Success Rate",
                  f"{(batch_success_rate*100):.1f}%" if batch_success_rate is not None else "—")
        c3.metric("Throughput (units, today)", throughput)

        st.caption("Batch considered successful if min in-spec across critical steps ≥ 95%.")    
        
        # Last-minute OOS rate across today's data.
        df["minute"] = df["ts"].dt.floor("min")
        minute_oos = (
            df.groupby("minute")["in_spec"]
            .apply(lambda x: 1 - x.mean())
            .reset_index(name="oos_rate")
            .sort_values("minute")
        )
        latest_oos = minute_oos.iloc[-1] if not minute_oos.empty else None
        st.metric(
            "OOS rate (last minute)",
            f"{(latest_oos['oos_rate']*100):.1f}%"
            if latest_oos is not None
            else "—",
        )

        left, right = st.columns((2, 1))
        with left:
            st.subheader("Mean value over time")
            mean_chart_with_limits(kpi_sel, step, sensor, "Mean value over time")
        with right:
            st.subheader("OOS rate over time")
            if not kpi_sel.empty:
                chart_df2 = kpi_sel.set_index("minute")[["oos_rate"]]
                st.line_chart(chart_df2)
            else:
                st.info("No KPI rows yet for this filter.")

        st.subheader("Latest raw readings")
        tail = df.sort_values("ts").tail(50)
        local_tz = datetime.now().astimezone().tzinfo
        tail["ts_local"] = tail["ts"].dt.tz_convert(local_tz)
        cols = [
                "ts_local",
                "ts",
                "value",
                "unit",
                "in_spec",
                "plant_id",
                "line_id",
                "step",
                "sensor",
                "batch_id",
         ]
        st.dataframe(tail[cols], use_container_width=True, height=300)

elif page == "Predictions":
    st.header("🔮 Next-Step Predictions")
    data_root = RAW_ROOT
    try:
        # Use full history so we include older batches; adjust max_files here if needed.
        step_stats = load_step_stats(data_root, max_files=None)
    except FileNotFoundError:
        st.error(f"No Parquet files found under {data_root}.")
        st.stop()

    try:
        transitions_df = build_transition_rows(step_stats)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if transitions_df.empty:
        st.info("No transitions available for training yet.")
        st.stop()

    transitions = sorted(transitions_df.transition.unique())
    transition = st.selectbox("Select transition", transitions, key="pred_transition")

    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="pred_test_size")
    n_estimators = st.slider("RandomForest estimators", 20, 300, 100, 20, key="pred_n_estimators")

    report = train_transition_model(
        transitions_df, transition=transition, test_size=test_size, n_estimators=n_estimators
    )

    st.subheader("Model performance")
    st.write(f"Rows: train={report['train_rows']} test={report['test_rows']}")
    for target, score in report["mae_by_target"].items():
        st.write(f"MAE {target}: {score:.4f}")

    # Compare predicted vs. actual for a recent window of rows
    subset = transitions_df[transitions_df["transition"] == transition].copy()
    target_cols = [c for c in subset.columns if c.startswith("target_") and subset[c].notna().any()]
    feature_cols = [c for c in subset.columns if c not in target_cols and c not in {"plant_id","line_id","batch_id","current_step","next_step","transition"}]

    subset = subset.dropna(subset=target_cols)
    feature_medians = subset[feature_cols].median()
    subset[feature_cols] = subset[feature_cols].fillna(feature_medians)
    window_size = 10
    tail = subset.tail(window_size).copy()
    preds = report["model"].predict(tail[feature_cols])

    for i, col in enumerate(target_cols):
        tail[f"pred_{col}"] = preds[:, i]

    # Persist the latest prediction window to Parquet so we have a record on disk.
    outpath = None
    if not tail.empty:
        ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_transition = transition.replace("->", "_")
        outpath = PREDICTIONS_ROOT / f"preds_{safe_transition}_{ts_str}.parquet"
        tail_to_save = tail.copy()
        tail_to_save["saved_at_utc"] = datetime.utcnow()
        tail_to_save["transition"] = transition
        tail_to_save.to_parquet(outpath, engine=ENGINE, index=False)

    st.subheader(f"Last {window_size} predictions vs ground truth")
    show_cols = ["batch_id", "current_step", "next_step"] + target_cols + [f"pred_{c}" for c in target_cols]
    st.dataframe(tail[show_cols], use_container_width=True, height=320)
    if outpath:
        st.caption(f"Saved snapshot to `{outpath}`")

    st.subheader(f"Prediction vs ground truth over last {window_size}")
    for col in target_cols:
        chart_df = tail[["batch_id", col, f"pred_{col}"]].copy()
        melted = chart_df.melt(id_vars="batch_id", var_name="series", value_name="value")
        if melted.empty:
            st.info(f"No values available for {col}.")
            continue
        chart = (
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x=alt.X("batch_id:N", title="Batch"),
                y=alt.Y("value:Q", title=col),
                color=alt.Color("series:N", title="Series"),
                tooltip=["batch_id", "series", alt.Tooltip("value:Q", format=".3f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
elif page == "Testing":
    st.header("🧪 Testing / QA Dashboard")

    @st.cache_data(show_spinner=False)
    def testing_demo_data() -> dict:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        defect_rate_over_time = [5, 4, 5, 6, 7, 7, 6, 7, 8, 9, 8, 14]
        quality_metric = [70, 67, 80, 83, 78, 80, 81, 84, 80, 74, 66, 72]
        defect_counts = [10, 8, 12, 9, 11, 10, 9, 11, 10, 13, 9, 16]
        rework_counts = [3, 2, 4, 5, 6, 5, 6, 7, 6, 5, 6, 7]
        scrap_counts = [8, 6, 7, 8, 9, 8, 9, 8, 9, 10, 8, 9]
        downtime_minutes = [120, 98, 110, 111, 120, 132, 142, 120, 125, 110, 95, 80]
        produced = [150, 142, 138, 140, 145, 148, 150, 146, 140, 135, 130, 125]
        good_production = [110, 120, 123, 130, 142, 138, 134, 132, 130, 121, 110, 100]

        return {
            "summary": {
                "defect_rate": "7%",
                "fpy": "91%",
                "scrap_rate": "8%",
                "total_defects": "99",
                "total_downtime_pct": "63%",
                "downtime_minutes": "426",
                "quality_overtime": 66,
                "equipment_availability": 95,
            },
            "defect_rate_df": pd.DataFrame({"month": months, "defect_rate": defect_rate_over_time}),
            "counts_df": pd.DataFrame(
                {
                    "month": months,
                    "Defect_Count": defect_counts,
                    "Rework_Count": rework_counts,
                    "Scrap_Count": scrap_counts,
                }
            ),
            "downtime_df": pd.DataFrame(
                {"month": months, "downtime_minutes": downtime_minutes, "total_produced": produced}
            ),
            "production_df": pd.DataFrame(
                {"month": months, "produced": produced, "good_production": good_production}
            ),
            "quality_metric_df": pd.DataFrame({"month": months, "quality_pct": quality_metric}),
        }

    def render_card(col, title: str, value: str, compact: bool = False):
        class_name = "qa-card compact" if compact else "qa-card"
        col.markdown(
            f"""
            <div class="{class_name}">
                <div class="qa-title">{title}</div>
                <div class="qa-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_metric_block(col, title: str, value: str):
        col.subheader(title)
        col.markdown(
            f"""
            <div class="metric-block">
                <div class="qa-card compact">
                    <div class="qa-title">{title}</div>
                    <div class="qa-value">{value}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def gauge_chart(value: float, title: str, color: str = "#f4b000"):
        source = pd.DataFrame(
            {
                "category": ["value", "rest"],
                "amount": [value, max(0, 100 - value)],
            }
        )
        base = alt.Chart(source).encode(theta=alt.Theta("amount:Q"), color=alt.Color("category:N", scale=None))
        foreground = base.transform_filter(alt.datum.category == "value").mark_arc(
            innerRadius=60, outerRadius=90, color=color
        )
        background = base.transform_filter(alt.datum.category == "rest").mark_arc(
            innerRadius=60, outerRadius=90, color="#e5e9f2"
        )
        text = (
            alt.Chart(pd.DataFrame({"text": [f"{value:.0f}%"]}))
            .mark_text(size=24, fontWeight="bold", color="#1f2937")
            .encode(text="text:N")
        )
        subtitle = (
            alt.Chart(pd.DataFrame({"text": [title]}))
            .mark_text(size=12, dy=22, color="#4a5568")
            .encode(text="text:N")
        )
        return alt.layer(background, foreground, text, subtitle).configure_view(stroke=None)

    data_demo = testing_demo_data()
    summary = data_demo["summary"]

    top = st.columns(6)
    render_card(top[0], "Defect Rate", summary["defect_rate"])
    render_card(top[1], "First Pass Yield (FPY)", summary["fpy"])
    render_card(top[2], "Scrap Rate", summary["scrap_rate"])
    render_card(top[3], "Total defects", summary["total_defects"])
    render_card(top[4], "Total downtime", summary["total_downtime_pct"])
    render_card(top[5], "Downtime minutes", summary["downtime_minutes"])

    st.markdown(" ")
    metric_col1, metric_col2 = st.columns(2)
    render_metric_block(metric_col1, "Quality Overtime", f"{summary['quality_overtime']}%")
    render_metric_block(metric_col2, "Equipment Availability", f"{summary['equipment_availability']}%")

    chart_defect = (
        alt.Chart(data_demo["defect_rate_df"])
        .mark_line(point=True, color="#f4b000", strokeWidth=3)
        .encode(
            x=alt.X("month:N", sort=None),
            y=alt.Y("defect_rate:Q", title="Defect rate (%)"),
            tooltip=["month", alt.Tooltip("defect_rate:Q", format=".1f")],
        )
        .properties(height=320)
    )

    counts_long = data_demo["counts_df"].melt("month", var_name="type", value_name="count")
    chart_counts = (
        alt.Chart(counts_long)
        .mark_bar()
        .encode(
            x=alt.X("month:N", sort=None),
            y=alt.Y("count:Q"),
            color=alt.Color("type:N", scale=alt.Scale(range=["#f4b000", "#4c5565", "#a0616a"])),
            tooltip=["month", "type", "count"],
        )
        .properties(height=320)
    )

    chart_downtime = (
        # Pre-melt to avoid Altair dtype inference errors on transform-generated fields
        alt.Chart(data_demo["downtime_df"].melt("month", var_name="series", value_name="value"))
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", sort=None),
            y=alt.Y("value:Q", title="Minutes / Units"),
            color=alt.Color("series:N", legend=None, scale=alt.Scale(range=["#f4b000", "#4c5565"])),
            tooltip=["month", "series", alt.Tooltip("value:Q", format=".1f")],
        )
        .properties(height=320)
    )

    prod_long = data_demo["production_df"].melt("month", var_name="type", value_name="units")
    chart_prod = (
        alt.Chart(prod_long)
        .mark_bar()
        .encode(
            x=alt.X("month:N", sort=None),
            y=alt.Y("units:Q"),
            color=alt.Color("type:N", scale=alt.Scale(range=["#f4b000", "#4c5565"]), title=""),
            tooltip=["month", "type", "units"],
        )
        .properties(height=320)
    )

    chart_quality = (
        alt.Chart(data_demo["quality_metric_df"])
        .mark_line(point=True, color="#f4b000", strokeWidth=3)
        .encode(
            x=alt.X("month:N", sort=None),
            y=alt.Y("quality_pct:Q", title="Quality (%)"),
            tooltip=["month", alt.Tooltip("quality_pct:Q", format=".1f")],
        )
        .properties(height=320)
    )

    st.markdown(" ")
    bars_row_left, bars_row_right = st.columns(2)
    bars_row_left.subheader("Defect, Rework and Scrap Counts by Month")
    bars_row_left.altair_chart(chart_counts, use_container_width=True)
    bars_row_right.subheader("Monthly Production vs Good Production")
    bars_row_right.altair_chart(chart_prod, use_container_width=True)

    st.markdown(" ")
    row_other_col1, row_other_col2, row_other_col3 = st.columns(3)
    row_other_col1.subheader("Defect Rate Over Time")
    row_other_col1.altair_chart(chart_defect, use_container_width=True)

    row_other_col2.subheader("Downtime Impact on Production")
    row_other_col2.altair_chart(chart_downtime, use_container_width=True)
    row_other_col2.markdown(
        '<span style="color:#f4b000;">downtime_minutes</span> • '
        '<span style="color:#4c5565;">total_produced</span>',
        unsafe_allow_html=True,
    )

    row_other_col3.subheader("Quality Metric Over Time")
    row_other_col3.altair_chart(chart_quality, use_container_width=True)
else:
    st.info(
        f"No raw parts found in {RAW_ROOT}/date={sel_date.isoformat()}/ yet. "
        "Keep the beer simulator & consumer running to generate data."
    )

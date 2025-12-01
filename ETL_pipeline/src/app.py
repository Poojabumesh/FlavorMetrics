# app.py — Beer production live KPIs
import glob
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Prediction helpers
from predict_next_step import (
    build_transition_rows,
    load_step_stats,
    train_transition_model,
)

RAW_ROOT = Path("data/raw")
MART_ROOT = Path("data/marts")
ENGINE = "fastparquet"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Beer Production KPIs", layout="wide")

# --- sidebar controls ---
st.sidebar.title("Filters")
page = st.sidebar.radio("Page", ["Overview", "Predictions"], index=0)
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
            if not kpi_sel.empty:
                chart_df = kpi_sel.set_index("minute")[["mean_value"]]
                st.line_chart(chart_df)
            else:
                st.info("No KPI rows yet for this filter.")
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
        step_stats = load_step_stats(data_root, max_files=600)
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

    # Compare predicted vs. actual for last 5 rows of this transition
    subset = transitions_df[transitions_df["transition"] == transition].copy()
    target_cols = [c for c in subset.columns if c.startswith("target_") and subset[c].notna().any()]
    feature_cols = [c for c in subset.columns if c not in target_cols and c not in {"plant_id","line_id","batch_id","current_step","next_step","transition"}]

    subset = subset.dropna(subset=target_cols)
    feature_medians = subset[feature_cols].median()
    subset[feature_cols] = subset[feature_cols].fillna(feature_medians)
    tail = subset.tail(5).copy()
    preds = report["model"].predict(tail[feature_cols])

    for i, col in enumerate(target_cols):
        tail[f"pred_{col}"] = preds[:, i]

    st.subheader("Last 5 predictions vs actual")
    show_cols = ["batch_id", "current_step", "next_step"] + target_cols + [f"pred_{c}" for c in target_cols]
    st.dataframe(tail[show_cols], use_container_width=True, height=300)

    st.subheader("Prediction vs actual over last 5")
    for col in target_cols:
        chart_df = tail[["batch_id", col, f"pred_{col}"]].set_index("batch_id")
        st.line_chart(chart_df)
else:
    st.info(
        f"No raw parts found in {RAW_ROOT}/date={sel_date.isoformat()}/ yet. "
        "Keep the beer simulator & consumer running to generate data."
    )

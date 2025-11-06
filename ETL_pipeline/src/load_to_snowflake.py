import os, glob
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

RAW_ROOT = Path("data/raw")

ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT")
USER      = os.getenv("SNOWFLAKE_USER")
PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "BEER")
SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
TABLE     = os.getenv("SNOWFLAKE_TABLE", "RAW_READINGS")

engine = create_engine(
    "snowflake://",
    connect_args=dict(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),         # XHGUFHO-QFB25468
        user=os.getenv("SNOWFLAKE_USER"),               # POOJABUMESH
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),     # COMPUTE_WH
        database=os.getenv("SNOWFLAKE_DATABASE"),       # BEER
        schema=os.getenv("SNOWFLAKE_SCHEMA"),           # PUBLIC
        # authenticator="externalbrowser",              # if you use SSO
        # role=os.getenv("SNOWFLAKE_ROLE","ACCOUNTADMIN"),
    ),
)


def concat_day(day_iso: str) -> pd.DataFrame:
    parts = glob.glob(f"data/raw/date={day_iso}/*.parquet")
    if not parts:
        raise SystemExit(f"No parts for {day_iso}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.rename(columns={"param": "sensor"})
    cols = ["plant_id","line_id","batch_id","step","sensor","value","unit","ts","in_spec"]
    df = df[cols]
    return df

def load_day(day_iso: str):
    df = concat_day(day_iso)
    with engine.begin() as con:
        con.exec_driver_sql(f"""
        create table if not exists {TABLE} (
          plant_id string, line_id string, batch_id string,
          step string, sensor string, value float, unit string,
          ts timestamp_tz, in_spec boolean
        );""")
        df.to_sql(TABLE, con=con, if_exists="append", index=False)

if __name__ == "__main__":
    import sys
    day = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.utcnow().date().isoformat()
    load_day(day)
    print(f"Loaded {day} -> {TABLE}")

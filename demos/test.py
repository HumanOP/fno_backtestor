import duckdb
import pandas as pd
with duckdb.connect("nifty_1min_desiquant.duckdb") as conn:
    nifty_df = conn.execute('SELECT * FROM "nifty_2022_01_03"').fetchdf()
    # nifty_df = conn.execute('SHOW TABLES').fetch_df()

    print(nifty_df)
nifty_df["timestamp"] = nifty_df["timestamp"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
nifty_df["expiry_date"] = nifty_df["expiry_date"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
print(nifty_df)
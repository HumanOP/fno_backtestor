import requests
import duckdb
import pandas as pd
from requests.auth import HTTPBasicAuth

class StreamingDataReader:
    def __init__(self, host, port, https, username, password, batch_size=1000000, db_path="nifty_opt_icici_1min.duckdb"):
        self.username = username
        self.password = password
        self.host = f"https://{host}:{port}" if https else f"http://{host}:{port}"
        self.batch_size = batch_size
        self.last_timestamp = None
        self.db_path = db_path

    def stream_data(self, table_name, time_column="timestamp"):
        """Generator function to fetch data in batches."""
        while True:
            if self.last_timestamp:
                query = f"SELECT * FROM {table_name} WHERE {time_column} > '{self.last_timestamp}' ORDER BY {time_column} ASC LIMIT {self.batch_size}"
            else:
                query = f"SELECT * FROM {table_name} ORDER BY {time_column} ASC LIMIT {self.batch_size}"

            response = requests.get(f"{self.host}/exec", params={"query": query}, auth=HTTPBasicAuth(self.username, self.password))

            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
                break

            data = response.json()
            if not data["dataset"]:  # Stop when no more data
                break

            df = pd.DataFrame(data["dataset"], columns=[col["name"] for col in data["columns"]])

            # Convert timestamp columns, handling numeric timestamps
            for col in ["timestamp", "expiry_date"]:
                if col in df.columns:
                    if df[col].dtype in ["float64", "int64"]:
                        df[col] = pd.to_datetime(df[col], unit="us", errors="coerce")  # Assume microseconds for QuestDB
                    else:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

            self.last_timestamp = df["timestamp"].max()  # Update last timestamp

            yield df  # Yield data chunk

    def store_data(self, df):
        """Stores data in DuckDB tables, creating separate tables based on timestamp date."""
        with duckdb.connect(self.db_path) as conn:
            date_groups = df.groupby(df["timestamp"].dt.date)  # Group by timestamp date (only date part)
            
            for date, date_df in date_groups:
                table_name = f"nifty_{date.strftime('%Y_%m_%d')}"  # Table name format: nifty_YYYY_MM_DD
                
                # *Fixing the Naming Issue*: Enclose in double quotes
                table_name = f'"{table_name}"'  # Enclose in quotes for DuckDB

                # Ensure table exists
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    ticker TEXT,
                    timestamp TIMESTAMP,
                    open FLOAT, high FLOAT, low FLOAT, close FLOAT,
                    volume FLOAT, open_interest FLOAT, strike_price FLOAT,
                    instrument_type TEXT, expiry_date TIMESTAMP, 
                    Time_to_expiry FLOAT, spot_price FLOAT,
                    r FLOAT, iv FLOAT, delta FLOAT, gamma FLOAT, vega FLOAT,
                    theta FLOAT, rho FLOAT
                )
                """
                conn.execute(create_table_query)

                # Insert data with explicit column mapping
                conn.register("temp_df", date_df)
                insert_query = f"""
                INSERT INTO {table_name} (
                    ticker, timestamp, open, high, low, close, volume, open_interest,
                    strike_price, instrument_type, expiry_date, Time_to_expiry,
                    spot_price, r, iv, delta, gamma, vega, theta, rho
                )
                SELECT
                    ticker, timestamp, open, high, low, close, volume, open_interest,
                    strike_price, instrument_type, expiry_date, Time_to_expiry,
                    spot_price, r, iv, delta, gamma, vega, theta, rho
                FROM temp_df
                """
                conn.execute(insert_query)

# Initialize Streaming Reader
reader = StreamingDataReader(host='qdb2.twocc.in', port=443, https=True, username='2Cents', password='2Cents1012cc')

# Stream Data & Store in DuckDB by Timestamp Date
table_name = "ohlc_indian_options_icici_nifty_1min"
for chunk in reader.stream_data(table_name, time_column="timestamp"):
    reader.store_data(chunk)
    print(f"Inserted {len(chunk)} rows across date tables")

print("Finished streaming all data.")
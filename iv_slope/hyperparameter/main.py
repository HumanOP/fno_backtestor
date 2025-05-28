import os
import re
import duckdb
import pandas as pd
import traceback
from backtester import parse_table_name
from walk_forward_optimizer import WalkForwardOptimizer

if __name__ == "__main__":
    db_path = "C:\\Users\\Administrator\\Desktop\\Divyanshu desiquant\\iv_slope\\nifty_1min_desiquant.duckdb"
    conn = None
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file '{db_path}' not found in {os.getcwd()}")
        conn = duckdb.connect(db_path)
        
        table_names = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchdf()
        if table_names.empty:
            raise ValueError("No tables found in the database")
        
        print("Table names DataFrame columns:", table_names.columns.tolist())
        print("First few rows of table_names:")
        print(table_names.head())
        
        table_names = table_names[90:703]
        pattern = re.compile(r'nifty_\d{4}_\d{2}_\d{2}')
        table_names = table_names[table_names['table_name'].str.match(pattern)]
        if table_names.empty:
            raise ValueError("No dates available in the specified range [90:703]")
        
        dates = pd.Series([parse_table_name(name) for name in table_names['table_name']], index=table_names.index)
        dates = dates.dropna()
        if dates.empty:
            raise ValueError("No valid dates parsed from table names")
        dates = dates.apply(lambda x: x.strftime('%Y-%m-%d'))
        
        legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg3': {'type': 'CE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg4': {'type': 'PE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None}
        }
        hyperparameter_grid = {
            "upper_gamma": [0.0, 0.02, 0.04],
            "upper_buffer": [-0.02, 0.00, 0.02],
            "lower_buffer": [-0.03, -0.06, -0.09],
            "lower_gamma": [-0.05, -0.08, -0.11]
        }
        
        print("Running Walk-Forward Optimization...")
        optimizer = WalkForwardOptimizer(legs, hyperparameter_grid, conn, dates, in_sample_ratio=0.6/4, out_sample_ratio=0.2/4)
        best_params, avg_performance, wfo_results = optimizer.optimize()
        
        print(f"Best parameters: {best_params}")
        print(f"Average out-of-sample Sharpe Ratio: {avg_performance}")
        print("WFO results saved to 'wfo_results.json'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the database file exists in the current working directory or provide the correct path.")
    except PermissionError:
        print(f"Error: Permission denied accessing '{db_path}'")
        print("Please check file permissions and ensure you have read/write access.")
    except duckdb.IOException as e:
        print(f"Error: Failed to open database: {e}")
        print("The database file may be corrupted or locked. Try restoring from a backup or checking for open connections.")
    except duckdb.ConnectionException as e:
        print(f"Connection error: {e}")
        print("The database connection was closed unexpectedly. Check for connection timeouts or resource issues.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        if conn is not None:
            try:
                conn.close()
                print("Database connection closed successfully.")
            except Exception as e:
                print(f"Error closing connection: {e}")
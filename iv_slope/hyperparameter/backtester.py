import duckdb
import pandas as pd
import math
import time
from datetime import datetime, date
import re
from trade_manager import TradeManager

iv = {}
time_arr = {}
data = None

def calculate_performance(trader: TradeManager) -> float:
    trades = trader.build_tradebook()
    if trades.empty:
        return 0.0
    closed_trades = trades[trades['exit_price'].notna()]
    if closed_trades.empty:
        return 0.0
    profits = (closed_trades['pnl'] * closed_trades['qty']).to_numpy()
    returns = pd.Series(profits)
    if len(returns) < 2:
        return 0.0
    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)
    return sharpe_ratio if not pd.isna(sharpe_ratio) else 0.0

def parse_table_name(table_name: str):
    """Convert table name like 'YYYY-MM-DD' or 'nifty_YYYY_MM_DD' to a date object."""
    if not isinstance(table_name, str):
        print(f"Invalid table name: {table_name} (type: {type(table_name)})")
        return None
    if re.match(r'\d{4}-\d{2}-\d{2}', table_name):
        try:
            return datetime.strptime(table_name, "%Y-%m-%d").date()
        except ValueError as e:
            print(f"Error parsing date from {table_name}: {e}")
            return None
    match = re.match(r'nifty_(\d{4})_(\d{2})_(\d{2})', table_name)
    if match:
        year, month, day = match.groups()
        print(f"Parsed {table_name}: year={year}, month={month}, day={day}")
        try:
            year, month, day = int(year), int(month), int(day)
            if not (1 <= month <= 12) or not (1 <= day <= 31):
                print(f"Invalid date components in {table_name}")
                return None
            return date(year, month, day)
        except ValueError as e:
            print(f"Error parsing date from {table_name}: {e}")
            return None
    print(f"Cannot parse table name: {table_name}")
    return None

def backtest(legs: dict, iv_slope_thresholds: dict, duckdb_conn, trader: TradeManager, dates: pd.Series, quantity : int = 1) -> float:
    global iv, time_arr, data
    signal = 0
    position_id = 1
    counter = 1
    iv_slope = 0
    parsed_dates = [parse_table_name(dt) for dt in dates]
    parsed_dates = [dt for dt in parsed_dates if dt is not None]
    if not parsed_dates:
        print("Error: No valid dates parsed from table names")
        return 0.0

    for dt in parsed_dates:
        date_str = f"nifty_{dt.strftime('%Y_%m_%d')}"
        start_time = time.time()
        print(f"Processing table: {date_str} at {counter}")
        counter += 1
        try:
            data_df = duckdb_conn.execute(f"SELECT * FROM {date_str} ORDER BY timestamp").fetchdf()
        except Exception as e:
            print(f"Error querying table {date_str}: {e}")
            continue
        time_to_expiry = sorted(data_df["Time_to_expiry"].unique())
        data_df.set_index("timestamp", inplace=True)
        ticker_map = {ticker: group for ticker, group in data_df.groupby("ticker", sort=False)}
        empty_df_template = data_df.iloc[0:0]

        for leg in legs.values():
            try:
                valid_tte = min(tte for tte in time_to_expiry if any(lower <= tte <= upper for lower, upper in [leg["expiry_range"]]))
                leg["expiry"] = data_df.loc[data_df["Time_to_expiry"] == valid_tte, "expiry_date"].iloc[0]
                if not isinstance(leg["expiry"], (datetime, pd.Timestamp, date)):
                    print(f"Invalid expiry {leg['expiry']} for leg {leg['type']}")
                    leg["expiry"] = pd.NaT
            except (ValueError, IndexError) as e:
                print(f"Error setting expiry for leg {leg['type']}: {e}")
                continue

        spot = data_df[["spot_price"]][~data_df.index.duplicated(keep="first")]
        for row in spot.itertuples():
            if row.spot_price is None or pd.isna(row.spot_price):
                continue
            atm = round(row.spot_price / 50) * 50
            for leg in legs.values():
                if leg["target_strike"] == "ATM":
                    leg["strike"] = float(atm)
                try:
                    if pd.isna(leg["expiry"]):
                        raise ValueError("Expiry is NaT")
                    contract = f"NIFTY{pd.Timestamp(leg['expiry']).strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
                except (ValueError, TypeError) as e:
                    print(f"Error forming contract for leg {leg['type']}: {e}")
                    continue
                leg["contract"] = contract
                subset_df = ticker_map.get(contract, empty_df_template)
                data = subset_df
                avl_time = subset_df.index.asof(row.Index) if not subset_df.empty else None
                leg["data"] = subset_df.loc[avl_time] if not pd.isna(avl_time) else None

            missing_legs = [leg["contract"] for leg in legs.values() if leg["data"] is None]
            if missing_legs:
                continue

            if (pd.Timestamp("15:29:00").time() <= pd.Timestamp(row.Index).time() <= pd.Timestamp("15:30:00").time()):
                try:
                    iv_slope = math.log((legs["leg1"]["data"]["iv"] + legs["leg2"]["data"]["iv"]) /
                                        (legs["leg3"]["data"]["iv"] + legs["leg4"]["data"]["iv"]), 10)
                    iv[row.Index] = (iv_slope, row.spot_price)
                except (ValueError, TypeError) as e:
                    print(f"Error calculating iv_slope: {e}")
                    continue

            new_signal = (
                (iv_slope > iv_slope_thresholds["upper_gamma"]) * 3 +
                (iv_slope_thresholds["upper_gamma"] >= iv_slope > iv_slope_thresholds["upper_buffer"]) * 2 +
                (iv_slope_thresholds["upper_buffer"] >= iv_slope > 0) * 1 +
                (0 >= iv_slope > iv_slope_thresholds["lower_buffer"]) * -1 +
                (iv_slope_thresholds["lower_buffer"] >= iv_slope > iv_slope_thresholds["lower_gamma"]) * -2 +
                (iv_slope_thresholds["lower_gamma"] >= iv_slope) * -3
            )

            active_trades = trader.active_trades()
            if (not active_trades) and (pd.Timestamp(row.Index).time() < pd.Timestamp("15:00:00").time()):
                if new_signal in (-2, 2):
                    continue
                elif new_signal == 1:
                    entry_type_dict = {'weekly': 'BUY', 'monthly': 'SELL'}
                elif new_signal == -1:
                    entry_type_dict = {'weekly': 'SELL', 'monthly': 'BUY'}
                elif new_signal in (-3, 3):
                    entry_type_dict = {'weekly': 'BUY', 'monthly': None}

                for leg_id, leg in legs.items():
                    entry_type = entry_type_dict.get(leg["expiry_type"])
                    if entry_type is None:
                        continue
                    entry_data = {
                        'strategy_id': 'strat1',
                        'position_id': position_id,
                        'leg_id': leg_id,
                        'symbol': leg["contract"],
                        'entry_date': pd.Timestamp(row.Index).date(),
                        'entry_time': pd.Timestamp(row.Index).time(),
                        'entry_price': leg["data"]["close"],
                        'qty': quantity,
                        'entry_type': entry_type,
                        'entry_spot': row.spot_price,
                        'stop_loss': None,
                        'take_profit': None,
                        'entry_reason': f'{new_signal} signal entry',
                    }
                    trader.place_order(entry_data)
                position_id += 1
            else:
                near_expiry = None
                for index, trade in active_trades:
                    try:
                        expiry = datetime.strptime(trade["symbol"][-14:-7], "%d%b%y").date()
                        near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing expiry from trade symbol {trade['symbol']}: {e}")
                        continue
                exit_reason = (
                    "End of Data reached" if ((dt == parsed_dates[-1]) and
                        (pd.Timestamp(row.Index).time() > pd.Timestamp("15:00:00").time())) else
                    "Near Expiry reached" if (pd.Timestamp(row.Index).date() == near_expiry) else
                    "Signal changed" if (signal != new_signal) else
                    None
                )
                if exit_reason:
                    for index, trade in active_trades:
                        contract = trade["symbol"]
                        subset_df = ticker_map.get(contract, empty_df_template)
                        subset_df = subset_df.loc[:row.Index]
                        close_price = subset_df.iloc[-1]["close"] if not subset_df.empty else trade["entry_price"]
                        exit_data = {
                            'exit_date': pd.Timestamp(row.Index).date(),
                            'exit_time': pd.Timestamp(row.Index).time(),
                            'exit_price': close_price,
                            'exit_spot': row.spot_price,
                            'exit_reason': exit_reason,
                        }
                        trader.square_off(index, exit_data)
                if signal == new_signal:
                    leg_strike = legs["leg2"]["strike"]
                    if not ((row.spot_price * 0.99) <= leg_strike <= (row.spot_price * 1.01)):
                        for leg_id, leg in legs.items():
                            entry_type = entry_type_dict.get(leg["expiry_type"])
                            if entry_type is None:
                                continue
                            entry_data = {
                                'strategy_id': 'strat1',
                                'position_id': position_id,
                                'leg_id': leg_id,
                                'symbol': leg["contract"],
                                'entry_date': pd.Timestamp(row.Index).date(),
                                'entry_time': pd.Timestamp(row.Index).time(),
                                'entry_price': leg["data"]["close"],
                                'qty': 1,
                                'entry_type': entry_type,
                                'entry_spot': row.spot_price,
                                'stop_loss': None,
                                'take_profit': None,
                                'entry_reason': 'Adjustment Calendar',
                            }
                            trader.place_order(entry_data)
                        position_id += 1
            signal = new_signal
        time_arr[date_str] = time.time() - start_time

    active_trades = trader.active_trades()
    if active_trades:
        print(f"Force-closing {len(active_trades)} open trades")
        last_date = parsed_dates[-1] if parsed_dates else date.today()
        for index, trade in active_trades:
            contract = trade["symbol"]
            subset_df = ticker_map.get(contract, empty_df_template)
            close_price = subset_df.iloc[-1]["close"] if not subset_df.empty else trade["entry_price"]
            exit_data = {
                'exit_date': last_date,
                'exit_time': pd.Timestamp("15:30:00").time(),
                'exit_price': close_price,
                'exit_spot': spot.iloc[-1]["spot_price"] if not spot.empty else trade["entry_spot"],
                'exit_reason': "End of backtest",
            }
            trader.square_off(index, exit_data)

    return calculate_performance(trader)
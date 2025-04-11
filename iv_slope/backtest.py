import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.trade_manager import TradeManager
from core.statistics_builder import StatisticsBuilder

import duckdb
import pandas as pd
import math
import time
from datetime import datetime, timedelta


def backtest(legs, iv_lower_slope, iv_upper_slope, dates, duckdb, trader):
    signal = 0
    postion_id = 1
    iv_slope = 0
    for date_str in dates:
        print(f"Processing table: {date_str}")
        data_df = duckdb.execute(f"SELECT * FROM {date_str} ORDER BY timestamp").fetchdf()

        data_df["timestamp"] = data_df["timestamp"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
        data_df["expiry_date"] = data_df["expiry_date"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)

        time_to_expiry = sorted(data_df["Time_to_expiry"].unique())

        data_df.set_index("timestamp", inplace=True)

        ticker_map = {
            ticker: group
            for ticker, group in data_df.groupby("ticker", sort=False)
        }
        empty_df_template = data_df.iloc[0:0]
        
        for leg in legs.values():
            valid_tte = min(tte for tte in time_to_expiry if any(lower <= tte <= upper for lower, upper in [leg["expiry_range"]]))
            leg["expiry"] = data_df.loc[data_df["Time_to_expiry"] == valid_tte, "expiry_date"].iloc[0]

        spot = data_df[["spot_price"]][~data_df.index.duplicated(keep="first")]

        for row in spot.itertuples():
            atm = round(row.spot_price / 50) * 50

            for leg in legs.values():
                if leg["target_strike"] == "ATM":
                    leg["strike"] = float(atm)
                contract = f"NIFTY{pd.Timestamp(leg['expiry']).strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
                leg["contract"] = contract
                subset_df = ticker_map.get(contract, empty_df_template)
                avl_time = subset_df.index.asof(row.Index) if not subset_df.empty else None
                leg["data"] = subset_df.loc[avl_time] if not pd.isna(avl_time) else None

            missing_legs = [leg["contract"] for leg in legs.values() if leg["data"] is None]
            if missing_legs:
                print(f"IV not found for {row.Index}. Spot: {row.spot_price} Missing legs: {missing_legs}")
                continue
            
            # Taking IV from the last available time
            if (pd.Timestamp("15:29:00").time() <= pd.Timestamp(row.Index).time() <= pd.Timestamp("15:30:00").time()):
                iv_slope = math.log((legs["leg1"]["data"]["iv"] + legs["leg2"]["data"]["iv"]) / (legs["leg3"]["data"]["iv"] + legs["leg4"]["data"]["iv"]) ,10)
                print(f"{row.spot_price}  {legs['leg1']['data']['iv']} {legs['leg2']['data']['iv']} {legs['leg3']['data']['iv']} {legs['leg4']['data']['iv']}")



            new_signal = (iv_slope > iv_upper_slope) * 2 + (iv_upper_slope >= iv_slope > 0) * 1 + (0 >= iv_slope > iv_lower_slope) * -1 + (iv_slope <= iv_lower_slope) * -2
            
            active_trades = trader.active_trades()
            if (not active_trades) and (pd.Timestamp(row.Index).time() < pd.Timestamp("15:00:00").time()):
                if new_signal == -2 or new_signal == 2:         # Ignoring gamma zone
                    continue
                elif new_signal == 1:
                    entry_type_dict = {'weekly': 'SELL', 'monthly': 'BUY'}
                elif new_signal == -1:
                    entry_type_dict = {'weekly': 'BUY', 'monthly': 'SELL'}


                for leg_id, leg in legs.items():
                    entry_data = {
                        'strategy_id': 'strat1',
                        'position_id': postion_id,
                        'leg_id': leg_id,
                        'symbol': leg["contract"],
                        'entry_date': pd.Timestamp(row.Index).date(),
                        'entry_time': pd.Timestamp(row.Index).time(),
                        'entry_price': leg["data"]["close"],
                        'qty': 1,
                        'entry_type': entry_type_dict.get(leg["expiry_type"]),
                        'entry_spot': row.spot_price,
                        'stop_loss': None,
                        'take_profit': None,
                        'entry_reason': f'{new_signal} signal entry',
                    }
                    trader.place_order(entry_data)
                postion_id += 1
                
            else:
                # Exit if near expiry date is reached
                near_expiry = None
                for index, trade in active_trades:
                    expiry = datetime.strptime(trade["symbol"][-14:-7], "%d%b%y").date()
                    near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
                exit_reason = (
                    "End of Data reached" if ((date_str == dates.values[-1]) and (pd.Timestamp(row.Index).time() > pd.Timestamp("15:00:00").time())) else
                    "Near Expiry reached" if (pd.Timestamp(row.Index).date() == near_expiry) else
                    "Signal changed" if (signal != new_signal) else
                    None
                )
                if exit_reason:
                    for index, trade in active_trades:
                        contract = trade["symbol"]
                        subset_df = ticker_map.get(contract, empty_df_template)
                        subset_df = subset_df.loc[:row.Index]
                        if not subset_df.empty:
                            close_price = subset_df.iloc[-1]["close"]
                        else:
                            print(f"Alert: Not able to exit leg: {trade["leg_id"]}, trade id: {trade["position_id"]}")
                            close_price = trade["entry_price"]
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
                    if (row.spot_price*0.99) <= leg_strike <= (row.spot_price*1.01):
                        # Case (a)
                        continue
                    else:
                        # Case (b)
                        # take new ATM Calendar
                        for leg_id, leg in legs.items():
                            entry_data = {
                                'strategy_id': 'strat1',
                                'position_id': postion_id,
                                'leg_id': leg_id,
                                'symbol': leg["contract"],
                                'entry_date': pd.Timestamp(row.Index).date(),
                                'entry_time': pd.Timestamp(row.Index).time(),
                                'entry_price': leg["data"]["close"],
                                'qty': 1,
                                'entry_type': entry_type_dict.get(leg["expiry_type"]),
                                'entry_spot': row.spot_price,
                                'stop_loss': None,
                                'take_profit': None,
                                'entry_reason': f'Adjustment Calendar',
                            }
                            trader.place_order(entry_data)
                        postion_id += 1

            signal = new_signal



trader = TradeManager()        
iv_upper_slope = 0.0052
iv_lower_slope = -0.0387

portfolio_sl = 0.01
portfolio_tp = 0.03
legs = {
    'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 17], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
    'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 17], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
    'leg3': {'type': 'CE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
    'leg4': {'type': 'PE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None}
    }   
  
conn = duckdb.connect("iv_slope/nifty_opt_1min.duckdb")
try:
    table_names = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchdf()
    dates = table_names["table_name"][1880:]    
    backtest(legs, iv_lower_slope, iv_upper_slope, dates, conn, trader)
except KeyboardInterrupt:
    print("Backtest interrupted by user. Gracefully exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    conn.close()
    print("Connection closed.")


tradebook_df = trader.build_tradebook()
tradebook_df.to_pickle("tradebook.pkl")

stats_builder = StatisticsBuilder(tradebook_df)
performance_stats = stats_builder.build_stats()

print("Performance Statistics:")
for key, value in performance_stats.items():
    print(f"{key}: {value}")

stats_builder.plot_pnl()
stats_builder.plot_drawdown()


### Quantstats Report
import quantstats as qs

tradebook_df = pd.read_pickle("tradebook.pkl")
stats_builder = StatisticsBuilder(tradebook_df)
stats_builder.build_stats()
returns = stats_builder.daily_pnl_sum
# Ensure the index is converted to a DatetimeIndex
returns.index = pd.to_datetime(returns.index, errors='coerce')
# Convert returns to percentage with respect to 400
returns = (returns / 400)   # actually here we will divide by cash used for entry
# Drop any rows with invalid dates
returns = returns.dropna()

print(returns)
qs.reports.html(
    returns,
    compound=False,
    output="tradebook.html",
    title="Tradebook Report",
    title_size=20,
    title_color="blue")
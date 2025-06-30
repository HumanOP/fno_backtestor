import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
# from core.backtesting_opt import _Data, Strategy, Backtest
from core.fw_testing.live_data_fetcher import Endpoint
from core.fw_testing.sync_ibkr_br_adapter import IBKRBrokerAdapter
from core.fw_testing.forwardtesting_opt import Strategy, AlgoRunner

import pandas as pd
import math
from datetime import datetime, date as DateObject # Added DateObject
iv = {}
class IV_Slope(Strategy):
    # Define parameters as class variables for optimization/flexibility
    iv: dict = None
    position_id= 0
    signal= 0
    iv_slope_thresholds = {
        "upper_gamma": 0.15,
        "upper_buffer": 0.05,
        "lower_buffer": -0.10,
        "lower_gamma": -0.3
    }

    portfolio_sl = 0.01
    portfolio_tp = 0.03
    legs = {
        'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
        'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
        'leg3': {'type': 'CE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None},
        'leg4': {'type': 'PE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss':None, 'take_profit':None}
        }
    

    def init(self):
        super().init()
        self.entry_type_dict = None
        print(self.legs)

    def next(self):
        super().next()
        if self.spot is None or pd.isna(self.spot):
            return
        atm = round(self.spot / 50) * 50

        for leg in self.legs.values():
            valid_tte = min(tte for tte in self.tte_to_expiry.keys() if any(lower <= tte <= upper for lower, upper in [leg["expiry_range"]]))
            leg["expiry"] = self.tte_to_expiry[valid_tte]

        for leg in self.legs.values():
            if leg["target_strike"] == "ATM":
                leg["strike"] = float(atm)
            contract = f"NIFTY{pd.Timestamp(leg['expiry']).strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
            leg["contract"] = contract
            leg["data"] = self._data.get_ticker_data(contract)

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            print(f"IV not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return
        
        # if (pd.Timestamp("15:29:00").time() <= pd.Timestamp(row.Index).time() <= pd.Timestamp("15:30:00").time()):
        iv_slope = math.log((self.legs["leg1"]["data"]["iv"] + self.legs["leg2"]["data"]["iv"]) / (self.legs["leg3"]["data"]["iv"] + self.legs["leg4"]["data"]["iv"]) ,10)
        # print(f"{self.spot}  {self.legs['leg1']['data']['iv']} {self.legs['leg2']['data']['iv']} {self.legs['leg3']['data']['iv']} {self.legs['leg4']['data']['iv']}")
        iv[self.time]=(iv_slope, self.spot)

        new_signal = (iv_slope > self.iv_slope_thresholds["upper_gamma"]) * 3 + (self.iv_slope_thresholds["upper_gamma"] >= iv_slope > self.iv_slope_thresholds["upper_buffer"]) * 2 + (self.iv_slope_thresholds["upper_buffer"] >= iv_slope > 0) * 1\
            + (0 >= iv_slope > self.iv_slope_thresholds["lower_buffer"]) * -1 + (self.iv_slope_thresholds["lower_buffer"] >= iv_slope > self.iv_slope_thresholds["lower_gamma"]) * -2 + (self.iv_slope_thresholds["lower_gamma"] >= iv_slope) * -3

        print(f"Signal: {self.signal}, new_signal: {new_signal} IV Slope: {iv_slope} Spot: {self.spot} Time: {self.time}")
        # print(f"New Signal: {new_signal} IV Slope: {iv_slope} Spot: {self.spot}")
        active_trades = self.active_trades

        if (not active_trades) and (pd.Timestamp(self.time).time() < pd.Timestamp("15:00:00").time()):
            if new_signal == -2 or new_signal == 2:         # No trade entry if buffer zone and no active position
                return
            elif new_signal == 1:
                self.entry_type_dict = {'weekly': 'BUY', 'monthly': 'SELL'}          # Original
                # entry_type_dict = {'weekly': 'SELL', 'monthly': 'BUY'}
                # continue
            elif new_signal == -1:
                self.entry_type_dict = {'weekly': 'SELL', 'monthly': 'BUY'}          # Original
                # entry_type_dict = {'weekly': 'BUY', 'monthly': 'SELL'}  
                # continue
            elif new_signal == -3 or new_signal == 3:
                self.entry_type_dict = {'weekly': 'BUY', 'monthly': None}          # Original
                # entry_type_dict = {'weekly': 'SELL', 'monthly': None}
                # continue
            
            placed_any_leg = False
            for leg_id, leg in self.legs.items():
                entry_type = self.entry_type_dict.get(leg["expiry_type"])
                order_fn = {'BUY': self.buy, 'SELL': self.sell}.get(entry_type)
                if order_fn is None:
                    continue

                order_fn(
                    strategy_id='strat1',
                    position_id=self.position_id,
                    leg_id=leg_id,
                    ticker=leg["contract"],
                    quantity=1,
                    stop_loss=None,
                    take_profit=None,
                    tag=f'{new_signal} signal entry'
                )
                placed_any_leg = True
            if placed_any_leg:
                self.position_id += 1
            
        else:
            print(f"MOMOrders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")

            # Exit if near expiry date is reached
            near_expiry = None
            for trade in active_trades:
                expiry = datetime.strptime(trade.ticker[-14:-7], "%d%b%y").date()
                near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
            exit_reason = (
                "Near Expiry reached" if (pd.Timestamp(self.time).date() == near_expiry) else
                "Signal changed" if (self.signal != new_signal) else
                None
            )
            if exit_reason:
                for trade in active_trades:
                    contract = trade.ticker
                    print("closing position")
                    trade.close(trade.size, tag=exit_reason)
            print(f"Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")


            if self.signal == new_signal:
                leg_strike = self.legs["leg2"]["strike"]
                if (self.spot*0.99) <= leg_strike <= (self.spot*1.01):
                    # Case (a)
                    pass
                else:
                    # Case (b)
                    # take new ATM Calendar
                    placed_any_leg = False
                    for leg_id, leg in self.legs.items():
                        entry_type = self.entry_type_dict.get(leg["expiry_type"])
                        order_fn = {'BUY': self.buy, 'SELL': self.sell}.get(entry_type)
                        if order_fn is None:
                            continue

                        order_fn(
                            strategy_id='strat1',
                            position_id=self.position_id,
                            leg_id=leg_id,
                            ticker=leg["contract"],
                            quantity=1,
                            stop_loss=None,
                            take_profit=None,
                            tag=f'Adjustment Calendar'
                        )
                        placed_any_leg = True
                    if placed_any_leg:
                        self.position_id += 1

        self.signal = new_signal



# Run the backtest pipeline
if __name__ == "__main__":
    ft = AlgoRunner(
        endpoint=Endpoint(db_path=db_path),
        strategy=IV_Slope,
        broker=IBKRBrokerAdapter,
        broker_creds=None,
        update_interval=1.0,
        end_time=None
    )
    processed_orders, final_positions, closed_trades, orders = ft.run()
    print("Processed Orders:", processed_orders)
    print("Final Positions:", final_positions)
    import pprint
    pprint.pprint(legs)




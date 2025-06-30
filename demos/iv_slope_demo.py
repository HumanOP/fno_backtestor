import sys
import os
# Add the parent directory to the Python path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtesting_opt1 import _Data, Strategy, Backtest
import pandas as pd
import math
from datetime import datetime, date as DateObject # Added DateObject
from core.risk_manager import FixedLossPositionSizing
iv = {}
class IV_Slope(Strategy):
    # Define parameters as class variables for optimization/flexibility
    iv_slope_thresholds: dict = None
    legs: dict = None
    iv: dict = None
    position_id= 0
    signal= 0
    portfolio_tp = 0
    portfolio_sl = 0
    

    def init(self):
        super().init()
        self.entry_type_dict = None
        print(legs)

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
            leg["data"] = self.get_ticker_data(contract)

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            print(f"IV not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return

        iv_slope = math.log((self.legs["leg1"]["data"]["iv"] + self.legs["leg2"]["data"]["iv"]) / (self.legs["leg3"]["data"]["iv"] + self.legs["leg4"]["data"]["iv"]), 10)
        iv[self.time] = (iv_slope, self.spot)

        new_signal = (iv_slope > self.iv_slope_thresholds["upper_gamma"]) * 3 + \
                    (self.iv_slope_thresholds["upper_gamma"] >= iv_slope > self.iv_slope_thresholds["upper_buffer"]) * 2 + \
                    (self.iv_slope_thresholds["upper_buffer"] >= iv_slope > 0) * 1 + \
                    (0 >= iv_slope > self.iv_slope_thresholds["lower_buffer"]) * -1 + \
                    (self.iv_slope_thresholds["lower_buffer"] >= iv_slope > self.iv_slope_thresholds["lower_gamma"]) * -2 + \
                    (self.iv_slope_thresholds["lower_gamma"] >= iv_slope) * -3

        print(f"Signal: {self.signal}, new_signal: {new_signal} IV Slope: {iv_slope} Spot: {self.spot} Time: {self.time}")
        active_trades = self.active_trades

        if (not active_trades) and (pd.Timestamp(self.time).time() < pd.Timestamp("15:00:00").time()):
            if new_signal == -2 or new_signal == 2:
                return
            elif new_signal == 1:
                self.entry_type_dict = {'weekly': 'BUY', 'monthly': 'SELL'}
            elif new_signal == -1:
                self.entry_type_dict = {'weekly': 'SELL', 'monthly': 'BUY'}
            elif new_signal == -3 or new_signal == 3:
                self.entry_type_dict = {'weekly': 'BUY', 'monthly': None}

            # Calculate net premium for the trade
            net_premium = 0
            for leg_id, leg in self.legs.items():
                entry_type = self.entry_type_dict.get(leg["expiry_type"])
                if entry_type is None:
                    continue
                premium = leg["data"]["close"]
                if pd.isna(premium):
                    print(f"Skipping leg {leg['contract']}: Missing premium")
                    return
                # BUY: negative (debit), SELL: positive (credit)
                net_premium += (-1 if entry_type == 'BUY' else 1) * premium

            # Total margin as net premium (absolute value for position sizing)
            current_equity = self._broker.margin_available
            total_margin = abs(net_premium) * 75  # Per contract
            print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

            # Calculate quantity using FixedLossPositionSizing
            quantity = FixedLossPositionSizing(
                account_value=current_equity,
                margin=total_margin,
                max_loss_percentage=1
            ).size()
            print(f"Quantity: {quantity}")

            if quantity <= 0:
                print(f"Skipping trade: Invalid quantity ({quantity})")
                return

            # Place orders
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
                    quantity=quantity,
                    stop_loss=None,
                    take_profit=None,
                    tag=f'{new_signal} signal entry'
                )
                placed_any_leg = True
            if placed_any_leg:
                self.position_id += 1

        else:
            print(f"MOMOrders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")

            # Exit logic remains unchanged
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
                    print("Closing position")
                    trade.close(trade.size, tag=exit_reason)
            print(f"Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")

            if self.signal == new_signal:
                leg_strike = self.legs["leg2"]["strike"]
                if (self.spot * 0.99) <= leg_strike <= (self.spot * 1.01):
                    # Case (a)
                    pass
                else:
                    # Case (b)
                    # take new ATM Calendar    pass
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
                            quantity=quantity,
                            stop_loss=None,
                            take_profit=None,
                            tag=f'Adjustment Calendar'
                        )
                        placed_any_leg = True
                    if placed_any_leg:
                        self.position_id += 1

        self.signal = new_signal
        

from core.hyperparameter_optimizer import HyperParameterOptimizer
from core.backtesting_opt1 import Backtest
import pandas as pd

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"
    hyperparameter_grid = {
        "upper_gamma": [0.15, 0.18],
        "upper_buffer": [0.05],
        "lower_buffer": [-0.10],
        "lower_gamma": [-0.3],
        "portfolio_sl": [0.02],  # Optimize stop loss
        "portfolio_tp": [0.03]   # Optimize take profit
    }
    
    params = {
        "upper_gamma": 0.15,
        "upper_buffer": 0.05,
        "lower_buffer": -0.10,
        "lower_gamma": -0.3,
        "portfolio_sl": 0.02,  # Optimize stop loss
        "portfolio_tp": 0.03   # Optimize take profit
    }

    legs = {
        'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
        'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
        'leg3': {'type': 'CE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
        'leg4': {'type': 'PE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None}
    }

    
    
    bt = Backtest(
        db_path=db_path,
        strategy=IV_Slope,
        cash=10000000,
        commission_per_contract=0.65,
        option_multiplier=75
    )
    
    
    #stats = bt.run(iv_slope_thresholds=params, legs=legs)
    #bt.tear_sheet()
    
    #print(stats)
    

    
    result = bt.run_window(start_date="2021-01-01", end_date="2024-01-01", iv_slope_thresholds=params, legs=legs)
    bt.tear_sheet()
    print(result)

    """hp = HyperParameterOptimizer(
         db_path=db_path,
         strategy=IV_Slope,
         cash=10000000,
         commission_per_contract=0.65,
         option_multiplier=75,
         legs=legs
     )


    best_params, best_sharpe, results_df = hp.optimize(
         hyperparameter_grid=hyperparameter_grid,
         maximize='Sharpe Ratio',
         method='grid',
         start_date="2022-01-01",
         end_date="2023-04-30"
     )

    print(f"Best Parameters: {best_params}")
    print(f"Best Sharpe Ratio: {best_sharpe}")
    print("Results:")
    print(results_df)"""

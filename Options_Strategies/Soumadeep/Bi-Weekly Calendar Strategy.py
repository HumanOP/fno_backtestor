import sys
import os
from pathlib import Path
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from core.backtesting_opt import _Data, Strategy, Backtest
import math
from datetime import datetime, date as DateObject

class DoubleCalendarStrategy(Strategy):
    # Define parameters as class variables for optimization/flexibility
    legs: dict = None
    position_id = 0
    portfolio_tp = 0
    portfolio_sl = 0

    """Parameters which can be defined here:"""
    stop_loss_percent: float = 0.01  # 1% per trade
    take_profit_percent: float = 0.02  # 2% per trade

    def init(self):
        super().init()
        self.legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 16], 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 16], 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg3': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [20, 23], 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None},
            'leg4': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [20, 23], 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None}
        }
        self.entry_premium = None
        self.entry_spot = None
        print(f"Strategy initialized with legs: {self.legs}")

    def next(self):
        super().next()
        if self.spot is None or pd.isna(self.spot) or not math.isfinite(self.spot):
            print(f"[{self.time}] Invalid spot price: {self.spot}, skipping")
            return

        atm = round(self.spot / 50) * 50
        if not math.isfinite(atm):
            print(f"[{self.time}] Invalid ATM strike: {atm}, skipping")
            return

        # Debug available TTEs
        print(f"[{self.time}] Available TTEs: {list(self.tte_to_expiry.keys())}")

        # Assign expiries
        for leg_id, leg in self.legs.items():
            lower, upper = leg["expiry_range"]
            valid_ttes = [tte for tte in self.tte_to_expiry.keys() if lower <= tte <= upper]
            if not valid_ttes:
                # Fallback: Select closest TTE to the range
                valid_ttes = self.tte_to_expiry.keys()
                if not valid_ttes:
                    print(f"[{self.time}] No TTEs available for {leg_id}, skipping")
                    return
                valid_tte = min(valid_ttes, key=lambda x: min(abs(x - lower), abs(x - upper)))
                print(f"[{self.time}] No TTE in range {leg['expiry_range']} for {leg_id}, using closest TTE: {valid_tte}")
            else:
                valid_tte = min(valid_ttes)
            leg["expiry"] = self.tte_to_expiry[valid_tte]

        # Assign strikes and contracts
        for leg in self.legs.values():
            if leg["target_strike"] == "ATM":
                leg["strike"] = float(atm)
            expiry = leg["expiry"]
            if not isinstance(expiry, (pd.Timestamp, datetime)):
                try:
                    expiry = pd.to_datetime(expiry)
                except (ValueError, TypeError) as e:
                    print(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                    return
            contract = f"NIFTY{expiry.strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
            leg["contract"] = contract
            leg["data"] = self._data.get_ticker_data(contract)

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            print(f"[{self.time}] Data not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return

        # Calculate net premium
        net_premium = 0
        for leg in self.legs.values():
            premium = leg["data"]["close"]
            if pd.isna(premium) or not math.isfinite(premium):
                print(f"[{self.time}] Skipping leg {leg['contract']}: Invalid premium {premium}")
                return
            net_premium += (1 if leg['action'] == 'SELL' else -1) * premium

        if not math.isfinite(net_premium) or net_premium == 0:
            print(f"[{self.time}] Invalid or zero net premium: {net_premium}, skipping")
            return

        current_equity = self._broker.margin_available
        total_margin = abs(net_premium) * 75  # Per contract, as in IV_Slope
        print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

        active_trades = self.active_trades
        quantity = 1  # Fixed quantity as in IV_Slope

        # Entry logic
        if not active_trades and pd.Timestamp(self.time).time() < pd.Timestamp("15:00:00").time():
            placed_any_leg = False
            print(f"[{self.time}] Selected legs: {[leg['contract'] for leg in self.legs.values()]}")
            for leg_id, leg in self.legs.items():
                order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                order_fn(
                    strategy_id='strat1',
                    position_id=self.position_id,
                    leg_id=leg_id,
                    ticker=leg["contract"],
                    quantity=quantity,
                    stop_loss=None,
                    take_profit=None,
                    tag='Entry'
                )
                placed_any_leg = True
            if placed_any_leg:
                self.entry_premium = abs(net_premium)
                self.entry_spot = self.spot
                self.position_id += 1
                print(f"[{self.time}] Entered position: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}")

        else:
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")

            # Calculate current position value
            current_premium = 0
            for leg in self.legs.values():
                premium = leg["data"]["close"]
                if pd.isna(premium) or not math.isfinite(premium):
                    print(f"[{self.time}] Invalid premium for {leg['contract']}: {premium}")
                    return
                current_premium += (1 if leg['action'] == 'SELL' else -1) * premium

            # P&L percentage
            pl_percent = (net_premium - current_premium) / abs(net_premium) if net_premium != 0 else 0

            # Exit logic
            near_expiry = None
            for trade in active_trades:
                expiry = datetime.strptime(trade.ticker[-14:-7], "%d%b%y").date()
                near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
            exit_reason = (
                "Near Expiry reached" if (pd.Timestamp(self.time).date() == near_expiry) else
                "Stop Loss Hit" if pl_percent <= -self.stop_loss_percent else
                "Take Profit Hit" if pl_percent >= self.take_profit_percent else
                None
            )
            if exit_reason:
                for trade in active_trades:
                    print(f"[{self.time}] Closing position: {exit_reason}")
                    trade.close(trade.size, tag=exit_reason)
                self.entry_premium = None
                self.entry_spot = None
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, closed_trades: {len(self.closed_trades)}")

            # Adjustment logic
            if not exit_reason and self.entry_spot:
                leg_strike = self.legs["leg2"]["strike"]
                if (self.spot * 0.99) <= leg_strike <= (self.spot * 1.01):
                    pass
                else:
                    placed_any_leg = False
                    for leg_id, leg in self.legs.items():
                        leg["strike"] = float(atm)
                        expiry = leg["expiry"]
                        if not isinstance(expiry, (pd.Timestamp, datetime)):
                            try:
                                expiry = pd.to_datetime(expiry)
                            except (ValueError, TypeError) as e:
                                print(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                                return
                        contract = f"NIFTY{expiry.strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
                        leg["contract"] = contract
                        leg["data"] = self._data.get_ticker_data(contract)
                        order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                        order_fn(
                            strategy_id='strat1',
                            position_id=self.position_id,
                            leg_id=leg_id,
                            ticker=leg["contract"],
                            quantity=quantity,
                            stop_loss=None,
                            take_profit=None,
                            tag='Adjustment Calendar'
                        )
                        placed_any_leg = True
                    if placed_any_leg:
                        self.position_id += 1
                        self.entry_spot = self.spot
                        self.entry_premium = abs(current_premium)

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"

    bt = Backtest(
        db_path=db_path,
        strategy=DoubleCalendarStrategy,
        cash=10000000,
        commission_per_contract=0.02,
        option_multiplier=75,
        #start_date="2025-06-20",
        #end_date="2025-06-20"
    )

    stats = bt.run()

    print(stats)
    bt.tear_sheet()
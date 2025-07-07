import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from datetime import datetime, date as DateObject

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtesting_opt import _Data, Strategy, Backtest

class DoubleCalendarStrategy(Strategy):
    def __init__(self, broker, _data, params):
        super().__init__(broker, _data, params)
        # Extract hyperparameters from params with defaults
        self.stop_loss_percent = params.get('stop_loss_percent', 0.01)
        self.take_profit_percent = params.get('take_profit_percent', 0.02)
        self.near_expiry_range = params.get('near_expiry_range', [14, 16])
        self.far_expiry_range = params.get('far_expiry_range', [20, 23])
        self.entry_premium_ratio = params.get('entry_premium_ratio', 0.6)
        self.adjust_premium_ratio = params.get('adjust_premium_ratio', 0.65)
        self.spot_move = params.get('spot_move', 150)
        self.strikes_away = params.get('strikes_away', 1)
        self.premium_move = params.get('premium_move', 0.3)
        self.strike_step = params.get('strike_step', 50)
        self.quantity = params.get('quantity', 1)
        self.legs = None
        self.position_id = 0
        self.portfolio_tp = 0
        self.portfolio_sl = 0
        self.active_trades_by_leg = None
        self.entry_premium = None
        self.entry_spot = None
        self.ticker_date_format = '%d%b%y'

    def init(self):
        super().init()
        self.legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': self.near_expiry_range, 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': self.near_expiry_range, 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg3': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': self.far_expiry_range, 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None},
            'leg4': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': self.far_expiry_range, 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None}
        }
        self.active_trades_by_leg = {}
        print(f"[{self.time}] Strategy initialized with legs: {self.legs}")

    def next(self):
        super().next()
        if self.spot is None or pd.isna(self.spot) or not math.isfinite(self.spot):
            print(f"[{self.time}] Invalid spot price: {self.spot}, skipping")
            return

        atm = round(self.spot / self.strike_step) * self.strike_step
        if not math.isfinite(atm):
            print(f"[{self.time}] Invalid ATM strike: {atm}, skipping")
            return

        print(f"[{self.time}] Available TTEs: {list(self.tte_to_expiry.keys())}")

        # Define ±2 day windows for near and far expiry ranges
        near_lower, near_upper = self.near_expiry_range
        far_lower, far_upper = self.far_expiry_range
        near_window = [near_lower - 2, near_upper + 2]
        far_window = [far_lower - 2, far_upper + 2]

        # Select TTEs within the windows, prioritizing closest to lower bound
        near_ttes = [tte for tte in self.tte_to_expiry.keys() if near_window[0] <= tte <= near_window[1]]
        far_ttes = [tte for tte in self.tte_to_expiry.keys() if far_window[0] <= tte <= far_window[1]]

        if not near_ttes or not far_ttes:
            print(f"[{self.time}] No valid TTEs available: near_ttes={near_ttes}, far_ttes={far_ttes}, skipping")
            return

        # Select TTE closest to the lower bound of each range
        near_tte = min(near_ttes, key=lambda x: abs(x - near_lower))
        far_tte = min(far_ttes, key=lambda x: abs(x - far_lower))

        # Ensure near_tte < far_tte
        if near_tte >= far_tte:
            print(f"[{self.time}] Invalid TTEs: near_tte={near_tte} >= far_tte={far_tte}, skipping")
            return

        # Assign expiries to legs
        for leg_id, leg in self.legs.items():
            leg["expiry"] = self.tte_to_expiry[near_tte] if leg_id in ['leg1', 'leg2'] else self.tte_to_expiry[far_tte]
            print(f"[{self.time}] Assigned TTE for {leg_id}: {leg['expiry']} (TTE={near_tte if leg_id in ['leg1', 'leg2'] else far_tte} days)")

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
            contract = f"NIFTY{expiry.strftime(self.ticker_date_format).upper()}{int(leg['strike'])}{leg['type']}"
            print(f"[{self.time}] Generated contract for {leg['type']}: {contract}")
            leg["contract"] = contract
            leg["data"] = self._data.get_ticker_data(contract)

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            print(f"[{self.time}] Data not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return

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
        total_margin = abs(net_premium) * 75
        print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

        active_trades = self.active_trades

        if not active_trades:
            call_sell_premium = self.legs['leg1']["data"]["close"]
            call_buy_premium = self.legs['leg3']["data"]["close"]
            put_sell_premium = self.legs['leg2']["data"]["close"]
            put_buy_premium = self.legs['leg4']["data"]["close"]

            if any(pd.isna(p) or not math.isfinite(p) for p in [call_sell_premium, call_buy_premium, put_sell_premium, put_buy_premium]):
                print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                return

            if call_buy_premium == 0 or put_buy_premium == 0:
                print(f"[{self.time}] Zero premium for CE Buy={call_buy_premium} or PE Buy={put_buy_premium}, skipping entry")
                return

            call_premium_ratio = call_sell_premium / call_buy_premium
            put_premium_ratio = put_sell_premium / put_buy_premium

            if call_premium_ratio > self.entry_premium_ratio and put_premium_ratio > self.entry_premium_ratio:
                placed_any_leg = False
                print(f"[{self.time}] Selected legs: {[leg['contract'] for leg in self.legs.values()]}")
                print(f"[{self.time}] Premium ratios: CE={call_premium_ratio:.2f}, PE={put_premium_ratio:.2f}")
                for leg_id, leg in self.legs.items():
                    order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                    order = order_fn(
                        strategy_id='strat1',
                        position_id=str(self.position_id),
                        leg_id=leg_id,
                        ticker=leg["contract"],
                        quantity=self.quantity,
                        stop_loss=None,
                        take_profit=None,
                        tag=f'Entry_{leg_id}'
                    )
                    # Store order for tracking; actual Trade object will be retrieved from active_trades
                    if order:
                        placed_any_leg = True
                        print(f"[{self.time}] Placed order for {leg_id}: {leg['contract']}")
                    else:
                        print(f"[{self.time}] Failed to place order for {leg_id}: {leg['contract']}")
                if placed_any_leg:
                    # Update active_trades_by_leg with Trade objects after orders are filled
                    for trade in self.active_trades:
                        leg_id = getattr(trade, 'leg_id', None)
                        if leg_id in self.legs:
                            self.active_trades_by_leg[leg_id] = trade
                    self.entry_premium = abs(net_premium)
                    self.entry_spot = self.spot
                    self.position_id += 1
                    print(f"[{self.time}] Entered position: Net Premium={net_premium:.2f}, Quantity={self.quantity}, Spot={self.entry_spot}")
            else:
                print(f"[{self.time}] Premium ratios not satisfied: CE={call_premium_ratio:.2f}, PE={put_premium_ratio:.2f}, skipping entry")

        else:
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            current_premium = 0
            for leg in self.legs.values():
                premium = leg["data"]["close"]
                if pd.isna(premium) or not math.isfinite(premium):
                    print(f"[{self.time}] Invalid premium for {leg['contract']}: {premium}")
                    return
                current_premium += (1 if leg['action'] == 'SELL' else -1) * premium

            pl_percent = (net_premium - current_premium) / abs(net_premium) if net_premium != 0 else 0

            near_expiry = None
            for trade in active_trades:
                try:
                    expiry = datetime.strptime(trade.ticker[-14:-7], self.ticker_date_format).date()
                    near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
                except ValueError as e:
                    print(f"[{self.time}] Error parsing expiry from ticker {trade.ticker}: {e}")
                    return
            exit_reason = (
                "Near Expiry reached" if (pd.Timestamp(self.time).date() == near_expiry) else
                "Stop Loss Hit" if pl_percent <= -self.stop_loss_percent else
                "Take Profit Hit" if pl_percent >= self.take_profit_percent else
                None
            )
            if exit_reason:
                for leg_id in list(self.active_trades_by_leg.keys()):
                    trade = self.active_trades_by_leg.get(leg_id)
                    if trade and hasattr(trade, 'close'):
                        print(f"[{self.time}] Closing position: {exit_reason}, Ticker: {trade.ticker}")
                        trade.close(tag=exit_reason)
                        self.active_trades_by_leg.pop(leg_id)
                self.entry_premium = None
                self.entry_spot = None
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            if active_trades:
                print(f"[{self.time}] Active Trades Debug:")
                for trade in active_trades:
                    print(f"  Trade: ticker={trade.ticker}, leg_id={getattr(trade, 'leg_id', 'N/A')}, entry_tag={getattr(trade, 'entry_tag', 'N/A')}, size={trade.size}")

            if not exit_reason and self.entry_spot and self.entry_premium:
                spot_move = abs(self.spot - self.entry_spot)
                premium_move = self.premium_move * self.entry_premium
                adjustment_triggered = spot_move >= self.spot_move

                call_sell_premium = self.legs['leg1']["data"]["close"]
                call_buy_premium = self.legs['leg3']["data"]["close"]
                put_sell_premium = self.legs['leg2']["data"]["close"]
                put_buy_premium = self.legs['leg4']["data"]["close"]

                if pd.isna(call_sell_premium) or pd.isna(call_buy_premium) or pd.isna(put_sell_premium) or pd.isna(put_buy_premium):
                    print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                    return

                call_premium_ratio = call_sell_premium / call_buy_premium if call_buy_premium != 0 else float('inf')
                put_premium_ratio = put_sell_premium / put_buy_premium if put_buy_premium != 0 else float('inf')
                premium_adjustment_needed = call_premium_ratio < self.adjust_premium_ratio or put_premium_ratio < self.adjust_premium_ratio

                total_margin = abs(current_premium) * 75 * self.quantity
                if self._broker.margin_available < total_margin:
                    print(f"[{self.time}] Insufficient margin for adjustment: Required {total_margin:.2f}, Available {self._broker.margin_available:.2f}")
                    return

                placed_any_adj = False
                exited_contracts = []
                new_contracts = []

                is_up_move = self.spot > self.entry_spot
                itm_legs = ['leg1', 'leg3'] if is_up_move else ['leg2', 'leg4']
                otm_legs = ['leg2', 'leg4'] if is_up_move else ['leg1', 'leg3']
                itm_strike = self.legs[itm_legs[0]]["strike"]
                strikes_away = abs(self.spot - itm_strike) / self.strike_step

                if strikes_away > self.strikes_away:
                    print(f"[{self.time}] ITM adjustment triggered: Spot {self.spot}, ITM strike {itm_strike}, Strikes away {strikes_away:.2f}")
                    for leg_id in itm_legs:
                        trade = self.active_trades_by_leg.get(leg_id)
                        if trade and hasattr(trade, 'close'):
                            print(f"[{self.time}] Closing ITM leg {leg_id}: {trade.ticker} (strikes away > {self.strikes_away})")
                            exited_contracts.append(trade.ticker)
                            trade.close(tag=f'ITM Adjust_{leg_id}')
                            self.active_trades_by_leg.pop(leg_id, None)
                        else:
                            print(f"[{self.time}] No valid trade to close for ITM leg {leg_id}")
                        leg = self.legs[leg_id]
                        leg["strike"] = float(atm)
                        expiry = leg["expiry"]
                        if not isinstance(expiry, (pd.Timestamp, datetime)):
                            try:
                                expiry = pd.to_datetime(expiry)
                            except (ValueError, TypeError) as e:
                                print(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                                return
                        contract = f"NIFTY{expiry.strftime(self.ticker_date_format).upper()}{int(leg['strike'])}{leg['type']}"
                        leg["contract"] = contract
                        leg["data"] = self._data.get_ticker_data(contract)
                        if leg["data"] is None:
                            print(f"[{self.time}] Data not found for new ITM contract {contract}")
                            return
                        new_contracts.append(contract)
                        order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                        order = order_fn(
                            strategy_id='strat1',
                            position_id=str(self.position_id),
                            leg_id=leg_id,
                            ticker=leg["contract"],
                            quantity=self.quantity,
                            stop_loss=None,
                            take_profit=None,
                            tag=f'ITM Adjust_{leg_id}'
                        )
                        if order:
                            # Update active_trades_by_leg with Trade object after order is filled
                            for trade in self.active_trades:
                                if getattr(trade, 'leg_id', None) == leg_id and getattr(trade, 'entry_tag', None) == f'ITM Adjust_{leg_id}':
                                    self.active_trades_by_leg[leg_id] = trade
                                    placed_any_adj = True
                                    print(f"[{self.time}] Re-entered ITM leg {leg_id}: {leg['contract']}")
                                    break
                        else:
                            print(f"[{self.time}] Failed to re-enter ITM leg {leg_id}: {leg['contract']}")

                elif adjustment_triggered or premium_adjustment_needed:
                    print(f"[{self.time}] OTM adjustment triggered: Spot move {spot_move}, Call ratio {call_premium_ratio:.2f}, Put ratio {put_premium_ratio:.2f}")
                    for leg_id in otm_legs:
                        leg = self.legs[leg_id]
                        buy_leg_id = 'leg3' if leg_id == 'leg1' else 'leg4' if leg_id == 'leg2' else leg_id
                        sell_premium = leg["data"]["close"] if leg['action'] == 'SELL' else self.legs[buy_leg_id]["data"]["close"]
                        buy_premium = self.legs[buy_leg_id]["data"]["close"] if leg['action'] == 'SELL' else leg["data"]["close"]
                        if pd.isna(sell_premium) or pd.isna(buy_premium):
                            print(f"[{self.time}] Invalid premiums for {leg['contract']}: sell={sell_premium}, buy={buy_premium}")
                            continue
                        if sell_premium < self.adjust_premium_ratio * buy_premium:
                            leg["strike"] = leg["strike"] + self.strike_step if is_up_move else leg["strike"] - self.strike_step
                            expiry = leg["expiry"]
                            if not isinstance(expiry, (pd.Timestamp, datetime)):
                                try:
                                    expiry = pd.to_datetime(expiry)
                                except (ValueError, TypeError) as e:
                                    print(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                                    return
                            contract = f"NIFTY{expiry.strftime(self.ticker_date_format).upper()}{int(leg['strike'])}{leg['type']}"
                            leg["contract"] = contract
                            leg["data"] = self._data.get_ticker_data(contract)
                            if leg["data"] is None:
                                print(f"[{self.time}] Data not found for rolled contract {contract}")
                                continue
                            new_sell_premium = leg["data"]["close"] if leg['action'] == 'SELL' else self.legs[buy_leg_id]["data"]["close"]
                            if new_sell_premium < self.adjust_premium_ratio * buy_premium:
                                print(f"[{self.time}] Rolled premium for {leg['contract']} still below {self.adjust_premium_ratio} of buy: {new_sell_premium}/{buy_premium}")
                                continue
                            trade = self.active_trades_by_leg.get(leg_id)
                            if trade and hasattr(trade, 'close'):
                                print(f"[{self.time}] Closing OTM leg {leg_id}: {trade.ticker} for rolling")
                                exited_contracts.append(trade.ticker)
                                trade.close(tag=f'OTM Roll_{leg_id}')
                                self.active_trades_by_leg.pop(leg_id, None)
                            else:
                                print(f"[{self.time}] No valid trade to close for OTM leg {leg_id}")
                            new_contracts.append(contract)
                            order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                            order = order_fn(
                                strategy_id='strat1',
                                position_id=str(self.position_id),
                                leg_id=leg_id,
                                ticker=leg["contract"],
                                quantity=self.quantity,
                                stop_loss=None,
                                take_profit=None,
                                tag=f'OTM Roll_{leg_id}'
                            )
                            if order:
                                # Update active_trades_by_leg with Trade object after order is filled
                                for trade in self.active_trades:
                                    if getattr(trade, 'leg_id', None) == leg_id and getattr(trade, 'entry_tag', None) == f'OTM Roll_{leg_id}':
                                        self.active_trades_by_leg[leg_id] = trade
                                        placed_any_adj = True
                                        print(f"[{self.time}] Rolled OTM leg {leg_id}: {leg['contract']}")
                                        break
                            else:
                                print(f"[{self.time}] Failed to roll OTM leg {leg_id}: {leg['contract']}")

                if placed_any_adj:
                    print(f"[{self.time}] Adjustment Summary: Exited Contracts: {exited_contracts}, New Contracts: {new_contracts}")
                    new_premium = 0
                    for leg in self.legs.values():
                        premium = leg["data"]["close"]
                        if pd.isna(premium) or not math.isfinite(premium):
                            print(f"[{self.time}] Invalid premium for {leg['contract']} after adjustment: {premium}")
                            new_premium = current_premium
                            break
                        new_premium += (1 if leg['action'] == 'SELL' else -1) * premium
                    if not math.isfinite(new_premium) or new_premium == 0:
                        print(f"[{self.time}] Invalid new net premium after adjustment: {new_premium}, reverting to current premium")
                        new_premium = current_premium
                    self.position_id += 1
                    self.entry_spot = self.spot
                    self.entry_premium = abs(new_premium)
                    print(f"[{self.time}] Adjusted position: New Spot={self.entry_spot}, New Premium={self.entry_premium:.2f}")

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"

    bt = Backtest(
        db_path=db_path,
        strategy=DoubleCalendarStrategy,
        cash=10000000,
        commission_per_contract=0.02,
        option_multiplier=75
    )

    stats, heatmap = bt.optimize(
        stop_loss_percent=[0.005, 0.01, 0.015, 0.02],
        take_profit_percent=[0.01, 0.015, 0.02, 0.025],
        near_expiry_range=[14, 16],
        far_expiry_range=[20, 23],
        entry_premium_ratio=[0.5, 0.6, 0.7],
        adjust_premium_ratio=[0.5, 0.6, 0.7],
        spot_move=[100, 150, 200],
        strikes_away=[0.5, 1, 1.5],
        premium_move=[0.2, 0.3, 0.4],
        strike_step=[75],
        quantity=[1],
        #entry_time=["14:00:00", "14:30:00", "15:00:00"],
        maximize='Sharpe Ratio',
        method='sambo',
        max_tries=100,
        random_state=0,
        return_heatmap=True,
        #constraint=lambda p: p.near_expiry_range[1] < p.far_expiry_range[0]
    )

    print(stats)
    bt.tear_sheet()
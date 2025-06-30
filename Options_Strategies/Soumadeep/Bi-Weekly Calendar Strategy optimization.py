"""import sys
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
    # Define parameters as class variables for optimization/flexibility
    legs: dict = None
    position_id = 0
    portfolio_tp = 0
    portfolio_sl = 0

    #Parameters which can be defined here:
    stop_loss_percent: float = 0.01  # 1% per trade
    take_profit_percent: float = 0.02  # 2% per trade
    ticker_date_format: str = '%d%b%y'  # Configurable: '%d%b%y' (e.g., 21JUN25) or '%Y%m%d' (e.g., 20250621)

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
        print(f"[{self.time}] Strategy initialized with legs: {self.legs}")

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
            contract = f"NIFTY{expiry.strftime(self.ticker_date_format).upper()}{int(leg['strike'])}{leg['type']}"
            print(f"[{self.time}] Generated contract for {leg['type']}: {contract}")
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
        total_margin = abs(net_premium) * 75  # Per contract
        print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

        active_trades = self.active_trades
        quantity = 1  # Fixed quantity

        # Entry logic
        if not active_trades and pd.Timestamp(self.time).time() < pd.Timestamp("15:15:00").time():
            placed_any_leg = False
            print(f"[{self.time}] Selected legs: {[leg['contract'] for leg in self.legs.values()]}")
            for leg_id, leg in self.legs.items():
                order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                order_fn(
                    strategy_id='strat1',
                    position_id=str(self.position_id),
                    leg_id=leg_id,
                    ticker=leg["contract"],
                    quantity=quantity,
                    stop_loss=None,
                    take_profit=None,
                    tag=f'Entry_{leg_id}'
                )
                placed_any_leg = True
            if placed_any_leg:
                self.entry_premium = abs(net_premium)
                self.entry_spot = self.spot
                self.position_id += 1
                print(f"[{self.time}] Entered position: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}")

        else:
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

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
                for trade in active_trades:
                    print(f"[{self.time}] Closing position: {exit_reason}, Ticker: {trade.ticker}")
                    trade.close(trade.size, tag=exit_reason)
                self.entry_premium = None
                self.entry_spot = None
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            # Debug active trades
            if active_trades:
                print(f"[{self.time}] Active Trades Debug:")
                for trade in active_trades:
                    print(f"  Trade: ticker={trade.ticker}, leg_id={getattr(trade, 'leg_id', 'N/A')}, entry_tag={getattr(trade, 'entry_tag', 'N/A')}, size={trade.size}")

            # Adjustment logic
            if not exit_reason and self.entry_spot and self.entry_premium:
                spot_move = abs(self.spot - self.entry_spot)
                premium_move = 0.3 * self.entry_premium  # 30% of entry premium
                strike_step = 50  # TODO: Derive from database
                adjustment_triggered = spot_move >= 150

                # Check premium ratios
                call_sell_premium = self.legs['leg1']["data"]["close"]
                call_buy_premium = self.legs['leg3']["data"]["close"]
                put_sell_premium = self.legs['leg2']["data"]["close"]
                put_buy_premium = self.legs['leg4']["data"]["close"]

                if pd.isna(call_sell_premium) or pd.isna(call_buy_premium) or pd.isna(put_sell_premium) or pd.isna(put_buy_premium):
                    print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                    return

                call_premium_ratio = call_sell_premium / call_buy_premium if call_buy_premium != 0 else float('inf')
                put_premium_ratio = put_sell_premium / put_buy_premium if put_buy_premium != 0 else float('inf')
                premium_adjustment_needed = call_premium_ratio < 0.65 or put_premium_ratio < 0.65

                # Check margin availability
                total_margin = abs(current_premium) * 75 * quantity
                if self._broker.margin_available < total_margin:
                    print(f"[{self.time}] Insufficient margin for adjustment: Required {total_margin:.2f}, Available {self._broker.margin_available:.2f}")
                    return

                # Initialize adjustment tracking
                placed_any_adj = False
                exited_contracts = []
                new_contracts = []

                # Step 1: Close ITM calendar if > 1 strike away
                is_up_move = self.spot > self.entry_spot
                itm_legs = ['leg1', 'leg3'] if is_up_move else ['leg2', 'leg4']
                otm_legs = ['leg2', 'leg4'] if is_up_move else ['leg1', 'leg3']
                itm_strike = self.legs[itm_legs[0]]["strike"]
                strikes_away = abs(self.spot - itm_strike) / strike_step

                if strikes_away > 1:
                    print(f"[{self.time}] ITM adjustment triggered: Spot {self.spot}, ITM strike {itm_strike}, Strikes away {strikes_away:.2f}")
                    for leg_id in itm_legs:
                        trades_to_close = [
                            trade for trade in active_trades
                            if getattr(trade, 'leg_id', None) == leg_id or getattr(trade, 'entry_tag', '').startswith(f'Entry_{leg_id}')
                        ]
                        if not trades_to_close:
                            print(f"[{self.time}] No trades found for ITM leg {leg_id}, skipping closure")
                            continue
                        for trade in trades_to_close:
                            print(f"[{self.time}] Closing ITM leg {leg_id}: {trade.ticker} (strikes away > 1)")
                            exited_contracts.append(trade.ticker)
                            trade.close(trade.size, tag=f'ITM Adjust_{leg_id}')
                        # Re-enter at ATM
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
                        order_fn(
                            strategy_id='strat1',
                            position_id=str(self.position_id),
                            leg_id=leg_id,
                            ticker=leg["contract"],
                            quantity=quantity,
                            stop_loss=None,
                            take_profit=None,
                            tag=f'ITM Adjust_{leg_id}'
                        )
                        placed_any_adj = True

                # Step 2: Roll OTM side if premium ratio < 0.5 or spot move >= 150
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
                        if sell_premium < 0.65 * buy_premium:
                            # Roll strike by 1
                            leg["strike"] = leg["strike"] + strike_step if is_up_move else leg["strike"] - strike_step
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
                            if new_sell_premium < 0.65 * buy_premium:
                                print(f"[{self.time}] Rolled premium for {leg['contract']} still below 50% of buy: {new_sell_premium}/{buy_premium}")
                                continue
                            # Close existing OTM leg
                            trades_to_close = [
                                trade for trade in active_trades
                                if getattr(trade, 'leg_id', None) == leg_id or getattr(trade, 'entry_tag', '').startswith(f'Entry_{leg_id}')
                            ]
                            for trade in trades_to_close:
                                print(f"[{self.time}] Closing OTM leg {leg_id}: {trade.ticker} for rolling")
                                exited_contracts.append(trade.ticker)
                                trade.close(trade.size, tag=f'OTM Roll_{leg_id}')
                            new_contracts.append(contract)
                            order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                            order_fn(
                                strategy_id='strat1',
                                position_id=str(self.position_id),
                                leg_id=leg_id,
                                ticker=leg["contract"],
                                quantity=quantity,
                                stop_loss=None,
                                take_profit=None,
                                tag=f'OTM Roll_{leg_id}'
                            )
                            placed_any_adj = True

                # Print adjustment summary
                if placed_any_adj:
                    print(f"[{self.time}] Adjustment Summary: Exited Contracts: {exited_contracts}, New Contracts: {new_contracts}")
                    # Recalculate net premium for new position
                    new_premium = 0
                    for leg in self.legs.values():
                        premium = leg["data"]["close"]
                        if pd.isna(premium) or not math.isfinite(premium):
                            print(f"[{self.time}] Invalid premium for {leg['contract']} after adjustment: {premium}")
                            new_premium = current_premium  # Fallback
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
        option_multiplier=75,
        #start_date="2025-06-01",  # Set to ensure data availability
        #end_date="2025-06-20"
    )

    stats = bt.run()
    print(stats)
    bt.tear_sheet()"""
    
    
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
    # Define parameters as class variables for optimization/flexibility
    legs: dict = None
    position_id = 0
    portfolio_tp = 0
    portfolio_sl = 0

    """Parameters which can be defined here:"""
    stop_loss_percent: float = 0.01  # 1% per trade
    take_profit_percent: float = 0.02  # 2% per trade
    ticker_date_format: str = '%d%b%y'  # Configurable: '%d%b%y' (e.g., 21JUN25) or '%Y%m%d' (e.g., 20250621)

    def init(self):
        super().init()
        self.legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [14, 16], 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [14, 16], 'target_strike': 'ATM', 'action': 'SELL', 'stop_loss': None, 'take_profit': None},
            'leg3': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [20, 23], 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None},
            'leg4': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [20, 23], 'target_strike': 'ATM', 'action': 'BUY', 'stop_loss': None, 'take_profit': None}
        }
        self.entry_premium = None
        self.entry_spot = None
        print(f"[{self.time}] Strategy initialized with legs: {self.legs}")

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
            contract = f"NIFTY{expiry.strftime(self.ticker_date_format).upper()}{int(leg['strike'])}{leg['type']}"
            print(f"[{self.time}] Generated contract for {leg['type']}: {contract}")
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
        total_margin = abs(net_premium) * 75  # Per contract
        print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

        active_trades = self.active_trades
        quantity = 1  # Fixed quantity

        # Entry logic
        if not active_trades and pd.Timestamp(self.time).time() < pd.Timestamp("15:15:00").time():
            # Check premium ratios: leg1/leg3 > 0.6 and leg2/leg4 > 0.6
            call_sell_premium = self.legs['leg1']["data"]["close"]
            call_buy_premium = self.legs['leg3']["data"]["close"]
            put_sell_premium = self.legs['leg2']["data"]["close"]
            put_buy_premium = self.legs['leg4']["data"]["close"]

            # Validate premiums
            if any(
                pd.isna(premium) or not math.isfinite(premium)
                for premium in [call_sell_premium, call_buy_premium, put_sell_premium, put_buy_premium]
            ):
                print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}, skipping entry")
                return

            if call_buy_premium == 0 or put_buy_premium == 0:
                print(f"[{self.time}] Zero premium for CE Buy={call_buy_premium} or PE Buy={put_buy_premium}, skipping entry")
                return

            # Calculate premium ratios
            call_premium_ratio = call_sell_premium / call_buy_premium
            put_premium_ratio = put_sell_premium / put_buy_premium

            # Check if ratios satisfy the condition
            if call_premium_ratio > 0.6 and put_premium_ratio > 0.6:
                placed_any_leg = False
                print(f"[{self.time}] Selected legs: {[leg['contract'] for leg in self.legs.values()]}")
                print(f"[{self.time}] Premium ratios: CE={call_premium_ratio:.2f}, PE={put_premium_ratio:.2f}")
                for leg_id, leg in self.legs.items():
                    order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                    order_fn(
                        strategy_id='strat1',
                        position_id=str(self.position_id),
                        leg_id=leg_id,
                        ticker=leg["contract"],
                        quantity=quantity,
                        stop_loss=None,
                        take_profit=None,
                        tag=f'Entry_{leg_id}'
                    )
                    placed_any_leg = True
                if placed_any_leg:
                    self.entry_premium = abs(net_premium)
                    self.entry_spot = self.spot
                    self.position_id += 1
                    print(f"[{self.time}] Entered position: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}")
            else:
                print(f"[{self.time}] Premium ratios not satisfied: CE={call_premium_ratio:.2f}, PE={put_premium_ratio:.2f}, skipping entry")

        else:
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

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
                for trade in active_trades:
                    print(f"[{self.time}] Closing position: {exit_reason}, Ticker: {trade.ticker}")
                    trade.close(trade.size, tag=exit_reason)
                self.entry_premium = None
                self.entry_spot = None
            print(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            # Debug active trades
            if active_trades:
                print(f"[{self.time}] Active Trades Debug:")
                for trade in active_trades:
                    print(f"  Trade: ticker={trade.ticker}, leg_id={getattr(trade, 'leg_id', 'N/A')}, entry_tag={getattr(trade, 'entry_tag', 'N/A')}, size={trade.size}")

            # Adjustment logic
            if not exit_reason and self.entry_spot and self.entry_premium:
                spot_move = abs(self.spot - self.entry_spot)
                premium_move = 0.3 * self.entry_premium  # 30% of entry premium
                strike_step = 50  # TODO: Derive from database
                adjustment_triggered = spot_move >= 150

                # Check premium ratios
                call_sell_premium = self.legs['leg1']["data"]["close"]
                call_buy_premium = self.legs['leg3']["data"]["close"]
                put_sell_premium = self.legs['leg2']["data"]["close"]
                put_buy_premium = self.legs['leg4']["data"]["close"]

                if pd.isna(call_sell_premium) or pd.isna(call_buy_premium) or pd.isna(put_sell_premium) or pd.isna(put_buy_premium):
                    print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                    return

                call_premium_ratio = call_sell_premium / call_buy_premium if call_buy_premium != 0 else float('inf')
                put_premium_ratio = put_sell_premium / put_buy_premium if put_buy_premium != 0 else float('inf')
                premium_adjustment_needed = call_premium_ratio < 0.65 or put_premium_ratio < 0.65

                # Check margin availability
                total_margin = abs(current_premium) * 75 * quantity
                if self._broker.margin_available < total_margin:
                    print(f"[{self.time}] Insufficient margin for adjustment: Required {total_margin:.2f}, Available {self._broker.margin_available:.2f}")
                    return

                # Initialize adjustment tracking
                placed_any_adj = False
                exited_contracts = []
                new_contracts = []

                # Step 1: Close ITM calendar if > 1 strike away
                is_up_move = self.spot > self.entry_spot
                itm_legs = ['leg1', 'leg3'] if is_up_move else ['leg2', 'leg4']
                otm_legs = ['leg2', 'leg4'] if is_up_move else ['leg1', 'leg3']
                itm_strike = self.legs[itm_legs[0]]["strike"]
                strikes_away = abs(self.spot - itm_strike) / strike_step

                if strikes_away > 1:
                    print(f"[{self.time}] ITM adjustment triggered: Spot {self.spot}, ITM strike {itm_strike}, Strikes away {strikes_away:.2f}")
                    for leg_id in itm_legs:
                        trades_to_close = [
                            trade for trade in active_trades
                            if getattr(trade, 'leg_id', None) == leg_id or getattr(trade, 'entry_tag', '').startswith(f'Entry_{leg_id}')
                        ]
                        if not trades_to_close:
                            print(f"[{self.time}] No trades found for ITM leg {leg_id}, skipping closure")
                            continue
                        for trade in trades_to_close:
                            print(f"[{self.time}] Closing ITM leg {leg_id}: {trade.ticker} (strikes away > 1)")
                            exited_contracts.append(trade.ticker)
                            trade.close(trade.size, tag=f'ITM Adjust_{leg_id}')
                        # Re-enter at ATM
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
                        order_fn(
                            strategy_id='strat1',
                            position_id=str(self.position_id),
                            leg_id=leg_id,
                            ticker=leg["contract"],
                            quantity=quantity,
                            stop_loss=None,
                            take_profit=None,
                            tag=f'ITM Adjust_{leg_id}'
                        )
                        placed_any_adj = True

                # Step 2: Roll OTM side if premium ratio < 0.5 or spot move >= 150
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
                        if sell_premium < 0.65 * buy_premium:
                            # Roll strike by 1
                            leg["strike"] = leg["strike"] + strike_step if is_up_move else leg["strike"] - strike_step
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
                            if new_sell_premium < 0.65 * buy_premium:
                                print(f"[{self.time}] Rolled premium for {leg['contract']} still below 50% of buy: {new_sell_premium}/{buy_premium}")
                                continue
                            # Close existing OTM leg
                            trades_to_close = [
                                trade for trade in active_trades
                                if getattr(trade, 'leg_id', None) == leg_id or getattr(trade, 'entry_tag', '').startswith(f'Entry_{leg_id}')
                            ]
                            for trade in trades_to_close:
                                print(f"[{self.time}] Closing OTM leg {leg_id}: {trade.ticker} for rolling")
                                exited_contracts.append(trade.ticker)
                                trade.close(trade.size, tag=f'OTM Roll_{leg_id}')
                            new_contracts.append(contract)
                            order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                            order_fn(
                                strategy_id='strat1',
                                position_id=str(self.position_id),
                                leg_id=leg_id,
                                ticker=leg["contract"],
                                quantity=quantity,
                                stop_loss=None,
                                take_profit=None,
                                tag=f'OTM Roll_{leg_id}'
                            )
                            placed_any_adj = True

                # Print adjustment summary
                if placed_any_adj:
                    print(f"[{self.time}] Adjustment Summary: Exited Contracts: {exited_contracts}, New Contracts: {new_contracts}")
                    # Recalculate net premium for new position
                    new_premium = 0
                    for leg in self.legs.values():
                        premium = leg["data"]["close"]
                        if pd.isna(premium) or not math.isfinite(premium):
                            print(f"[{self.time}] Invalid premium for {leg['contract']} after adjustment: {premium}")
                            new_premium = current_premium  # Fallback
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
        option_multiplier=75,
        #start_date="2025-06-01",  # Set to ensure data availability
        #end_date="2025-06-20"
    )

    stats = bt.run()
    print(stats)
    bt.tear_sheet()
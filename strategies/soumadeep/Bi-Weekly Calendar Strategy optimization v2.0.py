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

    # Parameters
    stop_loss_percent: float = 0.01  # Default 1% per trade
    take_profit_percent: float = 0.02  # Default 2% per trade
    ticker_date_format: str = '%d%b%y'  # e.g., 21JUL25
    min_premium_ratio: float = 0.6  # Default ratio for ATM options
    vol_slope_threshold_contango: float = 0.002  # Default IV slope per day for contango
    vol_slope_threshold_backwardation: float = -0.001  # Default IV slope per day for backwardation
    iv_slope_interval_hours: float = 4  # Default check IV slope every 4 hours
    max_adjustments: int = 2

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
        self.vol_slope = None
        self.strategy_type = None
        self.adjustment_count = 0
        self.last_iv_slope_time = None
        self.price_history = []  # Store recent spot prices for BB and volatility
        print(f"[{self.time}] Strategy initialized with legs: {self.legs}")

    def update_price_history(self):
        # Store up to 20 periods (1-minute data) for indicators
        self.price_history.append(self.spot)
        if len(self.price_history) > 20:
            self.price_history.pop(0)

    def get_bollinger_band_width(self, period=20):
        if len(self.price_history) < period:
            return 0.05  # Default for insufficient data
        prices = np.array(self.price_history[-period:])
        sma = np.mean(prices)
        std = np.std(prices)
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        bb_width = (upper_band - lower_band) / sma
        return bb_width

    def get_realized_volatility(self, period=20):
        if len(self.price_history) < period + 1:
            return 0.2  # Default 20%
        returns = np.diff(self.price_history[-period-1:]) / self.price_history[-period-1:-1]
        return np.std(returns) * np.sqrt(252 * 390)  # Annualized, 390 minutes/day

    def detect_market_environment(self):
        bb_width = self.get_bollinger_band_width()
        realized_vol = self.get_realized_volatility()
        if bb_width < 0.05 and realized_vol < 0.15:
            return "range_bound"
        elif realized_vol > 0.25 and bb_width > 0.1:
            return "volatile"
        elif bb_width >= 0.05 and realized_vol >= 0.15:
            return "trending"
        return "neutral"

    def set_dynamic_parameters(self):
        market_type = self.detect_market_environment()
        if market_type == "volatile":
            self.stop_loss_percent = 0.02
            self.take_profit_percent = 0.04
            self.min_premium_ratio = 0.5
            self.vol_slope_threshold_backwardation = -0.001
            self.iv_slope_interval_hours = 2
            self.max_adjustments = 3
        elif market_type == "trending":
            self.stop_loss_percent = 0.015
            self.take_profit_percent = 0.03
            self.min_premium_ratio = 0.55
            self.iv_slope_interval_hours = 2
            self.max_adjustments = 2
        else:  # range_bound or neutral
            self.stop_loss_percent = 0.01
            self.take_profit_percent = 0.02
            self.min_premium_ratio = 0.6
            self.iv_slope_interval_hours = 4
            self.max_adjustments = 2
        print(f"[{self.time}] Market type: {market_type}, Parameters: SL={self.stop_loss_percent}, TP={self.take_profit_percent}, Premium Ratio={self.min_premium_ratio}, IV Slope Interval={self.iv_slope_interval_hours}, Max Adjustments={self.max_adjustments}")

    def is_iv_slope_time(self):
        current_time = pd.Timestamp(self.time)
        market_open = current_time.replace(hour=9, minute=15, second=0)
        interval_minutes = self.iv_slope_interval_hours * 60
        intervals = [market_open + pd.Timedelta(minutes=i * interval_minutes) for i in range(int(6.25 * 60 / interval_minutes) + 1)]
        for interval in intervals:
            if abs((current_time - interval).total_seconds()) <= 300:  # 5-minute window
                return True
        return False

    def calculate_vol_slope(self, near_tte, far_tte):
        try:
            near_iv = (self.legs["leg1"]["data"]["iv"] + self.legs["leg2"]["data"]["iv"]) / 2
            far_iv = (self.legs["leg3"]["data"]["iv"] + self.legs["leg4"]["data"]["iv"]) / 2
            if near_iv <= 0 or far_iv <= 0:
                print(f"[{self.time}] Invalid IV data: near_iv={near_iv}, far_iv={far_iv}")
                return None
            dte_diff = far_tte - near_tte
            if dte_diff <= 0:
                print(f"[{self.time}] Invalid DTE difference: {dte_diff}")
                return None
            iv_slope = (far_iv - near_iv) / dte_diff
            print(f"[{self.time}] Volatility slope: {iv_slope:.4f} (Near IV avg={near_iv:.2f}%, Far IV avg={far_iv:.2f}%, DTE diff={dte_diff})")
            return iv_slope
        except (KeyError, TypeError, ValueError) as e:
            print(f"[{self.time}] Error calculating IV slope: {e}")
            return None

    def next(self):
        super().next()
        if self.spot is None or pd.isna(self.spot) or not math.isfinite(self.spot):
            print(f"[{self.time}] Invalid spot price: {self.spot}, skipping")
            return

        self.update_price_history()  # Update price history for indicators
        self.set_dynamic_parameters()  # Adjust parameters based on market environment

        atm = round(self.spot / 50) * 50
        if not math.isfinite(atm):
            print(f"[{self.time}] Invalid ATM strike: {atm}, skipping")
            return

        print(f"[{self.time}] Available TTEs: {list(self.tte_to_expiry.keys())}")

        # Define TTE windows
        near_lower, near_upper = self.legs['leg1']['expiry_range']
        far_lower, far_upper = self.legs['leg3']['expiry_range']
        near_window = [near_lower - 2, near_upper + 2]
        far_window = [far_lower - 2, far_upper + 2]
        near_ttes = [tte for tte in self.tte_to_expiry.keys() if near_window[0] <= tte <= near_window[1]]
        far_ttes = [tte for tte in self.tte_to_expiry.keys() if far_window[0] <= tte <= far_window[1]]

        if not near_ttes or not far_ttes:
            print(f"[{self.time}] No valid TTEs: near_ttes={near_ttes}, far_ttes={far_ttes}, skipping")
            return

        near_tte = min(near_ttes, key=lambda x: abs(x - near_lower))
        far_tte = min(far_ttes, key=lambda x: abs(x - far_lower))
        if near_tte >= far_tte:
            print(f"[{self.time}] Invalid TTEs: near_tte={near_tte} >= far_tte={far_tte}, skipping")
            return

        # Assign expiries and strikes
        for leg_id, leg in self.legs.items():
            leg["expiry"] = self.tte_to_expiry[near_tte] if leg_id in ['leg1', 'leg2'] else self.tte_to_expiry[far_tte]
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

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            print(f"[{self.time}] Data not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return

        # Validate IV data
        for leg_id, leg in self.legs.items():
            if "iv" not in leg["data"] or not math.isfinite(leg["data"]["iv"]) or leg["data"]["iv"] <= 0:
                print(f"[{self.time}] Invalid or missing IV for {leg['contract']}: {leg['data'].get('iv', 'N/A')}")
                return

        # Calculate IV slope at specified intervals
        if self.is_iv_slope_time():
            self.vol_slope = self.calculate_vol_slope(near_tte, far_tte)
            self.last_iv_slope_time = pd.Timestamp(self.time)
            if self.vol_slope is None:
                print(f"[{self.time}] Unable to calculate volatility slope, skipping")
                return

            # Select strategy based on volatility slope, confirmed by market environment
            market_type = self.detect_market_environment()
            near_iv = (self.legs["leg1"]["data"]["iv"] + self.legs["leg2"]["data"]["iv"]) / 2
            far_iv = (self.legs["leg3"]["data"]["iv"] + self.legs["leg4"]["data"]["iv"]) / 2
            avg_iv = (near_iv + far_iv) / 2
            if self.vol_slope > self.vol_slope_threshold_contango and market_type in ["range_bound", "neutral"]:
                self.strategy_type = 'double_calendar'
                self.legs['leg1']['action'] = 'SELL'
                self.legs['leg2']['action'] = 'SELL'
                self.legs['leg3']['action'] = 'BUY'
                self.legs['leg4']['action'] = 'BUY'
            elif self.vol_slope < self.vol_slope_threshold_backwardation and market_type in ["volatile", "trending"] and avg_iv > 15:
                self.strategy_type = 'reverse_calendar'
                self.legs['leg1']['action'] = 'BUY'
                self.legs['leg2']['action'] = 'BUY'
                self.legs['leg3']['action'] = 'SELL'
                self.legs['leg4']['action'] = 'SELL'
            else:
                self.strategy_type = None
                print(f"[{self.time}] Flat or insufficient slope ({self.vol_slope:.4f}) or incompatible market type ({market_type}, avg IV={avg_iv:.2f}%), skipping entry")
                return
            print(f"[{self.time}] Strategy selected: {self.strategy_type}")
        elif self.vol_slope is None or self.last_iv_slope_time is None:
            print(f"[{self.time}] No IV slope calculated yet, skipping")
            return
        else:
            print(f"[{self.time}] Using last IV slope: {self.vol_slope:.4f} from {self.last_iv_slope_time}")

        # Skip trading if no strategy is selected
        if self.strategy_type is None:
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
        total_margin = abs(net_premium) * 75
        print(f"[{self.time}] Current Equity: {current_equity:.2f}, Net Premium Margin: {total_margin:.2f}")

        active_trades = self.active_trades
        quantity = 1

        # Entry logic
        if not active_trades and pd.Timestamp(self.time).time() < pd.Timestamp("15:15:00").time():
            call_sell_premium = self.legs['leg1' if self.strategy_type == 'double_calendar' else 'leg3']["data"]["close"]
            call_buy_premium = self.legs['leg3' if self.strategy_type == 'double_calendar' else 'leg1']["data"]["close"]
            put_sell_premium = self.legs['leg2' if self.strategy_type == 'double_calendar' else 'leg4']["data"]["close"]
            put_buy_premium = self.legs['leg4' if self.strategy_type == 'double_calendar' else 'leg2']["data"]["close"]

            if any(pd.isna(p) or not math.isfinite(p) for p in [call_sell_premium, call_buy_premium, put_sell_premium, put_buy_premium]):
                print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                return

            if call_buy_premium == 0 or put_buy_premium == 0:
                print(f"[{self.time}] Zero premium for CE Buy={call_buy_premium} or PE Buy={put_buy_premium}, skipping entry")
                return

            call_premium_ratio = call_sell_premium / call_buy_premium
            put_premium_ratio = put_sell_premium / put_buy_premium

            if call_premium_ratio > self.min_premium_ratio and put_premium_ratio > self.min_premium_ratio:
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
                    print(f"[{self.time}] Entered {self.strategy_type}: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}")
            else:
                print(f"[{self.time}] Premium ratios not satisfied: CE={call_premium_ratio:.2f}, PE={put_premium_ratio:.2f}")

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
            current_slope = self.vol_slope
            if self.is_iv_slope_time():
                current_slope = self.calculate_vol_slope(near_tte, far_tte)
                if current_slope is not None:
                    self.vol_slope = current_slope
                    self.last_iv_slope_time = pd.Timestamp(self.time)

            if current_slope is None:
                print(f"[{self.time}] No valid IV slope, skipping exit")
                return

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
                "Vol Slope Reversal" if (
                    (self.strategy_type == 'double_calendar' and current_slope < 0) or
                    (self.strategy_type == 'reverse_calendar' and current_slope > 0)
                ) else
                None
            )

            if exit_reason:
                for trade in active_trades:
                    print(f"[{self.time}] Closing position: {exit_reason}, Ticker: {trade.ticker}")
                    trade.close(trade.size, tag=exit_reason)
                self.entry_premium = None
                self.entry_spot = None
                self.adjustment_count = 0
                self.strategy_type = None
                return

            # Adjustment logic
            if not exit_reason and self.entry_spot and self.entry_premium:
                market_type = self.detect_market_environment()
                strikes_away_threshold = 1 if market_type == "volatile" else 2
                slope_change_threshold = 0.01 if market_type == "volatile" else 0.005

                spot_move = abs(self.spot - self.entry_spot)
                slope_change = abs(current_slope - self.vol_slope) if self.vol_slope is not None else 0
                adjustment_triggered = spot_move >= 150 or slope_change > slope_change_threshold

                if self.adjustment_count >= self.max_adjustments:
                    print(f"[{self.time}] Max adjustments ({self.max_adjustments}) reached, skipping")
                    return

                call_sell_premium = self.legs['leg1' if self.strategy_type == 'double_calendar' else 'leg3']["data"]["close"]
                call_buy_premium = self.legs['leg3' if self.strategy_type == 'double_calendar' else 'leg1']["data"]["close"]
                put_sell_premium = self.legs['leg2' if self.strategy_type == 'double_calendar' else 'leg4']["data"]["close"]
                put_buy_premium = self.legs['leg4' if self.strategy_type == 'double_calendar' else 'leg2']["data"]["close"]

                if pd.isna(call_sell_premium) or pd.isna(call_buy_premium) or pd.isna(put_sell_premium) or pd.isna(put_buy_premium):
                    print(f"[{self.time}] Invalid premiums: CE Sell={call_sell_premium}, CE Buy={call_buy_premium}, PE Sell={put_sell_premium}, PE Buy={put_buy_premium}")
                    return

                call_premium_ratio = call_sell_premium / call_buy_premium if call_buy_premium != 0 else float('inf')
                put_premium_ratio = put_sell_premium / put_buy_premium if put_buy_premium != 0 else float('inf')
                premium_adjustment_needed = call_premium_ratio < self.min_premium_ratio or put_premium_ratio < self.min_premium_ratio

                total_margin = abs(current_premium) * 75 * quantity
                if self._broker.margin_available < total_margin:
                    print(f"[{self.time}] Insufficient margin for adjustment: Required {total_margin:.2f}, Available {self._broker.margin_available:.2f}")
                    return

                placed_any_adj = False
                exited_contracts = []
                new_contracts = []

                # Adjust ITM legs
                is_up_move = self.spot > self.entry_spot
                itm_legs = ['leg1', 'leg3'] if is_up_move else ['leg2', 'leg4']
                otm_legs = ['leg2', 'leg4'] if is_up_move else ['leg1', 'leg3']
                itm_strike = self.legs[itm_legs[0]]["strike"]
                strike_step = 50
                strikes_away = abs(self.spot - itm_strike) / strike_step

                if strikes_away > strikes_away_threshold:
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
                            print(f"[{self.time}] Closing ITM leg {leg_id}: {trade.ticker} (strikes away > {strikes_away_threshold})")
                            exited_contracts.append(trade.ticker)
                            trade.close(trade.size, tag=f'ITM Adjust_{leg_id}')
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
                        print(f"[{self.time}] Re-entered ITM leg {leg_id}: {leg['contract']}")

                # Roll OTM side if needed
                elif adjustment_triggered or premium_adjustment_needed:
                    print(f"[{self.time}] OTM adjustment triggered: Spot move {spot_move}, Slope change {slope_change:.4f}, Call ratio {call_premium_ratio:.2f}, Put ratio {put_premium_ratio:.2f}")
                    for leg_id in otm_legs:
                        leg = self.legs[leg_id]
                        buy_leg_id = 'leg3' if leg_id == 'leg1' else 'leg4' if leg_id == 'leg2' else leg_id
                        sell_premium = leg["data"]["close"] if leg['action'] == 'SELL' else self.legs[buy_leg_id]["data"]["close"]
                        buy_premium = self.legs[buy_leg_id]["data"]["close"] if leg['action'] == 'SELL' else leg["data"]["close"]
                        if pd.isna(sell_premium) or pd.isna(buy_premium):
                            print(f"[{self.time}] Invalid premiums for {leg['contract']}: sell={sell_premium}, buy={buy_premium}")
                            continue
                        if sell_premium < self.min_premium_ratio * buy_premium:
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
                            if new_sell_premium < self.min_premium_ratio * buy_premium:
                                print(f"[{self.time}] Rolled premium for {leg['contract']} still below {self.min_premium_ratio} of buy: {new_sell_premium}/{buy_premium}")
                                continue
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
                            print(f"[{self.time}] Rolled OTM leg {leg_id}: {leg['contract']}")

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
                    self.adjustment_count += 1
                    print(f"[{self.time}] Adjusted position: New Spot={self.entry_spot}, New Premium={self.entry_premium:.2f}")

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"

    bt = Backtest(
        db_path=db_path,
        strategy=DoubleCalendarStrategy,
        cash=10000000,
        commission_per_contract=0.02,
        option_multiplier=75,
    )
    stats = bt.run(start_date="2022-02-01", end_date="2023-12-31")
    print(stats)
    bt.tear_sheet()
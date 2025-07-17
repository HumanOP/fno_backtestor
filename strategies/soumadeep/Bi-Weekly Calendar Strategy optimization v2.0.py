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

# Global variables for backtesting period
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

class DoubleCalendarStrategy(Strategy):
    legs: dict = None
    position_id = 0
    portfolio_tp = 0
    portfolio_sl = 0

    # Parameters
    stop_loss_percent: float = 0.01
    take_profit_percent: float = 0.02
    ticker_date_format: str = '%d%b%y'
    min_premium_ratio: float = 0.6
    vol_slope_threshold_contango: float = 0.002
    iv_slope_interval_hours: float = 4
    max_adjustments: int = 2
    sma_short_period: int = 10
    sma_long_period: int = 50
    roc_period: int = 20
    sma_diff_threshold: float = 0.002  # 0.2% of spot
    roc_threshold: float = 0.5  # 0.5% ROC
    vix_threshold_low: float = 15  # Enter if VIX < 15
    vix_threshold_high: float = 20  # Exit if VIX > 20
    vix_spike_threshold: float = 5  # Exit if VIX increases by 5 points
    atr_period: int = 14
    atr_threshold_low: float = 0.005  # 0.5% of NIFTY close
    atr_threshold_high: float = 0.01  # 1% of NIFTY close

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
        self.price_history = []  # For SMA and ROC
        self.vix_history = []  # For VIX tracking
        self.entry_vix = None
        self.nifty_ohlc_history = []  # For ATR
        self.vix_is_decimal = False  # Flag for VIX scale

        # Load and validate CSV files
        try:
            self.vix_data = pd.read_csv(r"FnO-Synapse\demos\india_vix.csv")
            required_vix_cols = ['date', 'close']
            if not all(col in self.vix_data.columns for col in required_vix_cols):
                raise ValueError(f"India VIX CSV missing required columns: {required_vix_cols}")
            self.vix_data['date'] = pd.to_datetime(self.vix_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            self.vix_data.set_index('date', inplace=True)
            # Check VIX scale (assume decimal if max value < 1)
            if not self.vix_data.empty and self.vix_data['close'].max() < 1:
                self.vix_data['close'] *= 100
                self.vix_is_decimal = True
            # Filter to backtesting period using global variables
            self.vix_data = self.vix_data.loc[START_DATE:END_DATE]
            print(f"[{self.time}] Loaded India VIX data: {len(self.vix_data)} rows")
            print(f"[{self.time}] VIX columns: {self.vix_data.columns.tolist()}")
            print(f"[{self.time}] VIX sample:\n{self.vix_data.head()}")
            print(f"[{self.time}] VIX date range: {self.vix_data.index.min()} to {self.vix_data.index.max()}")
        except Exception as e:
            print(f"[{self.time}] Error loading India VIX CSV: {e}")
            self.vix_data = pd.DataFrame()

        try:
            self.nifty_data = pd.read_csv(r"FnO-Synapse\demos\nifty_50.csv")
            required_nifty_cols = ['date', 'open', 'high', 'low', 'close']
            if not all(col in self.nifty_data.columns for col in required_nifty_cols):
                raise ValueError(f"NIFTY spot CSV missing required columns: {required_nifty_cols}")
            self.nifty_data['date'] = pd.to_datetime(self.nifty_data['date'], utc=True).dt.tz_convert('Asia/Kolkata')
            self.nifty_data.set_index('date', inplace=True)
            # Filter to backtesting period using global variables
            self.nifty_data = self.nifty_data.loc[START_DATE:END_DATE]
            print(f"[{self.time}] Loaded NIFTY spot data: {len(self.nifty_data)} rows")
            print(f"[{self.time}] NIFTY columns: {self.nifty_data.columns.tolist()}")
            print(f"[{self.time}] NIFTY sample:\n{self.nifty_data.head()}")
            print(f"[{self.time}] NIFTY date range: {self.nifty_data.index.min()} to {self.nifty_data.index.max()}")
        except Exception as e:
            print(f"[{self.time}] Error loading NIFTY spot CSV: {e}")
            self.nifty_data = pd.DataFrame()

        print(f"[{self.time}] Strategy initialized with legs: {self.legs}")

    def get_nifty_vix_data(self):
        try:
            # Ensure current_time is a pd.Timestamp with IST timezone
            current_time = pd.Timestamp(self.time) if not isinstance(self.time, pd.Timestamp) else self.time
            if current_time.tz is None:
                current_time = current_time.tz_localize('Asia/Kolkata')
            else:
                current_time = current_time.tz_convert('Asia/Kolkata')

            # Fetch VIX
            current_vix = None
            if not self.vix_data.empty:
                vix_row = self.vix_data.loc[self.vix_data.index <= current_time].tail(1)
                print(f"[{self.time}] VIX row:\n{vix_row}")
                if not vix_row.empty and 'close' in vix_row.columns:
                    current_vix = float(vix_row['close'].iloc[0])
                    self.vix_history.append(current_vix)
                    if len(self.vix_history) > 1440:  # 24 hours (1440 minutes)
                        self.vix_history.pop(0)
                    print(f"[{self.time}] VIX: {current_vix:.2f}")

            # Fetch NIFTY for ATR
            current_atr = None
            if not self.nifty_data.empty:
                nifty_rows = self.nifty_data.loc[self.nifty_data.index <= current_time].tail(self.atr_period).copy()
                print(f"[{self.time}] NIFTY rows for ATR ({self.atr_period} periods):\n{nifty_rows}")
                if len(nifty_rows) >= self.atr_period and all(col in nifty_rows.columns for col in ['high', 'low', 'close']):
                    nifty_rows['tr'] = np.maximum(
                        nifty_rows['high'] - nifty_rows['low'],
                        np.maximum(
                            abs(nifty_rows['high'] - nifty_rows['close'].shift(1)),
                            abs(nifty_rows['low'] - nifty_rows['close'].shift(1))
                        )
                    )
                    latest_close = nifty_rows['close'].iloc[-1]
                    current_atr = nifty_rows['tr'].mean() / latest_close if latest_close != 0 else None
                    self.nifty_ohlc_history = nifty_rows[['open', 'high', 'low', 'close']].tail(self.atr_period).to_dict('records')
                    print(f"[{self.time}] ATR: {current_atr:.4f} (normalized by latest close)")

            return current_vix, current_atr
        except Exception as e:
            print(f"[{self.time}] Error fetching CSV data: {e}")
            return None, None

    def update_price_history(self):
        self.price_history.append(self.spot)
        if len(self.price_history) > self.sma_long_period:
            self.price_history.pop(0)

    def get_sma(self, period):
        if len(self.price_history) < period:
            return None
        return np.mean(self.price_history[-period:])

    def get_roc(self):
        if len(self.price_history) < self.roc_period + 1:
            return None
        current_price = self.price_history[-1]
        past_price = self.price_history[-self.roc_period - 1]
        return ((current_price - past_price) / past_price) * 100 if past_price != 0 else None

    def is_trending(self):
        sma_short = self.get_sma(self.sma_short_period)
        sma_long = self.get_sma(self.sma_long_period)
        roc = self.get_roc()
        if sma_short is None or sma_long is None or roc is None:
            return False
        sma_diff = abs(sma_short - sma_long) / self.spot if self.spot != 0 else 0
        if sma_diff > self.sma_diff_threshold or abs(roc) > self.roc_threshold:
            trend_direction = "uptrend" if sma_short > sma_long or roc > 0 else "downtrend"
            print(f"[{self.time}] Trend detected: {trend_direction}, SMA diff={sma_diff:.4f}, ROC={roc:.2f}%")
            return True
        return False

    def is_iv_slope_time(self):
        current_time = pd.Timestamp(self.time)
        if current_time.tz is None:
            current_time = current_time.tz_localize('Asia/Kolkata')
        else:
            current_time = current_time.tz_convert('Asia/Kolkata')
        market_open = current_time.replace(hour=9, minute=15, second=0)
        interval_minutes = self.iv_slope_interval_hours * 60
        intervals = [market_open + pd.Timedelta(minutes=i * interval_minutes) for i in range(int(6.25 * 60 / interval_minutes) + 1)]
        for interval in intervals:
            if abs((current_time - interval).total_seconds()) <= 600:  # Widen to ±10 minutes
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

        self.update_price_history()
        current_vix, current_atr = self.get_nifty_vix_data()

        atm = round(self.spot / 50) * 50
        if not math.isfinite(atm):
            print(f"[{self.time}] Invalid ATM strike: {atm}, skipping")
            return

        if not hasattr(self, 'tte_to_expiry') or not self.tte_to_expiry:
            print(f"[{self.time}] No TTE-to-expiry mapping available, skipping")
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

        # Calculate IV slope every 4 hours
        if self.is_iv_slope_time():
            self.vol_slope = self.calculate_vol_slope(near_tte, far_tte)
            self.last_iv_slope_time = pd.Timestamp(self.time).tz_localize('Asia/Kolkata')
            if self.vol_slope is None or current_vix is None or current_atr is None:
                print(f"[{self.time}] Unable to calculate volatility slope, VIX, or ATR, skipping")
                return

            # Select strategy based on IV slope, VIX, ATR, and trend indicators
            if (self.vol_slope > self.vol_slope_threshold_contango and 
                (current_vix is None or current_vix < self.vix_threshold_low) and 
                (current_atr is None or current_atr < self.atr_threshold_low) and 
                not self.is_trending()):
                self.strategy_type = 'double_calendar'
                self.legs['leg1']['action'] = 'SELL'
                self.legs['leg2']['action'] = 'SELL'
                self.legs['leg3']['action'] = 'BUY'
                self.legs['leg4']['action'] = 'BUY'
                self.entry_vix = current_vix
                print(f"[{self.time}] Strategy selected: {self.strategy_type}, VIX={current_vix if current_vix is not None else 'N/A'}, ATR={current_atr if current_atr is not None else 'N/A'}")
            else:
                self.strategy_type = None
                print(f"[{self.time}] Conditions not met: IV slope={self.vol_slope:.4f} (<= {self.vol_slope_threshold_contango}), VIX={current_vix if current_vix is not None else 'N/A'} (>= {self.vix_threshold_low}), ATR={current_atr if current_atr is not None else 'N/A'} (>= {self.atr_threshold_low}), or trending market, skipping entry")
                return
        elif self.vol_slope is None or self.last_iv_slope_time is None:
            print(f"[{self.time}] No IV slope calculated yet, skipping")
            return
        else:
            print(f"[{self.time}] Using last IV slope: {self.vol_slope:.4f} from {self.last_iv_slope_time}, VIX: {current_vix if current_vix is not None else 'N/A'}, ATR: {current_atr if current_atr is not None else 'N/A'}")

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
                    self.entry_vix = current_vix
                    self.position_id += 1
                    print(f"[{self.time}] Entered double_calendar: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}, VIX={current_vix if current_vix is not None else 'N/A'}, ATR={current_atr if current_atr is not None else 'N/A'}")
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
                    self.last_iv_slope_time = pd.Timestamp(self.time).tz_localize('Asia/Kolkata')

            if current_slope is None:
                print(f"[{self.time}] No valid IV slope, skipping exit checks requiring slope")
                current_slope = 0  # Neutral value to avoid errors in adjustment logic

            # Check for VIX spike
            vix_spike = current_vix - self.entry_vix if self.entry_vix is not None and current_vix is not None else 0

            near_expiry = None
            for trade in active_trades:
                try:
                    # Robust ticker parsing
                    ticker = trade.ticker
                    if len(ticker) < 14:
                        print(f"[{self.time}] Invalid ticker format: {ticker}")
                        continue
                    expiry_str = ticker[-14:-7]
                    expiry = datetime.strptime(expiry_str, self.ticker_date_format).date()
                    near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
                except ValueError as e:
                    print(f"[{self.time}] Error parsing expiry from ticker {trade.ticker}: {e}")
                    continue

            exit_reason = (
                "Near Expiry reached" if near_expiry and (pd.Timestamp(self.time).date() == near_expiry) else
                "Stop Loss Hit" if pl_percent <= -self.stop_loss_percent else
                "Take Profit Hit" if pl_percent >= self.take_profit_percent else
                "Vol Slope Reversal" if (self.strategy_type == 'double_calendar' and current_slope < 0) else
                "Trend Detected" if self.is_trending() else
                "High VIX" if current_vix is not None and current_vix > self.vix_threshold_high else
                "VIX Spike" if vix_spike > self.vix_spike_threshold else
                "High ATR" if current_atr is not None and current_atr > self.atr_threshold_high else
                None
            )

            if exit_reason:
                for trade in active_trades:
                    print(f"[{self.time}] Closing position: {exit_reason}, Ticker: {trade.ticker}, VIX={current_vix if current_vix is not None else 'N/A'}, ATR={current_atr if current_atr is not None else 'N/A'}")
                    trade.close(trade.size, tag=exit_reason)
                self.entry_premium = None
                self.entry_spot = None
                self.entry_vix = None
                self.adjustment_count = 0
                self.strategy_type = None
                return

            # Adjustment logic
            if not exit_reason and self.entry_spot and self.entry_premium:
                spot_move = abs(self.spot - self.entry_spot)
                slope_change = abs(current_slope - self.vol_slope) if self.vol_slope is not None else 0
                adjustment_triggered = spot_move >= 150 or slope_change > 0.005

                if self.adjustment_count >= self.max_adjustments:
                    print(f"[{self.time}] Max adjustments ({self.max_adjustments}) reached, skipping")
                    return

                call_sell_premium = self.legs['leg1']["data"]["close"]
                call_buy_premium = self.legs['leg3']["data"]["close"]
                put_sell_premium = self.legs['leg2']["data"]["close"]
                put_buy_premium = self.legs['leg4']["data"]["close"]

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

                # Adjust ITM legs if > 2 strikes away
                is_up_move = self.spot > self.entry_spot
                itm_legs = ['leg1', 'leg3'] if is_up_move else ['leg2', 'leg4']
                otm_legs = ['leg2', 'leg4'] if is_up_move else ['leg1', 'leg3']
                itm_strike = self.legs[itm_legs[0]]["strike"]
                strike_step = 50
                strikes_away = abs(self.spot - itm_strike) / strike_step if strike_step != 0 else 0

                if strikes_away > 2:
                    print(f"[{self.time}] ITM adjustment triggered: Spot {self.spot}, ITM strike {itm_strike}, Strikes away {strikes_away:.2f}")
                    for leg_id in itm_legs:
                        trades_to_close = [
                            trade for trade in active_trades
                            if hasattr(trade, 'leg_id') and trade.leg_id == leg_id or 
                            hasattr(trade, 'entry_tag') and trade.entry_tag.startswith(f'Entry_{leg_id}')
                        ]
                        if not trades_to_close:
                            print(f"[{self.time}] No trades found for ITM leg {leg_id}, skipping closure")
                            continue
                        for trade in trades_to_close:
                            print(f"[{self.time}] Closing ITM leg {leg_id}: {trade.ticker} (strikes away > 2)")
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
                                if hasattr(trade, 'leg_id') and trade.leg_id == leg_id or 
                                hasattr(trade, 'entry_tag') and trade.entry_tag.startswith(f'Entry_{leg_id}')
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
                    self.entry_vix = current_vix
                    self.adjustment_count += 1
                    print(f"[{self.time}] Adjusted position: New Spot={self.entry_spot}, New Premium={self.entry_premium:.2f}, VIX={current_vix if current_vix is not None else 'N/A'}, ATR={current_atr if current_atr is not None else 'N/A'}")

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"

    bt = Backtest(
        db_path=db_path,
        strategy=DoubleCalendarStrategy,
        cash=10000000,
        commission_per_contract=0.02,
        option_multiplier=75,
    )
    stats = bt.run(start_date=START_DATE, end_date=END_DATE)
    print(stats)
    bt.tear_sheet()
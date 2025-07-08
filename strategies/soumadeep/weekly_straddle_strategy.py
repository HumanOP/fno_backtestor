import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtesting_opt import _Data, Strategy, Backtest
import pandas as pd
import numpy as np
import math
from datetime import datetime, date as DateObject, timedelta
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import logging
import yfinance as yf
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Volatility functions
def close_to_close_volatility(close, window_size):
    returns = close.pct_change().dropna()
    return returns.rolling(window=window_size).std() * np.sqrt(252 * 390)

def parkinson_volatility(high, low, window_size):
    log_hl = np.log(high / low) ** 2
    return np.sqrt((1 / (4 * np.log(2)) * log_hl.rolling(window=window_size).mean()) * 252 * 390)

def rogers_satchell_volatility(high, low, open, close, window_size):
    log_ho = np.log(high / open)
    log_lo = np.log(low / open)
    log_co = np.log(close / open)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    return np.sqrt(rs.rolling(window=window_size).mean() * 252 * 390)

def yang_zhang_volatility(high, low, open, close, window_size):
    log_ho = np.log(high / open)
    log_lo = np.log(low / open)
    log_co = np.log(close / open)
    close_vol = close_to_close_volatility(close, window_size)
    open_vol = (np.log(open / close.shift(1))).rolling(window=window_size).std() * np.sqrt(252 * 390)
    rs_vol = rogers_satchell_volatility(high, low, open, close, window_size)
    k = 0.34 / (1 + (window_size + 1) / (window_size - 1))
    return np.sqrt((open_vol**2 + k * close_vol**2 + (1 - k) * rs_vol**2))

class VolatilityHybridStrategy(Strategy):
    # Strategy parameters
    stop_loss_percent_short: float = 0.02  # Short straddle: 2% loss
    take_profit_percent_short: float = 0.03  # Short straddle: 3% profit
    stop_loss_percent_long: float = 0.25  # Long straddle: 25% loss
    take_profit_percent_long: float = 0.50  # Long straddle: 50% profit
    vol_entry_threshold_short: float = 0.0  # Short straddle: volatility < mean
    vol_entry_threshold_long: float = 1.5   # Long straddle: volatility > mean + 1.5 std
    vol_exit_threshold_short: float = 1.5   # Short straddle exit: volatility > mean + 1.5 std
    vol_exit_threshold_long: float = 0.0    # Long straddle exit: volatility < mean
    confirmation_period: int = 2            # 2-minute confirmation
    max_holding_hours: float = 24          # Max 24 hours
    lot_size: int = 50                     # Nifty 50 lot size
    spread_factor: float = 0.005           # 0.5% bid-ask spread
    position_id = 0
    signal = 0  # 0: No position, 1: Short straddle, 2: Long straddle
    portfolio_tp = 0
    portfolio_sl = 0

    def init(self):
        super().init()
        self.legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'action': None, 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'action': None, 'stop_loss': None, 'take_profit': None}
        }
        self.entry_premium = None
        self.entry_spot = None
        self.entry_time = None
        self.strategy_type = None  # 'short' or 'long'
        # Event calendar (update with actual 2025 events)
        self.event_dates = [
            pd.Timestamp("2025-06-05"),  # RBI policy
            pd.Timestamp("2025-07-01"),  # Budget
            pd.Timestamp("2025-08-07")   # RBI policy
        ]
        logger.info(f"[{self.time}] Strategy initialized with legs: {self.legs}")

        # Fetch Yahoo Finance data for volatility features
        try:
            nifty50_ticker = "^NSEI"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            df = yf.download(nifty50_ticker, start=start_date, end=end_date, interval="1m")
        except Exception as e:
            logger.error(f"Error fetching Nifty 50 data: {str(e)}")
            raise

        if df.empty:
            logger.error("Nifty 50 data is empty")
            raise ValueError("Nifty 50 data is empty")

        # Fetch India VIX data
        try:
            vix_ticker = "^VIX"
            vix_df = yf.download(vix_ticker, start=start_date, end=end_date, interval="1m")
            vix_df = vix_df[['Close']].rename(columns={'Close': 'vix'}).fillna(method='ffill').fillna(method='bfill').dropna()
            vix_df['vix_mean'] = vix_df['vix'].rolling(window=20*390).mean()
            vix_df['vix_std'] = vix_df['vix'].rolling(window=20*390).std()
            self.vix_data = vix_df[['vix', 'vix_mean', 'vix_std']]
        except Exception as e:
            logger.error(f"Error fetching VIX data: {str(e)}")
            self.vix_data = None  # Fallback to no VIX confirmation

        # Rename columns
        df.columns = [col.lower() for col in df.columns]
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in Yahoo Finance data: {required_columns}. Got: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {required_columns}")

        # Handle NaNs
        df = df[required_columns].fillna(method='ffill').fillna(method='bfill').dropna()
        if df.empty:
            logger.error("Yahoo Finance data empty after NaN handling")
            raise ValueError("Data empty after NaN handling")

        # Calculate volatility features
        try:
            data = df.copy()
            data["vol_close_to_close_30"] = close_to_close_volatility(data['close'], window_size=30)
            data["vol_close_to_close_60"] = close_to_close_volatility(data['close'], window_size=60)
            data["vol_parkinson_30"] = parkinson_volatility(data['high'], data['low'], window_size=30)
            data["vol_parkinson_60"] = parkinson_volatility(data['high'], data['low'], window_size=60)
            data["vol_rogers_satchell_30"] = rogers_satchell_volatility(data['high'], data['low'], data['open'], data['close'], window_size=30)
            data["vol_rogers_satchell_60"] = rogers_satchell_volatility(data['high'], data['low'], data['open'], data['close'], window_size=60)
            data["vol_yang_zhang_30"] = yang_zhang_volatility(data['high'], data['low'], data['open'], data['close'], window_size=30)
            data["vol_yang_zhang_60"] = yang_zhang_volatility(data['high'], data['low'], data['open'], data['close'], window_size=60)
        except Exception as e:
            logger.error(f"Error calculating volatility features: {str(e)}")
            raise

        # Remove NaNs
        pre_len = len(data)
        data = data.dropna()
        if data.empty or len(data) < 100:
            logger.error(f"Data empty or insufficient ({len(data)} rows) after volatility calculations. Original length: {pre_len}")
            raise ValueError("Data empty or insufficient after volatility calculations")

        # Select volatility features
        vol_features = [col for col in data.columns if "vol_" in col]
        if not vol_features:
            logger.error("No volatility features found in data")
            raise ValueError("No volatility features found")

        # Standardize features
        try:
            scaler = StandardScaler()
            vol_scaled = scaler.fit_transform(data[vol_features])
            vol_scaled_df = pd.DataFrame(vol_scaled, index=data.index, columns=vol_features)
        except Exception as e:
            logger.error(f"Error standardizing volatility features: {str(e)}")
            raise

        # Apply Kernel PCA
        try:
            pca = KernelPCA(n_components=1)
            data["volatility"] = pca.fit_transform(vol_scaled_df)
        except Exception as e:
            logger.error(f"Error applying Kernel PCA: {str(e)}")
            raise

        # Calculate rolling mean and std
        data["vol_mean"] = data["volatility"].rolling(window=100).mean()
        data["vol_std"] = data["volatility"].rolling(window=100).std()
        data = data.dropna()
        if data.empty:
            logger.error("Data empty after rolling mean/std calculations")
            raise ValueError("Data empty after rolling calculations")

        # Store volatility data
        self.vol_data = data[["volatility", "vol_mean", "vol_std"]]
        logger.info(f"[{self.time}] Volatility data initialized with {len(self.vol_data)} rows")

        # Calculate SMAs from database data
        db_data = self._data.get_index_data()
        if db_data is None or db_data.empty:
            logger.error("Database index data not available or empty")
            raise ValueError("Database index data not available or empty")
        db_data['sma_50'] = db_data['spot_price'].rolling(window=50).mean()
        db_data['sma_200'] = db_data['spot_price'].rolling(window=200).mean()
        self.db_data = db_data[['sma_50', 'sma_200']].dropna()

    def next(self):
        super().next()
        if self.spot is None or pd.isna(self.spot) or not math.isfinite(self.spot):
            logger.warning(f"[{self.time}] Invalid spot price: {self.spot}, skipping")
            return

        # Get volatility data
        try:
            vol_row = self.vol_data.loc[self.vol_data.index[self.vol_data.index.get_loc(pd.Timestamp(self.time), method='nearest')]]
            current_vol = vol_row["volatility"]
            vol_mean = vol_row["vol_mean"]
            vol_std = vol_row["vol_std"]
        except KeyError:
            logger.warning(f"[{self.time}] Volatility data not available, skipping")
            return

        # Get VIX data
        vix_confirm_short = True
        vix_confirm_long = True
        if self.vix_data is not None:
            try:
                vix_row = self.vix_data.loc[self.vix_data.index[self.vix_data.index.get_loc(pd.Timestamp(self.time), method='nearest')]]
                vix_confirm_short = vix_row['vix'] < vix_row['vix_mean']
                vix_confirm_long = vix_row['vix'] > vix_row['vix_mean'] + vix_row['vix_std']
            except KeyError:
                logger.warning(f"[{self.time}] VIX data not available, skipping VIX confirmation")

        # Get trend data
        try:
            db_row = self.db_data.loc[self.db_data.index[self.db_data.index.get_loc(pd.Timestamp(self.time), method='nearest')]]
            trend_confirm = abs(db_row['sma_50'] - db_row['sma_200']) / db_row['sma_200'] < 0.01
        except KeyError:
            logger.warning(f"[{self.time}] SMA data not available, skipping trend confirmation")
            trend_confirm = True

        # Check event calendar
        event_free = all(
            abs((pd.Timestamp(self.time).date() - event.date()).days) > 2
            for event in self.event_dates
        )

        atm = round(self.spot / 50) * 50
        if not math.isfinite(atm):
            logger.warning(f"[{self.time}] Invalid ATM strike: {atm}, skipping")
            return

        # Assign expiries
        for leg_id, leg in self.legs.items():
            lower, upper = leg["expiry_range"]
            valid_ttes = [tte for tte in self.tte_to_expiry.keys() if lower <= tte <= upper]
            if not valid_ttes:
                valid_tte = min(self.tte_to_expiry.keys(), key=lambda x: min(abs(x - lower), abs(x - upper)))
                logger.info(f"[{self.time}] No TTE in range {leg['expiry_range']} for {leg_id}, using closest TTE: {valid_tte}")
            else:
                valid_tte = min(valid_ttes)
            leg["expiry"] = self.tte_to_expiry[valid_tte]

        # Assign strikes and contracts
        for leg in self.legs.values():
            leg["strike"] = float(atm)
            expiry = leg["expiry"]
            if not isinstance(expiry, (pd.Timestamp, datetime)):
                try:
                    expiry = pd.to_datetime(expiry)
                except (ValueError, TypeError) as e:
                    logger.warning(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                    return
            contract = f"NIFTY{expiry.strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
            leg["contract"] = contract
            leg["data"] = self._data.get_ticker_data(contract)

        missing_legs = [leg["contract"] for leg in self.legs.values() if leg["data"] is None]
        if missing_legs:
            logger.warning(f"[{self.time}] Data not found for {self.time}. Spot: {self.spot} Missing legs: {missing_legs}")
            return

        # Validate IV and premium with bid-ask spread
        net_premium = 0
        for leg in self.legs.values():
            premium = leg["data"]["close"]
            iv = leg["data"].get("iv")
            if pd.isna(premium) or not math.isfinite(premium) or pd.isna(iv) or not math.isfinite(iv):
                logger.warning(f"[{self.time}] Skipping leg {leg['contract']}: Invalid premium {premium} or IV {iv}")
                return
            adjusted_premium = premium * (1 - self.spread_factor) if leg['action'] == 'SELL' else premium * (1 + self.spread_factor)
            net_premium += (1 if leg['action'] == 'SELL' else -1) * adjusted_premium

        if not math.isfinite(net_premium) or net_premium == 0:
            logger.warning(f"[{self.time}] Invalid or zero net premium: {net_premium}, skipping")
            return

        # Calculate signal
        new_signal = 0
        try:
            current_idx = self.vol_data.index.get_loc(pd.Timestamp(self.time), method='nearest')
            vol_confirm_short = all(
                self.vol_data["volatility"].iloc[current_idx - j] < self.vol_data["vol_mean"].iloc[current_idx - j]
                for j in range(self.confirmation_period)
                if current_idx - j >= 0
            )
            vol_confirm_long = all(
                self.vol_data["volatility"].iloc[current_idx - j] > (self.vol_data["vol_mean"].iloc[current_idx - j] + self.vol_entry_threshold_long * self.vol_data["vol_std"].iloc[current_idx - j])
                for j in range(self.confirmation_period)
                if current_idx - j >= 0
            )
            if vol_confirm_short and vix_confirm_short and trend_confirm and event_free:
                new_signal = 1  # Short straddle
                self.strategy_type = 'short'
                self.legs['leg1']['action'] = 'SELL'
                self.legs['leg2']['action'] = 'SELL'
            elif vol_confirm_long and vix_confirm_long and trend_confirm and event_free:
                new_signal = 2  # Long straddle
                self.strategy_type = 'long'
                self.legs['leg1']['action'] = 'BUY'
                self.legs['leg2']['action'] = 'BUY'
            elif self.strategy_type == 'short' and current_vol > (vol_mean + self.vol_exit_threshold_short * vol_std):
                new_signal = -1  # Exit short straddle
            elif self.strategy_type == 'long' and current_vol < (vol_mean + self.vol_exit_threshold_long * vol_std):
                new_signal = -1  # Exit long straddle
        except Exception as e:
            logger.warning(f"[{self.time}] Error in volatility confirmation: {str(e)}")
            return

        logger.info(f"[{self.time}] Signal: {self.signal}, New Signal: {new_signal}, Volatility: {current_vol:.2f}, Vol Mean: {vol_mean:.2f}, Spot: {self.spot}")

        current_equity = self._broker.margin_available
        total_margin = abs(net_premium) * self.lot_size
        base_quantity = max(1, int(current_equity / (total_margin * 2)))
        vol_factor = min(1.5, max(0.5, 1 / (1 + current_vol / vol_mean))) if vol_mean != 0 else 1
        quantity = max(1, int(base_quantity * vol_factor))
        active_trades = self.active_trades

        # Entry logic
        if not active_trades and new_signal in [1, 2] and pd.Timestamp(self.time).time() < pd.Timestamp("15:00:00").time():
            placed_any_leg = False
            logger.info(f"[{self.time}] Selected legs: {[leg['contract'] for leg in self.legs.values()]}")
            for leg_id, leg in self.legs.items():
                order_fn = self.sell if leg['action'] == 'SELL' else self.buy
                try:
                    order_fn(
                        strategy_id='strat1',
                        position_id=self.position_id,
                        leg_id=leg_id,
                        ticker=leg["contract"],
                        quantity=quantity,
                        stop_loss=None,
                        take_profit=None,
                        tag=f"Entry {'Short' if new_signal == 1 else 'Long'} Straddle"
                    )
                    placed_any_leg = True
                except Exception as e:
                    logger.error(f"[{self.time}] Error placing order for {leg['contract']}: {str(e)}")
                    continue
            if placed_any_leg:
                self.entry_premium = abs(net_premium)
                self.entry_spot = self.spot
                self.entry_time = self.time
                self.position_id += 1
                logger.info(f"[{self.time}] Entered {'short' if new_signal == 1 else 'long'} straddle: Net Premium={net_premium:.2f}, Quantity={quantity}, Spot={self.entry_spot}")

        else:
            logger.info(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            # Calculate current position value
            current_premium = 0
            for leg in self.legs.values():
                premium = leg["data"]["close"]
                adjusted_premium = premium * (1 + self.spread_factor) if leg['action'] == 'SELL' else premium * (1 - self.spread_factor)
                if pd.isna(premium) or not math.isfinite(premium):
                    logger.warning(f"[{self.time}] Invalid premium for {leg['contract']}: {premium}")
                    return
                current_premium += (1 if leg['action'] == 'SELL' else -1) * adjusted_premium

            # P&L percentage
            pl_percent = (self.entry_premium - current_premium) / abs(self.entry_premium) if self.entry_premium and self.strategy_type == 'short' else \
                         (current_premium - self.entry_premium) / abs(self.entry_premium) if self.entry_premium else 0

            # Exit logic
            hours_held = ((pd.Timestamp(self.time) - pd.Timestamp(self.entry_time)).total_seconds() / 3600
                         if self.entry_time else 0)
            near_expiry = None
            for trade in active_trades:
                try:
                    expiry = datetime.strptime(trade.ticker[-14:-7], "%d%b%y").date()
                    near_expiry = expiry if near_expiry is None else min(near_expiry, expiry)
                except Exception as e:
                    logger.warning(f"[{self.time}] Error parsing expiry from ticker {trade.ticker}: {str(e)}")
                    continue
            stop_loss = self.stop_loss_percent_short if self.strategy_type == 'short' else self.stop_loss_percent_long
            take_profit = self.take_profit_percent_short if self.strategy_type == 'short' else self.take_profit_percent_long
            exit_reason = (
                "Near Expiry reached" if (near_expiry and pd.Timestamp(self.time).date() == near_expiry) else
                "Stop Loss Hit" if pl_percent <= -stop_loss else
                "Take Profit Hit" if pl_percent >= take_profit else
                "Volatility Spike" if new_signal == -1 else
                "VIX Spike" if self.vix_data is not None and vix_row['vix'] > (vix_row['vix_mean'] + vix_row['vix_std']) else
                "Max Holding Reached" if hours_held >= self.max_holding_hours else
                None
            )
            if exit_reason:
                for trade in active_trades:
                    try:
                        logger.info(f"[{self.time}] Closing position: {exit_reason}")
                        trade.close(trade.size, tag=exit_reason)
                    except Exception as e:
                        logger.error(f"[{self.time}] Error closing trade {trade.ticker}: {str(e)}")
                        continue
                self.entry_premium = None
                self.entry_spot = None
                self.entry_time = None
                self.strategy_type = None
                self.signal = 0
                logger.info(f"[{self.time}] Orders: {len(self.orders)}, Active Trades: {len(self.active_trades)}, Equity: {self.equity:.2f}, Closed Trades: {len(self.closed_trades)}")

            # Adjustment logic (short straddle only)
            if not exit_reason and self.entry_spot and self.strategy_type == 'short':
                leg_strike = self.legs["leg2"]["strike"]
                if not (self.spot * 0.99 <= leg_strike <= self.spot * 1.01):
                    placed_any_leg = False
                    for leg_id, leg in self.legs.items():
                        leg["strike"] = float(atm)
                        expiry = leg["expiry"]
                        if not isinstance(expiry, (pd.Timestamp, datetime)):
                            try:
                                expiry = pd.to_datetime(expiry)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"[{self.time}] Invalid expiry format for leg {leg}: {expiry}, error: {e}")
                                continue
                        contract = f"NIFTY{expiry.strftime('%d%b%y').upper()}{int(leg['strike'])}{leg['type']}"
                        leg["contract"] = contract
                        leg["data"] = self._data.get_ticker_data(contract)
                        if leg["data"] is None:
                            logger.warning(f"[{self.time}] No data for adjusted contract {contract}")
                            continue
                        order_fn = self.sell
                        try:
                            order_fn(
                                strategy_id='strat1',
                                position_id=self.position_id,
                                leg_id=leg_id,
                                ticker=leg["contract"],
                                quantity=quantity,
                                stop_loss=None,
                                take_profit=None,
                                tag='Adjustment Straddle'
                            )
                            placed_any_leg = True
                        except Exception as e:
                            logger.error(f"[{self.time}] Error placing adjustment order for {leg['contract']}: {str(e)}")
                            continue
                    if placed_any_leg:
                        self.position_id += 1
                        self.entry_spot = self.spot
                        self.entry_premium = abs(current_premium)
                        self.entry_time = self.time
                        logger.info(f"[{self.time}] Adjusted straddle: New Spot={self.entry_spot}")

        self.signal = new_signal

if __name__ == "__main__":
    db_path = r"FnO-Synapse\demos\nifty_1min_desiquant.duckdb"
    bt = Backtest(
        db_path=db_path,
        strategy=VolatilityHybridStrategy,
        cash=10000000,
        commission_per_contract=0.65,
        option_multiplier=75
    )
    try:
        stats = bt.run()
        logger.info(f"Backtest results: {stats}")
        bt.tear_sheet()
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from scipy.stats import norm
plt.style.use("seaborn-v0_8")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Volatility functions
def close_to_close_volatility(close, window_size):
    returns = close.pct_change().dropna()
    return returns.rolling(window=window_size).std() * np.sqrt(252 * 6)  # Annualized, 6 hours/day

def parkinson_volatility(high, low, window_size):
    log_hl = np.log(high / low) ** 2
    return np.sqrt((1 / (4 * np.log(2)) * log_hl.rolling(window=window_size).mean()) * 252 * 6)

def rogers_satchell_volatility(high, low, open, close, window_size):
    log_ho = np.log(high / open)
    log_lo = np.log(low / open)
    log_co = np.log(close / open)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    return np.sqrt(rs.rolling(window=window_size).mean() * 252 * 6)

def yang_zhang_volatility(high, low, open, close, window_size):
    log_ho = np.log(high / open)
    close_vol = close_to_close_volatility(close, window_size)
    open_vol = np.log(open / close.shift(1)).rolling(window=window_size).std() * np.sqrt(252 * 6)
    rs_vol = rogers_satchell_volatility(high, low, open, close, window_size)
    k = 0.34 / (1 + (window_size + 1) / (window_size - 1))
    return np.sqrt((open_vol**2 + k * close_vol**2 + (1 - k) * rs_vol**2))

# Simplified Black-Scholes for option pricing
def simple_option_price(S, K, sigma, t=14/365, r=0.0025):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(1.0, call), max(1.0, put)  # Ensure non-zero prices

# Strategy parameters
initial_capital = 10000000  # INR
lot_size = 50
commission_per_contract = 0.65
spread_factor = 0.005  # 0.5% bid-ask spread
stop_loss_percent_short = 0.02
take_profit_percent_short = 0.03
stop_loss_percent_long = 0.25
take_profit_percent_long = 0.50
vol_entry_threshold_short = 0.0
vol_entry_threshold_long = 1.5
vol_exit_threshold_short = 1.5
vol_exit_threshold_long = 0.0
confirmation_period = 2
max_holding_hours = 24
event_dates = [
    pd.Timestamp("2025-06-05"),  # RBI policy
    pd.Timestamp("2025-07-01"),  # Budget
    pd.Timestamp("2025-08-07")   # RBI policy
]

# Fetch Yahoo Finance data
start_date = pd.Timestamp("2024-06-24")
end_date = pd.Timestamp("2025-06-23")
try:
    nifty_df = yf.download("^NSEI", start=start_date, end=end_date, interval="1h")
    vix_df = yf.download("^VIX", start=start_date, end=end_date, interval="1h")
except Exception as e:
    logger.error(f"Error fetching Yahoo Finance data: {str(e)}")
    raise

# Process Nifty 50 data
if isinstance(nifty_df.columns, pd.MultiIndex):
    nifty_df.columns = [col[0] for col in nifty_df.columns]
else:
    nifty_df.columns = [str(col) for col in nifty_df.columns]
nifty_df.columns = [col.lower() for col in nifty_df.columns]
required_columns = ['open', 'high', 'low', 'close']
if not all(col in nifty_df.columns for col in required_columns):
    logger.error(f"Missing columns: {required_columns}. Got: {list(nifty_df.columns)}")
    raise ValueError(f"Missing required columns: {required_columns}")
nifty_df = nifty_df[required_columns].fillna(method='ffill').fillna(method='bfill').dropna()
if nifty_df.empty:
    logger.error("Nifty 50 data empty after NaN handling")
    raise ValueError("Nifty 50 data empty")

# Process VIX data
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = [col[0] for col in vix_df.columns]
else:
    vix_df.columns = [str(col) for col in vix_df.columns]
vix_df = vix_df[['Close']].rename(columns={'Close': 'vix'}).fillna(method='ffill').fillna(method='bfill')
# Ensure DatetimeIndex
vix_df.index = pd.to_datetime(vix_df.index)
vix_df['vix_mean'] = vix_df['vix'].rolling(window=20*6).mean()  # 20 days
vix_df['vix_std'] = vix_df['vix'].rolling(window=20*6).std()
vix_df = vix_df.dropna()
if vix_df.empty:
    logger.warning("VIX data empty, disabling VIX confirmation")
    vix_df = None

# Calculate volatility features
data = nifty_df.copy()
data["vol_close_to_close_30"] = close_to_close_volatility(data['close'], window_size=30)
data["vol_close_to_close_60"] = close_to_close_volatility(data['close'], window_size=60)
data["vol_parkinson_30"] = parkinson_volatility(data['high'], data['low'], window_size=30)
data["vol_parkinson_60"] = parkinson_volatility(data['high'], data['low'], window_size=60)
data["vol_rogers_satchell_30"] = rogers_satchell_volatility(data['high'], data['low'], data['open'], data['close'], window_size=30)
data["vol_rogers_satchell_60"] = rogers_satchell_volatility(data['high'], data['low'], data['open'], data['close'], window_size=60)
data["vol_yang_zhang_30"] = yang_zhang_volatility(data['high'], data['low'], data['open'], data['close'], window_size=30)
data["vol_yang_zhang_60"] = yang_zhang_volatility(data['high'], data['low'], data['open'], data['close'], window_size=60)
data = data.dropna()
if data.empty:
    logger.error("Data empty after volatility calculations")
    raise ValueError("Data empty")

# PCA for volatility
vol_features = [col for col in data.columns if "vol_" in col]
scaler = StandardScaler()
vol_scaled = scaler.fit_transform(data[vol_features])
pca = KernelPCA(n_components=1)
data["volatility"] = pca.fit_transform(vol_scaled)
data["vol_mean"] = data["volatility"].rolling(window=100).mean()
data["vol_std"] = data["volatility"].rolling(window=100).std()
data["sma_50"] = data['close'].rolling(window=50).mean()
data["sma_200"] = data['close'].rolling(window=200).mean()
data = data.dropna()
if data.empty:
    logger.error("Data empty after PCA and SMA calculations")
    raise ValueError("Data empty")

# Initialize backtest variables
capital = initial_capital
trades = []
equity_curve = []
position = None
entry_premium = 0
entry_spot = 0
entry_time = None
strategy_type = None
quantity = 0

# Backtest loop
for timestamp, row in data.iterrows():
    spot = row['close']
    if pd.isna(spot) or not np.isfinite(spot):
        logger.warning(f"[{timestamp}] Invalid spot price: {spot}, skipping")
        continue

    # Skip non-trading hours
    if not (pd.Timestamp("09:15:00").time() <= timestamp.time() <= pd.Timestamp("15:00:00").time()):
        continue

    # Get volatility and VIX data
    current_vol = row["volatility"]
    vol_mean = row["vol_mean"]
    vol_std = row["vol_std"]
    vix_confirm_short = True
    vix_confirm_long = True
    vix_row = None
    if vix_df is not None:
        try:
            # Find nearest timestamp
            nearest_idx = vix_df.index[np.argmin(np.abs(vix_df.index - timestamp))]
            vix_row = vix_df.loc[nearest_idx]
            vix_confirm_short = vix_row['vix'] < vix_row['vix_mean']
            vix_confirm_long = vix_row['vix'] > vix_row['vix_mean'] + vix_row['vix_std']
        except Exception as e:
            logger.warning(f"[{timestamp}] VIX data not available, skipping VIX confirmation: {str(e)}")

    # Trend filter
    trend_confirm = abs(row['sma_50'] - row['sma_200']) / row['sma_200'] < 0.01 if not pd.isna(row['sma_50']) else True

    # Event filter
    event_free = all(abs((timestamp.date() - event.date()).days) > 2 for event in event_dates)

    # Calculate ATM strike
    atm = round(spot / 50) * 50
    if not np.isfinite(atm):
        logger.warning(f"[{timestamp}] Invalid ATM strike: {atm}, skipping")
        continue

    # Estimate option prices
    implied_vol = 0.1 + 0.05 * (current_vol - data["volatility"].mean()) / data["volatility"].std()
    implied_vol = np.clip(implied_vol, 0.05, 0.5)
    call_premium, put_premium = simple_option_price(S=spot, K=atm, sigma=implied_vol)
    net_premium = call_premium + put_premium

    # Calculate signal
    signal = 0
    try:
        current_idx = data.index.get_loc(timestamp)
        if current_idx >= confirmation_period:
            vol_confirm_short = all(
                data["volatility"].iloc[current_idx - j] < data["vol_mean"].iloc[current_idx - j]
                for j in range(confirmation_period)
            )
            vol_confirm_long = all(
                data["volatility"].iloc[current_idx - j] > (data["vol_mean"].iloc[current_idx - j] + vol_entry_threshold_long * data["vol_std"].iloc[current_idx - j])
                for j in range(confirmation_period)
            )
            if vol_confirm_short and vix_confirm_short and trend_confirm and event_free:
                signal = 1  # Short straddle
                strategy_type = 'short'
            elif vol_confirm_long and vix_confirm_long and trend_confirm and event_free:
                signal = 2  # Long straddle
                strategy_type = 'long'
            elif position == 'short' and current_vol > (vol_mean + vol_exit_threshold_short * vol_std):
                signal = -1
            elif position == 'long' and current_vol < (vol_mean + vol_exit_threshold_long * vol_std):
                signal = -1
    except Exception as e:
        logger.warning(f"[{timestamp}] Error in volatility confirmation: {str(e)}")
        continue

    # Dynamic position sizing
    total_margin = abs(net_premium) * lot_size
    base_quantity = max(1, int(capital / (total_margin * 2)))
    vol_factor = min(1.5, max(0.5, 1 / (1 + current_vol / vol_mean))) if vol_mean != 0 else 1
    quantity = max(1, int(base_quantity * vol_factor))

    # Entry logic
    if not position and signal in [1, 2] and timestamp.time() < pd.Timestamp("15:00:00").time():
        action = 'SELL' if signal == 1 else 'BUY'
        entry_call = call_premium * (1 - spread_factor if action == 'SELL' else 1 + spread_factor)
        entry_put = put_premium * (1 - spread_factor if action == 'SELL' else 1 + spread_factor)
        entry_premium = entry_call + entry_put
        position = 'short' if signal == 1 else 'long'
        entry_spot = spot
        entry_time = timestamp
        if action == 'BUY':
            capital -= entry_premium * quantity * lot_size * (1 + spread_factor)
        else:
            capital += entry_premium * quantity * lot_size * (1 - spread_factor)
        capital -= 2 * quantity * commission_per_contract
        trades.append({
            'entry_time': timestamp,
            'type': position,
            'strike': atm,
            'quantity': quantity,
            'entry_premium': entry_premium,
            'entry_spot': spot
        })
        logger.info(f"[{timestamp}] Entered {position} straddle: Premium={entry_premium:.2f}, Quantity={quantity}, Spot={spot}")

    # Exit and adjustment logic
    elif position:
        current_call = call_premium * (1 + spread_factor if position == 'short' else 1 - spread_factor)
        current_put = put_premium * (1 + spread_factor if position == 'short' else 1 - spread_factor)
        current_premium = current_call + current_put
        pl_percent = (entry_premium - current_premium) / abs(entry_premium) if position == 'short' else \
                     (current_premium - entry_premium) / abs(entry_premium)
        hours_held = (timestamp - entry_time).total_seconds() / 3600
        stop_loss = stop_loss_percent_short if position == 'short' else stop_loss_percent_long
        take_profit = take_profit_percent_short if position == 'long' else take_profit_percent_long
        expiry_date = (entry_time + timedelta(days=14)).date()  # Approximate 14-day expiry
        exit_reason = (
            "Near Expiry" if timestamp.date() >= expiry_date else
            "Stop Loss" if pl_percent <= -stop_loss else
            "Take Profit" if pl_percent >= take_profit else
            "Volatility Exit" if signal == -1 else
            "VIX Spike" if vix_df is not None and vix_row is not None and vix_row['vix'] > (vix_row['vix_mean'] + vix_row['vix_std']) else
            "Max Holding" if hours_held >= max_holding_hours else
            None
        )
        if exit_reason:
            pl = (entry_premium - current_premium if position == 'short' else current_premium - entry_premium) * quantity * lot_size
            capital += (pl if position == 'long' else -pl) - 2 * quantity * commission_per_contract
            trades[-1].update({
                'exit_time': timestamp,
                'exit_premium': current_premium,
                'pl': pl,
                'exit_reason': exit_reason
            })
            logger.info(f"[{timestamp}] Exited {position} straddle: P&L={pl:.2f}, Reason={exit_reason}")
            position = None
            entry_premium = 0
            entry_spot = 0
            entry_time = None
            strategy_type = None
        elif position == 'short' and not (spot * 0.99 <= atm <= spot * 1.01):
            # Adjust short straddle
            new_call_premium, new_put_premium = simple_option_price(S=spot, K=atm, sigma=implied_vol)
            new_premium = new_call_premium + new_put_premium
            pl = (entry_premium - current_premium) * quantity * lot_size
            capital += pl - 2 * quantity * commission_per_contract
            trades[-1].update({
                'exit_time': timestamp,
                'exit_premium': current_premium,
                'pl': pl,
                'exit_reason': 'Adjustment'
            })
            entry_premium = new_premium * (1 - spread_factor)
            entry_spot = spot
            entry_time = timestamp
            capital += entry_premium * quantity * lot_size * (1 - spread_factor) - 2 * quantity * commission_per_contract
            trades.append({
                'entry_time': timestamp,
                'type': 'short',
                'strike': atm,
                'quantity': quantity,
                'entry_premium': entry_premium,
                'entry_spot': spot
            })
            logger.info(f"[{timestamp}] Adjusted short straddle: New Premium={entry_premium:.2f}, New Spot={spot}")

    equity_curve.append({'timestamp': timestamp, 'equity': capital})

# Process results
equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['pl'] > 0]) / num_trades
    profit_factor = trades_df[trades_df['pl'] > 0]['pl'].sum() / abs(trades_df[trades_df['pl'] < 0]['pl'].sum()) if any(trades_df['pl'] < 0) else np.inf
    sharpe_ratio = (trades_df['pl'].mean() / trades_df['pl'].std()) * np.sqrt(252 * 6) if trades_df['pl'].std() != 0 else 0
    max_drawdown = ((equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()).max() * 100
else:
    total_return = 0
    num_trades = 0
    win_rate = 0
    profit_factor = 0
    sharpe_ratio = 0
    max_drawdown = 0

print(f"Total Return: {total_return:.2f}%")
print(f"Number of Trades: {num_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
axes[0].plot(data.index, data['close'], label="Nifty 50 Close", color='black')
axes[0].set_title("Nifty 50 Close Price")
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(data.index, data["volatility"], label="PCA Volatility", color='blue')
axes[1].plot(data.index, data["vol_mean"] + vol_entry_threshold_long * data["vol_std"], color='r', linestyle='--', label="Long Entry Threshold")
axes[1].plot(data.index, data["vol_mean"], color='g', linestyle='--', label="Short Entry/Exit Threshold")
buy_signals = trades_df[trades_df['type'] == 'long']['entry_time']
sell_signals = trades_df[trades_df['type'] == 'short']['entry_time']
axes[1].scatter(buy_signals, data.loc[buy_signals, "volatility"], color='green', marker='^', label="Long Straddle Entry")
axes[1].scatter(sell_signals, data.loc[sell_signals, "volatility"], color='red', marker='v', label="Short Straddle Entry")
axes[1].set_title("PCA Volatility with Trading Signals")
axes[1].set_ylabel("Volatility")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(equity_df.index, equity_df["equity"], label="Equity", color='purple')
axes[2].set_title("Equity Curve")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Equity (INR)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
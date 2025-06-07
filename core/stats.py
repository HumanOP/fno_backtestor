import pandas as pd
import numpy as np
import warnings
from typing import List, Union

def _data_period(index):
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Timedelta(days=1)
    diff = index.to_series().diff().mean()
    return diff if not pd.isna(diff) else pd.Timedelta(days=1)

def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']

def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

def compute_stats(
    orders,
    trades,
    equity_curve,
    strategy_instance=None,
    risk_free_rate: float = 0.0,
    positions: dict = None
) -> pd.Series:
    assert -1 < risk_free_rate < 1

    # Ensure index is datetime
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.to_datetime(equity_curve.index)

    index = equity_curve.index
    equity = equity_curve.rename("Equity").to_frame()

    cummax = np.maximum.accumulate(equity['Equity'])
    dd = 1 - equity['Equity'] / cummax
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    # Convert durations to days (optionally: use np.busday_count for business days)
    dd_dur_days = dd_dur.dt.days if dd_dur.notna().any() else pd.Series(dtype='float64')

    # Format trades
    if isinstance(trades, pd.DataFrame):
        trades_df = trades.copy()
    else:
        trades_df = pd.DataFrame({
            'EntryTime': [t.entry_datetime for t in trades],
            'ExitTime': [t.exit_datetime for t in trades],
            'PnL': [t.pl for t in trades],
            'ReturnPct': [getattr(t, 'pl_pct', t.pl / t.entry_price if t.entry_price else 0) for t in trades],
            'EntryTag': [getattr(t, 'entry_tag', None) for t in trades],
            'ExitTag': [getattr(t, 'exit_tag', None) for t in trades],
        })

    trades_df = trades_df[trades_df['ExitTime'].notna()]
    trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
    trades_df = trades_df.sort_values('EntryTime')

    returns = trades_df['ReturnPct']
    pl = trades_df['PnL']
    durations = trades_df['Duration']

    # Period metrics
    period = _data_period(index).days or 1
    daily_equity = equity['Equity'].resample('1D').last().dropna()
    period_returns = daily_equity.pct_change().dropna()

    gmean_return = geometric_mean(period_returns)
    annual_periods = 252 if period <= 1 else 365 / period
    annual_return = (1 + gmean_return) ** annual_periods - 1

    vol_ann = np.sqrt(period_returns.var() * annual_periods)
    sharpe = (annual_return - risk_free_rate) / vol_ann if vol_ann else np.nan
    sortino = (annual_return - risk_free_rate) / (
        np.sqrt(np.mean(period_returns.clip(upper=0) ** 2)) * np.sqrt(annual_periods)
    ) if not period_returns.empty else np.nan
    calmar = annual_return / dd.max() if dd.max() else np.nan

    win_rate = (pl > 0).mean() if len(pl) else np.nan
    profit_factor = pl[pl > 0].sum() / -pl[pl < 0].sum() if (pl < 0).any() else np.nan
    kelly = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean()) if (pl > 0).any() and (pl < 0).any() else np.nan
    sqn = np.sqrt(len(pl)) * pl.mean() / pl.std() if pl.std() else np.nan

    stats = pd.Series({
        'Start': index[0],
        'End': index[-1],
        'Duration': index[-1] - index[0],
        'Total Trades': len(trades_df),
        'Win Rate [%]': win_rate * 100 if win_rate is not np.nan else np.nan,
        'Avg Trade Return [%]': geometric_mean(returns) * 100,
        'Best Trade [%]': returns.max() * 100,
        'Worst Trade [%]': returns.min() * 100,
        'Profit Factor': profit_factor,
        'Expectancy [%]': returns.mean() * 100,
        'Kelly Criterion': kelly,
        'SQN': sqn,
        'Max Trade Duration': durations.max(),
        'Avg Trade Duration': durations.mean(),
        'Final Equity': equity['Equity'].iloc[-1],
        'Initial Equity': equity['Equity'].iloc[0],
        'Return [%]': (equity['Equity'].iloc[-1] / equity['Equity'].iloc[0] - 1) * 100,
        'Return (Ann.) [%]': annual_return * 100,
        'Volatility (Ann.) [%]': vol_ann * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown [%]': dd.max() * 100,
        'Avg Drawdown [%]': dd_peaks.mean() * 100,
        'Max Drawdown Duration [days]': dd_dur_days.max() if not dd_dur_days.empty else np.nan,
        'Avg Drawdown Duration [days]': dd_dur_days.mean() if not dd_dur_days.empty else np.nan,
    })

    stats['_strategy'] = strategy_instance
    stats['_equity_curve'] = equity
    stats['_trades'] = trades_df
    stats['_positions'] = positions
    return stats

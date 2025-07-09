import warnings
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtesting_opt import Order, Strategy, Trade

def _data_period(index):
    """Estimate the data period from a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Timedelta(days=1)  # Default to daily
    diff = index.to_series().diff().mean()
    if pd.isna(diff):
        return pd.Timedelta(days=1)
    return diff

def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
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
        orders: Union[List['Order'], pd.DataFrame],
        trades: Union[List['Trade'], pd.DataFrame],
        equity: Union[pd.DataFrame, pd.Series],  # Updated to accept Series too
        strategy_instance: 'Strategy',
        risk_free_rate: float = 0,
        positions: dict = None,
        trade_start_bar: int = 0,
) -> pd.Series:
    assert -1 < risk_free_rate < 1

    # Handle empty equity curve first
    if len(equity) == 0:
        return pd.Series({
            'Start': pd.NaT,
            'End': pd.NaT,
            'Duration': pd.NaT,
            '# Trades': 0,
            'Win Rate [%]': np.nan,
            '_strategy': strategy_instance,
            '_equity_curve': pd.DataFrame({'Equity': []}),
            '_trades': pd.DataFrame(),
            '_orders': pd.DataFrame(),
            '_positions': positions,
            '_trade_start_bar': trade_start_bar
        })

    # Fix: Properly convert index to DatetimeIndex and create DataFrame with 'Equity' column
    if isinstance(equity, pd.Series):
        # Ensure index is datetime
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)
        
        # Create DataFrame with explicit column name
        equity_df = pd.DataFrame(equity.values, index=equity.index, columns=['Equity'])
        
    elif isinstance(equity, pd.DataFrame):
        # Ensure index is datetime
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)
            
        if 'Equity' in equity.columns:
            equity_df = equity.copy()
        else:
            # Assume first column is equity if 'Equity' column doesn't exist
            equity_df = equity.copy()
            equity_df.columns = ['Equity'] + list(equity_df.columns[1:]) if len(equity_df.columns) > 1 else ['Equity']
    else:
        raise ValueError(f"equity must be pandas Series or DataFrame, got {type(equity)}")

    index = equity_df.index

    dd = 1 - equity_df['Equity'] / np.maximum.accumulate(equity_df['Equity'])
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    if isinstance(orders, pd.DataFrame):
        orders_df = orders
    else:
        orders_df = pd.DataFrame({
            'SignalTime': [getattr(o, 'time', None) for o in orders],  # Use broker.time
            'Ticker': [getattr(o, 'ticker', 'UNKNOWN') for o in orders],
            'Side': ['Buy' if getattr(o, 'is_long', True) else 'Sell' for o in orders],
            'Size': [int(getattr(o, 'size', 0)) for o in orders],
        }).set_index('SignalTime')

    equity_df = pd.concat([equity_df, pd.DataFrame({'DrawdownPct': dd, 'DrawdownDuration': dd_dur}, index=index)], axis=1)
    if isinstance(trades, pd.DataFrame):
        trades_df = trades
    else:
        # Helper function for exact timestamp matching only
        def get_exact_bar(datetime_val, index):
            if datetime_val is None or pd.isna(datetime_val):
                return np.nan
            try:
                # Convert to same timezone/precision if needed
                if hasattr(datetime_val, 'tz_localize') and datetime_val.tz is None:
                    if hasattr(index, 'tz') and index.tz is not None:
                        datetime_val = datetime_val.tz_localize(index.tz)
                elif hasattr(datetime_val, 'tz_convert') and datetime_val.tz is not None:
                    if hasattr(index, 'tz') and index.tz is not None and datetime_val.tz != index.tz:
                        datetime_val = datetime_val.tz_convert(index.tz)
                
                # Only return position if exact match exists
                if datetime_val in index:
                    return index.get_loc(datetime_val)
                else:
                    return np.nan
            except (KeyError, TypeError, ValueError, AttributeError):
                return np.nan
        
        trades_df = pd.DataFrame({
            'EntryBar': [get_exact_bar(getattr(t, 'entry_datetime', None), index) for t in trades],
            'ExitBar': [get_exact_bar(getattr(t, 'exit_datetime', None), index) for t in trades],
            'Ticker': [getattr(t, 'ticker', 'UNKNOWN') for t in trades],
            'Size': [getattr(t, 'size', 0) for t in trades],
            'EntryPrice': [getattr(t, 'entry_price', 0) for t in trades],
            'ExitPrice': [getattr(t, 'exit_price', 0) for t in trades],
            'PnL': [getattr(t, 'pl', 0) for t in trades],
            'ReturnPct': [getattr(t, 'pl_pct', 0) for t in trades],
            'EntryTime': [getattr(t, 'entry_datetime', None) for t in trades],
            'ExitTime': [getattr(t, 'exit_datetime', None) for t in trades],
            'Tag': [getattr(t, 'entry_tag', None) for t in trades],  # Use entry_tag
            'Reason': [getattr(t, 'exit_tag', None) for t in trades]  # Use exit_tag
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
        # Filter out trades without exit times (open trades)
        trades_df = trades_df[trades_df['ExitTime'].notna()]

    # Handle empty trades
    if len(trades_df) == 0:
        pl = pd.Series(dtype='float64')
        returns = pd.Series(dtype='float64')
        durations = pd.Series(dtype='timedelta64[ns]')
    else:
        pl = trades_df['PnL']
        returns = trades_df['ReturnPct']
        durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta) or pd.isna(value):
            return value
        try:
            resolution = getattr(_period, 'resolution_string', None) or getattr(_period, 'resolution', 'D')
            return value.ceil(resolution)
        except (AttributeError, ValueError):
            # Fallback to simple rounding if resolution methods fail
            return value

    s = pd.Series(dtype=object)
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    have_position = np.zeros(len(index))
    for t in trades_df.itertuples(index=False):
        if pd.isna(t.EntryBar) or pd.isna(t.ExitBar):
            continue
        have_position[int(t.EntryBar):int(t.ExitBar) + 1] = 1

    s.loc['Exposure Time [%]'] = have_position.mean() * 100
    s.loc['Equity Final [$]'] = equity_df['Equity'].iloc[-1]
    s.loc['Equity Peak [$]'] = equity_df['Equity'].max()
    s.loc['Return [%]'] = (equity_df['Equity'].iloc[-1] - equity_df['Equity'].iloc[0]) / equity_df['Equity'].iloc[0] * 100
    # c = ohlc_data['Close'].values
    # s.loc['Buy & Hold Return [%]'] = (c[-1] - c[trade_start_bar]) / c[trade_start_bar] * 100

    gmean_period_return: float = 0
    period_returns = np.array(np.nan)
    annual_trading_periods = np.nan
    if isinstance(index, pd.DatetimeIndex):
        period = _data_period(index).days
        if period <= 1:
            period_returns = equity_df['Equity'].iloc[trade_start_bar:].resample('D').last().dropna().pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = float(
                365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * 0.6 else 252)
        elif period >= 28 and period <= 31:
            period_returns = equity_df['Equity'].iloc[trade_start_bar:].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 12
        elif period >= 365 and period <= 366:
            period_returns = equity_df['Equity'].iloc[trade_start_bar:].pct_change()
            gmean_period_return = geometric_mean(period_returns)
            annual_trading_periods = 1
        else:
            warnings.warn(f'Unsupported data period from index: {period} days.')

    annualized_return = (1 + gmean_period_return)**annual_trading_periods - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    s.loc['Volatility (Ann.) [%]'] = np.sqrt((period_returns.var(ddof=int(bool(period_returns.shape))) + (1 + gmean_period_return)**2)**annual_trading_periods - (1 + gmean_period_return)**(2*annual_trading_periods)) * 100
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate) / (s.loc['Volatility (Ann.) [%]'] or np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(period_returns.clip(-np.inf, 0)**2)) * np.sqrt(annual_trading_periods))
        except Warning:
            s.loc['Sortino Ratio'] = np.nan
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    s.loc['# Trades'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100 if not returns.empty else np.nan
    s.loc['Worst Trade [%]'] = returns.min() * 100 if not returns.empty else np.nan
    mean_return = geometric_mean(returns) if not returns.empty else np.nan
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max()) if not durations.empty else pd.NaT
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean()) if not durations.empty else pd.NaT
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan) if not returns.empty else np.nan
    s.loc['Expectancy [%]'] = returns.mean() * 100 if not returns.empty else np.nan
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan) if not pl.empty and pl.std() > 0 else np.nan
    
    # Kelly Criterion calculation with additional safety checks
    if not pl.empty and (pl > 0).any() and (pl < 0).any():
        avg_win = pl[pl > 0].mean()
        avg_loss = -pl[pl < 0].mean()
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else np.nan
    else:
        kelly = np.nan
    s.loc['Kelly Criterion'] = kelly

    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df
    s.loc['_orders'] = orders_df
    s.loc['_positions'] = positions
    s.loc['_trade_start_bar'] = trade_start_bar

    s = _Stats(s)
    return s

class _Stats(pd.Series):
    def __repr__(self):
        with pd.option_context('max_colwidth', 20):
            return super().__repr__()
import numpy as np
import pandas as pd
from numba import njit
from math import log

@njit(nogil=True)
def _close_to_close_estimator(close, window_size):
    """
    Compute Close-to-Close volatility using standard deviation of log returns.

    Parameters
    ----------
    close : np.ndarray
        Array of closing prices.
    window_size : int
        Rolling window size.

    Returns
    -------
    vol : np.ndarray
        Array containing the rolling Close-to-Close volatility.
    """
    n = close.shape[0]
    vol = np.empty(n)
    vol[:window_size] = np.nan

    # Compute log returns
    log_returns = np.empty(n)
    log_returns[0] = np.nan
    for i in range(1, n):
        if close[i-1] > 0:
            ratio = close[i] / close[i-1]
            log_returns[i] = log(ratio)  # Use math.log for scalar operation
        else:
            log_returns[i] = np.nan

    # Compute rolling standard deviation
    for i in range(window_size, n):
        window = log_returns[i-window_size:i]
        if np.any(np.isnan(window)):
            vol[i] = np.nan
            continue
        mean = np.mean(window)
        sum_squared_diff = np.sum((window - mean) ** 2)
        vol[i] = np.sqrt(sum_squared_diff / (window_size - 1))  # Sample standard deviation

    return vol

def close_to_close_volatility(close: pd.Series, window_size: int = 30) -> pd.Series:
    """
    Calculate Close-to-Close volatility using NumPy operations with Numba acceleration.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices.
    window_size : int, optional
        Number of periods for the rolling calculation (default is 30).

    Returns
    -------
    pd.Series
        Series containing the rolling Close-to-Close volatility, with NaN for the first `window_size` rows.

    Raises
    ------
    ValueError
        If close is empty, not a Series, window_size is invalid, prices are non-positive, or contains invalid data.
    """
    if not isinstance(close, pd.Series):
        raise ValueError(f"close must be a pandas Series, got {type(close)}.")
    if close.empty:
        raise ValueError("Close Series is empty.")
    if window_size <= 1 or window_size > len(close):
        raise ValueError(f"window_size must be > 1 and <= Series length ({len(close)}).")
    if not np.issubdtype(close.dtype, np.floating):
        raise ValueError(f"Close Series must contain floating-point values, got {close.dtype}.")
    if close.isna().any():
        raise ValueError("Close Series contains NaN values.")
    if (close <= 0).any():
        raise ValueError("Closing prices must be positive.")

    close_array = close.to_numpy(dtype=np.float64)
    vol_array = _close_to_close_estimator(close_array, window_size)
    return pd.Series(vol_array, name=f"vol_close_to_close_{window_size}", index=close.index)

@njit(nogil=True)
def _parkinson_estimator(high, low, window_size):
    """
    Compute Parkinson's volatility estimator.

    Parameters
    ----------
    high : np.ndarray
        Array of high prices.
    low : np.ndarray
        Array of low prices.
    window_size : int
        Rolling window size.

    Returns
    -------
    vol : np.ndarray
        Array containing the rolling Parkinson volatility.
    """
    n = high.shape[0]
    vol = np.empty(n)
    vol[:window_size] = np.nan

    for i in range(window_size, n):
        sum_squared = 0.0
        valid = True
        for j in range(i - window_size, i):
            if high[j] <= 0 or low[j] <= 0:
                valid = False
                break
            sum_squared += np.log(high[j] / low[j]) ** 2
        vol[i] = np.sqrt(sum_squared / (4 * window_size * np.log(2))) if valid else np.nan

    return vol

def parkinson_volatility(high: pd.Series, low: pd.Series, window_size: int = 30) -> pd.Series:
    """
    Calculate Parkinson's volatility estimator using NumPy operations with Numba acceleration.

    Parameters
    ----------
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.
    window_size : int, optional
        Number of periods for the rolling calculation (default is 30).

    Returns
    -------
    pd.Series
        Series containing the rolling Parkinson volatility, with NaN for the first `window_size` rows.

    Raises
    ------
    ValueError
        If inputs are empty, not Series, window_size is invalid, or prices are non-positive.
    """
    for name, series in [('high', high), ('low', low)]:
        if not isinstance(series, pd.Series):
            raise ValueError(f"{name} must be a pandas Series, got {type(series)}.")
        if series.empty:
            raise ValueError(f"{name} Series is empty.")
        if not np.issubdtype(series.dtype, np.floating):
            raise ValueError(f"{name} Series must contain floating-point values, got {series.dtype}.")
        if series.isna().any():
            raise ValueError(f"{name} Series contains NaN values.")
    if len(high) != len(low):
        raise ValueError(f"High and low Series must have the same length, got {len(high)} and {len(low)}.")
    if window_size <= 0 or window_size > len(high):
        raise ValueError(f"window_size must be > 0 and <= Series length ({len(high)}).")
    if (high <= 0).any() or (low <= 0).any():
        raise ValueError("High and low prices must be positive.")

    high_array = high.to_numpy(dtype=np.float64)
    low_array = low.to_numpy(dtype=np.float64)
    vol_array = _parkinson_estimator(high_array, low_array, window_size)
    return pd.Series(vol_array, name=f"vol_parkinson_{window_size}", index=high.index)

@njit(nogil=True)
def _rogers_satchell_estimator(high, low, open_, close, window_size):
    """
    Compute Rogers-Satchell volatility estimator.

    Parameters
    ----------
    high : np.ndarray
        Array of high prices.
    low : np.ndarray
        Array of low prices.
    open_ : np.ndarray
        Array of open prices.
    close : np.ndarray
        Array of close prices.
    window_size : int
        Rolling window size.

    Returns
    -------
    vol : np.ndarray
        Array containing the rolling Rogers-Satchell volatility.
    """
    n = high.shape[0]
    vol = np.empty(n)
    vol[:window_size] = np.nan

    for i in range(window_size, n):
        sum_val = 0.0
        valid = True
        for j in range(i - window_size, i):
            if high[j] <= 0 or low[j] <= 0 or open_[j] <= 0 or close[j] <= 0:
                valid = False
                break
            term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
            term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
            sum_val += term1 + term2
        vol[i] = np.sqrt(sum_val / window_size) if valid else np.nan

    return vol

def rogers_satchell_volatility(high: pd.Series, low: pd.Series, open: pd.Series, close: pd.Series, window_size: int = 30) -> pd.Series:
    """
    Calculate Rogers-Satchell volatility estimator using NumPy operations with Numba acceleration.

    Parameters
    ----------
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.
    open : pd.Series
        Series of open prices.
    close : pd.Series
        Series of close prices.
    window_size : int, optional
        Number of periods for the rolling calculation (default is 30).

    Returns
    -------
    pd.Series
        Series containing the rolling Rogers-Satchell volatility, with NaN for the first `window_size` rows.

    Raises
    ------
    ValueError
        If inputs are empty, not Series, window_size is invalid, or prices are non-positive.
    """
    for name, series in [('high', high), ('low', low), ('open', open), ('close', close)]:
        if not isinstance(series, pd.Series):
            raise ValueError(f"{name} must be a pandas Series, got {type(series)}.")
        if series.empty:
            raise ValueError(f"{name} Series is empty.")
        if not np.issubdtype(series.dtype, np.floating):
            raise ValueError(f"{name} Series must contain floating-point values, got {series.dtype}.")
        if series.isna().any():
            raise ValueError(f"{name} Series contains NaN values.")
    if not (len(high) == len(low) == len(open) == len(close)):
        raise ValueError(f"All Series must have the same length, got {len(high)}, {len(low)}, {len(open)}, {len(close)}.")
    if window_size <= 0 or window_size > len(high):
        raise ValueError(f"window_size must be > 0 and <= Series length ({len(high)}).")
    if (high <= 0).any() or (low <= 0).any() or (open <= 0).any() or (close <= 0).any():
        raise ValueError("High, low, open, and close prices must be positive.")

    high_array = high.to_numpy(dtype=np.float64)
    low_array = low.to_numpy(dtype=np.float64)
    open_array = open.to_numpy(dtype=np.float64)
    close_array = close.to_numpy(dtype=np.float64)
    vol_array = _rogers_satchell_estimator(high_array, low_array, open_array, close_array, window_size)
    return pd.Series(vol_array, name=f"vol_rogers_satchell_{window_size}", index=high.index)

@njit(nogil=True)
def _yang_zhang_estimator(high, low, open_, close, window_size, k=0.34):
    """
    Compute Yang-Zhang volatility estimator.

    Parameters
    ----------
    high : np.ndarray
        Array of high prices.
    low : np.ndarray
        Array of low prices.
    open_ : np.ndarray
        Array of open prices.
    close : np.ndarray
        Array of close prices.
    window_size : int
        Rolling window size.
    k : float, optional
        Weighting factor for open-to-close variance (default is 0.34).

    Returns
    -------
    vol : np.ndarray
        Array containing the rolling Yang-Zhang volatility.
    """
    n = high.shape[0]
    vol = np.empty(n)
    vol[:window_size] = np.nan

    for i in range(window_size, n):
        sum_oc = 0.0
        sum_cc = 0.0
        sum_rs = 0.0
        valid = True
        for j in range(i - window_size, i):
            if high[j] <= 0 or low[j] <= 0 or open_[j] <= 0 or close[j] <= 0:
                valid = False
                break
            term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
            term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
            sum_rs += term1 + term2
            diff_oc = np.log(open_[j] / close[j])
            sum_oc += diff_oc * diff_oc
            diff_cc = np.log(close[j] / open_[j])
            sum_cc += diff_cc * diff_cc
        if valid:
            sigma_oc = sum_oc / window_size
            sigma_cc = sum_cc / window_size
            sigma_rs = sum_rs / window_size
            vol[i] = np.sqrt(sigma_oc + k * sigma_cc + (1 - k) * sigma_rs)
        else:
            vol[i] = np.nan

    return vol

def yang_zhang_volatility(high: pd.Series, low: pd.Series, open: pd.Series, close: pd.Series, window_size: int = 30, k: float = 0.34) -> pd.Series:
    """
    Calculate Yang-Zhang volatility estimator using NumPy operations with Numba acceleration.

    Parameters
    ----------
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.
    open : pd.Series
        Series of open prices.
    close : pd.Series
        Series of close prices.
    window_size : int, optional
        Number of periods for the rolling calculation (default is 30).
    k : float, optional
        Weighting factor for open-to-close variance (default is 0.34).

    Returns
    -------
    pd.Series
        Series containing the rolling Yang-Zhang volatility, with NaN for the first `window_size` rows.

    Raises
    ------
    ValueError
        If inputs are empty, not Series, window_size is invalid, or prices are non-positive.
    """
    for name, series in [('high', high), ('low', low), ('open', open), ('close', close)]:
        if not isinstance(series, pd.Series):
            raise ValueError(f"{name} must be a pandas Series, got {type(series)}.")
        if series.empty:
            raise ValueError(f"{name} Series is empty.")
        if not np.issubdtype(series.dtype, np.floating):
            raise ValueError(f"{name} Series must contain floating-point values, got {series.dtype}.")
        if series.isna().any():
            raise ValueError(f"{name} Series contains NaN values.")
    if not (len(high) == len(low) == len(open) == len(close)):
        raise ValueError(f"All Series must have the same length, got {len(high)}, {len(low)}, {len(open)}, {len(close)}.")
    if window_size <= 0 or window_size > len(high):
        raise ValueError(f"window_size must be > 0 and <= Series length ({len(high)}).")
    if (high <= 0).any() or (low <= 0).any() or (open <= 0).any() or (close <= 0).any():
        raise ValueError("High, low, open, and close prices must be positive.")

    high_array = high.to_numpy(dtype=np.float64)
    low_array = low.to_numpy(dtype=np.float64)
    open_array = open.to_numpy(dtype=np.float64)
    close_array = close.to_numpy(dtype=np.float64)
    vol_array = _yang_zhang_estimator(high_array, low_array, open_array, close_array, window_size, k)
    return pd.Series(vol_array, name=f"vol_yang_zhang_{window_size}", index=high.index)
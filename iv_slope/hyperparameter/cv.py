import pandas as pd
import numpy as np
import itertools
import json
import numbers
from abc import abstractmethod
import duckdb
from datetime import datetime, date
import re
import time
import math
from typing import Dict, List, Tuple, Iterable, Any, Optional
import os
import sys
import traceback

from trade_manager import TradeManager
from backtester import backtest, parse_table_name, calculate_performance

class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.

    Reference: https://github.com/sam31415/timeseriescv/blob/master/timeseriescv/cross_validation.py
    """
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)} was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')
        if not pred_times.equals(pred_times.sort_values()):
            raise ValueError('pred_times should be sorted')
        if not eval_times.equals(eval_times.sort_values()):
            raise ValueError('eval_times should be sorted')
        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])

# Purged Walk-Forward Cross-Validation
class PurgedWalkForwardCV(BaseTimeSeriesCrossValidator):
    """
    Purged walk-forward cross-validation as described in Advances in Financial Machine Learning.
    """
    def __init__(self, n_splits=10, n_test_splits=1, min_train_splits=2, max_train_splits=None):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type {type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits >= self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits
        if not isinstance(min_train_splits, numbers.Integral):
            raise ValueError(f"The minimal number of train folds must be of Integral type. {min_train_splits} of type {type(min_train_splits)} was passed.")
        min_train_splits = int(min_train_splits)
        if min_train_splits <= 0 or min_train_splits >= self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting min_train_splits between 1 and n_splits - n_test_splits, got min_train_splits = {min_train_splits}.")
        self.min_train_splits = min_train_splits
        if max_train_splits is None:
            max_train_splits = self.n_splits - self.n_test_splits
        if not isinstance(max_train_splits, numbers.Integral):
            raise ValueError(f"The maximal number of train folds must be of Integral type. {max_train_splits} of type {type(max_train_splits)} was passed.")
        max_train_splits = int(max_train_splits)
        if max_train_splits <= 0 or max_train_splits > self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting max_train_split between 1 and n_splits - n_test_splits, got max_train_split = {max_train_splits}.")
        self.max_train_splits = max_train_splits
        self.fold_bounds = []

    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None, eval_times: pd.Series = None, split_by_time: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        super().split(X, y, pred_times, eval_times)
        self.fold_bounds = compute_fold_bounds(self, split_by_time)
        count_folds = 0
        for fold_bound in self.fold_bounds:
            if count_folds < self.min_train_splits:
                count_folds = count_folds + 1
                continue
            if self.n_splits - count_folds < self.n_test_splits:
                break
            test_indices = self.compute_test_set(fold_bound, count_folds)
            train_indices = self.compute_train_set(fold_bound, count_folds)
            count_folds = count_folds + 1
            yield train_indices, test_indices

    def compute_train_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        if count_folds > self.max_train_splits:
            start_train = self.fold_bounds[count_folds - self.max_train_splits]
        else:
            start_train = 0
        train_indices = np.arange(start_train, fold_bound)
        train_indices = purge(self, train_indices, fold_bound, self.indices[-1])
        return train_indices

    def compute_test_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        if self.n_splits - count_folds > self.n_test_splits:
            end_test = self.fold_bounds[count_folds + self.n_test_splits]
        else:
            end_test = self.indices[-1] + 1
        return np.arange(fold_bound, end_test)

# Combinatorial Purged K-Fold Cross-Validation
class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation.
    """
    def __init__(self, n_splits=10, n_test_splits=2, embargo_td=pd.Timedelta(minutes=0)):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type {type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits
        if not isinstance(embargo_td, pd.Timedelta):
            raise ValueError(f"The embargo time should be of type Pandas Timedelta. {embargo_td} of type {type(embargo_td)} was passed.")
        if embargo_td < pd.Timedelta(minutes=0):
            raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None, eval_times: pd.Series = None, split_by_time: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        super().split(X, y, pred_times, eval_times)
        fold_bounds = compute_fold_bounds(self, split_by_time)
        # Create fold boundary tuples (start, end) ensuring valid indices
        fold_bound_pairs = []
        for i in range(len(fold_bounds)):
            start = self.indices[fold_bounds[i]] if i < len(fold_bounds) else self.indices[-1]
            end_idx = fold_bounds[i + 1] if i + 1 < len(fold_bounds) else len(self.indices)
            end = self.indices[end_idx - 1] + 1 if end_idx <= len(self.indices) else self.indices[-1] + 1
            fold_bound_pairs.append((start, end))
        selected_fold_bounds = list(itertools.combinations(fold_bound_pairs, self.n_test_splits))
        selected_fold_bounds.reverse()
        for fold_bound_list in selected_fold_bounds:
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)
            yield train_indices, test_indices

    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        train_indices = np.setdiff1d(self.indices, test_indices)
        for test_fold_start, test_fold_end in test_fold_bounds:
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices

def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]

def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray, test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
    if not hasattr(cv, 'embargo_td'):
        raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo time.")
    last_test_eval_time = cv.eval_times.iloc[test_indices[test_indices <= test_fold_end]].max()
    min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices

def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray, test_fold_start: int, test_fold_end: int) -> np.ndarray:
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))


def backtest_cv(legs: Dict, hyperparameter_grid: Dict, duckdb_conn: duckdb.DuckDBPyConnection, dates: pd.Series, cv_type: str = 'purged_walk_forward', n_splits: int = 10, in_sample_ratio: float = 0.15, out_sample_ratio: float = 0.05, embargo_td: pd.Timedelta = pd.Timedelta(minutes=0)) -> Tuple[Dict, float, List[Dict]]:
    """
    Perform backtesting using PurgedWalkForwardCV or CombPurgedKFoldCV.

    Parameters
    ----------
    legs : dict
        Dictionary defining the trading legs.
    hyperparameter_grid : dict
        Dictionary containing lists of possible values for each hyperparameter.
    duckdb_conn : duckdb.DuckDBPyConnection
        Connection to the DuckDB database.
    dates : pd.Series
        Series of dates (as strings in 'YYYY-MM-DD' format).
    cv_type : str, default='purged_walk_forward'
        Type of cross-validation ('purged_walk_forward' or 'combinatorial_purged').
    n_splits : int, default=10
        Number of folds for cross-validation.
    in_sample_ratio : float, default=0.15
        Proportion of data for in-sample training.
    out_sample_ratio : float, default=0.05
        Proportion of data for out-of-sample testing.
    embargo_td : pd.Timedelta, default=pd.Timedelta(minutes=0)
        Embargo period for combinatorial cross-validation.

    Returns
    -------
    best_params : Dict
        Best hyperparameter combination based on out-of-sample Sharpe ratio.
    avg_performance : float
        Average out-of-sample Sharpe ratio for the best parameters.
    cv_results : List[Dict]
        Results for each fold and hyperparameter combination.
    """
    total_ratio = in_sample_ratio + out_sample_ratio
    if total_ratio >= 1.0:
        raise ValueError("Sum of in_sample_ratio and out_sample_ratio must be less than 1.0")
    n_test_splits = max(1, int(n_splits * out_sample_ratio))
    min_train_splits = max(1, int(n_splits * in_sample_ratio))
    if cv_type == 'purged_walk_forward':
        cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            min_train_splits=min_train_splits,
            max_train_splits=None
        )
    elif cv_type == 'combinatorial_purged':
        cv = CombPurgedKFoldCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            embargo_td=embargo_td
        )
    else:
        raise ValueError("cv_type must be 'purged_walk_forward' or 'combinatorial_purged'")

    pred_times = pd.Series(pd.to_datetime(dates), index=dates.index)
    eval_times = pred_times + pd.Timedelta(days=1)
    X = pd.DataFrame(index=dates.index)
    y = pd.Series(0, index=dates.index)

    keys, values = zip(*hyperparameter_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    folds = list(cv.split(X=X, y=y, pred_times=pred_times, eval_times=eval_times, split_by_time=True))
    cv_results = []
    param_performance = {tuple(params.items()): [] for params in param_combinations}

    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        train_dates = dates.iloc[train_indices]
        test_dates = dates.iloc[test_indices]
        for params in param_combinations:
            trader = TradeManager()
            sharpe_ratio = backtest(
                legs=legs,
                iv_slope_thresolds=params,
                duckdb=duckdb_conn,
                trader=trader,
                dates=test_dates
            )
            result = {
                'fold': fold_idx,
                'params': params,
                'train_dates': train_dates.tolist(),
                'test_dates': test_dates.tolist(),
                'sharpe_ratio': sharpe_ratio
            }
            cv_results.append(result)
            param_performance[tuple(params.items())].append(sharpe_ratio)

    avg_performances = {
        params: np.mean(sharpes) if sharpes else 0.0
        for params, sharpes in param_performance.items()
    }
    best_params_tuple = max(avg_performances, key=avg_performances.get, default=None)
    best_params = dict(best_params_tuple) if best_params_tuple else {}
    avg_performance = avg_performances.get(best_params_tuple, 0.0)

    with open('cv_results.json', 'w') as f:
        json.dump(cv_results, f, default=str)

    return best_params, avg_performance, cv_results


"""
Example:

if __name__ == "__main__":
    db_path = r"C:\Users\Administrator\Desktop\Divyanshu desiquant\iv_slope\nifty_1min_desiquant.duckdb"
    conn = None
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file '{db_path}' not found in {os.getcwd()}")
        conn = duckdb.connect(db_path)
        
        table_names = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchdf()
        if table_names.empty:
            raise ValueError("No tables found in the database")
        
        table_names = table_names[90:703]
        pattern = re.compile(r'nifty_\d{4}_\d{2}_\d{2}')
        table_names = table_names[table_names['table_name'].str.match(pattern)]
        if table_names.empty:
            raise ValueError("No dates available in the specified range [90:703]")
        
        # Wrapper function to parse table names correctly without changing parse_table_name
        def parse_table_name_wrapper(table_name: str):
            if not isinstance(table_name, str):
                print(f"Invalid table name: {table_name} (type: {type(table_name)})")
                return None
            if re.match(r'\d{4}-\d{2}-\d{2}', table_name):
                try:
                    return datetime.strptime(table_name, "%Y-%m-%d").date()
                except ValueError as e:
                    print(f"Error parsing date from {table_name}: {e}")
                    return None
            match = re.match(r'nifty_(\d{4})_(\d{2})_(\d{2})', table_name)
            if match:
                year, month, day = match.groups()
                try:
                    year, month, day = int(year), int(month), int(day)
                    if not (1 <= month <= 12) or not (1 <= day <= 31):
                        print(f"Invalid date components in {table_name}")
                        return None
                    print(f"Parsed {table_name}: year={year}, month={month}, day={day}")
                    return date(year, month, day)  # Use date class directly
                except ValueError as e:
                    print(f"Error parsing date from {table_name}: {e}")
                    return None
            print(f"Cannot parse table name: {table_name}")
            return None
        
        dates = pd.Series([parse_table_name(name) for name in table_names['table_name']], index=table_names.index)
        dates = dates.dropna()
        if dates.empty:
            raise ValueError("No valid dates parsed from table names")
        dates = dates.apply(lambda x: x.strftime('%Y-%m-%d'))
        
        legs = {
            'leg1': {'type': 'CE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg2': {'type': 'PE', 'expiry_type': 'weekly', 'expiry_range': [12, 20], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg3': {'type': 'CE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None},
            'leg4': {'type': 'PE', 'expiry_type': 'monthly', 'expiry_range': [26, 34], 'target_strike': 'ATM', 'stop_loss': None, 'take_profit': None}
        }
        hyperparameter_grid = {
            "upper_gamma": [0.04],
            "upper_buffer": [-0.02],
            "lower_buffer": [-0.06],
            "lower_gamma": [-0.09]
        }
        
        print("Running Cross-Validation Backtest...")
        best_params, avg_performance, cv_results = backtest_cv(
            legs=legs,
            hyperparameter_grid=hyperparameter_grid,
            duckdb_conn=conn,
            dates=dates,
            cv_type='purged_walk_forward',
            n_splits=10,
            in_sample_ratio=0.6/4,
            out_sample_ratio=0.2/4
        )
        print(f"PurgedWalkForwardCV - Best parameters: {best_params}")
        print(f"PurgedWalkForwardCV - Average out-of-sample Sharpe Ratio: {avg_performance}")
        
        best_params, avg_performance, cv_results = backtest_cv(
            legs=legs,
            hyperparameter_grid=hyperparameter_grid,
            duckdb_conn=conn,
            dates=dates,
            cv_type='combinatorial_purged',
            n_splits=10,
            in_sample_ratio=0.6/4,
            out_sample_ratio=0.2/4,
            embargo_td=pd.Timedelta(minutes=30)
        )
        print(f"CombPurgedKFoldCV - Best parameters: {best_params}")
        print(f"CombPurgedKFoldCV - Average out-of-sample Sharpe Ratio: {avg_performance}")
        print("CV results saved to 'cv_results.json'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the database file exists in the current working directory or provide the correct path.")
    except PermissionError:
        print(f"Error: Permission denied accessing '{db_path}'")
        print("Please check file permissions and ensure you have read/write access.")
    except duckdb.IOException as e:
        print(f"Error: Failed to open database: {e}")
        print("The database file may be corrupted or locked. Try restoring from a backup or checking for open connections.")
    except duckdb.ConnectionException as e:
        print(f"Connection error: {e}")
        print("The database connection was closed unexpectedly. Check for connection timeouts or resource issues.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        if conn is not None:
            try:
                conn.close()
                print("Database connection closed successfully.")
            except Exception as e:
                print(f"Error closing connection: {e}")


"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, sys, pandas as pd, traceback, shutil, concurrent.futures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local quantstats folder
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # core directory
    workspace_root = os.path.dirname(current_dir)  # FnO-Synapse directory
    quantstats_path = os.path.join(current_dir, 'quantstats')  # core/quantstats directory
    
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    if os.path.exists(quantstats_path) and os.path.isdir(quantstats_path):
        # Add the core directory to sys.path so we can import core.quantstats
        import core.quantstats as quantstats
        print(f"Successfully imported local quantstats from: {quantstats_path}")
    else:
        raise ImportError(f"Quantstats folder not found at: {quantstats_path}")
        
except ImportError as e:
    quantstats = None
    print(f"Failed to import local quantstats: {e}")
    print("Attempting to import quantstats from pip installation...")
    try:
        import quantstats
        print("Successfully imported quantstats from pip installation")
    except ImportError:
        quantstats = None
        print("No quantstats installation found. Tearsheet generation will be disabled.")

# Import plotting from quantstats
try:
    if quantstats:
        from core.quantstats._plotting import core as plot
        print("Successfully imported plotting module from local quantstats")
    else:
        plot = None
        print("Plotting module not available")
except ImportError as e:
    plot = None
    print(f"Failed to import plotting module: {e}")

# Import compute_stats from local core module
from core._stats import compute_stats

# Import Backtest from local core module
from core.backtesting_opt import Backtest as BaseBacktest

class Backtest(BaseBacktest):
    """Custom Backtest class that properly handles date filtering during optimization"""
    
    def __init__(self, db_path, strategy, **kwargs):
        super().__init__(db_path, strategy, **kwargs)
        self.db_path = db_path  # Store db_path for our custom logic
        self._opt_start_date = None
        self._opt_end_date = None
    
    def optimize(self, *, start_date=None, end_date=None, **kwargs):
        """Override optimize to store date parameters and enforce them during optimization"""
        self._opt_start_date = start_date
        self._opt_end_date = end_date
        
        # Add date parameters to kwargs for the parent optimize method
        if start_date is not None:
            kwargs['start_date'] = start_date
        if end_date is not None:
            kwargs['end_date'] = end_date
            
        try:
            result = super().optimize(**kwargs)
            return result
        except Exception as e:
            raise
    
    def run(self, **kwargs):
        """Override run to use stored optimization dates if no dates provided"""
        if 'start_date' not in kwargs and self._opt_start_date is not None:
            kwargs['start_date'] = self._opt_start_date
        if 'end_date' not in kwargs and self._opt_end_date is not None:
            kwargs['end_date'] = self._opt_end_date
        
        try:
            result = super().run(**kwargs)
            return result
        except Exception as e:
            raise


from threading import Lock, Thread
import time, math
from datetime import datetime
import json
import uuid
import re
import numpy as np
import inspect
from tqdm import tqdm
import multiprocessing


class WalkForwardOptimizer:
    def __init__(self, strategy,
                 optimization_params,
                 constraint=None,
                 maximize='Equity Final [$]', * ,
                 cash: float = 10000000,
                 holding: dict = {},
                 commission: float = 0.65,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 trade_start_date=None,
                 lot_size=1,
                 fail_fast=True,
                 storage: dict | None = None,
                 is_option: bool = False,
                 look_ahead_bias: bool = False,
                 database_name: str = 'wfobacktest',
                 load = 0.6):
        stack = inspect.stack()
        caller_frame = stack[1]
        self.caller_filename = caller_frame.filename
        self.constraint = constraint
        self.optimization_params=optimization_params
        self.strategy = strategy
        self.maximize=maximize
        self.cash = cash
        self.commission = commission
        self.holding = holding
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        self.trade_start_date = trade_start_date
        self.lot_size= lot_size
        self.fail_fast = fail_fast
        self.storage = storage
        self.is_option = is_option 
        self.model_path = None
        # self.reader = DataReader()
        # self.reader.initialize(host='qdb.satyvm.com', port=443, https = True, username='2Cents', password='2Cents$1012cc')
        self.look_ahead_bias = look_ahead_bias
        self.num_processes = (int)(os.cpu_count()*load)
        self.database_name = database_name

    def get_period_dates(self, start_date, period):
        if period[-1] == 'M':
            months = int(period[:-1])
            end_date = start_date + pd.DateOffset(months=months)
        elif period[-1] == 'D':
            days = int(period[:-1])
            end_date = start_date + pd.DateOffset(days=days)
        elif period[-1] == 'H':
            hours = int(period[:-1])
            end_date = start_date + pd.DateOffset(hours=hours)
        else:
            raise ValueError("Unsupported period format, use 'M' for months, 'D' for days, or 'H' for hours")
        return end_date
    
    def serialize_optimization_params(self,params):
        result = {}
        for k, v in params.items():
            if isinstance(v, range):
                result[k] = list(v)  # Convert range to list
            else:
                result[k] = v
        return result

    def get_class_source_paths(self,obj, allowed_prefixes=("Custom", "risk", "trade","level", "weighted", "alpha")):
        class_paths = {}
        for attr_name in dir(obj):
            if not attr_name.startswith(allowed_prefixes):
                continue
            attr = getattr(obj, attr_name)
            if hasattr(attr, "__class__"):
                try:
                    class_path = inspect.getfile(attr.__class__)
                    class_paths[attr_name] = class_path
                except TypeError:
                    pass 
        return class_paths
    

    def clean_result(self,result):
        """Convert complex pandas DataFrames inside result to dicts for safe multiprocessing transfer."""
        result_clean = result.copy()
        for key in ['_equity_curve', '_trades', '_orders']:
            if key in result_clean:
                df = result_clean[key]
                if hasattr(df, 'reset_index'):
                    result_clean[key] = df.reset_index(drop=True).where(pd.notnull(df.reset_index(drop=True)), None).to_dict(orient='records')
        return result_clean

    # For Progress Tracking
    def calculate_num_iterations(self, total_candles, training_candles, testing_candles):
        current_index = 0
        num_iterations = 0
        while current_index + training_candles + testing_candles - 1 < total_candles:
            num_iterations += 1
            current_index += testing_candles
        return num_iterations
    
    @staticmethod
    def progress_updater(queue: multiprocessing.Queue, progress_bars: dict[str, tqdm]):
        while True:
            message = queue.get()
            if message == 'DONE':
                break
            stock, increment = message
            if stock in progress_bars:
                progress_bars[stock].update(increment)

    def compare_at_ts(self,df_full: pd.DataFrame,
                    df_cut:  pd.DataFrame,
                    ts: pd.Timestamp,
                    tol: float = 1e-8
                    ) -> dict:
        """
        Compare the single row at timestamp ts in df_full vs df_cut
        Returns a dict { column_name: (full_value, cut_value), ... }
        If empty, they match.
        """
        row_f = df_full.loc[ts]
        row_c = df_cut .loc[ts]

        diffs = {}
        for col in row_f.index:
            a, b = row_f[col], row_c[col]
            # both NaN → ignore
            if pd.isna(a) and pd.isna(b):
                continue
            # numeric tolerance
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if abs(a - b) > tol:
                    diffs[col] = (a, b)
            else:
                if a != b:
                    diffs[col] = (a, b)
        return diffs

    def generate_report(self,summary: dict) -> str:
        """
        Turn the summary of check_lookahead_bias(...) into a report.
        """
        lines = []
        if not summary["has_bias"]:
            lines.append(f"No lookahead bias detected on {summary['total_checked']} bars.")
            return "\n".join(lines)

        lines.append(f"Lookahead bias detected on {len(summary['biases'])} of "
                    f"{summary['total_checked']} checked bars.")
        for b in summary["biases"]:
            ts = b["timestamp"]
            ck = b["Signal"]
            lines.append(f" • {ts} (Signal={ck}):")
            for col, (v_full, v_cut) in b["diffs"].items():
                lines.append(f"     {col}: full={v_full}  cut={v_cut}")
        return "\n".join(lines)

    def partial_backtest(self, data: pd.DataFrame, bt_kwargs: dict):
        bt = Backtest(data, self.strategy, **bt_kwargs)
        result = bt.run() # Pass the keyword arguments additionally passed

        # # Get complete DataFrame with all indicators
        strategy_instance = bt._results._strategy
        full_data = strategy_instance.data.df.copy()
        # Add all indicators to the DataFrame
        for name, ind in strategy_instance.indicator_attrs_np.items():
            # print(f"Adding indicator {name}:")
            # print(ind)
            full_data[name] = ind
        
        # Initialize check column with zeros
        full_data['check'] = 0
        
        # Get trades from results
        trades_df = result['_trades']
        
        # Mark entry and exit points with 1 based on conditions
        len_df = 0
        for _, trade in trades_df.iterrows():
            entry_time = trade['EntryTime']
            exit_time = trade['ExitTime']
            size = trade['Size']  # Use absolute value since size can be negative for short positions
            exit_reason = trade['Reason']  # Using get() in case ExitReason doesn't exist
            if exit_reason is None:
                len_df += 1

            # Mark entry bar
            if entry_time in full_data.index:
                if size > 0:
                    full_data.loc[entry_time, 'check'] = 1
                else:
                    full_data.loc[entry_time, 'check'] = -1
                
            # Mark exit bar 
            if exit_time in full_data.index and exit_reason is None:
                if size > 0:
                    full_data.loc[exit_time, 'check'] = 1
                else:
                    full_data.loc[exit_time, 'check'] = -1

        # print(f"Full data: {full_data}")
        # sleep(10)
        return full_data

    def look_ahead_bias_check(self, result, strategy_instance):

        print("Checking look-ahead bias.....")

        full_data = strategy_instance.data.df.copy()
        # Add all indicators to the DataFrame
        for name, ind in strategy_instance.indicator_attrs_np.items():
            # print(f"Adding indicator {name}:")
            # print(ind)
            full_data[name] = ind
        
        # Initialize check column with zeros
        full_data['check'] = 0
        
        # Get trades from results
        trades_df = result['_trades']
        
        # Mark entry and exit points with 1 based on conditions
        len_df = 0
        for _, trade in trades_df.iterrows():
            entry_time = trade['EntryTime']
            exit_time = trade['ExitTime']
            size = trade['Size']  # Use absolute value since size can be negative for short positions
            exit_reason = trade['Reason']  # Using get() in case ExitReason doesn't exist
            if exit_reason is None:
                len_df += 1

            # Mark entry bar
            if entry_time in full_data.index:
                if size > 0:
                    full_data.loc[entry_time, 'check'] = 1
                else:
                    full_data.loc[entry_time, 'check'] = -1
                
            # Mark exit bar 
            if exit_time in full_data.index and exit_reason is None:
                if size > 0:
                    full_data.loc[exit_time, 'check'] = 1
                else:
                    full_data.loc[exit_time, 'check'] = -1

        # ─── now invoke look-ahead check on _every_ trade ─────────
        try:
            to_check = full_data.index[ full_data["check"].ne(0) ]
            raw = strategy_instance.data.df.copy()
            bt_kwargs = dict(
                cash=self.cash,
                commission=self.commission,
                holding=self.holding,
                margin=self.margin,
                trade_on_close=self.trade_on_close,
                hedging=self.hedging,
                exclusive_orders=self.exclusive_orders,
                trade_start_date=self.trade_start_date,
                lot_size=self.lot_size,
                fail_fast=self.fail_fast,
                storage=self.storage,
                is_option=self.is_option
                )
            biases = []
            # print(f"Running look-ahead bias check on marked bars. {to_check}")
            for ts in to_check:
                cut_raw = raw.loc[:ts].copy()
                # print(f"Cut raw: {cut_raw}")
                # sleep(10)
                cut_df  = self.partial_backtest(cut_raw, bt_kwargs)
                # print(f"Cut df: {cut_df}")
                diffs   = self.compare_at_ts(full_data, cut_df, ts)
                # print(f"Differences: {diffs}")
                if diffs:
                    biases.append({
                        "timestamp": ts,
                        "Signal":     int(full_data.at[ts, "check"]),
                        "diffs":     diffs
                    })
                    break

            report = {
                "has_bias":     bool(biases),
                "total_checked": len(to_check),
                "biases":        biases
            }
            print(self.generate_report(report))
            # sleep(10)
        except Exception as e:
            print("Look-ahead bias check raised:", e)
            # sleep(10)

    def optimize(self, strategy_class, timeframe, stock, optimization_params, db_path, training_candles, testing_candles, method='grid',
                start_date=None, end_date=None, pbar=None, queue=None):
        """
        Walk-forward optimization using database path approach
        
        Parameters:
        -----------
        db_path : str
            Path to the DuckDB database file
        training_candles : int
            Number of candles/days for training period
        testing_candles : int  
            Number of candles/days for testing period
        start_date : str or datetime, optional
            Overall start date for optimization
        end_date : str or datetime, optional
            Overall end date for optimization
        """
        
        try:
            # Get data range efficiently without loading all data
            from core.backtesting_opt import _Data
            data_obj = _Data(db_path, start_date, end_date)
            
            tables = data_obj.get_tables_in_date_range()
            
            if not tables:
                return None, None, None, None, None, None
            
            # Get start and end dates by loading only first and last tables
            first_df = data_obj.load_table(tables[0])
            last_df = data_obj.load_table(tables[-1])
            
            if first_df.empty or last_df.empty:
                data_obj.close()
                return None, None, None, None, None, None
            
            data_start_date = first_df.index.min()
            data_end_date = last_df.index.max()
            data_obj.close()
            
            # Calculate timeframe in minutes and estimate total candles
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            
            total_minutes = (data_end_date - data_start_date).total_seconds() / 60
            estimated_total_candles = int(total_minutes / timeframe_minutes)
            
            # Check if we have enough data
            if training_candles + testing_candles > estimated_total_candles:
                return None, None, None, None, None, None
            
            # Calculate walk-forward periods using trading-aware time calculations
            if data_start_date.hour < 9 or (data_start_date.hour == 9 and data_start_date.minute < 15):
                train_start_date = data_start_date.replace(hour=9, minute=15)
            elif data_start_date.hour >= 15 and data_start_date.minute > 30:
                next_day = data_start_date.date() + pd.Timedelta(days=1)
                while next_day.weekday() >= 5:  # Skip weekends
                    next_day += pd.Timedelta(days=1)
                train_start_date = pd.Timestamp(next_day).replace(hour=9, minute=15)
            else:
                train_start_date = data_start_date
            
            # Calculate training end date using trading minutes
            train_end_date = self._add_trading_minutes(train_start_date, training_candles * timeframe_minutes)
            test_end_date = self._add_trading_minutes(train_start_date, (training_candles + testing_candles) * timeframe_minutes)

            results_list = []
            trades_list = []
            equity_curves = []
            orders_list = []
            last_strategy = None
            iteration_count = 0
            
            while test_end_date <= data_end_date:
                iteration_count += 1
                
                try:
                    # Optimize on training period
                    bt_train = Backtest(db_path, strategy_class,
                                       cash=self.cash,
                                       commission_per_contract=self.commission,
                                       option_multiplier=75)
                    
                    # Run optimization on training period
                    train_start_str = train_start_date.strftime('%Y-%m-%d %H:%M:%S')
                    train_end_str = train_end_date.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if not optimization_params:
                        break
                    
                    if self.constraint is None:
                        train_output = bt_train.optimize(
                            start_date=train_start_str, 
                            end_date=train_end_str,
                            maximize=self.maximize, 
                            method=method,
                            **optimization_params
                        )
                    else:    
                        train_output = bt_train.optimize(
                            start_date=train_start_str, 
                            end_date=train_end_str,
                            maximize=self.maximize, 
                            constraint=self.constraint, 
                            method=method,
                            **optimization_params
                        )
                    
                    if train_output is None:
                        break
                    
                    # Extract strategy from results
                    last_strategy = train_output.get('_strategy')
                    if last_strategy is None:
                        break
                        
                    # Extract optimized parameters from the strategy instance
                    best_params = {param: value for param, value in last_strategy._params.items() 
                                  if param in optimization_params}
                    
                except Exception as train_e:
                    break

                try:
                    # Create optimized strategy classes with best parameters
                    class TrainOptimizedStrategy(strategy_class):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            for param, value in best_params.items():
                                setattr(self, param, value)
                                
                        def init(self):
                            super().init()
                            
                    class TestOptimizedStrategy(strategy_class):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            for param, value in best_params.items():
                                setattr(self, param, value)
                                
                        def init(self):
                            super().init()
                            
                        def next(self):
                            # Only trade during testing period (after training end date)
                            # Use self.time property which accesses broker time
                            if self.time >= pd.Timestamp(train_end_date.date()):
                                super().next()
                    
                    # Run training backtest with optimized parameters
                    train_bt = Backtest(db_path, TrainOptimizedStrategy,
                                       cash=self.cash,
                                       commission_per_contract=self.commission,
                                       option_multiplier=75)
                    train_stats = train_bt.run(start_date=train_start_date.strftime('%Y-%m-%d %H:%M:%S'), 
                                             end_date=train_end_date.strftime('%Y-%m-%d %H:%M:%S'))

                    # Run testing backtest with optimized parameters (from training start to test end for indicators)
                    test_bt = Backtest(db_path, TestOptimizedStrategy,
                                      cash=self.cash,
                                      commission_per_contract=self.commission,
                                      option_multiplier=75)
                    test_stats = test_bt.run(start_date=train_start_date.strftime('%Y-%m-%d %H:%M:%S'), 
                                           end_date=test_end_date.strftime('%Y-%m-%d %H:%M:%S'))
                    
                except Exception as bt_e:
                    break
                
                try:
                    # Process results
                    trades = test_stats.get('_trades', pd.DataFrame())
                    if not trades.empty:
                        # Filter trades to testing period only
                        trades = trades[trades['EntryTime'] >= train_end_date].copy()
                        if not trades.empty:
                            trades['Period'] = f"{train_end_date} to {test_end_date}"
                            trades_list.append(trades)
                    
                    orders = test_stats.get('_orders', pd.DataFrame())
                    if not orders.empty:
                        # Filter orders to testing period only
                        orders = orders[orders.index >= train_end_date]
                        if not orders.empty:
                            orders_list.append(orders)
                    
                    equity_curve = test_stats.get('_equity_curve')
                    if equity_curve is not None and isinstance(equity_curve, (pd.Series, pd.DataFrame)) and not equity_curve.empty:
                        # Filter equity curve to testing period only
                        equity_curve = equity_curve[equity_curve.index >= train_end_date]
                        if not equity_curve.empty:
                            equity_pct_change = equity_curve.pct_change().dropna()
                            if isinstance(equity_pct_change, (pd.Series, pd.DataFrame)) and not equity_pct_change.empty:
                                equity_curves.append(equity_pct_change)

                    # Create summary statistics
                    train_summary_stats = dict(train_stats)
                    train_summary_stats.update({
                        "Start Date": train_start_date,
                        "End Date": train_end_date,
                        "Type": "Training",
                    })
                    train_summary_stats.update(best_params)

                    test_summary_stats = dict(test_stats)
                    test_summary_stats.update({
                        "Start Date": train_end_date,
                        "End Date": test_end_date,
                        "Type": "Testing",
                    })
                    test_summary_stats.update(best_params)

                    results_list.append(train_summary_stats)
                    results_list.append(test_summary_stats)

                except Exception as proc_e:
                    # Continue with next iteration instead of breaking
                    pass
                    
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                if queue is not None:
                    queue.put((stock, 1))

                # Move to next iteration using trading-aware time advancement
                try:
                    train_start_date = self._add_trading_minutes(train_start_date, testing_candles * timeframe_minutes)
                    train_end_date = self._add_trading_minutes(train_start_date, training_candles * timeframe_minutes)
                    test_end_date = self._add_trading_minutes(train_start_date, (training_candles + testing_candles) * timeframe_minutes)
                    
                    # Break if we've moved beyond the available data
                    if train_end_date > data_end_date:
                        break
                        
                except Exception as time_e:
                    break

        except Exception as main_e:
            return None, None, None, None, None, None

        try:
            # Store final results in database
            if results_list and test_summary_stats:
                result_temp = test_summary_stats.copy()
                result_temp['stock_name'] = stock
                result_temp['time_frame'] = timeframe
                result_temp = self.clean_result(result_temp)
                result_temp['rebalance_period'] = training_candles
                result_temp['maximize'] = self.maximize
                result_temp['optimization_params'] = json.dumps({
                    k: list(v) if isinstance(v, range) else v
                    for k, v in optimization_params.items()
                })
                if self.constraint is not None:
                    constraint_code = inspect.getsource(self.constraint)
                    result_temp['constraint'] = constraint_code

            # Prepare output DataFrames
            df_orders = pd.concat(orders_list, ignore_index=True) if orders_list else pd.DataFrame()
            
            df_results = pd.DataFrame(results_list)
            if not df_results.empty:
                df_results = df_results[['Start Date', 'End Date', 'Type'] + 
                                       [col for col in df_results.columns if col not in ['Start Date', 'End Date', 'Type']]]
                if 'Return [%]' in df_results.columns:
                    df_results.loc[df_results['Type'] == 'Testing', 'Cumulative Return [%]'] = \
                        df_results.loc[df_results['Type'] == 'Testing', 'Return [%]'].cumsum()
            
            df_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()

            # Create df_equity
            if equity_curves:
                df_equity = pd.concat(equity_curves)
            else:
                df_equity = pd.Series(dtype=float)
            
            # Compute final statistics using data date range
            stats = None
            if len(df_trades) > 0:
                actual_equity_curve = (df_equity + 1).cumprod() * self.cash
                # Create a simple date index for equity curve alignment
                date_index = pd.date_range(start=data_start_date, end=data_end_date, freq='D')
                if not actual_equity_curve.index.equals(date_index):
                    actual_equity_curve = actual_equity_curve.reindex(date_index).ffill().bfill()
                
                # Convert to DataFrame if it's a Series, otherwise keep as is
                if isinstance(actual_equity_curve, pd.Series):
                    actual_equity_curve = actual_equity_curve.to_frame(name='Equity')
                elif isinstance(actual_equity_curve, pd.DataFrame):
                    # If it's already a DataFrame, ensure it has the right column name
                    if actual_equity_curve.shape[1] == 1:
                        actual_equity_curve.columns = ['Equity']
                    else:
                        # If multiple columns, take the first one and rename it
                        actual_equity_curve = actual_equity_curve.iloc[:, 0].to_frame(name='Equity')
                
                # For stats computation, we'll use a simplified approach since we don't have OHLC data
                stats = {
                    'Start': data_start_date,
                    'End': data_end_date,
                    'Duration': str(data_end_date - data_start_date),
                    'Return [%]': ((actual_equity_curve['Equity'].iloc[-1] / self.cash) - 1) * 100 if len(actual_equity_curve) > 0 else 0,
                    'Max Drawdown [%]': ((actual_equity_curve['Equity'].min() / actual_equity_curve['Equity'].cummax().max()) - 1) * 100 if len(actual_equity_curve) > 0 else 0,
                    '# Trades': len(df_trades),
                    '_equity_curve': actual_equity_curve,
                    '_trades': df_trades,
                    '_orders': df_orders,
                    '_strategy': last_strategy
                }
                
            # Extract best parameters from the last strategy
            if last_strategy:
                best_params = {param: value for param, value in last_strategy._params.items() 
                              if param in optimization_params}
            else:
                best_params = {}

            return stats, df_results, df_trades, df_equity, best_params, last_strategy
            
        except Exception as final_e:
            return None, None, None, None, None, None

    def remove_timezone(self, df):
        for col in df.select_dtypes(include=['datetimetz']).columns:
            df[col] = df[col].dt.tz_localize(None)
        return df
    
    def get_root_dir(self):
        return os.getcwd()
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert timeframe string to minutes"""
        timeframe = timeframe.lower()
        if 'min' in timeframe:
            return int(timeframe.replace('min', ''))
        elif 'h' in timeframe or 'hour' in timeframe:
            hours = int(timeframe.replace('h', '').replace('hour', ''))
            return hours * 60
        elif 'd' in timeframe or 'day' in timeframe:
            return 375  # trading day minutes
        elif 'w' in timeframe or 'week' in timeframe:
            return 375 * 5  # trading week minutes
    
    def _add_trading_minutes(self, start_time, minutes_to_add, market_start_time="09:15", market_end_time="15:30"):
        """
        Add trading minutes to a timestamp, respecting market hours
        
        Parameters:
        -----------
        start_time : pd.Timestamp
            Starting timestamp
        minutes_to_add : int
            Number of trading minutes to add
        market_start_time : str
            Market opening time (default: "09:15" for Indian markets)
        market_end_time : str
            Market closing time (default: "15:30" for Indian markets)
        """
        current_time = start_time
        remaining_minutes = minutes_to_add
        
        # Parse market hours
        market_start_hour, market_start_min = map(int, market_start_time.split(':'))
        market_end_hour, market_end_min = map(int, market_end_time.split(':'))
        
        # Calculate minutes per trading day
        market_minutes_per_day = (market_end_hour * 60 + market_end_min) - (market_start_hour * 60 + market_start_min)
        
        while remaining_minutes > 0:
            # Get market start and end for current day
            current_date = current_time.date()
            market_start = pd.Timestamp(current_date).replace(hour=market_start_hour, minute=market_start_min)
            market_end = pd.Timestamp(current_date).replace(hour=market_end_hour, minute=market_end_min)
            
            # If current time is before market open, move to market open
            if current_time < market_start:
                current_time = market_start
            
            # If current time is after market close, move to next trading day
            if current_time >= market_end:
                # Move to next day's market open
                next_day = current_date + pd.Timedelta(days=1)
                # Skip weekends (Saturday=5, Sunday=6)
                while next_day.weekday() >= 5:
                    next_day += pd.Timedelta(days=1)
                current_time = pd.Timestamp(next_day).replace(hour=market_start_hour, minute=market_start_min)
                continue
            
            # Calculate how many minutes we can add in current trading session
            minutes_until_market_close = (market_end - current_time).total_seconds() / 60
            minutes_to_add_today = min(remaining_minutes, minutes_until_market_close)
            
            # Add the minutes
            current_time += pd.Timedelta(minutes=minutes_to_add_today)
            remaining_minutes -= minutes_to_add_today
            
            # If we've used up the trading day, move to next trading day
            if remaining_minutes > 0 and current_time >= market_end:
                next_day = current_date + pd.Timedelta(days=1)
                # Skip weekends
                while next_day.weekday() >= 5:
                    next_day += pd.Timedelta(days=1)
                current_time = pd.Timestamp(next_day).replace(hour=market_start_hour, minute=market_start_min)
        
        return current_time
    
    def calculate_wfa_metrics(self, df_results):
        """Calculate Walk Forward Analysis specific metrics"""
        wfa_metrics = {
            'Walk Forward Efficiency': (
                df_results[df_results['Type'] == 'Testing']['Return [%]'].mean() /
                df_results[df_results['Type'] == 'Training']['Return [%]'].mean()
            ),
            'Consistency Score': (
                (df_results[df_results['Type'] == 'Testing']['Return [%]'] > 0).mean()
            ),
            'Parameter Stability': self._calculate_parameter_stability(df_results),
            'Robustness Score': self._calculate_robustness_score(df_results)
        }
        return wfa_metrics
    def _calculate_parameter_stability(self, df_results):
        """Calculate stability of optimized parameters across walks"""
        param_stability = {}
        training_results = df_results[df_results['Type'] == 'Training']

        for param in self.optimization_params.keys():
            values = training_results[param]
            param_stability[param] = {
                'std': values.std(),
                'cv': values.std() / values.mean() if values.mean() != 0 else float('inf'),
                'range': values.max() - values.min()
            }
        return param_stability
    
    def monte_carlo_validation(self, strategy_class, data, num_iterations=10):
        """Perform Monte Carlo cross-validation"""
        mc_results = []
        for i in range(num_iterations):
            # Randomly select training/testing periods
            split_point = np.random.randint(
                len(data) * 0.4, 
                len(data) * 0.6
            )
            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]

            # Run optimization on random split
            stats, _, _, _, params, _, _, _ = self.optimize(
                strategy_class, 
                "MC_" + str(i), 
                self.optimization_params,
                train_data,
                len(train_data),
                len(test_data)
            )
            mc_results.append({
                'iteration': i,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_return': stats.get('Return [%]', 0),
                'parameters': params
            })
        return pd.DataFrame(mc_results)
    
    def optimize_stock(self, stock, timeframe, db_path, exchange=None, training_candles=2000, testing_candles=200, 
                      start_date=None, end_date=None):
        '''
        Function to process a single stock using db_path instead of fetching data
        
        Parameters:
        -----------
        stock : str
            Stock symbol to optimize
        timeframe : str
            Timeframe for the data (e.g., '1min', '5min', '1day')
        db_path : str
            Path to the DuckDB database file
        exchange : str, optional
            Exchange identifier
        training_candles : int, optional
            Number of candles for training period (default: 2000)
        testing_candles : int, optional
            Number of candles for testing period (default: 200)
        start_date : str or datetime, optional
            Start date for backtesting (e.g., '2023-01-01' or datetime object)
            If provided, overrides training_candles logic
        end_date : str or datetime, optional
            End date for backtesting (e.g., '2023-12-31' or datetime object)
            If provided, limits the backtesting period
        '''
        stock = stock.upper()
        
        try:
            # Validate date parameters
            if start_date is not None and end_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                if start_date >= end_date:
                    raise ValueError("start_date must be before end_date")
                
                print(f"Using date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            elif start_date is not None or end_date is not None:
                print("Warning: Both start_date and end_date should be provided for date-based backtesting")
            else:
                print(f"Using candle-based approach: {training_candles} training, {testing_candles} testing candles")
            
            # Set model path
            for attr in dir(self.strategy):
                if not attr.startswith("__"):
                    if attr == 'train_percentage':
                        print("Found train_percentage")
                        base, _ = os.path.splitext(self.caller_filename)
                        self.model_path = base + ".pkl"
       
            # Now perform the optimization and generate the tearsheet
            strategy_class = self.strategy
            optimization_params = self.optimization_params
            strategy_metrics = []

            # Calculate expected iterations based on timeframe and date range (much faster)
            try:
                # Convert timeframe to minutes for calculation
                timeframe_minutes = self._timeframe_to_minutes(timeframe)
                
                if start_date and end_date:
                    # Use provided date range
                    start_dt = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
                    end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
                    
                    # Calculate total minutes in the date range (approximate, assuming trading hours)
                    total_days = (end_dt - start_dt).days
                    # Assume 6.25 hours of trading per day (375 minutes for equity markets)
                    approx_total_candles = total_days * (375 // timeframe_minutes)
                else:
                    # Quick check to get approximate data range without loading all data
                    from core.backtesting_opt import _Data
                    data_obj = _Data(db_path, start_date, end_date)
                    tables = data_obj.get_tables_in_date_range()
                    if tables:
                        # Just load first table to get start date and last table to get end date
                        first_df = data_obj.load_table(tables[0])
                        last_df = data_obj.load_table(tables[-1])
                        if not first_df.empty and not last_df.empty:
                            start_dt = first_df.index.min()
                            end_dt = last_df.index.max()
                            total_days = (end_dt - start_dt).days
                            approx_total_candles = total_days * (375 // timeframe_minutes)
                        else:
                            approx_total_candles = 1000  # fallback
                    else:
                        approx_total_candles = 1000  # fallback
                    data_obj.close()
                
                expected_iterations = self.calculate_num_iterations(approx_total_candles, training_candles, testing_candles)
                
            except Exception as e:
                expected_iterations = 1
                
            pbar = tqdm(total=expected_iterations, desc=f"Walk-Forward Optimizing {stock}")

            try:
                stats, df_results, df_trades, df_equity, best_params, last_strategy = self.optimize(
                    strategy_class, timeframe, stock, optimization_params, db_path, training_candles,
                    testing_candles, start_date=start_date, end_date=end_date, pbar=pbar)
            except Exception as opt_e:
                pbar.close()
                return
            finally:
                pbar.close()

            # Prepare stats for output
            if stats is None:
                return
                
            # Convert best_params to JSON-serializable format
            best_params_clean = {key: int(value) if isinstance(value, np.int64) else value 
                               for key, value in best_params.items()}
            
            # Add metadata to stats
            stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
            stats_dict['best_params'] = json.dumps(best_params_clean)
            stats_dict['stock_name'] = stock 
            stats_dict['time_frame'] = timeframe
            
            # Create output directory
            output_root_directory = self.get_root_dir()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = self.strategy.__name__
            output_directory_path = os.path.join(output_root_directory, f'wfo_{strategy_name}_{stock}_{timeframe}_{timestamp}')
            self.output_directory = output_directory_path
            os.makedirs(output_directory_path, exist_ok=True)
            
            # Prepare metrics
            metrics = {'Symbol': stock, 'Timeframe': timeframe}
            metrics.update(stats_dict)
            strategy_metrics.append(metrics)
            
            # Save to Excel
            try:
                excel_path = os.path.join(output_directory_path, f'{stock}_{timeframe}.xlsx')
                with pd.ExcelWriter(excel_path) as writer:
                    self.remove_timezone(df_results).to_excel(writer, sheet_name='All Runs')
                    if not df_trades.empty:
                        self.remove_timezone(df_trades).to_excel(writer, sheet_name='Trades')
                    pd.DataFrame([stats_dict]).to_excel(writer, sheet_name='Summary')
            except Exception as excel_e:
                pass

            # Look-ahead bias check (simplified for now)
            if self.look_ahead_bias:
                # TODO: Implement look-ahead bias check for backtesting_opt
                pass

            # Generate tearsheet if trades exist
            if not df_trades.empty and len(df_trades) > 0:
                try:
                    filename = f"{stock}_{timeframe}.html"
                    tear_sheet_path = os.path.join(output_directory_path, filename)
                    
                    # Convert DataFrame to Series if needed
                    if isinstance(df_equity, pd.DataFrame):
                        df_equity = df_equity.iloc[:, 0]
                    
                    # Create benchmark
                    benchmark_series = None
                    try:
                        from core.backtesting_opt import _Data
                        data = _Data(db_path, start_date=start_date, end_date=end_date)
                        daily_benchmark_data = []
                        
                        for table in data._table_names:
                            try:
                                data.load_table(table)
                                if data._spot_table is not None and not data._spot_table.empty:
                                    daily_spot = data._spot_table['spot_price'].resample('1D').last()
                                    daily_benchmark_data.append(daily_spot)
                            except Exception as table_error:
                                print(f"Warning: Could not load table {table}: {table_error}")
                                continue
                        
                        data.close()
                        
                        if daily_benchmark_data:
                            benchmark_series = pd.concat(daily_benchmark_data).sort_index()
                            benchmark_series = benchmark_series[~benchmark_series.index.duplicated(keep='first')]
                            benchmark_series = benchmark_series.pct_change().dropna()
                            benchmark_series = benchmark_series.rename("Benchmark")
                            print(f"Created default benchmark with {len(benchmark_series)} data points")
                        else:
                            benchmark_series = None
                            print("No benchmark data available")
                    except Exception as e:
                        print(f"Warning: Could not create benchmark data: {e}")
                        benchmark_series = None
                        
                    if quantstats:
                        try:
                            quantstats.reports.html(
                                df_equity, 
                                stock_name=stock,
                                timeframe=timeframe,
                                benchmark=benchmark_series, 
                                output=tear_sheet_path,
                                strategy_title=f"{stock} Strategy",
                                benchmark_title="Benchmark"
                            )
                        except Exception as qs_e:
                            pass
                        
                except Exception as e:
                    pass
            
        except Exception as main_e:
            raise

    def create_tearsheets(self):
        for data_tuple in self.results:
            stock, tear_sheet_path, df_equity, benchmark = data_tuple
            try:
                quantstats.reports.html(df_equity, benchmark=benchmark, output=tear_sheet_path)
                print(f"Tear sheet generated and saved to {tear_sheet_path}")
            except Exception as e:
                print(f"Error while generating tearsheet for {stock} {e}")
                raise
        
    def process_stock_chunk(self, data_chunk, queue):
        results = []
        for data_tuple in data_chunk:
            stock, data, timeframe = data_tuple

            if len(data) == 0:
                print("No data available for stock",stock)
                continue
            
            print('WFO started for',stock)
            
            strategy_class = self.strategy
            optimization_params=self.optimization_params
            strategy_metrics = []
            data.index = data.index.tz_localize(None)
            raw_equity_curve = data['Close'].pct_change().dropna()
            raw_equity_curve.index = pd.to_datetime(raw_equity_curve.index)
            stats,df_results,df_trades,df_equity,best_params,last_strategy, rm_obj, tm_obj = self.optimize(strategy_class, timeframe, stock, optimization_params, data,
                                                                 self.training_candles, self.testing_candles, pbar=None, queue=queue)
            if stats is None:
                continue
            stats['_strategy'].risk_management_strategy = rm_obj
            stats['_strategy'].trade_management_strategy = tm_obj
            metrics = {'Symbol': stock, 'Timeframe': timeframe}
            metrics.update(stats)
            strategy_metrics.append(metrics)
            with pd.ExcelWriter(os.path.join(self.output_xl_sheets_path, f'{stock}_{timeframe}.xlsx')) as writer:
                self.remove_timezone(df_results).to_excel(writer, sheet_name='All Runs')
                self.remove_timezone(df_trades).to_excel(writer, sheet_name='Trades')
                pd.DataFrame(stats).to_excel(writer,sheet_name='Summary')
            
            print("WFO done for",stock)
            
            if(len(df_trades)>0):
                print("Number of trades are: ",stock, len(df_trades))
                df_trades.set_index('ExitTime', inplace=True)
                df_trades.index = pd.to_datetime(df_trades.index)
                filename=f"{stock}_{timeframe}.html"
                tear_sheet_path = os.path.join(self.output_tearsheets_path, filename)
                combined_df = pd.DataFrame({
                    'Benchmark': raw_equity_curve,
                })
                combined_df.index=raw_equity_curve.index
                try:
                    quantstats.reports.html(
                        df_equity,
                        stock_name=stock,
                        timeframe=timeframe,
                        equity_df=stats['_equity_curve'], 
                        benchmark=combined_df['Benchmark'], 
                        output=tear_sheet_path,
                        strategy_title=f"{stock} Strategy",
                        benchmark_title="Benchmark"
                    )
                    print(f"Tear sheet generated and saved to {tear_sheet_path}")
                except Exception as e:
                    print(f"Error while generating tearsheet for {stock} : {e}")
                    continue
                # return (stock, tear_sheet_path, df_equity, combined_df['Benchmark'])
                results.append((stock,tear_sheet_path,stats,  df_equity,combined_df['Benchmark']))
            else:
                print(f"Tradebook is empty for {stock}. Tearsheet not generated.")
                # return None
        return results
    def split_list(self, data, n):
        chunk_size = math.ceil(len(data) / n)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def generate_summary(self, output_directory):
        # Create a DataFrame from results
        stock_data = []
        stock_names = []

        for _ in self.results:
            print("this is start" , _)
            result_series =  _[2]
            stock_name = _[0]
            stock_data.append(result_series)
            stock_names.append(stock_name)
            # print(result_series)
        df = pd.DataFrame(stock_data, index=stock_names)
        df.index.name = "Stock"
        
        # Drop unwanted columns
        df = df.drop(columns=["_equity_curve", "_trades", "_orders"], errors="ignore")

        # Sort DataFrame by Sharpe Ratio in descending order (if column exists)
        sharpe_column = 'Sharpe Ratio'
        if sharpe_column in df.columns:
            df = df.sort_values(by=sharpe_column, ascending=False)
            print(f"Results sorted by {sharpe_column} (highest first)")
        else:
            print(f"Warning: '{sharpe_column}' column not found in results, using original order")
        
        # Compute statistical metrics for numerical columns
        metrics = {}
        for column in df.select_dtypes(include="number").columns:
            column_data = df[column].dropna()  # Exclude NaN values
            metrics[column] = {
                "Mean": column_data.mean(),
                "Median": column_data.median(),
                "Max": column_data.max(),
                "Min": column_data.min(),
                "Standard Deviation": column_data.std(),
                "Top 10 Percentile Average": column_data[column_data >= column_data.quantile(0.9)].mean(),
                "Top 25 Percentile Average": column_data[column_data >= column_data.quantile(0.75)].mean(),
                "Top 50 Percentile Average": column_data[column_data >= column_data.quantile(0.5)].mean(),
                "Top 75 Percentile Average": column_data[column_data >= column_data.quantile(0.25)].mean(),
            }
        
        metrics_df = pd.DataFrame(metrics).T  
        metrics_df.index.name = "Metric"

        # temp = MultiAssetWFOUploader()
        # temp.update_output_data(metrics_df,self.universe_name,self.timeframe,self.caller_filename, 'MULTI_ASSET_WFO',self.results)

        output_path = os.path.join(output_directory, f"summary.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Stock Results")
            metrics_df.to_excel(writer, sheet_name="Metrics")
        
        print(f"Summary sheet saved to {output_path}")



    def optimize_universe(self, universe, timeframe, exchange = None, training_candles=2000, testing_candles=200):
        # First get data for the universe
        stack = inspect.stack()
        caller_frame = stack[1]
        self.caller_filename = caller_frame.filename
        
        self.universe_name = universe 
        self.timeframe = timeframe


        self.training_candles = training_candles
        self.testing_candles = testing_candles
        try:
            data = self.reader.fetch_universe(universe, timeframe, exchange)
        except Exception as e:
            print(f"Error fetching data for stocks for {universe}: {e}")
            raise

        print("Data fetched for",universe)

        output_root_directory = self.get_root_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategy_name = self.strategy.__name__
        output_directory_path = os.path.join(output_root_directory, f'wfo_{strategy_name}_{universe}_{timeframe}_{timestamp}')
        self.output_directory = output_directory_path
        os.makedirs(output_directory_path)
        
        print("Output saved to",output_directory_path)
        
        os.makedirs(os.path.join(output_directory_path, 'output_xl_sheets'))
        os.makedirs(os.path.join(output_directory_path, 'output_tearsheets'))
        self.output_xl_sheets_path = os.path.join(output_directory_path, 'output_xl_sheets')
        self.output_tearsheets_path = os.path.join(output_directory_path, 'output_tearsheets')
        

        if self.look_ahead_bias:    # Only check for look-ahead bias if the flag is set

            test_data = self.reader.fetch_stock('USDJPY', '1day', 'forex', 'metaquotes')

            test_bt = Backtest(test_data, self.strategy,
                                cash = self.cash,
                                commission = self.commission,
                                holding = self.holding,
                                margin = self.margin,
                                trade_on_close = self.trade_on_close,
                                hedging = self.hedging,
                                exclusive_orders = self.exclusive_orders,   
                                trade_start_date = self.trade_start_date,
                                lot_size= self.lot_size,
                                fail_fast = self.fail_fast,
                                storage = self.storage,
                                is_option = self.is_option
                                )
            test_result = test_bt.run()

            self.look_ahead_bias_check(test_result, test_bt._results._strategy)

        data_tuples = [(stock, stock_data, timeframe) for (stock,stock_data) in data.items()]
        data_chunks = list(self.split_list(data_tuples, self.num_processes))
        
        progress_bars: dict[str, tqdm] = {}
        for i, (stock, stock_data, _) in enumerate(data_tuples):
            num_iter = self.calculate_num_iterations(
                len(stock_data),
                training_candles,
                testing_candles
            )
            progress_bars[stock] = tqdm(
                total=num_iter,
                desc=stock,
                position=i,
                leave=False
            )
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        updater_thread = Thread(
            target=WalkForwardOptimizer.progress_updater,
            args=(queue, progress_bars),
            daemon=True
        )
        updater_thread.start()
        outer_pbar = tqdm(total=len(data_tuples), desc=f"Collecting results for '{universe}'")

        self.results = []
        
        # Launch worker processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self.process_stock_chunk, chunk, queue)
                for chunk in data_chunks
            ]

            # This outer tqdm is just a progress over "how many total stock‐chunks have returned results."
            outer_pbar = tqdm(total=len(data_tuples), desc=f"Collecting results for '{universe}'")

            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                except Exception as e:
                    print(f"Error in one of the worker processes: {e}")
                    chunk_results = []

                for result_tuple in chunk_results:
                    self.results.append(result_tuple)
                    outer_pbar.update(1)
            outer_pbar.close()

        # Tell the updater thread to stop, then join
        queue.put('DONE')
        updater_thread.join()

        # Close all per-stock progress bars
        for pb in progress_bars.values():
            pb.close()

        self.generate_summary(output_directory_path)

if __name__ == "__main__":
    print("Here in main.")
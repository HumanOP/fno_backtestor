import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import duckdb
from hyperparameter_optimizer import HyperParameterOptimizer
from backtesting_opt1 import Backtest


class WalkForwardOptimizer:
    """
    Performs walk-forward optimization using HyperParameterOptimizer.
    """
    def __init__(self, duckdb_path, strategy, cash, legs: dict, hyperparameter_grid: Dict[str, List[float]], 
                start_date: str, end_date: str, in_sample_months: int = 6, out_sample_months: int = 3, 
                commission_per_contract: float = 0.65, option_multiplier: int = 75):
        self.cash = cash
        self.strategy = strategy
        self.legs = legs
        self.hyperparameter_grid = hyperparameter_grid
        self.duckdb = duckdb_path
        self.commission_per_contract = commission_per_contract
        self.option_multiplier = option_multiplier
        self.in_sample_months = in_sample_months
        self.out_sample_months = out_sample_months
        
        # Convert date strings to datetime objects
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"WFO Date range: {start_date} to {end_date}")
        print(f"In-sample period: {self.in_sample_months} months")
        print(f"Out-sample period: {self.out_sample_months} months")
        
        # Calculate walk-forward windows
        self.windows = self._calculate_windows()
        print(f"Total walk-forward windows: {len(self.windows)}")
        
    
    def _calculate_windows(self) -> List[Tuple[str, str, str, str]]:
        """
        Calculate rolling walk-forward windows based on months.
        Each window advances by the out-sample period, creating overlapping in-sample periods.
        Returns list of tuples: (in_sample_start, in_sample_end, out_sample_start, out_sample_end)
        """
        windows = []
        current_start = self.start_date
        
        while True:
            # Calculate in-sample period (fixed length)
            in_sample_start = current_start
            in_sample_end_dt = in_sample_start + relativedelta(months=self.in_sample_months) - relativedelta(days=1)
            
            # Calculate out-sample period (immediately follows in-sample)
            out_sample_start_dt = in_sample_end_dt + relativedelta(days=1)
            out_sample_end_dt = out_sample_start_dt + relativedelta(months=self.out_sample_months) - relativedelta(days=1)
            
            # Check if we have enough data for this window
            if out_sample_end_dt > self.end_date:
                print(f" Stopping window calculation: out_sample_end ({out_sample_end_dt.date()}) > end_date ({self.end_date.date()})")
                break
            
            # Convert dates to string format
            in_sample_start_str = in_sample_start.strftime('%Y-%m-%d')
            in_sample_end_str = in_sample_end_dt.strftime('%Y-%m-%d')
            out_sample_start_str = out_sample_start_dt.strftime('%Y-%m-%d')
            out_sample_end_str = out_sample_end_dt.strftime('%Y-%m-%d')
            
            windows.append((in_sample_start_str, in_sample_end_str, out_sample_start_str, out_sample_end_str))
            print(f"Window {len(windows)}: In-sample {in_sample_start_str} to {in_sample_end_str}, Out-sample {out_sample_start_str} to {out_sample_end_str}")
            
            # Move to next window: advance by out-sample period for rolling windows
            # This creates overlapping in-sample periods which is proper for walk-forward optimization
            current_start = current_start + relativedelta(months=self.out_sample_months)
            
        return windows

    def optimize(self) -> Tuple[Dict, float, List[Dict]]:
        """
        Run walk-forward optimization across all calculated windows.
        
        Returns:
        - best_params: dict, best parameters from the window with highest out-sample performance
        - avg_out_sample_performance: float, average out-sample performance across all windows
        - results: list, detailed results for each window
        """
        results = []
        out_sample_performances = []
        
        # Collect all out-sample data for consolidated tearsheet
        all_out_sample_equity_curves = []
        all_out_sample_trades = []
        out_sample_periods = []
        
        if not self.windows:
            raise ValueError("No valid walk-forward windows found. Check your date range and month parameters.")
        
        
        # def constraint(params: Dict) -> bool:
        #     return params["upper_gamma"] > params["upper_buffer"] > 0 > params["lower_buffer"] > params["lower_gamma"]

        for window_idx, (in_sample_start, in_sample_end, out_sample_start, out_sample_end) in enumerate(self.windows):
            print(f"\n{'='*60}")
            print(f"PROCESSING WINDOW {window_idx + 1}/{len(self.windows)}")
            print(f"{'='*60}")
            print(f"In-sample period: {in_sample_start} to {in_sample_end}")
            print(f"Out-sample period: {out_sample_start} to {out_sample_end}")
            print(f"Window progress: {window_idx + 1}/{len(self.windows)} ({((window_idx + 1)/len(self.windows)*100):.1f}%)")
            
            # Create in-sample optimizer
            optimizer = HyperParameterOptimizer(
                db_path=self.duckdb,
                strategy=self.strategy,
                cash=self.cash,
                commission_per_contract=self.commission_per_contract,
                option_multiplier=self.option_multiplier,
                legs=self.legs
            )
            
            # Run in-sample optimization
            try:
                print("Running in-sample hyperparameter optimization...")
                optimization_result = optimizer.optimize(
                    hyperparameter_grid=self.hyperparameter_grid,
                    maximize='Sharpe Ratio',
                    method='grid',
                    start_date=in_sample_start,
                    end_date=in_sample_end
                )
                
                # Handle different return formats
                if optimization_result is None:
                    print(f"Warning: Optimization returned None for window {window_idx + 1}. Skipping.")
                    continue
                    
                if len(optimization_result) != 3:
                    print(f"Warning: Unexpected optimization result format for window {window_idx + 1}. Skipping.")
                    continue
                
                best_in_sample_params, best_in_sample_performance, results_df = optimization_result
                
                if best_in_sample_params is None or best_in_sample_performance is None:
                    print(f"Warning: No valid parameters found for window {window_idx + 1}. Skipping.")
                    continue
                    
                print(f" In-sample optimization complete!")
                print(f"Best in-sample Sharpe Ratio: {best_in_sample_performance:.4f}")
                print(f"Best parameters: {best_in_sample_params}")
                print(f"Results shape: {results_df.shape if results_df is not None else 'None'}")
                
                # Debug parameter extraction for out-sample test
                if isinstance(best_in_sample_params, dict):
                    # Extract parameters that can be passed to backtest.run_window
                    if 'iv_slope_thresholds' in best_in_sample_params:
                        # Parameters are nested, extract them
                        flat_params = best_in_sample_params.copy()
                        print(f"Using nested parameter structure for out-sample test")
                    else:
                        # Parameters are already flat, construct nested structure
                        flat_params = best_in_sample_params
                        print(f"Using flat parameter structure for out-sample test")
                else:
                    print(f"Warning: Unexpected parameter type: {type(best_in_sample_params)}")
                    flat_params = best_in_sample_params
                
                # Save in-sample results
                # results_df.to_csv(f'in_sample_results_window_{window_idx + 1}.csv', index=False)
                print(f"In-sample results saved to: in_sample_results_window_{window_idx + 1}.csv")
                
            except Exception as e:
                print(f" Error during in-sample optimization for window {window_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Run out-sample test with best parameters
            try:
                print("Running out-sample testing with best parameters...")
                backtest = Backtest(
                    db_path=self.duckdb,
                    strategy=self.strategy,
                    cash=self.cash,
                    commission_per_contract=self.commission_per_contract,
                    option_multiplier=self.option_multiplier
                )
                
                out_sample_results = backtest.run_window(
                    start_date=out_sample_start,
                    end_date=out_sample_end,
                    **flat_params
                )
                
                if out_sample_results is None:
                    print(f"Warning: Out-sample backtest returned None for window {window_idx + 1}. Skipping.")
                    continue
                
                out_sample_performance = out_sample_results.get('Sharpe Ratio', 0)
                if pd.isna(out_sample_performance):
                    print(f"Warning: Out-sample Sharpe Ratio is NaN for window {window_idx + 1}. Using 0.")
                    out_sample_performance = 0
                    
                print(f" Out-sample testing complete!")
                print(f"Out-sample Sharpe Ratio: {out_sample_performance:.4f}")
                print(f"Out-sample results keys: {list(out_sample_results.keys()) if isinstance(out_sample_results, dict) else 'Not a dict'}")
                
                # Save out-sample results
                # out_sample_results.to_csv(f'out_sample_results_window_{window_idx + 1}.csv')
                print(f"Out-sample results saved to: out_sample_results_window_{window_idx + 1}.csv")
                
            except Exception as e:
                print(f" Error during out-sample testing for window {window_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Store results for this window
            window_result = {
                'window': (in_sample_start, out_sample_end),
                'in_sample_period': (in_sample_start, in_sample_end),
                'out_sample_period': (out_sample_start, out_sample_end),
                'in_sample_params': best_in_sample_params,
                'in_sample_performance': best_in_sample_performance,
                'out_sample_performance': out_sample_performance
            }
            results.append(window_result)
            out_sample_performances.append(out_sample_performance)
            
            # Collect out-sample data for consolidated tearsheet
            if out_sample_results is not None:
                # Extract equity curve and trades from out-sample results
                equity_curve = out_sample_results.get('_equity_curve')
                trades_data = out_sample_results.get('_trades')
                
                if equity_curve is not None and not equity_curve.empty:
                    all_out_sample_equity_curves.append(equity_curve)
                
                if trades_data is not None and not trades_data.empty:
                    all_out_sample_trades.append(trades_data)
                
                out_sample_periods.append((out_sample_start, out_sample_end))
                
                print(f" Collected out-sample data: {len(equity_curve)} equity points, {len(trades_data)} trades")
            
            print(f"Window {window_idx + 1}/{len(self.windows)} completed successfully!")
            print(f"   Remaining windows: {len(self.windows) - (window_idx + 1)}")
            
            # Force flush output to ensure we see progress
            import sys
            sys.stdout.flush()
        # Final results processing and summary
        if not results:
            raise ValueError("No valid results from walk-forward optimization. All windows failed.")
        
        print(f"Successfully completed {len(results)} out of {len(self.windows)} windows")
        
        # Calculate performance statistics
        avg_out_sample_performance = sum(out_sample_performances) / len(out_sample_performances)
        best_result = max(results, key=lambda x: x['out_sample_performance'])
        worst_result = min(results, key=lambda x: x['out_sample_performance'])
        best_params = best_result['in_sample_params']
        
        # Clean best_params for saving (remove legs data which is not a hyperparameter)
        if isinstance(best_params, dict) and 'legs' in best_params:
            best_params_clean = best_params.copy()
            best_params_clean.pop('legs', None)  # Remove legs from saved params
        else:
            best_params_clean = best_params
        
        print(f"Average out-sample Sharpe Ratio: {avg_out_sample_performance:.4f}")
        print(f"Best out-sample Sharpe Ratio: {best_result['out_sample_performance']:.4f}")
        print(f"Worst out-sample Sharpe Ratio: {worst_result['out_sample_performance']:.4f}")
        print(f"Standard deviation of out-sample Sharpe: {pd.Series(out_sample_performances).std():.4f}")
        
        # Calculate win rate (positive out-sample Sharpe ratios)
        positive_results = sum(1 for x in out_sample_performances if x > 0)
        win_rate = positive_results / len(out_sample_performances) * 100
        print(f"Win rate (positive out-sample Sharpe): {win_rate:.1f}% ({positive_results}/{len(out_sample_performances)})")
        
        # Best performing window details
        best_window_idx = out_sample_performances.index(best_result['out_sample_performance'])
        print(f"\nBest performing window: {best_window_idx + 1}")
        print(f"Best window out-sample period: {best_result['out_sample_period'][0]} to {best_result['out_sample_period'][1]}")
        print(f"Best parameters: {best_params}")
        
        # Generate consolidated tearsheet from all out-sample periods
        if all_out_sample_equity_curves and all_out_sample_trades:
            print("\n" + "="*70)
            print(" GENERATING OUT-SAMPLE TEARSHEET")
            print("="*70)
            self._generate_tearsheet(
                all_out_sample_equity_curves,
                all_out_sample_trades,
                out_sample_periods,
                best_params,
                avg_out_sample_performance,
                results
            )
        else:
            print("\n No out-sample data available for tearsheet generation")
        
        # Create comprehensive output
        wfo_output = {
            'summary': {
                'total_windows_attempted': len(self.windows),
                'successful_windows': len(results),
                'success_rate': len(results) / len(self.windows) * 100,
                'in_sample_months': self.in_sample_months,
                'out_sample_months': self.out_sample_months,
                'avg_out_sample_sharpe_ratio': avg_out_sample_performance,
                'best_out_sample_sharpe_ratio': best_result['out_sample_performance'],
                'worst_out_sample_sharpe_ratio': worst_result['out_sample_performance'],
                'std_out_sample_sharpe_ratio': pd.Series(out_sample_performances).std(),
                'win_rate_percent': win_rate,
                'best_window_id': best_window_idx + 1
            },
            'best_params': best_params_clean,
            'results': [
                {
                    'window_id': idx + 1,
                    'overall_window': (str(result['window'][0]), str(result['window'][1])),
                    'in_sample_period': (str(result['in_sample_period'][0]), str(result['in_sample_period'][1])),
                    'out_sample_period': (str(result['out_sample_period'][0]), str(result['out_sample_period'][1])),
                    'in_sample_params': result['in_sample_params'],
                    'in_sample_sharpe_ratio': result['in_sample_performance'],
                    'out_sample_sharpe_ratio': result['out_sample_performance']
                } for idx, result in enumerate(results)
            ]
        }
        
        # Save results to JSON file
        output_filename = 'wfo_results.json'
        
        # Clean the output for JSON serialization
        def clean_for_json(obj):
            """Convert complex objects to JSON-serializable format"""
            from datetime import date
            
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, datetime, date)):
                return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
            elif isinstance(obj, pd.Series):
                return f"<pandas.Series with {len(obj)} elements>"
            elif isinstance(obj, pd.DataFrame):
                return f"<pandas.DataFrame {obj.shape}>"
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
                return f"<{type(obj).__name__} object>"
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        try:
            cleaned_output = clean_for_json(wfo_output)
            with open(output_filename, 'w') as f:
                json.dump(cleaned_output, f, indent=4)
            print(f"\nDetailed results saved to: {output_filename}")
        except Exception as e:
            print(f"\nWarning: Could not save full results to JSON: {e}")
            # Save a simplified version
            simplified_output = {
                'summary': wfo_output['summary'],
                'best_params_simplified': {
                    'iv_slope_thresholds': best_params.get('iv_slope_thresholds', {}),
                    'portfolio_sl': best_params.get('portfolio_sl', 0.02),
                    'portfolio_tp': best_params.get('portfolio_tp', 0.03),
                    'legs_info': f"Complex legs data with {len(self.legs)} legs"
                }
            }
            with open(output_filename, 'w') as f:
                json.dump(simplified_output, f, indent=4)
            print(f"\nSimplified results saved to: {output_filename}")
        
        print(f"\n Walk-Forward Optimization completed successfully!")
        return best_params, avg_out_sample_performance, results

    def _generate_tearsheet(self, all_out_sample_equity_curves, all_out_sample_trades, out_sample_periods, 
                                         best_params, avg_out_sample_performance, results):
        """
        Generate a tearsheet from all out-sample periods by concatenating 
        equity curves and trades, then using existing tearsheet functionality.
        
        Parameters:
        - all_out_sample_equity_curves: List of equity curves from out-sample periods
        - all_out_sample_trades: List of trades from out-sample periods  
        - out_sample_periods: List of out-sample periods
        - best_params: Best parameters from WFO
        - avg_out_sample_performance: Average out-sample performance
        - results: List of all window results
        """
        try:
            print(" Concatenating equity curves from all out-sample periods...")
            print(f"   Number of equity curves to concatenate: {len(all_out_sample_equity_curves)}")
            
            # Debug: Check the structure of equity curves
            for i, eq_curve in enumerate(all_out_sample_equity_curves):
                print(f"   Equity curve {i+1}: {type(eq_curve)}, shape: {eq_curve.shape if hasattr(eq_curve, 'shape') else 'N/A'}")
                print(f"      Columns: {eq_curve.columns.tolist() if hasattr(eq_curve, 'columns') else 'N/A'}")
            
            # Concatenate all equity curves
            consolidated_equity_curve = pd.concat(all_out_sample_equity_curves, ignore_index=False)
            consolidated_equity_curve = consolidated_equity_curve.sort_index()
            
            # Handle duplicate timestamps by taking the last value (in case of overlaps)
            if consolidated_equity_curve.index.duplicated().any():
                print(" Found duplicate timestamps, taking last values...")
                consolidated_equity_curve = consolidated_equity_curve.groupby(consolidated_equity_curve.index).last()
            
            print(" Concatenating trades from all out-sample periods...")
            print(f"   Number of trade DataFrames to concatenate: {len(all_out_sample_trades)}")
            
            # Debug: Check the structure of trades
            for i, trades in enumerate(all_out_sample_trades):
                print(f"   Trades {i+1}: {type(trades)}, shape: {trades.shape if hasattr(trades, 'shape') else 'N/A'}")
                print(f"      Columns: {trades.columns.tolist() if hasattr(trades, 'columns') else 'N/A'}")
            
            # Concatenate all trades 
            consolidated_trades = pd.concat(all_out_sample_trades, ignore_index=True)
            
            # Sort trades by entry time if available
            if 'EntryTime' in consolidated_trades.columns:
                consolidated_trades = consolidated_trades.sort_values('EntryTime')
            elif 'EntryBar' in consolidated_trades.columns:
                consolidated_trades = consolidated_trades.sort_values('EntryBar')
            
            # Create consolidated results object matching the format expected by tearsheet
            print(" Computing consolidated statistics...")
            try:
                from stats import compute_stats
                
                consolidated_results = compute_stats(
                    orders=[],  # We don't have orders, only completed trades
                    trades=consolidated_trades,
                    equity_curve=consolidated_equity_curve
                )
                
                print(" Statistics computed successfully")
                print(f"   Available stats keys: {list(consolidated_results.keys())}")
                
            except Exception as stats_error:
                print(f" Error computing stats: {stats_error}")
                # Create a minimal results object manually
                consolidated_results = pd.Series({
                    '_equity_curve': consolidated_equity_curve,
                    '_trades': consolidated_trades,
                    'Sharpe Ratio': 0.0,  # Placeholder
                    'Total Trades': len(consolidated_trades),
                    'Start': consolidated_equity_curve.index.min(),
                    'End': consolidated_equity_curve.index.max()
                })
                print(" Created minimal results object")
            
            # Add WFO-specific metadata
            consolidated_results['WFO_Windows'] = len(results)
            consolidated_results['WFO_Avg_Performance'] = avg_out_sample_performance
            consolidated_results['WFO_Best_Params'] = best_params
            consolidated_results['WFO_Period_Start'] = out_sample_periods[0][0] if out_sample_periods else None
            consolidated_results['WFO_Period_End'] = out_sample_periods[-1][1] if out_sample_periods else None
            
            print(" Creating backtest instance for tearsheet generation...")
            
            # Create a backtest instance to use its tearsheet functionality
            backtest = Backtest(
                db_path=self.duckdb,
                strategy=self.strategy,
                cash=self.cash,
                commission_per_contract=self.commission_per_contract,
                option_multiplier=self.option_multiplier
            )
            
            # Create benchmark using the new get_spot_prices_for_duration method
            print(" Creating benchmark series for out-sample periods...")
            benchmark_series = None
            try:
                from backtesting_opt1 import _Data
                
                # Get the actual date range from the consolidated equity curve
                start_date = consolidated_equity_curve.index.min().strftime('%Y-%m-%d')
                end_date = consolidated_equity_curve.index.max().strftime('%Y-%m-%d')
                print(f"   Benchmark date range: {start_date} to {end_date}")
                
                # Use the new method to get spot prices for the duration
                with _Data(self.duckdb) as data:
                    spot_data = data.get_spot_prices(start_date, end_date)
                
                if spot_data is not None and not spot_data.empty:
                    print(f"   Retrieved spot data: {len(spot_data)} data points")
                    print(f"   Spot data range: {spot_data.index.min()} to {spot_data.index.max()}")
                    
                    # Convert to daily data by resampling to 1D interval
                    daily_spot = spot_data.resample('1D').last().dropna()
                    print(f"   Daily spot data after resampling: {len(daily_spot)} data points")
                    
                    if not daily_spot.empty:
                        # Calculate returns for benchmark
                        benchmark_series = daily_spot.pct_change().dropna()
                        benchmark_series = benchmark_series.rename("Benchmark")
                        
                        print(f" Benchmark created: {len(benchmark_series)} daily returns")
                        print(f"   Benchmark range: {benchmark_series.index.min()} to {benchmark_series.index.max()}")
                        print(f"   Sample benchmark values: {benchmark_series.head(3).tolist()}")
                        print(f"   Benchmark stats: mean={benchmark_series.mean():.6f}, std={benchmark_series.std():.6f}")
                    else:
                        print(" No daily spot data available after resampling")
                        benchmark_series = None
                else:
                    print(" No spot data found for the out-sample date range")
                    benchmark_series = None
                    
            except Exception as benchmark_error:
                print(f" Could not create benchmark: {benchmark_error}")
                import traceback
                traceback.print_exc()
                benchmark_series = None
            
            # Generate tearsheet filename
            strategy_name = getattr(self.strategy, '__name__', 'Strategy')
            tearsheet_filename = f"WFO_Consolidated_OutSample_{strategy_name}.html"
            
            print(f" Generating consolidated tearsheet: {tearsheet_filename}")
            print(f"   Results type: {type(consolidated_results)}")
            print(f"   Results keys: {list(consolidated_results.keys()) if hasattr(consolidated_results, 'keys') else 'N/A'}")
            print(f"   Benchmark available: {'Yes' if benchmark_series is not None else 'No'}")
            
            # Generate the tearsheet using existing functionality
            try:
                # Debug: Print detailed information about results structure
                print(f" Results structure before tearsheet:")
                print(f"   Type: {type(consolidated_results)}")
                print(f"   Keys: {list(consolidated_results.keys()) if hasattr(consolidated_results, 'keys') else 'No keys'}")
                
                if '_equity_curve' in consolidated_results:
                    eq_curve = consolidated_results['_equity_curve']
                    print(f"   Equity curve type: {type(eq_curve)}")
                    print(f"   Equity curve shape: {eq_curve.shape if hasattr(eq_curve, 'shape') else 'No shape'}")
                    print(f"   Equity curve columns: {eq_curve.columns.tolist() if hasattr(eq_curve, 'columns') else 'No columns'}")
                    print(f"   Equity curve index range: {eq_curve.index.min()} to {eq_curve.index.max()}")
                    print(f"   Sample equity values: {eq_curve.head(3) if hasattr(eq_curve, 'head') else 'Cannot show sample'}")
                
                if '_trades' in consolidated_results:
                    trades = consolidated_results['_trades']
                    print(f"   Trades type: {type(trades)}")
                    print(f"   Trades shape: {trades.shape if hasattr(trades, 'shape') else 'No shape'}")
                    print(f"   Trades columns: {trades.columns.tolist() if hasattr(trades, 'columns') else 'No columns'}")
                
                # Set the benchmark in the backtest instance if we have one
                if benchmark_series is not None:
                    # Store benchmark for tearsheet generation
                    consolidated_results['_benchmark'] = benchmark_series
                    print(f"   Added benchmark to consolidated_results with {len(benchmark_series)} data points")
                    print(f"   Benchmark sample values: {benchmark_series.head(3).tolist()}")
                    print(f"   Benchmark date range: {benchmark_series.index.min()} to {benchmark_series.index.max()}")
                else:
                    print("    No benchmark available - tearsheet will be generated without benchmark")
                
                print(f" Calling tear_sheet method...")
                backtest.tear_sheet(
                    results=consolidated_results,
                    filename=tearsheet_filename,
                    open_browser=True,
                    generate_trade_logs=True
                )
                
            except Exception as tearsheet_error:
                print(f" Error type: {type(tearsheet_error).__name__}")
                print(" Tearsheet method call failed. Checking method availability...")
                print(f"   Backtest instance methods: {[method for method in dir(backtest) if not method.startswith('_')]}")
                
                # Print full traceback for debugging
                import traceback
                traceback.print_exc()
                
                # Try to see if quantstats is available
                try:
                    import quantstats_lumi as quantstats
                    print(f" quantstats_lumi is available: {quantstats.__version__}")
                except ImportError as qs_error:
                    print(f" quantstats_lumi import error: {qs_error}")
                
                raise tearsheet_error
            
        except Exception as e:
            import traceback
            backtest.tear_sheet(
                results=consolidated_results,
                filename=tearsheet_filename,
                open_browser=True,
                generate_trade_logs=True
            )
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
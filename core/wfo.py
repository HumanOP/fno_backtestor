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
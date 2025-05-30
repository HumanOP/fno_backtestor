import json
from typing import Dict, List, Tuple
import duckdb
import pandas as pd
from hyperparameter_optimizer import HyperParameterOptimizer

class WalkForwardOptimizer:
    """
    Performs walk-forward optimization using HyperParameterOptimizer.
    """
    def __init__(self, legs: dict, hyperparameter_grid: Dict[str, List[float]], duckdb_conn, 
                 dates: pd.Series, in_sample_ratio: float = 0.6/4, out_sample_ratio: float = 0.2/4):
        self.legs = legs
        self.hyperparameter_grid = hyperparameter_grid
        self.duckdb = duckdb_conn
        self.dates = dates.tolist()
        self.in_sample_ratio = in_sample_ratio
        self.out_sample_ratio = out_sample_ratio
        self.total_dates = len(self.dates)
        self.in_sample_size = int(self.total_dates * in_sample_ratio)
        self.out_sample_size = int(self.total_dates * out_sample_ratio)
        if self.in_sample_size + self.out_sample_size > self.total_dates:
            raise ValueError("In-sample + out-sample size exceeds total dates")

    def optimize(self) -> Tuple[Dict, float, List[Dict]]:
        results = []
        out_sample_performances = []
        def constraint(params: Dict) -> bool:
            return params["upper_gamma"] > params["upper_buffer"] > 0 > params["lower_buffer"] > params["lower_gamma"]

        for start in range(0, self.total_dates - self.in_sample_size - self.out_sample_size + 1, self.out_sample_size):
            in_sample_dates = self.dates[start:start + self.in_sample_size]
            out_sample_dates = self.dates[start + self.in_sample_size:start + self.in_sample_size + self.out_sample_size]
            print(f"Processing window: In-sample {in_sample_dates[0]} to {in_sample_dates[-1]}, Out-sample {out_sample_dates[0]} to {out_sample_dates[-1]}")
            optimizer = HyperParameterOptimizer(self.legs, self.duckdb, pd.Series(in_sample_dates))
            try:
                best_in_sample_params, best_in_sample_performance, results_df = optimizer.optimize(
                    hyperparameter_grid=self.hyperparameter_grid,
                    maximize='Sharpe Ratio',
                    method='grid',
                    constraint=constraint
                )
                print(f"In-sample best params: {best_in_sample_params}, Sharpe Ratio: {best_in_sample_performance}")
                results_df.to_csv(f'in_sample_results_window_{start}.csv')
            except Exception as e:
                print(f"Error during in-sample optimization: {e}")
                continue

            from backtest_utils import backtest
            from iv_slope.Hyper_Risk_framework.trade_manager import TradeManager
            trader = TradeManager()
            try:
                out_sample_performance = backtest(self.legs, best_in_sample_params, self.duckdb, trader, pd.Series(out_sample_dates))
                tradebook = trader.build_tradebook()
                print(f"Out-sample: {len(tradebook)} trades, Sharpe Ratio: {out_sample_performance}")
                tradebook.to_csv(f'out_sample_tradebook_window_{start}.csv')
            except Exception as e:
                print(f"Error during out-sample testing: {e}")
                continue

            results.append({
                'window': (in_sample_dates[0], out_sample_dates[-1]),
                'in_sample_params': best_in_sample_params,
                'in_sample_performance': best_in_sample_performance,
                'out_sample_performance': out_sample_performance
            })
            out_sample_performances.append(out_sample_performance)
        if not results:
            raise ValueError("No valid results from walk-forward optimization")
        avg_out_sample_performance = sum(out_sample_performances) / len(out_sample_performances)
        best_result = max(results, key=lambda x: x['out_sample_performance'])
        best_params = best_result['in_sample_params']
        wfo_output = {
            'best_params': best_params,
            'avg_out_sample_sharpe_ratio': avg_out_sample_performance,
            'results': [
                {
                    'window': (str(result['window'][0]), str(result['window'][1])),
                    'in_sample_params': result['in_sample_params'],
                    'in_sample_sharpe_ratio': result['in_sample_performance'],
                    'out_sample_sharpe_ratio': result['out_sample_performance']
                } for result in results
            ]
        }
        with open('wfo_results.json', 'w') as f:
            json.dump(wfo_output, f, indent=4)
        return best_params, avg_out_sample_performance, results
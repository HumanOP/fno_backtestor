import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
from itertools import product
import itertools
import datetime
from openpyxl.drawing.image import Image
import json
from functools import partial
from backtesting_opt1 import Backtest

class HyperParameterOptimizer:
    def __init__(self, db_path: str, strategy, *, cash: float = 100000, commission_per_contract: float = 0.65, 
                 option_multiplier: int = 75, legs: dict):
        self._backtest_factory = partial(
            Backtest, db_path=db_path, strategy=strategy, cash=cash,
            commission_per_contract=commission_per_contract, option_multiplier=option_multiplier
        )
        self.legs = legs

    def optimize(self, hyperparameter_grid: dict, maximize: str = 'Sharpe Ratio', 
                 method: str = 'grid', start_date: str = None, end_date: str = None,
                 max_tries: int = None, constraint: callable = None):
        """
        Optimize hyperparameters using grid search.
        
        Parameters:
        - hyperparameter_grid: dict, parameter combinations to test
        - maximize: str, metric to maximize (default: 'Sharpe Ratio')
        - method: str, optimization method (only 'grid' supported)
        - max_tries: int, maximum parameter combinations to test
        - constraint: callable, function to filter parameter combinations
        - start_date: str, start date for backtest window (YYYY-MM-DD format)
        - end_date: str, end date for backtest window (YYYY-MM-DD format)
        
        Returns:
        - best_params: dict, best parameter combination
        - best_score: float, best metric value
        - results_df: pd.DataFrame, all results
        """
        if method != 'grid':
            raise ValueError("Only 'grid' method is supported in this implementation")
            
        # Log the optimization window if specified
        if start_date or end_date:
            print(f"Optimizing on date window: {start_date} to {end_date}")
        else:
            print("Optimizing on full dataset")
        param_keys = list(hyperparameter_grid.keys())
        param_values = [hyperparameter_grid[key] for key in param_keys]
        param_combinations = [dict(zip(param_keys, combo)) for combo in product(*param_values)]

        if constraint:
            param_combinations = [params for params in param_combinations if constraint(params)]
        if max_tries and max_tries < len(param_combinations):
            param_combinations = param_combinations[:max_tries]
        if not param_combinations:
            raise ValueError("No admissible parameter combinations to test")
            
        print(f"Total parameter combinations to test: {len(param_combinations)}")
        results = []
        best_sharpe = -np.inf
        best_params = None

        for idx, flat_params in enumerate(param_combinations, 1):
            print(f"Testing parameters {idx}/{len(param_combinations)}: {flat_params}")

            # Construct the full parameter dictionary
            params = {
                "iv_slope_thresholds": {
                    "upper_gamma": flat_params["upper_gamma"],
                    "upper_buffer": flat_params["upper_buffer"],
                    "lower_buffer": flat_params["lower_buffer"],
                    "lower_gamma": flat_params["lower_gamma"]
                },
                "portfolio_sl": flat_params.get("portfolio_sl", 0.01),  # Default if not optimized
                "portfolio_tp": flat_params.get("portfolio_tp", 0.03),  # Default if not optimized
                "legs": self.legs  # Pass legs as a parameter
            }

            try:
                backtest = self._backtest_factory()
                
                # Use run_window if date range is specified, otherwise use regular run
                if start_date is not None or end_date is not None:
                    result = backtest.run_window(start_date=start_date, end_date=end_date, **params)
                else:
                    result = backtest.run(**params)
                    
                print(f"Results: {result}")

                if maximize not in result or pd.isna(result[maximize]):
                    print(f"Invalid result for parameters {flat_params}: missing or NaN {maximize}")
                    continue

                sharpe_ratio = result[maximize]
                results.append({**flat_params, 'Sharpe Ratio': sharpe_ratio})
                print(f"Sharpe ratio: {sharpe_ratio}")

                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_params = params  # Store the full params dict
                    print(f"New best Sharpe ratio: {best_sharpe}")
                    
            except Exception as e:
                print(f"Error testing parameters {flat_params}: {e}")
                continue

        results_df = pd.DataFrame(results)
        
        # Print optimization summary
        if results:
            print(f"\n=== Optimization Complete ===")
            print(f"Tested {len(param_combinations)} parameter combinations")
            print(f"Valid results: {len(results)}")
            print(f"Best {maximize}: {best_sharpe:.4f}")
            print(f"Best parameters: {best_params}")
            
            if len(results) > 1:
                print(f"Worst {maximize}: {min(results, key=lambda x: x['Sharpe Ratio'])['Sharpe Ratio']:.4f}")
                print(f"Average {maximize}: {results_df['Sharpe Ratio'].mean():.4f}")
        else:
            print("Warning: No valid results found during optimization")
        
        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # output_directory_path = os.path.join(os.getcwd(), f'hyperparameter_optimizer_output_{timestamp}')
        # os.makedirs(output_directory_path, exist_ok=True)
        # sheet_path = os.path.join(output_directory_path, 'summary.xlsx')
        # self.generate_heatmaps(results_df, sheet_path, output_directory_path)
        # self.delete_png_files(output_directory_path)

        return best_params, best_sharpe, results_df

    def generate_heatmaps(self, results_df: 'pd.DataFrame', output_file: str, png_directory: str) -> None:
        workbook = Workbook()
        sheet = workbook.create_sheet(title="Sharpe Ratio")
        for row_idx, row in enumerate([results_df.columns.tolist()] + results_df.values.tolist(), start=1):
            for col_idx, value in enumerate(row, start=1):
                sheet.cell(row=row_idx, column=col_idx, value=value)
        start_row = len(results_df) + 3
        parameter_columns = [col for col in results_df.columns if col != "Sharpe Ratio"]
        parameter_pairs = list(itertools.combinations(parameter_columns, 2))
        for param_x, param_y in parameter_pairs:
            grouped = results_df.groupby([param_x, param_y])["Sharpe Ratio"].mean().unstack(fill_value=np.nan)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color="white")
            plt.figure(figsize=(5, 4))
            ax = plt.gca()
            heatmap = ax.imshow(grouped.values, cmap=cmap, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(grouped.columns)))
            ax.set_yticks(np.arange(len(grouped.index)))
            ax.set_xticklabels(grouped.columns, rotation=45)
            ax.set_yticklabels(grouped.index)
            ax.set_xlabel(param_y)
            ax.set_ylabel(param_x)
            ax.set_title(f"Heatmap of Sharpe Ratio: {param_x} vs {param_y}")
            plt.colorbar(heatmap, ax=ax)
            image_file = os.path.join(png_directory, f"{param_x}_{param_y}_Sharpe_Ratio_heatmap.png")
            plt.savefig(image_file, bbox_inches="tight", dpi=150)
            plt.close()
            from openpyxl.drawing.image import Image
            img = Image(image_file)
            img.anchor = f"A{start_row}"
            sheet.add_image(img)
            start_row += 30
        if "Sheet" in workbook.sheetnames:
            workbook.remove(workbook["Sheet"])
        workbook.save(output_file)
        print(f"Heatmaps saved to {output_file}")

    def delete_png_files(self, png_directory: str) -> None:
        for filename in os.listdir(png_directory):
            if filename.endswith(".png"):
                file_path = os.path.join(png_directory, filename)
                os.remove(file_path)
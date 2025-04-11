import sys
import os

# Go one directory up from notebooks/ and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.trade_manager import TradeManager
from core.statistics_builder import StatisticsBuilder
from core.report_generator import ReportGenerator
import pandas as pd


def run_backtest():
    """Example of using the refactored components."""
    # Initialize the trade manager
    tm = TradeManager()
    
    # Place some sample trades
    entry_data = {
        'strategy_id': 'strat1',
        'position_id': 'pos1',
        'leg_id': 'leg1',
        'symbol': 'NIFTY21JUN2432540CE',
        'entry_date': '2021-06-01',
        'entry_time': '09:15:00',
        'entry_price': 15000,
        'qty': 1,
        'entry_type': 'BUY',
        'entry_spot': 15000,
        'stop_loss': 14900,
        'take_profit': 15100,
        'entry_reason': 'breakout'
    }
    tm.place_order(entry_data)
    
    # Close the trade
    exit_data = {
        'exit_date': '2021-06-02',
        'exit_time': '15:30:00',
        'exit_price': 15100,
        'exit_spot': 15100,
        'exit_reason': 'target_hit'
    }
    tm.square_off(0, exit_data)
    
    # Build the tradebook
    tradebook_df = tm.build_tradebook()
    
    # Calculate statistics
    stats_builder = StatisticsBuilder(tradebook_df)
    daily_pnl, maxdd, expirywise_pnl, monthly_pnl, yearly_pnl, monthwise_pnl, daywise_pnl = stats_builder.generate_report()
    performance_stats = stats_builder.build_stats()
    
    # Display stats
    print("Performance Statistics:")
    for key, value in performance_stats.items():
        print(f"{key}: {value}")
    
    # Create visualization
    stats_builder.plot_pnl()
    stats_builder.plot_drawdown()
    
    # Generate Excel report
    input_params = pd.DataFrame([{'strategy': 'Calendar Spread', 'timeframe': 'Daily'}])
    ReportGenerator.excel_output(
        input_params,
        daily_pnl,
        maxdd,
        expirywise_pnl, 
        monthly_pnl,
        yearly_pnl,
        monthwise_pnl,
        daywise_pnl,
        tradebook_df,
        'backtest_results.xlsx'
    )


if __name__ == "__main__":
    run_backtest()

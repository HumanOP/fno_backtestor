from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import plotly.express as px
from datetime import datetime


class StatisticsBuilder:
    """
    Handles calculation of performance statistics and visualization.
    
    Responsible for:
    - Calculating trading performance metrics
    - Generating PnL and drawdown statistics
    - Visualizing performance through charts
    """

    def __init__(self, tradebook: pd.DataFrame) -> None:
        """
        Initialize StatisticsBuilder with a tradebook.
        
        Args:
            tradebook: DataFrame containing trade data
        """
        self.tradebook = tradebook
        self.daily_pnl = None
        self.maxdd = None
        self.expirywise_pnl = None
        self.monthly_pnl = None
        self.yearly_pnl = None
        self.monthwise_pnl = None
        self.daywise_pnl = None
        self.daily_pnl_sum = None
        self.daily_drawdown = None
        self.stats = None

    def generate_report(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate various performance reports from the tradebook.
        
        Returns:
            Tuple of DataFrames containing different PnL breakdowns
        """
        daily_pnl = pd.DataFrame()
        expirywise_pnl = pd.DataFrame()
        monthly_pnl = pd.DataFrame()
        yearly_pnl = pd.DataFrame()
        monthwise_pnl = pd.DataFrame()
        daywise_pnl = pd.DataFrame()

        daily_pnl['exit_date'] = pd.to_datetime(self.tradebook['exit_date']).dt.strftime('%d-%m-%Y')
        daily_pnl['pnl'] = self.tradebook['pnl']
        pnl = daily_pnl['pnl'].cumsum()
        daily_pnl['ddcalc'] = pnl - pnl.cummax()  # Calculate drawdown from peak
        maxdd = pd.DataFrame({'maxdd': [daily_pnl['ddcalc'].min()]})
        
        expirywise_pnl = self.tradebook.groupby(self.tradebook['expiry'])['pnl'].sum().reset_index()
        expirywise_pnl.columns = ['expiry', 'pnl']
        expirywise_pnl['expiry'] = pd.to_datetime(expirywise_pnl['expiry']).dt.strftime('%d-%m-%Y')

        monthly_pnl = self.tradebook.groupby(self.tradebook['exit_date'].dt.to_period('M'))['pnl'].sum().reset_index()
        monthly_pnl.columns = ['exit_time', 'pnl']

        yearly_pnl = self.tradebook.groupby(self.tradebook['exit_date'].dt.to_period('Y'))['pnl'].sum().reset_index()
        yearly_pnl.columns = ['exit_year', 'pnl']

        monthwise_pnl = self.tradebook.groupby(self.tradebook['exit_date'].dt.month_name())['pnl'].sum().reset_index()
        monthwise_pnl.columns = ['exit_month', 'pnl']
        monthwise_pnl['exit_month'] = pd.Categorical(
            monthwise_pnl['exit_month'], 
            categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], 
            ordered=True
        )
        monthwise_pnl = monthwise_pnl.sort_values('exit_month').reset_index(drop=True)

        daywise_pnl = self.tradebook.groupby(self.tradebook['exit_date'].dt.day_name())['pnl'].sum().reset_index()
        daywise_pnl.columns = ['exit_day', 'pnl']
        daywise_pnl['exit_day'] = pd.Categorical(
            daywise_pnl['exit_day'], 
            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
            ordered=True
        )
        daywise_pnl = daywise_pnl.sort_values('exit_day').reset_index(drop=True)

        self.daily_pnl = daily_pnl
        self.maxdd = maxdd
        self.expirywise_pnl = expirywise_pnl
        self.monthly_pnl = monthly_pnl
        self.yearly_pnl = yearly_pnl
        self.monthwise_pnl = monthwise_pnl
        self.daywise_pnl = daywise_pnl
        
        return daily_pnl, maxdd, expirywise_pnl, monthly_pnl, yearly_pnl, monthwise_pnl, daywise_pnl


    def build_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics and performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.tradebook.empty:
            print("No trades executed.")
            return {}

        self.tradebook["date"] = pd.to_datetime(self.tradebook['exit_datetime']).dt.date
        daily_pnl_sum = self.tradebook.groupby('date')['pnl'].sum()

        win_rate = (daily_pnl_sum > 0).mean()  # Calculating win rate

        # Average gain of winning trades and average loss of losing trades
        average_gain = daily_pnl_sum[daily_pnl_sum > 0].mean()
        average_loss = daily_pnl_sum[daily_pnl_sum < 0].mean()

        rr = abs(average_gain / average_loss)  # Risk-Reward Ratio

        expectancy = rr * win_rate - (1 - win_rate)  # Expectancy

        # Calculating Drawdown
        roll_max = daily_pnl_sum.cumsum().cummax()
        daily_drawdown = roll_max - daily_pnl_sum.cumsum()
        max_drawdown = daily_drawdown.max()

        # Calmar Ratio (requires annual return and max drawdown)
        annual_return = daily_pnl_sum.sum() / len(daily_pnl_sum) * 365  # Simplified annual return
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else float('inf')

        # Recovery days for each drawdown period
        drawdown_periods = daily_drawdown > 0  # Identify drawdown periods
        recovery_days = []
        current_recovery = 0
        for date, is_drawdown in drawdown_periods.items():
            if is_drawdown:
                current_recovery += 1
                if date == drawdown_periods.keys()[-1]:  # If the last date is a drawdown period
                    recovery_days.append(current_recovery)
                    current_recovery = 0
            elif current_recovery > 0:
                recovery_days.append(current_recovery)
                current_recovery = 0
                
        max_recovery_days = max(recovery_days) if recovery_days else 0

        self.stats = {
            "Risk-Reward Ratio": rr,
            "Win Rate": win_rate,
            "Expectancy": expectancy,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Max Recovery Days": max_recovery_days
        }
        
        self.daily_pnl_sum = daily_pnl_sum
        self.daily_drawdown = daily_drawdown

        if max_recovery_days == recovery_days[-1]:
            self.stats["Alert"] = ("Couldn't recover the largest drawdown")
            
        return self.stats
    
    def plot_pnl(self) -> None:
        """Generate and display a cumulative PnL chart."""
        fig = px.line(self.daily_pnl_sum.cumsum(), title='Cumulative PnL Over Time')
        fig.show()
        
    def plot_drawdown(self) -> None:
        """Generate and display a drawdown chart."""
        fig = px.line((self.daily_drawdown * -1), title='Daily Drawdown Over Time')
        fig.add_scatter(x=self.daily_drawdown.index, y=self.daily_drawdown * -1, fill='tozeroy', mode='none', fillcolor='red')
        fig.show()

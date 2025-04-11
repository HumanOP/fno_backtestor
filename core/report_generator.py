import pandas as pd
from typing import Optional, Dict, Any


class ReportGenerator:
    """
    Handles generation of reports and Excel output.
    
    Responsible for:
    - Formatting and exporting trade data to Excel
    - Creating properly formatted sheets for different metrics
    - Applying appropriate formatting to Excel output
    """

    @staticmethod
    def excel_output(
        inputs: pd.DataFrame,
        daily_pnl: pd.DataFrame, 
        maxdd: pd.DataFrame,
        expirywise_pnl: pd.DataFrame,
        monthly_pnl: pd.DataFrame,
        yearly_pnl: pd.DataFrame,
        monthwise_pnl: pd.DataFrame,
        daywise_pnl: pd.DataFrame,
        transaction_df: pd.DataFrame,
        output_path: str = 'output.xlsx'
    ) -> None:
        """
        Export all trading data to a well-formatted Excel workbook.
        
        Args:
            inputs: DataFrame containing input parameters
            daily_pnl: DataFrame with daily PnL
            maxdd: DataFrame with maximum drawdown
            expirywise_pnl: DataFrame with PnL by expiry
            monthly_pnl: DataFrame with PnL by month
            yearly_pnl: DataFrame with PnL by year
            monthwise_pnl: DataFrame with PnL by month name
            daywise_pnl: DataFrame with PnL by day of week
            transaction_df: DataFrame with all trade transactions
            output_path: Path to save the Excel file (default: 'output.xlsx')
        """
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Write input data
            inputs.to_excel(writer, sheet_name='Input', index=False)
            
            # Write statistics
            daily_pnl.to_excel(writer, sheet_name='Stats', startrow=0, startcol=0, index=False) 
            maxdd.to_excel(writer, sheet_name='Stats', startrow=0, startcol=3, index=False)
            expirywise_pnl.to_excel(writer, sheet_name='Stats', startrow=0, startcol=5, index=False)
            monthly_pnl.to_excel(writer, sheet_name='Stats', startrow=0, startcol=8, index=False)  
            yearly_pnl.to_excel(writer, sheet_name='Stats', startrow=0, startcol=11, index=False) 
            monthwise_pnl.to_excel(writer, sheet_name='Stats', startrow=len(yearly_pnl) + 2, startcol=11, index=False)  
            daywise_pnl.to_excel(writer, sheet_name='Stats', startrow=len(daywise_pnl) + 2, startcol=11, index=False)  
            
            # Write transaction details
            transaction_df.to_excel(writer, sheet_name='Transaction', index=False)
            
            # Format workbook
            workbook = writer.book
            worksheet = writer.sheets['Transaction']

            # Add date formatting
            date_format = workbook.add_format({'num_format': 'dd-mm-yyyy'})
            datetime_format = workbook.add_format({'num_format': 'dd-mm-yyyy hh:mm:ss'})

            # Apply date formatting to datetime columns
            for col_num, col_name in enumerate(transaction_df.columns):
                if pd.api.types.is_datetime64_any_dtype(transaction_df[col_name]):
                    worksheet.set_column(col_num, col_num, 20, datetime_format)

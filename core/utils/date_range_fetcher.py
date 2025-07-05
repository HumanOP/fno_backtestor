"""
Date Range Data Fetcher for NIFTY Options Database
Utility functions for fetching data within specific date ranges
"""

import duckdb
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, date
import re

class DateRangeDataFetcher:
    """
    Enhanced data fetcher with date range capabilities for NIFTY options data
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the data fetcher
        
        Parameters:
        -----------
        db_path : str
            Path to the DuckDB database file
        """
        self.db_path = db_path
        self._conn = None
        self.connect()
    
    def connect(self):
        """Connect to the database"""
        if self._conn is not None:
            self._conn.close()
        self._conn = duckdb.connect(self.db_path)
    
    def close(self):
        """Close database connection"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def get_all_tables(self) -> pd.DataFrame:
        """
        Get all table names in the database
        
        Returns:
        --------
        pd.DataFrame: DataFrame with table names
        """
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        return self._conn.execute(query).fetchdf()
    
    def get_tables_by_date_pattern(self, year: int = None, month: int = None, day: int = None) -> List[str]:
        """
        Get tables matching a specific date pattern
        
        Parameters:
        -----------
        year : int, optional
            Year to filter by
        month : int, optional  
            Month to filter by
        day : int, optional
            Day to filter by
            
        Returns:
        --------
        List[str]: List of matching table names
        """
        tables = self.get_all_tables()
        pattern_parts = ['nifty']
        
        if year:
            pattern_parts.append(f"{year:04d}")
        else:
            pattern_parts.append(r'\d{4}')
            
        if month:
            pattern_parts.append(f"{month:02d}")
        else:
            pattern_parts.append(r'\d{2}')
            
        if day:
            pattern_parts.append(f"{day:02d}")
        else:
            pattern_parts.append(r'\d{2}')
        
        pattern = '_'.join(pattern_parts)
        return tables[tables['table_name'].str.match(pattern)]['table_name'].tolist()
    
    def get_table_date_info(self, table_name: str) -> Dict:
        """
        Get date information for a specific table
        
        Parameters:
        -----------
        table_name : str
            Name of the table
            
        Returns:
        --------
        Dict: Dictionary with min_date, max_date, and record_count
        """
        try:
            query = f"""
            SELECT 
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(*) as record_count
            FROM {table_name}
            """
            result = self._conn.execute(query).fetchone()
            return {
                'table_name': table_name,
                'min_date': result[0],
                'max_date': result[1], 
                'record_count': result[2]
            }
        except Exception as e:
            return {
                'table_name': table_name,
                'min_date': None,
                'max_date': None,
                'record_count': 0,
                'error': str(e)
            }
    
    def find_tables_in_date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Find tables that contain data within the specified date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str  
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        List[str]: List of table names containing data in the range
        """
        all_tables = self.get_all_tables()['table_name'].tolist()
        matching_tables = []
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        print(f"Searching for tables with data between {start_date} and {end_date}")
        
        for table in all_tables:
            try:
                table_info = self.get_table_date_info(table)
                if table_info['min_date'] is None:
                    continue
                    
                table_min = pd.to_datetime(table_info['min_date'])
                table_max = pd.to_datetime(table_info['max_date'])
                
                # Check if table's date range overlaps with requested range
                if not (table_max < start_ts or table_min > end_ts):
                    matching_tables.append(table)
                    print(f"✓ {table}: {table_min.date()} to {table_max.date()} ({table_info['record_count']} records)")
                
            except Exception as e:
                print(f"⚠ Error checking table {table}: {e}")
                continue
        
        print(f"Found {len(matching_tables)} tables with data in the specified range")
        return matching_tables
    
    def fetch_data_for_date_range(self, start_date: str, end_date: str, 
                                  columns: List[str] = None, 
                                  filters: Dict = None) -> pd.DataFrame:
        """
        Fetch data for a specific date range across multiple tables
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format  
        columns : List[str], optional
            Specific columns to fetch. If None, fetches all columns
        filters : Dict, optional
            Additional filters like {'Time_to_expiry': [7, 30]}
            
        Returns:
        --------
        pd.DataFrame: Combined data from all tables in the range
        """
        tables = self.find_tables_in_date_range(start_date, end_date)
        
        if not tables:
            print("No tables found for the specified date range")
            return pd.DataFrame()
        
        all_data = []
        columns_str = ', '.join(columns) if columns else '*'
        
        for table in tables:
            try:
                query = f"""
                SELECT {columns_str}
                FROM {table}
                WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
                """
                
                # Add additional filters
                if filters:
                    for column, values in filters.items():
                        if isinstance(values, list) and len(values) == 2:
                            query += f" AND {column} BETWEEN {values[0]} AND {values[1]}"
                        elif isinstance(values, (str, int, float)):
                            query += f" AND {column} = '{values}'"
                
                query += " ORDER BY timestamp"
                
                df = self._conn.execute(query).fetchdf()
                if not df.empty:
                    all_data.append(df)
                    print(f"✓ Loaded {len(df)} records from {table}")
                
            except Exception as e:
                print(f"⚠ Error loading data from {table}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Total records loaded: {len(combined_df)}")
        return combined_df
    
    def get_spot_prices_for_range(self, start_date: str, end_date: str) -> pd.Series:
        """
        Get spot prices for a specific date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.Series: Series with timestamp index and spot prices
        """
        df = self.fetch_data_for_date_range(
            start_date, end_date, 
            columns=['timestamp', 'spot_price']
        )
        
        if df.empty:
            return pd.Series()
        
        # Remove duplicates and set index
        df = df.drop_duplicates(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df['spot_price'].dropna()
    
    def get_options_data_for_range(self, start_date: str, end_date: str,
                                   option_type: str = None,
                                   expiry_range: Tuple[int, int] = None) -> pd.DataFrame:
        """
        Get options data for a specific date range with filters
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        option_type : str, optional
            'CE' for call options, 'PE' for put options
        expiry_range : Tuple[int, int], optional
            Time to expiry range in days
            
        Returns:
        --------
        pd.DataFrame: Filtered options data
        """
        filters = {}
        if expiry_range:
            filters['Time_to_expiry'] = expiry_range
        
        df = self.fetch_data_for_date_range(start_date, end_date, filters=filters)
        
        if df.empty:
            return df
        
        # Filter by option type
        if option_type:
            df = df[df['ticker'].str.contains(option_type, na=False)]
        
        return df
    
    def get_summary_stats_for_range(self, start_date: str, end_date: str) -> Dict:
        """
        Get summary statistics for a date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        Dict: Summary statistics
        """
        tables = self.find_tables_in_date_range(start_date, end_date)
        
        if not tables:
            return {}
        
        total_records = 0
        unique_tickers = set()
        min_spot = float('inf')
        max_spot = float('-inf')
        
        for table in tables:
            try:
                query = f"""
                SELECT 
                    COUNT(*) as records,
                    COUNT(DISTINCT ticker) as tickers,
                    MIN(spot_price) as min_spot,
                    MAX(spot_price) as max_spot
                FROM {table}
                WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
                """
                
                result = self._conn.execute(query).fetchone()
                total_records += result[0]
                
                if result[2] is not None:
                    min_spot = min(min_spot, result[2])
                if result[3] is not None:
                    max_spot = max(max_spot, result[3])
                
            except Exception as e:
                print(f"Error getting stats from {table}: {e}")
                continue
        
        return {
            'date_range': f"{start_date} to {end_date}",
            'tables_processed': len(tables),
            'total_records': total_records,
            'spot_price_range': f"{min_spot:.2f} - {max_spot:.2f}" if min_spot != float('inf') else "N/A"
        }

# Example usage and utility functions
def quick_date_query(db_path: str, start_date: str, end_date: str, table_name: str = None):
    """
    Quick utility function for simple date range queries
    
    Parameters:
    -----------
    db_path : str
        Path to the database
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    table_name : str, optional
        Specific table name to query
        
    Returns:
    --------
    pd.DataFrame: Query results
    """
    conn = duckdb.connect(db_path)
    
    try:
        if table_name:
            query = f"""
            SELECT timestamp, ticker, spot_price, expiry_date, Time_to_expiry
            FROM {table_name}
            WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
            ORDER BY timestamp
            """
        else:
            # This would need to be adapted based on your specific table structure
            query = f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main' 
              AND table_name LIKE 'nifty_%'
            """
        
        return conn.execute(query).fetchdf()
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Example usage
    db_path = "nifty_1min_desiquant.duckdb"
    
    # Initialize fetcher
    fetcher = DateRangeDataFetcher(db_path)
    
    try:
        # Example 1: Get summary for a date range
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        print("=== Summary Statistics ===")
        stats = fetcher.get_summary_stats_for_range(start_date, end_date)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Example 2: Get spot prices
        print("\n=== Spot Prices ===")
        spot_prices = fetcher.get_spot_prices_for_range(start_date, end_date)
        print(f"Spot price data points: {len(spot_prices)}")
        if not spot_prices.empty:
            print(f"Price range: {spot_prices.min():.2f} - {spot_prices.max():.2f}")
        
        # Example 3: Get options data with filters
        print("\n=== Options Data ===")
        options_data = fetcher.get_options_data_for_range(
            start_date, end_date,
            option_type="CE",  # Call options only
            expiry_range=(7, 30)  # 7-30 days to expiry
        )
        print(f"Options records: {len(options_data)}")
        
    finally:
        fetcher.close() 
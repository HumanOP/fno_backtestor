from typing import List, Dict, Tuple, Any
import pandas as pd
from datetime import datetime

class TradeManager:
    """
    Manages the execution of trades and trade book maintenance.
    
    Responsible for:
      - Placing new trade orders
      - Closing open positions
      - Tracking active trades
      - Building and maintaining the trade book
    """
    
    def __init__(self) -> None:
        """Initialize the TradeManager with an empty tradebook."""
        print("TradeManager initialized")
        self.tradebook: List[Dict[str, Any]] = []
        self.tradebook_built: bool = False
        
    def place_order(self, entry_data: Dict[str, Any]) -> None:
        """
        Place a new trade order and add it to the tradebook.
        """
        trade = {
            'strategy_id': entry_data['strategy_id'],
            'position_id': entry_data['position_id'],
            'leg_id': entry_data['leg_id'],
            'symbol': entry_data['symbol'],
            'entry_date': entry_data['entry_date'],
            'entry_time': entry_data['entry_time'],
            'exit_date': None,
            'exit_time': None,
            'entry_price': entry_data['entry_price'],
            'exit_price': None,
            'qty': entry_data['qty'],
            'entry_type': entry_data['entry_type'],
            'entry_spot': entry_data['entry_spot'],
            'exit_spot': None,
            'stop_loss': entry_data['stop_loss'],
            'take_profit': entry_data['take_profit'],
            'entry_reason': entry_data['entry_reason'],
            'exit_reason': None,
            'status': 'open',
        }
        self.tradebook.append(trade)
        
    def square_off(self, trade_index: int, exit_data: Dict[str, Any]) -> None:
        """
        Close an open position.
        """
        self.tradebook[trade_index]['status'] = 'closed'
        self.tradebook[trade_index]['exit_date'] = exit_data["exit_date"]
        self.tradebook[trade_index]['exit_time'] = exit_data["exit_time"]
        self.tradebook[trade_index]['exit_price'] = exit_data["exit_price"]
        self.tradebook[trade_index]['exit_spot'] = exit_data["exit_spot"]
        self.tradebook[trade_index]['exit_reason'] = exit_data["exit_reason"]
    
    def active_trades(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Return all active trades.
        """
        return [(index, trade) for index, trade in enumerate(self.tradebook) if trade['status'] == 'open']

    def build_tradebook(self) -> pd.DataFrame:
        """
        Convert tradebook to DataFrame and add calculated columns.
        """
        if self.tradebook_built and isinstance(self.tradebook, pd.DataFrame) and not self.tradebook.empty:
            print("Returning existing tradebook DataFrame")
            return self.tradebook

        if isinstance(self.tradebook, list) and not self.tradebook:
            print("Tradebook is empty. No trades to build.")
            return pd.DataFrame()
        elif isinstance(self.tradebook, pd.DataFrame) and self.tradebook.empty:
            print("Tradebook DataFrame is empty. No trades to build.")
            return pd.DataFrame()

        if isinstance(self.tradebook, list):
            df = pd.DataFrame(self.tradebook)
        else:
            df = self.tradebook.copy()

        expected_cols = ['symbol', 'entry_date', 'exit_date', 'entry_time', 'exit_time']
        if all(col in df.columns for col in expected_cols):
            print("Raw tradebook before processing:")
            print(df[expected_cols].head())
        else:
            print(f"Missing expected columns: {[col for col in expected_cols if col not in df.columns]}")
            print("Available columns:", df.columns.tolist())

        def parse_expiry(symbol):
            if not isinstance(symbol, str) or len(symbol) < 14:
                print(f"Invalid symbol format: {symbol}")
                return pd.NaT
            try:
                return datetime.strptime(symbol[-14:-7], "%d%b%y")
            except (ValueError, TypeError) as e:
                print(f"Error parsing expiry from symbol {symbol}: {e}")
                return pd.NaT

        df["expiry"] = pd.to_datetime(df["symbol"].apply(parse_expiry), errors='coerce')
        df["instrument_type"] = df["symbol"].apply(lambda x: x[-2:] if isinstance(x, str) else None)
        df["strike"] = df["symbol"].apply(lambda x: x[-7:-2] if isinstance(x, str) else None)

        df["entry_date"] = pd.to_datetime(df["entry_date"], errors='coerce')
        closed_trades = df['status'] == 'closed'
        df.loc[closed_trades, "exit_date"] = pd.to_datetime(df.loc[closed_trades, "exit_date"], errors='coerce')

        print(f"entry_date null count: {df['entry_date'].isna().sum()}")
        print(f"exit_date null count (closed trades): {df.loc[closed_trades, 'exit_date'].isna().sum()}")
        print(f"expiry null count: {df['expiry'].isna().sum()}")

        if df['entry_date'].isna().any():
            print("Rows with null entry_date:")
            print(df[df['entry_date'].isna()][['symbol', 'entry_date', 'entry_time']])
        if df.loc[closed_trades, 'exit_date'].isna().any():
            print("Rows with null exit_date (closed trades):")
            print(df.loc[closed_trades & df['exit_date'].isna()][['symbol', 'exit_date', 'exit_time']])
        if df['expiry'].isna().any():
            print("Rows with null expiry:")
            print(df[df['expiry'].isna()][['symbol', 'expiry']])

        df["entry_datetime"] = df["entry_date"] + pd.to_timedelta(df["entry_time"].astype(str), errors='coerce')
        df.loc[closed_trades, "exit_datetime"] = (
            df.loc[closed_trades, "exit_date"] + 
            pd.to_timedelta(df.loc[closed_trades, "exit_time"].astype(str), errors='coerce')
        )

        df["pnl"] = df.apply(
            lambda row: (row["exit_price"] - row["entry_price"] if row["entry_type"] == "BUY" 
                         else row["entry_price"] - row["exit_price"])
            if row["status"] == "closed" and pd.notna(row["exit_price"]) and pd.notna(row["entry_price"]) else None,
            axis=1
        )
        self.tradebook = df
        self.tradebook_built = True
        return df
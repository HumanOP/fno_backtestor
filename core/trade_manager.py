from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import pandas as pd


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
        self.tradebook: List[Dict[str, Any]] = []
        self.tradebook_built: bool = False
        
    def place_order(self, entry_data: Dict[str, Any]) -> None:
        """
        Place a new trade order and add it to the tradebook.
        
        Args:
            entry_data: Dictionary containing trade entry details
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
            'entry_type': entry_data['entry_type'],     # buy/sell
            'entry_spot': entry_data['entry_spot'], 
            'exit_spot': None,
            'stop_loss': entry_data['stop_loss'],
            'take_profit': entry_data['take_profit'],
            'entry_reason': entry_data['entry_reason'], # reason for entry
            'exit_reason': None,
            'status': 'open',                           # open/closed
        }
        self.tradebook.append(trade)
        
    def square_off(self, trade_index: int, exit_data: Dict[str, Any]) -> None:
        """
        Close an open position.
        
        Args:
            trade_index: Index of the trade in tradebook
            exit_data: Dictionary containing trade exit details
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
        
        Returns:
            List of tuples containing trade index and trade details
        """
        return [(index, trade) for index, trade in enumerate(self.tradebook) if trade['status'] == 'open']

    def build_tradebook(self) -> pd.DataFrame:
        """
        Convert tradebook to DataFrame and add calculated columns.
        
        Returns:
            DataFrame containing processed tradebook
        """
        self.tradebook = pd.DataFrame(self.tradebook)
        self.tradebook["expiry"] = self.tradebook["symbol"].apply(lambda x: datetime.strptime(x[-14:-7], "%d%b%y"))
        self.tradebook["instrument_type"] = self.tradebook["symbol"].apply(lambda x: x[-2:])
        self.tradebook["strike"] = self.tradebook["symbol"].apply(lambda x: x[-7:-2])
        self.tradebook["entry_date"] = pd.to_datetime(self.tradebook["entry_date"])
        self.tradebook["exit_date"] = pd.to_datetime(self.tradebook["exit_date"])
        self.tradebook["entry_day_name"] = self.tradebook["entry_date"].dt.day_name()
        self.tradebook["exit_day_name"] = self.tradebook["exit_date"].dt.day_name()
        self.tradebook["entry_datetime"] = pd.to_datetime(self.tradebook["entry_date"]) + pd.to_timedelta(self.tradebook["entry_time"].astype(str))
        self.tradebook["exit_datetime"] = pd.to_datetime(self.tradebook["exit_date"]) + pd.to_timedelta(self.tradebook["exit_time"].astype(str))
        self.tradebook["dte"] = (self.tradebook["expiry"] - self.tradebook["entry_date"]).dt.days
        self.tradebook["pnl"] = self.tradebook.apply(
            lambda row: row["exit_price"] - row["entry_price"] if row["entry_type"] == "BUY" else row["entry_price"] - row["exit_price"],
            axis=1
        )
        self.tradebook_built = True
        return self.tradebook

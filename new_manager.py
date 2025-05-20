from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd

@dataclass
class Trade:
    strategy_id: str
    position_id: str
    leg_id: str
    symbol: str
    entry_date: str
    entry_time: str
    entry_price: float
    qty: int
    entry_type: str
    entry_spot: float
    stop_loss: float
    take_profit: float
    entry_reason: str

    exit_date: Optional[str] = None
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_spot: Optional[float] = None
    exit_reason: Optional[str] = None
    status: str = "open"

    def close(self, exit_data: dict):
        self.exit_date = exit_data["exit_date"]
        self.exit_time = exit_data["exit_time"]
        self.exit_price = exit_data["exit_price"]
        self.exit_spot = exit_data["exit_spot"]
        self.exit_reason = exit_data["exit_reason"]
        self.status = "closed"

    def is_open(self) -> bool:
        return self.status == "open"


class Order:
    def __init__(self):
        self.trades: List[Trade] = []

    def place_order(self, entry_data: dict):
        trade = Trade(**entry_data)
        self.trades.append(trade)

    def square_off(self, index: int, exit_data: dict):
        self.trades[index].close(exit_data)

    def get_active_trades(self) -> List[tuple]:
        return [(i, t) for i, t in enumerate(self.trades) if t.is_open()]
    

class Position:
    def __init__(self, trades: List[Trade]):
        self.trades = trades

    def build_tradebook(self) -> pd.DataFrame:
        trade_dicts = [t.__dict__ for t in self.trades]
        df = pd.DataFrame(trade_dicts)

        df["expiry"] = df["symbol"].apply(lambda x: datetime.strptime(x[-14:-7], "%d%b%y"))
        df["instrument_type"] = df["symbol"].apply(lambda x: x[-2:])
        df["strike"] = df["symbol"].apply(lambda x: x[-7:-2])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df["entry_day_name"] = df["entry_date"].dt.day_name()
        df["exit_day_name"] = df["exit_date"].dt.day_name()
        df["entry_datetime"] = df["entry_date"] + pd.to_timedelta(df["entry_time"])
        df["exit_datetime"] = df["exit_date"] + pd.to_timedelta(df["exit_time"])
        df["dte"] = (df["expiry"] - df["entry_date"]).dt.days
        df["pnl"] = df.apply(
            lambda row: row["exit_price"] - row["entry_price"] if row["entry_type"] == "BUY"
            else row["entry_price"] - row["exit_price"],
            axis=1
        )

        return df


class Strategy:
    def __init__(self):
        self.order = Order()

    def place_trade(self, entry_data: dict):
        self.order.place_order(entry_data)

    def close_trade(self, index: int, exit_data: dict):
        self.order.square_off(index, exit_data)

    def active_trades(self):
        return self.order.get_active_trades()

    def tradebook_df(self):
        position = Position(self.order.trades)
        return position.build_tradebook()

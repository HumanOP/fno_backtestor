import functools
import os
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from copy import copy
from datetime import datetime, date as DateObject # Added DateObject
from functools import partial # lru_cache removed
from math import copysign
from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import webbrowser

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq

import duckdb
import pandas as pd
from typing import Dict, Optional, List

class _Data:
    def __init__(self, db_path: str):
        self._conn = None
        self._table_names: List[str] = []                     # List of table names in the database
        self._ticker_map: Dict[str, pd.DataFrame] = {}        # Ticker-wise grouped data
        self._data_df_template: Optional[pd.DataFrame] = None # Template for data
        self._spot: Optional[pd.DataFrame] = None
        self._time_to_expiry: Optional[List] = None
        self._leg_data_map: Dict[str, pd.Series] = {}         # Contract: Series (1 row time-aligned)
        self.connect_db(db_path)                              # Initialize connection and load table names

    def connect_db(self, db_path: str):
        """
        Connect to the database.
        """
        if self._conn is not None:
            self._conn.close()
        self._conn = duckdb.connect(db_path)
        self._table_names = self._conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchdf()["table_name"]
        self._ticker_map.clear()  # Clear ticker map when reconnecting
        self._spot = None
        self._leg_data_map.clear()

    def load_table(self, table_name: str):
        df = self._conn.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
        df["expiry_date"] = df["expiry_date"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
        self._time_to_expiry = sorted(df["Time_to_expiry"].unique())
        df.set_index("timestamp", inplace=True)
        self._build_ticker_map(df)
        self._data_df_template = df.iloc[0:0]
        self._spot = df[["spot_price"]][~df.index.duplicated(keep="first")]
        return df

    def _build_ticker_map(self, df):
        self._ticker_map = {ticker: group for ticker, group in df.groupby("ticker", sort=False)}

    # def get_leg_data(self, contract: str, timestamp: pd.Timestamp) -> Optional[pd.Series]:
    #     if contract not in self._ticker_map:
    #         return None
    #     df = self._ticker_map[contract]
    #     asof_time = df.index.asof(timestamp)
    #     if pd.isna(asof_time):
    #         return None
    #     return df.loc[asof_time]

    # def build_leg_data_map(self, legs: Dict[str, Dict], timestamp: pd.Timestamp):
    #     self._leg_data_map.clear()
    #     for leg_id, leg in legs.items():
    #         contract = leg["contract"]
    #         leg_data = self.get_leg_data(contract, timestamp)
    #         if leg_data is not None:
    #             self._leg_data_map[leg_id] = leg_data

    @property
    def ticker_map(self):
        return self._ticker_map

    @property
    def leg_data(self):
        return self._leg_data_map

    def close(self):
        self._conn.close()

    def __repr__(self):
        # keys = list(self._cache.keys())
        # return f"<Data cached_tables={keys} current_time_range={[self._current_index[0], self._current_index[-1]] if self._current_index is not None else None}>"
        return str(self._table_names)

class _OptionsData:
    """
    Wrapper for the current day's options chain data.
    """
    def __init__(self, initial_df: Optional[pd.DataFrame] = None):
        self._df = initial_df if initial_df is not None else pd.DataFrame()
        self.index = self._df.index # Assuming options data might have its own index

    def _set_data(self, df: pd.DataFrame):
        self._df = df
        self.index = self._df.index

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key: str) -> pd.Series: # Access columns like 'Last', 'Bid', 'Ask'
        if key not in self._df.columns:
            raise KeyError(f"Column '{key}' not in current options data.")
        return self.df[key]

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy() # Return a copy to prevent modification

    def get_contract_data(self, ticker: str) -> Optional[pd.Series]:
        # Assuming 'OptionSymbol' is a column in the options_df
        if 'OptionSymbol' not in self._df.columns:
            # Try index if OptionSymbol is the index
            if ticker in self._df.index:
                 return self._df.loc[ticker]
            warnings.warn("Options data DataFrame does not have 'OptionSymbol' column or index.")
            return None
        
        contract_row = self._df[self._df['OptionSymbol'] == ticker]
        if not contract_row.empty:
            return contract_row.iloc[0]
        return None


class Strategy(ABC):
    def __init__(self, broker: '_Broker', spot_data: _SpotData, params: dict):
        self._broker: _Broker = broker
        self._spot_data: _SpotData = spot_data # For current spot bar
        self.options_data: _OptionsData = _OptionsData() # Will be updated by Backtest.run()
        self._params = self._check_params(params)
        self._records = {}
        self._data_index = spot_data.index.copy() # Main timeline from spot
        self._start_on_day = 0


    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    @abstractmethod
    def init(self):
        """Initialize the strategy. Declare spot indicators, etc."""
        pass

    @abstractmethod
    def next(self):
        """
        Called for each spot data bar AFTER options_data for the current day is loaded.
        Access `self.spot_data` for current spot bar.
        Access `self.options_data` for current day's option chain.
        """
        pass

    def buy(self, *,
                   ticker: str,
                   quantity: float, # Number of contracts
                   limit: Optional[float] = None,
                   # stop: Optional[float] = None, # REMOVED
                   tag: object = None):
        assert quantity > 0, "Quantity for buying options must be positive"
        return self._broker.new_order(ticker, quantity, limit, tag)

    def sell(self, *,
                    ticker: str,
                    quantity: float, # Number of contracts
                    limit: Optional[float] = None,
                    # stop: Optional[float] = None, # REMOVED
                    tag: object = None):
        # Negative quantity for selling (to open short or close long)
        assert quantity > 0, "Quantity for selling options must be positive (use negative for broker call)"
        return self._broker.new_order(ticker, -quantity, limit, tag)

    # --- record() can be kept for custom logging ---
    def record(self, name: str = None, plot: bool = True, overlay: bool = None, color: str = None, scatter: bool = False, **kwargs):
        for k, v in kwargs.items():
            current_time = self._broker.now # Spot data timestamp
            if isinstance(v, dict) or isinstance(v, pd.Series):
                v = dict(v)
                if k not in self._records:
                    self._records[k] = pd.DataFrame(index=self._data_index, columns=v.keys())
                self._records[k].loc[current_time, list(v.keys())] = list(v.values())
            else:
                if k not in self._records:
                    self._records[k] = pd.Series(index=self._data_index)
                self._records[k].loc[current_time] = v # Use .loc for safety
            
            # Store plotting attributes if needed later, but actual plotting needs rework
            if not hasattr(self._records[k], 'attrs'): self._records[k].attrs = {}
            self._records[k].name = name or k # Ensure name is set
            self._records[k].attrs.update({'name': name or k, 'plot': plot, 'overlay': overlay,
                                           'color': color, 'scatter': scatter})


    @property
    def equity(self) -> float:
        return self._broker.equity()

    @property
    def spot_data(self) -> _SpotData: # Renamed from data
        return self._spot_data

    def position(self, ticker: str) -> 'Position': # Now takes ticker
        return self._broker.positions.get(ticker, Position(self._broker, ticker, 0)) # Return empty if not found

    @property
    def orders(self) -> 'List[Order]':
        return self._broker.orders

    # trades() and closed_trades() now refer to option trades
    def trades(self, ticker: str = None) -> 'Tuple[Trade, ...]':
        if ticker:
            return tuple(self._broker.trades.get(ticker, []))
        return tuple(trade for trades_list in self._broker.trades.values() for trade in trades_list)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        return tuple(self._broker.closed_trades)

    def start_on_day(self, n: int):
        assert 0 <= n < len(self._spot_data), f"day must be within [0, {len(self._spot_data)-1}]"
        self._start_on_day = n
    

class Position:
    def __init__(self, broker: '_Broker', ticker: str, initial_size: float = 0): # Added initial_size
        self.__broker = broker
        self.__ticker = ticker
        # Position size is now managed directly by summing trades in _Broker,
        # or _Broker can update a size attribute here.
        # For simplicity, let's make Position mostly a query interface.

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in number of contracts. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades.get(self.__ticker, []))

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position."""
        return sum(trade.pl for trade in self.__broker.trades.get(self.__ticker, []))

    @property
    def pl_pct(self) -> float:
        """Profit/loss in percent. This is harder for options without average entry cost.
           Could be (current_value - cost_basis) / cost_basis.
           For simplicity, we might omit this or calculate it based on an average entry price.
        """
        # This needs a clear definition for options.
        # One way: Sum of P/L of open trades / Sum of initial values of open trades
        current_val = sum(trade.value for trade in self.__broker.trades.get(self.__ticker, []))
        entry_val = sum(trade.entry_price * abs(trade.size) * self.__broker._option_multiplier
                        for trade in self.__broker.trades.get(self.__ticker, [])
                        if trade.size != 0) # abs because short sales also have entry value
        if entry_val == 0: return np.nan
        # P/L for shorts is (entry_price - current_price), for longs (current_price - entry_price)
        # The self.pl already accounts for this.
        return self.pl / entry_val if entry_val else np.nan


    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    def close(self, portion: float = 1.):
        """Close portion of position by closing `portion` of each active trade."""
        # This needs to iterate through trades of this specific ticker
        for trade in list(self.__broker.trades.get(self.__ticker, [])): # Iterate copy
            trade.close(portion)

    def __repr__(self):
        num_trades = len(self.__broker.trades.get(self.__ticker, []))
        return f'<Position: {self.__ticker} Size={self.size} ({num_trades} trades)>'


class _OutOfMoneyError(Exception):
    pass


class Order:
    def __init__(self, broker: '_Broker',
                 ticker: str, # Changed from ticker
                 size: float, # Number of contracts
                 entry_time: datetime = None,
                 tag: object = None,
                 reason: object = None): 
        self.__broker = broker
        self.__ticker = ticker
        assert size != 0
        self.__size = size # Positive for buy, negative for sell
        self.__entry_time = entry_time # Timestamp from spot data
        self.__tag = tag
        self.__reason = reason

    def _replace(self, **kwargs): # Keep for internal use
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return f'<Order {self.__ticker} Size={self.__size} Time={self.__entry_time} Tag={self.__tag}>'


    def cancel(self):
        """Cancel the order."""
        if self in self.__broker.orders: # Check if still in list
            self.__broker.orders.remove(self)

    @property
    def ticker(self) -> str: # Renamed
        return self.__ticker

    @property
    def size(self) -> float:
        return self.__size
    
    @size.setter
    def size(self, size):
        self.__size = size

    @property
    def tag(self):
        return self.__tag
    
    @property
    def reason(self) -> Optional['Order']:
        return self.__reason

    @reason.setter
    def reason(self, value: Optional['Order']):
        self.__reason = value

    @property
    def is_long(self): # True if this order is to buy
        return self.__size > 0

    @property
    def is_short(self): # True if this order is to sell
        return self.__size < 0

    @property
    def entry_time(self) -> datetime: # Spot data timestamp
        return self.__entry_time


class Trade:
    def __init__(self, broker: '_Broker', ticker: str, size: int,
                 entry_price: float, entry_bar_index: int, # entry_bar_index from spot data
                 tag, reason: Optional['Order'] = None):
        self.__broker = broker
        self.__ticker = ticker
        self.__size = size # Number of contracts, negative for short
        self.__entry_price = entry_price # Price per unit (premium)
        self.__exit_price: Optional[float] = None
        self.__entry_bar_index: int = entry_bar_index # Index in spot_data
        self.__exit_bar_index: Optional[int] = None
        self.__tag = tag
        self.__reason = reason # From closing order

    def __repr__(self):
        return (f'<Trade {self.__ticker} Size={self.__size} EntryBar={self.__entry_bar_index} ExitBar={self.__exit_bar_index or ""} '
                f'EntryPrice={self.__entry_price:.2f} ExitPrice={self.__exit_price or "":.2f} PnL={self.pl:.2f} Tag={self.__tag}>')


    def _replace(self, **kwargs): # Keep
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs): # Keep
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1., finalize=False): # finalize is for end of backtest
        assert 0 < portion < 1.000000000000001, "portion must be a fraction between 0 and 1"
        # Size to close is opposite of trade size
        size_to_close = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)

        order = Order(self.__broker, self.__ticker, size_to_close,
                      parent_trade=self, entry_time=self.__broker.now, tag=self.__tag)
        if finalize:
            return order # For _Broker.finalize()
        else:
            self.__broker.orders.insert(0, order) # Prioritize closing orders


    @property
    def ticker(self): # Renamed
        return self.__ticker

    @property
    def size(self):
        return self.__size

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        return self.__exit_price

    @property
    def entry_bar(self) -> int: # Index in spot_data
        return self.__entry_bar_index

    @property
    def exit_bar(self) -> Optional[int]: # Index in spot_data
        return self.__exit_bar_index

    @property
    def tag(self):
        return self.__tag
    
    @property
    def reason(self) -> Optional['Order']:
        return self.__reason

    @reason.setter
    def reason(self, value: Optional['Order']):
        self.__reason = value
        

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]: # Timestamp from spot_data
        return self.__broker._spot_data.index[self.__entry_bar_index]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        if self.__exit_bar_index is None:
            return None
        return self.__broker._spot_data.index[self.__exit_bar_index]

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return not self.is_long

    @property
    def pl(self):
        """Trade profit (positive) or loss (negative) in cash units."""
        if self.__exit_price is None: # Mark-to-market P&L for open trade
            current_option_price = self.__broker.get_option_last_price(self.__ticker)
            if current_option_price is None: return 0 # Not in current chain / no price
            price_diff = current_option_price - self.__entry_price
        else: # Realized P&L for closed trade
            price_diff = self.__exit_price - self.__entry_price

        return self.__size * price_diff * self.__broker._option_multiplier

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent of initial premium paid/received."""
        if self.__entry_price == 0: return np.nan # Avoid division by zero

        if self.__exit_price is None:
            current_option_price = self.__broker.get_option_last_price(self.__ticker)
            if current_option_price is None: return np.nan
            price_diff = current_option_price - self.__entry_price
        else:
            price_diff = self.__exit_price - self.__entry_price

        # For longs, pl_pct = price_diff / entry_price
        # For shorts, pl_pct = -price_diff / entry_price (since entry_price was received)
        # Simpler: (pnl / (abs(size) * entry_price * multiplier))
        initial_value_per_contract = self.__entry_price
        pnl_per_contract = price_diff * copysign(1, self.__size) # To align with P&L direction
        
        return pnl_per_contract / initial_value_per_contract if initial_value_per_contract else np.nan


    @property
    def value(self):
        """Trade current market value in cash (contracts * price * multiplier)."""
        price = self.__exit_price or self.__broker.get_option_last_price(self.__ticker)
        if price is None: return 0 # No current price available
        return self.__size * price * self.__broker._option_multiplier


class _Broker:
    def __init__(self, *,
                 data: _Data,
                 cash: float,
                 commission_per_contract: float, # Changed commission model
                 option_multiplier: int,
                 ):
        assert 0 < cash, f"cash should be > 0, is {cash}"
        assert commission_per_contract >= 0, "commission_per_contract should be >= 0"
        assert option_multiplier > 0, "option_multiplier must be positive"

        self._data = data
        self._cash = cash
        self._commission_per_contract = commission_per_contract
        self._option_multiplier = option_multiplier

        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {} # Updated as trades occur


    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self._cash:.2f}, Equity={self.equity():.2f} ({active_trades} open trades)>'

    def new_order(self,
                  ticker: str,
                  size: float, # Number of contracts
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None): # trade is parent_trade for closing orders
        assert size != 0, "Order size must be non-zero"

        order = Order(self, ticker, size, parent_trade=trade, entry_time=self.now, tag=tag)

        if trade: # This is a closing order for an existing trade
            self.orders.insert(0, order)
        else: # New opening order
            # if self._exclusive_orders: # Not typically used this way for options
            #    pass
            self.orders.append(order)
        return order

    def get_option_contract_details(self, ticker: str) -> Optional[pd.Series]:
        return self._current_options_data.get_contract_data(ticker)

    def get_option_execution_price(self, ticker: str, is_buy_order: bool, limit_price: Optional[float]) -> Optional[float]:
        """
        Determines execution price for an option.
        Simple logic: if limit, use limit if marketable. Else, use 'Last' or 'Mid'.
        A real system uses NBBO and considers liquidity.
        """
        contract = self.get_option_contract_details(ticker)
        if contract is None:
            # warnings.warn(f"No data for option {ticker} in current chain. Order cannot fill.")
            return None

        # Prioritize 'Last', then 'Ask' for buy, 'Bid' for sell, then calculated Mid
        fill_price = None
        last_price = contract.get('Last')
        ask_price = contract.get('Ask')
        bid_price = contract.get('Bid')

        if is_buy_order:
            market_price = None
            if pd.notna(ask_price) and ask_price > 0: market_price = ask_price
            elif pd.notna(last_price) and last_price > 0 : market_price = last_price
            elif pd.notna(bid_price) and bid_price > 0: market_price = bid_price # Less likely fill for buy
            else: # Market Buy
                fill_price = market_price
        else: # Sell order
            market_price = None
            if pd.notna(bid_price) and bid_price > 0: market_price = bid_price
            elif pd.notna(last_price) and last_price > 0: market_price = last_price
            elif pd.notna(ask_price) and ask_price > 0: market_price = ask_price # Less likely fill for sell
            else: # Market Sell
                fill_price = market_price
        
        return fill_price if pd.notna(fill_price) and fill_price > 0 else None


    def get_option_last_price(self, ticker: str) -> Optional[float]:
        """ Gets the last known price for an option, typically for MTM P&L. """
        contract = self.get_option_contract_details(ticker)
        if contract is None: return None
        
        # Prefer 'Last', then 'Mid' (Bid+Ask)/2, then Bid or Ask if only one exists
        last = contract.get('Last')
        if pd.notna(last) and last > 0: return last
        
        bid = contract.get('Bid')
        ask = contract.get('Ask')
        if pd.notna(bid) and bid > 0 and pd.notna(ask) and ask > 0:
            return (bid + ask) / 2
        if pd.notna(bid) and bid > 0: return bid
        if pd.notna(ask) and ask > 0: return ask
        return None


    def equity(self, ticker: str = None) -> float: # MTM equity
        if ticker:
            return sum(trade.value for trade in self.trades.get(ticker, []))
        else:
            mtm_value_of_open_positions = sum(
                trade.value for trades_list in self.trades.values() for trade in trades_list
            )
            return self._cash + mtm_value_of_open_positions

    @property
    def margin_available(self) -> float:
        # Simplified: Equity minus initial margin for any short positions (if allowed).
        # For now, assume all cash is available if no complex margin.
        # If selling options: self.equity() - margin_used_for_shorts
        # A very basic model:
        value_of_long_options = sum(
            abs(trade.entry_price * trade.size * self._option_multiplier)
            for trades_list in self.trades.values()
            for trade in trades_list if trade.is_long
        )
        # This is not standard margin, but represents cash tied up if we assume full payment for longs.
        # True margin_available is more complex. For now, let's use cash as a proxy for buying power.
        return self._cash # Simplification: assumes buying power is current cash.

    @property
    def all_trades(self) -> List[Trade]: # Active trades
        return [trade for trades_list in self.trades.values() for trade in trades_list]

    @property
    def now(self): # Current spot data timestamp
        return self._spot_data.now if len(self._spot_data) > 0 else None


    def handle_expirations(self, current_date: DateObject):
        """
        Handle option expirations.
        current_date is the date part of the current spot bar.
        """
        symbols_to_remove_from_trades = []
        for ticker, trade_list in list(self.trades.items()): # Iterate copy of items
            if not trade_list:
                symbols_to_remove_from_trades.append(ticker)
                continue

            # Get expiry from the first trade (assume all trades for a symbol have same expiry)
            # This requires ticker to be parsable or options_data to have Expiry
            contract_details = self.get_option_contract_details(ticker)
            if not contract_details or 'Expiry' not in contract_details:
                # warnings.warn(f"Cannot determine expiry for {ticker}. Skipping expiration check.")
                continue

            try:
                expiry_date = contract_details['Expiry']
                if isinstance(expiry_date, (pd.Timestamp, datetime)):
                    expiry_date = expiry_date.date()
                # Add more robust parsing if expiry_date is string
            except Exception as e:
                # warnings.warn(f"Could not parse expiry for {ticker}: {e}. Skipping.")
                continue

            if current_date >= expiry_date:
                # Option expired or is expiring today
                # print(f"Option {ticker} expiring on {expiry_date} (current_date: {current_date})")
                underlying_price_at_expiry = self._spot_data.Close[-1] # Spot close on expiry day

                for trade in list(trade_list): # Iterate copy of trade_list
                    intrinsic_value = 0
                    if contract_details['Type'].upper() == 'CALL':
                        intrinsic_value = max(0, underlying_price_at_expiry - contract_details['Strike'])
                    elif contract_details['Type'].upper() == 'PUT':
                        intrinsic_value = max(0, contract_details['Strike'] - underlying_price_at_expiry)
                    
                    exit_price_at_expiry = intrinsic_value
                    
                    # Close the trade at this exit_price
                    self._close_trade(trade, exit_price_at_expiry, len(self._spot_data) - 1, reason="Expired")
                
                if not self.trades.get(ticker): # If all trades for this symbol are closed
                    symbols_to_remove_from_trades.append(ticker)
        
        for sym in symbols_to_remove_from_trades:
            if sym in self.trades and not self.trades[sym]: # double check if empty
                del self.trades[sym]
                if sym in self.positions:
                    del self.positions[sym]


    def finalize(self): # Close all open option positions at last available price
        current_bar_idx = len(self._spot_data) - 1
        for ticker in list(self.trades.keys()):
            for trade in list(self.trades.get(ticker, [])): # Iterate copy
                last_price = self.get_option_last_price(ticker)
                if last_price is None: # If no price, assume worthless (simplification)
                    # A better approach might be to use intrinsic value if near expiry
                    contract = self.get_option_contract_details(ticker)
                    underlying_close = self._spot_data.Close[-1]
                    if contract and contract['Type'].upper() == 'CALL':
                        last_price = max(0, underlying_close - contract['Strike'])
                    elif contract and contract['Type'].upper() == 'PUT':
                        last_price = max(0, contract['Strike'] - underlying_close)
                    else:
                        last_price = 0 # Fallback
                
                # print(f"Finalizing trade for {ticker} at price {last_price}")
                self._close_trade(trade, last_price, current_bar_idx, reason="EndOfTest")


    def next(self): # Called for each spot bar
        # Log equity
        current_bar_idx = len(self._spot_data) -1
        self._equity[current_bar_idx] = self.equity()

        if self.equity() <= 0:
            # If equity is negative, set all to 0 and stop the simulation
            # For options, equity can go negative with short positions.
            # The definition of "out of money" might be different (e.g., margin call).
            # For simplicity, we'll keep the "stop if equity <= 0" rule.
            if self._equity[current_bar_idx] <=0:
                warnings.warn("Equity is zero or negative. Stopping simulation.", UserWarning)
                self.finalize() # Close all positions
                self._cash = 0
                self._equity[current_bar_idx:] = 0 # Fill future equity with 0
                raise _OutOfMoneyError
        
        self._process_orders()


    def _process_orders(self):
        current_bar_idx = len(self._spot_data) - 1 # Index in the full spot_data

        for order in list(self.orders): # Iterate a copy
            if order not in self.orders: continue # Already processed/canceled

            # Option orders are typically GTD (Good 'Til Day) or GTC.
            # For simplicity, assume orders persist until filled/canceled or end of day.
            # More realistically, unfilled day orders would be canceled by EOD.
            # Here, we process against the current options_data snapshot.

            exec_price = self.get_option_execution_price(order.ticker, order.is_long, order.limit)

            if exec_price is None:
                # print(f"Order for {order.ticker} ({order.size} @ LMT {order.limit}) could not fill at current market.")
                # In a real system, unfilled limit orders might persist.
                # For this backtester, if it doesn't fill on this "tick" (daily snapshot), it waits.
                # User might need to cancel it via strategy logic.
                continue

            # Order can be filled
            # 1. Cost/Proceeds
            trade_value = exec_price * abs(order.size) * self._option_multiplier
            commission_cost = self._commission_per_contract * abs(order.size)

            # 2. Cash Check (Simplified)
            if order.is_long: # Buying an option
                required_cash = trade_value + commission_cost
                if self._cash < required_cash:
                    if self._fail_fast:
                        raise RuntimeError(f"Not enough cash for {order}. Has {self._cash:.2f}, needs {required_cash:.2f}. Aborting.")
                    else:
                        warnings.warn(f"Not enough cash for {order}. Has {self._cash:.2f}, needs {required_cash:.2f}. Order skipped.")
                        self.orders.remove(order)
                        continue
            # else: Shorting an option - margin would be checked here. Skipped for now.

            # 3. Update Cash
            if order.is_long:
                self._cash -= (trade_value + commission_cost)
            else: # Selling an option
                self._cash += (trade_value - commission_cost)


            # 4. Handle Trade Creation / Closing
            if order.parent_trade: # This is an order to close an existing trade
                trade_to_close = order.parent_trade
                # Ensure the size matches what's being closed (order.size is opposite of trade_to_close.size portion)
                # e.g. trade_to_close.size = 10 (long), order.size = -10 (sell to close)
                if abs(order.size) > abs(trade_to_close.size):
                    warnings.warn(f"Closing order {order} size {order.size} exceeds parent trade size {trade_to_close.size}. Adjusting.")
                    order.size = copysign(abs(trade_to_close.size), order.size)

                if trade_to_close in self.trades.get(order.ticker, []):
                     self._reduce_trade(trade_to_close, exec_price, order.size, current_bar_idx, order.reason or "Closed")
                else: # Parent trade already closed or doesn't exist
                    warnings.warn(f"Parent trade for closing order {order} not found or already closed.")
                self.orders.remove(order)
                continue

            # This is a new opening order
            # If not hedging, check for existing opposite positions to close first
            if not self._hedging:
                for existing_trade in list(self.trades.get(order.ticker, [])):
                    if existing_trade.is_long != order.is_long: # Opposite position
                        # Close existing_trade fully if new order is larger or equal
                        if abs(order.size) >= abs(existing_trade.size):
                            size_to_close_from_existing = -existing_trade.size
                            self._reduce_trade(existing_trade, exec_price, size_to_close_from_existing, current_bar_idx, "Offset")
                            order.size += size_to_close_from_existing # Reduce new order by amount offset
                        else: # New order is smaller, partially close existing_trade
                            size_to_close_from_existing = -order.size # Close amount = new order's full size
                            self._reduce_trade(existing_trade, exec_price, size_to_close_from_existing, current_bar_idx, "Offset")
                            order.size = 0 # New order fully offset
                        if order.size == 0: break
            
            if order.size != 0: # If any part of the order remains to be opened
                self._open_trade(order.ticker, exec_price, int(order.size), current_bar_idx, order.tag)

            self.orders.remove(order) # Order processed

    def _reduce_trade(self, trade: Trade, price: float, size_change: float, time_index: int, reason: str = "Reduced"):
        # size_change is the amount by which the trade's size is changing.
        # e.g., trade.size = 10 (long), size_change = -5 (closing 5 contracts) -> new size = 5
        # e.g., trade.size = 10 (long), size_change = -10 (closing all) -> new size = 0
        assert trade.size * size_change <= 0, "size_change must be opposite or reduce existing trade size"
        assert abs(trade.size) >= abs(size_change)

        size_left = trade.size + size_change # e.g. 10 + (-5) = 5
        
        # Create a "closing" trade record for the portion being closed
        closed_portion_trade = trade._copy(
            size=-size_change, # The amount that was actually transacted to reduce
            exit_price=price,
            exit_bar_index=time_index,
            reason=reason
        )
        # Update P&L for broker's cash based on this realized portion
        # self._cash += closed_portion_trade.pl # This is implicitly handled by _close_trade logic

        if size_left == 0: # Trade is fully closed
            self._close_trade(trade, price, time_index, reason)
        else: # Trade is partially closed
            trade._replace(size=size_left)
            # Add the record of the closed portion
            self.closed_trades.append(closed_portion_trade)
            # self._cash needs to be adjusted by the P&L of the closed_portion_trade.
            # The initial P&L calculation for closed_portion_trade might be off if its entry_price is the original.
            # Let's make it simpler: when reducing, the original trade.pl is on its full size.
            # The cash impact is: `(-size_change) * price * multiplier - commission_for_closing_leg`
            # This is already handled when the closing order is processed.
            # What we need is to correctly record the P&L of the part that was closed.
            # closed_portion_trade.entry_price is the original entry.
            # closed_portion_trade.exit_price is current `price`.
            # closed_portion_trade.size is `-size_change`.
            # Its P&L `(-size_change) * (price - trade.entry_price) * multiplier` is the realized P&L for this chunk.
            # This closed_portion_trade object will reflect that.

    def _close_trade(self, trade: Trade, price: float, time_index: int, reason: str = "Closed"):
        if trade in self.trades.get(trade.ticker, []):
            self.trades[trade.ticker].remove(trade)
            if not self.trades[trade.ticker]: # List is now empty
                del self.trades[trade.ticker]
                if trade.ticker in self.positions:
                    del self.positions[trade.ticker] # Remove from active positions map

        trade._replace(exit_price=price, exit_bar_index=time_index, reason=reason)
        self.closed_trades.append(trade)
        # Cash adjustment for the closing trade should have happened when its order was processed.
        # The `trade.pl` is now final.

    def _open_trade(self, ticker: str, price: float, size: int,
                    time_index: int, tag):
        trade = Trade(self, ticker, size, price, time_index, tag)
        if ticker not in self.trades:
            self.trades[ticker] = []
        self.trades[ticker].append(trade)

        if ticker not in self.positions:
            self.positions[ticker] = Position(self, ticker)
        # Position object will query self.trades for its size/pl dynamically


class Backtest:
    def __init__(self,
                 db_path: str,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 100000,
                 commission_per_contract: float = 0.65, # New commission model
                 option_multiplier: int = 75, # New
                 trade_start_date: Optional[Union[str, datetime]] = None, # Can be str or datetime
                 ):

        self.db_path = db_path 

        self._broker_factory = partial(
            _Broker, cash=cash, commission_per_contract=commission_per_contract,
            option_multiplier=option_multiplier,
            trade_start_date=trade_start_date)
        self._strategy = strategy
        self._results: Optional[pd.Series] = None

    def run(self, **kwargs) -> pd.Series:
        data = _Data(self.db_path)
        broker: _Broker = self._broker_factory(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)
        processed_orders: List[Order] = []
        final_positions = None

        try:
            strategy.init()
        except Exception as e:
            print(f'Strategy initialization failed: {e}')
            traceback.print_exc()
            return pd.Series(name="StrategyInitError") # Return empty/error series

        progress_bar = _tqdm(data._table_names, desc="Backtesting Options Strategy")

        for table in data._table_names:
            data.load_table(table)
            
            for row in data.spot.itertuples():
                # Process each row in the spot data

                # Broker actions: expirations, process orders, update equity
                try:
                    # broker.handle_expirations(new_spot_day_obj) # Pass date part
                    broker.next() # This will call _process_orders
                except _OutOfMoneyError:
                    print('Strategy ran out of money.')
                    progress_bar.close()
                    break
                except Exception as e:
                    print(f"Error in broker.next() or handle_expirations() on {table}: {e}")
                    traceback.print_exc()
                    progress_bar.close()
                    break # Critical error

                # Strategy decision making
                try:
                    strategy.next()
                    processed_orders.extend(broker.orders)
                except Exception as e:
                    print(f"Error in strategy.next() on {table}: {e}")
                    traceback.print_exc()
                    progress_bar.close()
                    break # Critical error
                
                # Log processed orders from this step (optional, for detailed analysis)
                # processed_orders_during_run.extend(broker.orders) # Careful, broker.orders is dynamic

            progress_bar.update(1)

        else: # Loop completed without break
            final_positions = ({t: p.size for t, p in broker.positions.items()}
                                   | {'Cash': int(broker.margin_available)})
            
            broker.finalize() # Close any open positions at the end
        
        progress_bar.close()

        return processed_orders, final_positions
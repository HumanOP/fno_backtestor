# --- START OF FILE options_backtesting.py ---

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
        self._conn = duckdb.connect(db_path)
        self._cache: Dict[str, pd.DataFrame] = {}             # Loaded tables cache
        self._ticker_map: Dict[str, pd.DataFrame] = {}        # Ticker-wise grouped data
        self._current_df: Optional[pd.DataFrame] = None       # Active dataframe for the date
        self._current_index: Optional[pd.DatetimeIndex] = None
        self._time_series: Optional[pd.DataFrame] = None      # Spot time series (indexed)
        self._leg_data_map: Dict[str, pd.Series] = {}         # Contract: Series (1 row time-aligned)

    def load_table(self, table_name: str):
        if table_name in self._cache:
            df = self._cache[table_name]
        else:
            df = self._conn.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
            df["expiry_date"] = df["expiry_date"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
            df.set_index("timestamp", inplace=True)
            self._cache[table_name] = df
        self._current_df = df
        self._current_index = df.index
        self._build_ticker_map()
        return df

    def _build_ticker_map(self):
        if self._current_df is not None:
            self._ticker_map = {ticker: group for ticker, group in self._current_df.groupby("ticker", sort=False)}

    def update(self, timestamp: pd.Timestamp):
        if self._current_df is None:
            raise ValueError("No data loaded. Call load_table() first.")
        if self._time_series is None:
            self._time_series = self._current_df[["spot_price"]][~self._current_df.index.duplicated(keep="first")]
        self.build_leg_data_map(self._leg_data_map, timestamp)

    def get_leg_data(self, contract: str, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        if contract not in self._ticker_map:
            return None
        df = self._ticker_map[contract]
        asof_time = df.index.asof(timestamp)
        if pd.isna(asof_time):
            return None
        return df.loc[asof_time]

    def build_leg_data_map(self, legs: Dict[str, Dict], timestamp: pd.Timestamp):
        self._leg_data_map.clear()
        for leg_id, leg in legs.items():
            contract = leg["contract"]
            leg_data = self.get_leg_data(contract, timestamp)
            if leg_data is not None:
                self._leg_data_map[leg_id] = leg_data

    def set_time_series(self, df: pd.DataFrame):
        self._time_series = df

    @property
    def spot_series(self) -> pd.DataFrame:
        return self._time_series if self._time_series is not None else pd.DataFrame()

    @property
    def current_df(self):
        return self._current_df

    @property
    def time_index(self):
        return self._current_index

    @property
    def ticker_map(self):
        return self._ticker_map

    @property
    def leg_data(self):
        return self._leg_data_map

    def close(self):
        self._conn.close()

    def __repr__(self):
        keys = list(self._cache.keys())
        return f"<Data cached_tables={keys} current_time_range={[self._current_index[0], self._current_index[-1]] if self._current_index is not None else None}>"


class _SpotData(_Data): # Keep a similar structure for spot
    pass

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

    def get_contract_data(self, opt_contract: str) -> Optional[pd.Series]:
        # Assuming 'OptionSymbol' is a column in the options_df
        if 'OptionSymbol' not in self._df.columns:
            # Try index if OptionSymbol is the index
            if opt_contract in self._df.index:
                 return self._df.loc[opt_contract]
            warnings.warn("Options data DataFrame does not have 'OptionSymbol' column or index.")
            return None
        
        contract_row = self._df[self._df['OptionSymbol'] == opt_contract]
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
                   opt_contract: str,
                   quantity: float, # Number of contracts
                   limit: Optional[float] = None,
                   # stop: Optional[float] = None, # REMOVED
                   tag: object = None):
        assert quantity > 0, "Quantity for buying options must be positive"
        return self._broker.new_order(opt_contract, quantity, limit, tag)

    def sell(self, *,
                    opt_contract: str,
                    quantity: float, # Number of contracts
                    limit: Optional[float] = None,
                    # stop: Optional[float] = None, # REMOVED
                    tag: object = None):
        # Negative quantity for selling (to open short or close long)
        assert quantity > 0, "Quantity for selling options must be positive (use negative for broker call)"
        return self._broker.new_order(opt_contract, -quantity, limit, tag)

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

    def position(self, opt_contract: str) -> 'Position': # Now takes opt_contract
        return self._broker.positions.get(opt_contract, Position(self._broker, opt_contract, 0)) # Return empty if not found

    @property
    def orders(self) -> 'List[Order]':
        return self._broker.orders

    # trades() and closed_trades() now refer to option trades
    def trades(self, opt_contract: str = None) -> 'Tuple[Trade, ...]':
        if opt_contract:
            return tuple(self._broker.trades.get(opt_contract, []))
        return tuple(trade for trades_list in self._broker.trades.values() for trade in trades_list)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        return tuple(self._broker.closed_trades)

    def start_on_day(self, n: int):
        assert 0 <= n < len(self._spot_data), f"day must be within [0, {len(self._spot_data)-1}]"
        self._start_on_day = n
    

class Position:
    def __init__(self, broker: '_Broker', opt_contract: str, initial_size: float = 0): # Added initial_size
        self.__broker = broker
        self.__opt_contract = opt_contract
        # Position size is now managed directly by summing trades in _Broker,
        # or _Broker can update a size attribute here.
        # For simplicity, let's make Position mostly a query interface.

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in number of contracts. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades.get(self.__opt_contract, []))

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position."""
        return sum(trade.pl for trade in self.__broker.trades.get(self.__opt_contract, []))

    @property
    def pl_pct(self) -> float:
        """Profit/loss in percent. This is harder for options without average entry cost.
           Could be (current_value - cost_basis) / cost_basis.
           For simplicity, we might omit this or calculate it based on an average entry price.
        """
        # This needs a clear definition for options.
        # One way: Sum of P/L of open trades / Sum of initial values of open trades
        current_val = sum(trade.value for trade in self.__broker.trades.get(self.__opt_contract, []))
        entry_val = sum(trade.entry_price * abs(trade.size) * self.__broker._option_multiplier
                        for trade in self.__broker.trades.get(self.__opt_contract, [])
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
        # This needs to iterate through trades of this specific opt_contract
        for trade in list(self.__broker.trades.get(self.__opt_contract, [])): # Iterate copy
            trade.close(portion)

    def __repr__(self):
        num_trades = len(self.__broker.trades.get(self.__opt_contract, []))
        return f'<Position: {self.__opt_contract} Size={self.size} ({num_trades} trades)>'


class _OutOfMoneyError(Exception):
    pass


class Order:
    def __init__(self, broker: '_Broker',
                 opt_contract: str, # Changed from ticker
                 size: float, # Number of contracts
                 limit_price: Optional[float] = None,
                 parent_trade: Optional['Trade'] = None,
                 entry_time: datetime = None, # This will be spot timestamp
                 tag: object = None,
                 reason: object = None): # Kept for consistency
        self.__broker = broker
        self.__opt_contract = opt_contract
        assert size != 0
        self.__size = size # Positive for buy, negative for sell
        self.__limit_price = limit_price
        self.__parent_trade = parent_trade
        self.__entry_time = entry_time # Timestamp from spot data
        self.__tag = tag
        self.__reason = reason

    def _replace(self, **kwargs): # Keep for internal use
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return f'<Order {self.__opt_contract} Size={self.__size} Limit={self.__limit_price} Tag={self.__tag}>'


    def cancel(self):
        """Cancel the order."""
        if self in self.__broker.orders: # Check if still in list
            self.__broker.orders.remove(self)

    @property
    def opt_contract(self) -> str: # Renamed
        return self.__opt_contract

    @property
    def size(self) -> float:
        return self.__size
    
    @size.setter
    def size(self, size):
        self.__size = size


    @property
    def limit(self) -> Optional[float]:
        return self.__limit_price


    @property
    def parent_trade(self):
        return self.__parent_trade

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
    def is_contingent(self): # Only true if it's closing a parent_trade
        return bool(self.__parent_trade)

    @property
    def entry_time(self) -> datetime: # Spot data timestamp
        return self.__entry_time


class Trade:
    def __init__(self, broker: '_Broker', opt_contract: str, size: int,
                 entry_price: float, entry_bar_index: int, # entry_bar_index from spot data
                 tag, reason: Optional['Order'] = None):
        self.__broker = broker
        self.__opt_contract = opt_contract
        self.__size = size # Number of contracts, negative for short
        self.__entry_price = entry_price # Price per unit (premium)
        self.__exit_price: Optional[float] = None
        self.__entry_bar_index: int = entry_bar_index # Index in spot_data
        self.__exit_bar_index: Optional[int] = None
        self.__tag = tag
        self.__reason = reason # From closing order

    def __repr__(self):
        return (f'<Trade {self.__opt_contract} Size={self.__size} EntryBar={self.__entry_bar_index} ExitBar={self.__exit_bar_index or ""} '
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

        order = Order(self.__broker, self.__opt_contract, size_to_close,
                      parent_trade=self, entry_time=self.__broker.now, tag=self.__tag)
        if finalize:
            return order # For _Broker.finalize()
        else:
            self.__broker.orders.insert(0, order) # Prioritize closing orders


    @property
    def opt_contract(self): # Renamed
        return self.__opt_contract

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
            current_option_price = self.__broker.get_option_last_price(self.__opt_contract)
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
            current_option_price = self.__broker.get_option_last_price(self.__opt_contract)
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
        price = self.__exit_price or self.__broker.get_option_last_price(self.__opt_contract)
        if price is None: return 0 # No current price available
        return self.__size * price * self.__broker._option_multiplier


class _Broker:
    def __init__(self, *,
                 spot_data: _SpotData, # Changed from data
                 current_options_data_ref: _OptionsData, # Reference to Backtest's current options
                 cash: float,
                 # holding: dict = {}, # Holding structure might change for options
                 commission_per_contract: float, # Changed commission model
                 option_multiplier: int,
                 # margin: float = 1., # Margin is more complex, simplified for now
                 # trade_on_close=False, # Less relevant, execution is vs option's quotes
                 hedging=True, # Hedging is usually default for options (buy SPY C & SPY P)
                 # exclusive_orders=False, # Less relevant
                 trade_start_date=None, # From spot_data index
                 # lot_size=1, # Options are usually traded in single contract increments
                 fail_fast=True,
                 # storage, # REMOVED
                 # is_option # REMOVED, this IS an options backtester
                 ):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert commission_per_contract >= 0, "commission_per_contract should be >= 0"
        assert option_multiplier > 0, "option_multiplier must be positive"

        self._spot_data = spot_data
        self._current_options_data = current_options_data_ref # Reference, updated by Backtest
        self._cash = cash
        # self._holding = holding # How to initialize option holdings? Typically start flat.
        self._commission_per_contract = commission_per_contract
        self._option_multiplier = option_multiplier
        # self._leverage = 1 / margin # REMOVED
        self._hedging = hedging # If False, buying a call would close a short call of same strike/expiry
        self._trade_start_date = trade_start_date

        self._equity = np.tile(np.nan, len(spot_data.index)) # Simpler equity array for now
        self.orders: List[Order] = []
        # Trades: dict keyed by opt_contract, then list of trades for that symbol
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {} # Updated as trades occur

        self._trade_start_bar = 0
        if self._trade_start_date:
            # Find the index in spot_data that corresponds to trade_start_date
            # This assumes spot_data.index is sorted and unique.
            try:
                # Ensure tz-awareness matches or handle appropriately
                if self._spot_data.index.tz is not None and self._trade_start_date.tzinfo is None:
                    self._trade_start_date = self._trade_start_date.replace(tzinfo=self._spot_data.index.tz)
                elif self._spot_data.index.tz is None and self._trade_start_date.tzinfo is not None:
                    self._trade_start_date = self._trade_start_date.replace(tzinfo=None)

                self._trade_start_bar = self._spot_data.index.get_loc(self._trade_start_date, method='bfill')
            except KeyError:
                # If date not found, start from beginning or end based on policy
                # For now, if date is before start, start at 0. If after end, effectively no trades.
                if self._trade_start_date < self._spot_data.index[0]:
                    self._trade_start_bar = 0
                elif self._trade_start_date > self._spot_data.index[-1]:
                    self._trade_start_bar = len(self._spot_data.index) # No trades will occur
                else: # Should be caught by get_loc, but as a fallback
                    self._trade_start_bar = (self._spot_data.index < self._trade_start_date).sum()


    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self._cash:.2f}, Equity={self.equity():.2f} ({active_trades} open trades)>'

    def new_order(self,
                  opt_contract: str,
                  size: float, # Number of contracts
                  limit: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None): # trade is parent_trade for closing orders
        
        # Ignore orders before trade_start_date (based on spot_data index)
        current_bar_index = len(self._spot_data) -1 # Current index in the sliced spot_data
        if current_bar_index < self._trade_start_bar:
             # print(f"Order for {opt_contract} ignored: current bar {current_bar_index} < trade_start_bar {self._trade_start_bar}")
             return None

        size = float(size)
        limit = limit and float(limit)

        # Basic validation (more complex for options, e.g., price increments)
        # For options, a buy order is size > 0, sell is size < 0.
        # Limit price should be positive.
        if limit is not None and limit <= 0:
            warnings.warn(f"Order for {opt_contract} has non-positive limit price {limit}. Ignoring.")
            return None

        order = Order(self, opt_contract, size, limit,
                      parent_trade=trade, entry_time=self.now, tag=tag)

        if trade: # This is a closing order for an existing trade
            self.orders.insert(0, order)
        else: # New opening order
            # if self._exclusive_orders: # Not typically used this way for options
            #    pass
            self.orders.append(order)
        return order

    def get_option_contract_details(self, opt_contract: str) -> Optional[pd.Series]:
        return self._current_options_data.get_contract_data(opt_contract)

    def get_option_execution_price(self, opt_contract: str, is_buy_order: bool, limit_price: Optional[float]) -> Optional[float]:
        """
        Determines execution price for an option.
        Simple logic: if limit, use limit if marketable. Else, use 'Last' or 'Mid'.
        A real system uses NBBO and considers liquidity.
        """
        contract = self.get_option_contract_details(opt_contract)
        if contract is None:
            # warnings.warn(f"No data for option {opt_contract} in current chain. Order cannot fill.")
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

            if limit_price is not None: # Limit Buy
                if market_price is not None and limit_price >= market_price:
                    fill_price = market_price # Fill at market if limit is competitive or better
                else:
                   
                    if market_price is None or limit_price >= market_price:
                        fill_price = limit_price # Fill at limit (optimistic if market_price is worse or unknown)
                    else: # limit_price < market_price (e.g. Ask for buy)
                        return None # Cannot fill at this limit
            else: # Market Buy
                fill_price = market_price
        else: # Sell order
            market_price = None
            if pd.notna(bid_price) and bid_price > 0: market_price = bid_price
            elif pd.notna(last_price) and last_price > 0: market_price = last_price
            elif pd.notna(ask_price) and ask_price > 0: market_price = ask_price # Less likely fill for sell

            if limit_price is not None: # Limit Sell
                if market_price is not None and limit_price <= market_price:
                    fill_price = market_price # Fill at market if limit is competitive or better
                else:
                    if market_price is None or limit_price <= market_price:
                        fill_price = limit_price
                    else: # limit_price > market_price (e.g. Bid for sell)
                        return None # Cannot fill
            else: # Market Sell
                fill_price = market_price
        
        return fill_price if pd.notna(fill_price) and fill_price > 0 else None


    def get_option_last_price(self, opt_contract: str) -> Optional[float]:
        """ Gets the last known price for an option, typically for MTM P&L. """
        contract = self.get_option_contract_details(opt_contract)
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


    def equity(self, opt_contract: str = None) -> float: # MTM equity
        if opt_contract:
            return sum(trade.value for trade in self.trades.get(opt_contract, []))
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
        for opt_contract, trade_list in list(self.trades.items()): # Iterate copy of items
            if not trade_list:
                symbols_to_remove_from_trades.append(opt_contract)
                continue

            # Get expiry from the first trade (assume all trades for a symbol have same expiry)
            # This requires opt_contract to be parsable or options_data to have Expiry
            contract_details = self.get_option_contract_details(opt_contract)
            if not contract_details or 'Expiry' not in contract_details:
                # warnings.warn(f"Cannot determine expiry for {opt_contract}. Skipping expiration check.")
                continue

            try:
                expiry_date = contract_details['Expiry']
                if isinstance(expiry_date, (pd.Timestamp, datetime)):
                    expiry_date = expiry_date.date()
                # Add more robust parsing if expiry_date is string
            except Exception as e:
                # warnings.warn(f"Could not parse expiry for {opt_contract}: {e}. Skipping.")
                continue

            if current_date >= expiry_date:
                # Option expired or is expiring today
                # print(f"Option {opt_contract} expiring on {expiry_date} (current_date: {current_date})")
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
                
                if not self.trades.get(opt_contract): # If all trades for this symbol are closed
                    symbols_to_remove_from_trades.append(opt_contract)
        
        for sym in symbols_to_remove_from_trades:
            if sym in self.trades and not self.trades[sym]: # double check if empty
                del self.trades[sym]
                if sym in self.positions:
                    del self.positions[sym]


    def finalize(self): # Close all open option positions at last available price
        current_bar_idx = len(self._spot_data) - 1
        for opt_contract in list(self.trades.keys()):
            for trade in list(self.trades.get(opt_contract, [])): # Iterate copy
                last_price = self.get_option_last_price(opt_contract)
                if last_price is None: # If no price, assume worthless (simplification)
                    # A better approach might be to use intrinsic value if near expiry
                    contract = self.get_option_contract_details(opt_contract)
                    underlying_close = self._spot_data.Close[-1]
                    if contract and contract['Type'].upper() == 'CALL':
                        last_price = max(0, underlying_close - contract['Strike'])
                    elif contract and contract['Type'].upper() == 'PUT':
                        last_price = max(0, contract['Strike'] - underlying_close)
                    else:
                        last_price = 0 # Fallback
                
                # print(f"Finalizing trade for {opt_contract} at price {last_price}")
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

            exec_price = self.get_option_execution_price(order.opt_contract, order.is_long, order.limit)

            if exec_price is None:
                # print(f"Order for {order.opt_contract} ({order.size} @ LMT {order.limit}) could not fill at current market.")
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

                if trade_to_close in self.trades.get(order.opt_contract, []):
                     self._reduce_trade(trade_to_close, exec_price, order.size, current_bar_idx, order.reason or "Closed")
                else: # Parent trade already closed or doesn't exist
                    warnings.warn(f"Parent trade for closing order {order} not found or already closed.")
                self.orders.remove(order)
                continue

            # This is a new opening order
            # If not hedging, check for existing opposite positions to close first
            if not self._hedging:
                for existing_trade in list(self.trades.get(order.opt_contract, [])):
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
                self._open_trade(order.opt_contract, exec_price, int(order.size), current_bar_idx, order.tag)

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
        if trade in self.trades.get(trade.opt_contract, []):
            self.trades[trade.opt_contract].remove(trade)
            if not self.trades[trade.opt_contract]: # List is now empty
                del self.trades[trade.opt_contract]
                if trade.opt_contract in self.positions:
                    del self.positions[trade.opt_contract] # Remove from active positions map

        trade._replace(exit_price=price, exit_bar_index=time_index, reason=reason)
        self.closed_trades.append(trade)
        # Cash adjustment for the closing trade should have happened when its order was processed.
        # The `trade.pl` is now final.

    def _open_trade(self, opt_contract: str, price: float, size: int,
                    time_index: int, tag):
        trade = Trade(self, opt_contract, size, price, time_index, tag)
        if opt_contract not in self.trades:
            self.trades[opt_contract] = []
        self.trades[opt_contract].append(trade)

        if opt_contract not in self.positions:
            self.positions[opt_contract] = Position(self, opt_contract)
        # Position object will query self.trades for its size/pl dynamically


class Backtest:
    def __init__(self,
                 spot_data_df: pd.DataFrame, # Changed
                 options_data_loader: Callable[[DateObject], pd.DataFrame], # New
                 strategy: Type[Strategy],
                 cash: float = 100000,
                 commission_per_contract: float = 0.65, # New commission model
                 option_multiplier: int = 75, # New
                 trade_start_date: Optional[Union[str, datetime]] = None, # Can be str or datetime
                 fail_fast=True,
                 ):

        self.spot_data_full = spot_data_df.copy() # Keep the full spot data
        
        self.spot_data_full.index.name = 'Date' # Ensure index has a name

        self.options_data_loader = options_data_loader
        self._current_options_data_wrapper = _OptionsData() # Broker and Strategy will get this

        # if isinstance(trade_start_date, str):
        #     trade_start_date = pd.to_datetime(trade_start_date).replace(tzinfo=self.spot_data_full.index.tz)
        # elif isinstance(trade_start_date, datetime) and self.spot_data_full.index.tz:
        #      if trade_start_date.tzinfo is None: # Make trade_start_date tz-aware if spot_data is
        #          trade_start_date = trade_start_date.replace(tzinfo=self.spot_data_full.index.tz)


        self._broker_factory = partial(
            _Broker, cash=cash, commission_per_contract=commission_per_contract,
            option_multiplier=option_multiplier,
            trade_start_date=trade_start_date,
            fail_fast=fail_fast,
            current_options_data_ref=self._current_options_data_wrapper # Pass the reference
        )
        self._strategy_class = strategy # Renamed from _strategy
        self._results: Optional[pd.Series] = None

        # Reference for Buy & Hold (on spot)
        self._spot_ohlc_ref = self.spot_data_full.copy()


    def run(self, **kwargs) -> pd.Series:
        spot_data = _SpotData(self.spot_data_full.copy(deep=False)) # For strategy.spot_data
        broker: _Broker = self._broker_factory(spot_data=spot_data)
        strategy: Strategy = self._strategy_class(broker, spot_data, kwargs)

        processed_orders_during_run: List[Order] = [] # For stats if needed
        # final_positions = None # For stats if needed

        try:
            strategy.init()
        except Exception as e:
            print(f'Strategy initialization failed: {e}')
            traceback.print_exc()
            return pd.Series(name="StrategyInitError") # Return empty/error series

        start_bar_index = strategy._start_on_day # Default start
        start_bar_index = max(0, min(start_bar_index, len(self.spot_data_full) - 1))


        # Main loop over spot data days
        current_spot_day_obj: Optional[DateObject] = None

        unique_spot_dates = pd.Series(self.spot_data_full.index.date).unique()
        # Filter unique_spot_dates to start from the actual trading start date
        start_datetime_obj = self.spot_data_full.index[start_bar_index].date()
        tradeable_spot_dates = [d for d in unique_spot_dates if d >= start_datetime_obj]

        progress_bar = _tqdm(tradeable_spot_dates, desc="Backtesting Options Strategy")


        for i in range(start_bar_index, len(self.spot_data_full)):
            spot_data._set_length(i + 1) # Update strategy's view of spot_data

            # Check if it's a new day for loading options data
            new_spot_day_obj = spot_data.index[-1].date()

            if new_spot_day_obj != current_spot_day_obj:
                # print(f"New day: {new_spot_day_obj}. Loading options chain.")
                try:
                    options_df_for_day = self.options_data_loader(new_spot_day_obj)
                    if options_df_for_day is None or options_df_for_day.empty:
                        # warnings.warn(f"No options data loaded for {new_spot_day_obj}. Strategy may not find contracts.")
                        self._current_options_data_wrapper._set_data(pd.DataFrame()) # Empty df
                    else:
                        self._current_options_data_wrapper._set_data(options_df_for_day)
                    
                    strategy.options_data = self._current_options_data_wrapper # Update strategy's view
                    current_spot_day_obj = new_spot_day_obj
                    if new_spot_day_obj in progress_bar.iterable: # Check if this date is in our tqdm list
                        progress_bar.update(1)

                except Exception as e:
                    warnings.warn(f"Error loading options data for {new_spot_day_obj}: {e}")
                    self._current_options_data_wrapper._set_data(pd.DataFrame()) # Set empty on error
                    strategy.options_data = self._current_options_data_wrapper
                    current_spot_day_obj = new_spot_day_obj # Still update day to avoid re-fetch attempt
                    if new_spot_day_obj in progress_bar.iterable:
                         progress_bar.update(1)


            # Broker actions: expirations, process orders, update equity
            try:
                broker.handle_expirations(new_spot_day_obj) # Pass date part
                broker.next() # This will call _process_orders
            except _OutOfMoneyError:
                print('Strategy ran out of money.')
                progress_bar.close()
                break
            except Exception as e:
                print(f"Error in broker.next() or handle_expirations() on {new_spot_day_obj}: {e}")
                traceback.print_exc()
                progress_bar.close()
                break # Critical error

            # Strategy decision making
            if i >= broker._trade_start_bar: # Ensure strategy logic only runs after trade_start_bar
                try:
                    strategy.next()
                except Exception as e:
                    print(f"Error in strategy.next() on {new_spot_day_obj}: {e}")
                    traceback.print_exc()
                    progress_bar.close()
                    break # Critical error
            
            # Log processed orders from this step (optional, for detailed analysis)
            # processed_orders_during_run.extend(broker.orders) # Careful, broker.orders is dynamic

        else: # Loop completed without break
            broker.finalize() # Close any open positions at the end
        
        progress_bar.close()

        # Prepare results
        # This needs a new `compute_stats_options` function
        equity_series = pd.Series(broker._equity[:len(spot_data_sliced)], index=spot_data_sliced.index).dropna()
        
        # For a simplified result:
        results = pd.Series(dtype=object)
        results['Start'] = equity_series.index[0] if not equity_series.empty else self.spot_data_full.index[0]
        results['End'] = equity_series.index[-1] if not equity_series.empty else self.spot_data_full.index[-1]
        results['Duration'] = results['End'] - results['Start']
        results['Initial Equity [$]'] = cash
        results['Final Equity [$]'] = broker.equity() # Final equity after finalize
        results['Return [%]'] = (results['Final Equity [$]'] / cash - 1) * 100 if cash > 0 else 0.
        
        # Spot Buy & Hold
        spot_return = (self.spot_data_full['Close'].iloc[-1] / self.spot_data_full['Close'].iloc[start_bar_index] - 1) * 100
        results['Buy & Hold Return (Spot) [%]'] = spot_return
        
        results['# Trades'] = len(broker.closed_trades)
        win_trades = [t for t in broker.closed_trades if t.pl > 0]
        results['Win Rate [%]'] = (len(win_trades) / len(broker.closed_trades) * 100) if broker.closed_trades else 0.
        
        if broker.closed_trades:
            results['Avg. Trade PnL [$]'] = np.mean([t.pl for t in broker.closed_trades])
            results['Best Trade PnL [$]'] = np.max([t.pl for t in broker.closed_trades])
            results['Worst Trade PnL [$]'] = np.min([t.pl for t in broker.closed_trades])
        else:
            results['Avg. Trade PnL [$]'] = 0
            results['Best Trade PnL [$]'] = 0
            results['Worst Trade PnL [$]'] = 0

        results['_equity_curve'] = equity_series
        results['_trades'] = pd.DataFrame([
            {
                'OptionSymbol': t.opt_contract, 'Size': t.size,
                'EntryBar': t.entry_bar, 'ExitBar': t.exit_bar,
                'EntryTime': t.entry_time, 'ExitTime': t.exit_time,
                'EntryPrice': t.entry_price, 'ExitPrice': t.exit_price,
                'PnL': t.pl, 'Tag': t.tag, 'Reason': t.reason
            } for t in broker.closed_trades
        ])
        results['_strategy_instance'] = strategy # For potential later inspection
        results['_broker_instance'] = broker     # For potential later inspection
        
        self._results = results
        return self._results.copy()
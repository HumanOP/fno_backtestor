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
        self._spot_table: Optional[pd.DataFrame] = None
        self._time: Optional[pd.Timestamp] = None
        self._spot: Optional[float] = None
        self._tte_to_expiry: Optional[Dict] = None
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
        self._spot_table = None

    def load_table(self, table_name: str):
        df = self._conn.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
        # df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
        # df["expiry_date"] = df["expiry_date"].dt.tz_localize("Asia/Kolkata").dt.tz_convert(None)
        self._tte_to_expiry = df.drop_duplicates("expiry_date").set_index("Time_to_expiry")["expiry_date"].to_dict()
        df.set_index("timestamp", inplace=True)
        self._build_ticker_map(df)
        self._data_df_template = df.iloc[0:0]
        self._spot_table = df[["spot_price"]][~df.index.duplicated(keep="first")]
        return df

    def _build_ticker_map(self, df):
        self._ticker_map = {ticker: group for ticker, group in df.groupby("ticker", sort=False)}

    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        # 1. Need to add a cache for this
        # 2. Need to check the data directly at that time and if not available then do the following
        subset_df = self._ticker_map.get(ticker, self._data_df_template)
        asof_time = subset_df.index.asof(self._time) if not subset_df.empty else None
        if pd.isna(asof_time):
            return None
        return subset_df.loc[asof_time]

    def close(self):
        self._conn.close()

    def __repr__(self):
        # keys = list(self._cache.keys())
        # return f"<Data cached_tables={keys} current_time_range={[self._current_index[0], self._current_index[-1]] if self._current_index is not None else None}>"
        return str(self._table_names)


class Strategy(ABC):
    def __init__(self, broker: '_Broker', _data: _Data, params: dict):
        self._broker: _Broker = broker
        self._data: _Data = _data
        self._params = self._check_params(params)
        self._records = {}
        self._start_on_day = 0


    def __repr__(self):
        return '<Strategy ' + str(self) + '>'
    
#   def __str__(self): not defined as in synapse
    
    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with.")
            setattr(self, k, v)
        return params

    @abstractmethod
    def init(self):
        """Initialize the strategy. Declare spot indicators, etc."""
        print("Initializing strategy...")

    @abstractmethod
    def next(self):
        """
        Called for each spot data bar AFTER options_data for the current day is loaded.
        Access `self.spot_data` for current spot bar.
        Access `self.options_data` for current day's option chain.
        """
        # print("Strategy next() called")
        pass

    def buy(self, *,
        strategy_id: str,
        position_id: str,
        leg_id: str,
        ticker: str,
        quantity: float, # Number of contracts
        stop_loss: float, take_profit: float, tag: str):
        assert quantity > 0, "Quantity for buying options must be positive"
        return self._broker.new_order(strategy_id, position_id, leg_id, ticker, quantity, stop_loss, take_profit, tag)

    def sell(self, *,
        strategy_id: str,
        position_id: str,
        leg_id: str,
        ticker: str,
        quantity: float, # Number of contracts
        stop_loss: float, take_profit: float, tag: str):
        # Negative quantity for selling (to open short or close long)
        assert quantity > 0, "Quantity for selling options must be positive (use negative for broker call)"
        return self._broker.new_order(strategy_id, position_id, leg_id, ticker, -quantity, stop_loss, take_profit, tag)


    # Doubt on how can it be used and have to correct the logic
    # --- record() can be kept for custom logging ---
    # def record(self, name: str = None, plot: bool = True, overlay: bool = None, color: str = None, scatter: bool = False, **kwargs):
    #     for k, v in kwargs.items():
    #         current_time = self._broker.time # Spot data timestamp
    #         if isinstance(v, dict) or isinstance(v, pd.Series):
    #             v = dict(v)
    #             if k not in self._records:
    #                 self._records[k] = pd.DataFrame(index=self._data_index, columns=v.keys())
    #             self._records[k].loc[current_time, list(v.keys())] = list(v.values())
    #         else:
    #             if k not in self._records:
    #                 self._records[k] = pd.Series(index=self._data_index)
    #             self._records[k].loc[current_time] = v # Use .loc for safety
            
    #         # Store plotting attributes if needed later, but actual plotting needs rework
    #         if not hasattr(self._records[k], 'attrs'): self._records[k].attrs = {}
    #         self._records[k].name = name or k # Ensure name is set
    #         self._records[k].attrs.update({'name': name or k, 'plot': plot, 'overlay': overlay,
    #                                        'color': color, 'scatter': scatter})
    
    @property
    def time(self):
        return self._data._time
    
    @property
    def spot(self):
        return self._data._spot

    @property
    def tte_to_expiry(self):
        return self._data._tte_to_expiry

    @property
    def equity(self) -> float:
        return self._broker.equity()

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
    def active_trades(self) -> 'Tuple[Trade, ...]':
        return tuple(trade for trades_list in self._broker.trades.values() for trade in trades_list)


    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        return tuple(self._broker.closed_trades)
    

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
        # current_val = sum(trade.value for trade in self.__broker.trades.get(self.__ticker, []))
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

    # have to fix this, currently trade.close takes size and tag
    def close(self, tag, portion: float = 1.):
        """Close portion of position by closing `portion` of each active trade."""
        # This needs to iterate through trades of this specific ticker
        for trade in list(self.__broker.trades.get(self.__ticker, [])): # Iterate copy
            trade.close(tag, portion)

    def __repr__(self):
        num_trades = len(self.__broker.trades.get(self.__ticker, []))
        return f'<Position: {self.__ticker} Size={self.size} ({num_trades} trades)>'


class _OutOfMoneyError(Exception):
    pass


class Order:
    def __init__(self, broker: '_Broker', strategy_id: str, position_id: str, leg_id: str,
                 ticker: str, # Changed from ticker
                 size: float, # Number of contracts
                 stop_loss: float = None, take_profit: float = None,
                 tag: object = None, trade: 'Trade' = None): 
        self.__broker = broker
        self.__strategy_id = strategy_id
        self.__position_id = position_id
        self.__leg_id = leg_id
        self.__ticker = ticker
        assert size != 0
        self.__size = size # Positive for buy, negative for sell
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        self.__tag = tag
        self.__trade = trade

    def _replace(self, **kwargs): # Keep for internal use
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return f'<Order {self.__ticker} Size={self.__size} Tag={self.__tag}>'


    def cancel(self):
        """Cancel the order."""
        if self in self.__broker.orders: # Check if still in list
            self.__broker.orders.remove(self)

    @property
    def strategy_id(self) -> str:
        return self.__strategy_id

    @property
    def position_id(self) -> str:
        return self.__position_id

    @property
    def leg_id(self) -> str:
        return self.__leg_id

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
    def stop_loss(self) -> Optional[float]:
        return self.__stop_loss
    
    @stop_loss.setter
    def stop_loss(self, stop_loss: Optional[float]):
        self.__stop_loss = stop_loss

    @property
    def take_profit(self) -> Optional[float]:
        return self.__take_profit

    @take_profit.setter
    def take_profit(self, take_profit: Optional[float]):
        self.__take_profit = take_profit

    @property
    def tag(self):
        return self.__tag
    
    @property
    def trade(self) -> Optional['Trade']:
        """Associated trade if this order is part of a trade."""
        return self.__trade

    @property
    def is_long(self): # True if this order is to buy
        return self.__size > 0

    @property
    def is_short(self): # True if this order is to sell
        return self.__size < 0


class Trade:
    def __init__(self, broker: '_Broker', strategy_id: str, position_id: str, leg_id: str,
                 ticker: str, size: int, entry_price: float, entry_datetime: pd.Timestamp,
                 entry_spot: float, stop_loss: float = None, take_profit: float = None, entry_tag: str = None):

        self.__broker = broker
        self.__strategy_id = strategy_id
        self.__position_id = position_id
        self.__leg_id = leg_id
        self.__ticker = ticker
        self.__size = size
        self.__entry_price = entry_price
        self.__entry_datetime = entry_datetime
        self.__entry_spot = entry_spot
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        self.__entry_tag = entry_tag


        self.__exit_price: Optional[float] = None
        self.__exit_datetime: Optional[pd.Timestamp] = None
        self.__exit_spot: Optional[float] = None
        self.__exit_tag: Optional[str] = None

    def __repr__(self):
        return (f'<Trade {self.__ticker} Size={self.__size} EntryDatetime={self.__entry_datetime} ExitDatetime={self.__exit_datetime or ""} '
                f'EntryPrice={self.__entry_price:.2f} ExitPrice={self.__exit_price or "":.2f} PnL={self.pl:.2f} Tag={self.__entry_tag or ""}>')

    def _replace(self, **kwargs): # Keep
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs): # Keep
        return copy(self)._replace(**kwargs)

    def close(self, size, tag, finalize=False): # finalize is for end of backtest
        assert 0 < abs(size) <= abs(self.__size)

        order_size = -size      # Note we are sending reverse size for reverse side order
    
        order = Order(self.__broker, self.__strategy_id, self.__position_id, self.__leg_id,
                      self.__ticker, order_size, stop_loss=self.__stop_loss,
                      take_profit=self.__take_profit, tag=tag, trade=self) 
        if finalize:
            return order        # For _Broker.finalize()
        else:
            self.__broker.orders.insert(0, order) # Prioritize closing orders
            return order


    @property
    def ticker(self):
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
    def entry_datetime(self) -> int: # Index in spot_data
        return self.__entry_datetime

    @property
    def exit_datetime(self) -> Optional[int]: # Index in spot_data
        return self.__exit_datetime

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
            current_ticker_price = self.__broker.get_ticker_last_price(self.__ticker)
            if current_ticker_price is None: return 0 # Not in current chain / no price
            price_diff = current_ticker_price - self.__entry_price
        else: # Realized P&L for closed trade
            price_diff = self.__exit_price - self.__entry_price

        return self.__size * price_diff * self.__broker._option_multiplier

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent of initial premium paid/received."""
        if self.__entry_price == 0: return np.nan # Avoid division by zero

        if self.__exit_price is None:
            current_ticker_price = self.__broker.get_ticker_last_price(self.__ticker)
            if current_ticker_price is None: return np.nan
            price_diff = current_ticker_price - self.__entry_price
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
        price = self.__exit_price or self.__broker.get_ticker_last_price(self.__ticker)
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
        self._equity: Dict[pd.Timestamp, float] = {}

    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self._cash:.2f}, Equity={self.equity():.2f} ({active_trades} open trades)>'

    def new_order(self, strategy_id: str, position_id: str, leg_id: str,
                  ticker: str,
                  size: float, # Number of contracts
                  stop_loss: float = None, take_profit: float = None,
                  tag: object = None, trade: Trade = None) -> Order:
        assert size != 0, "Order size must be non-zero"

        order = Order(self, strategy_id, position_id, leg_id, ticker, size,
                      stop_loss=stop_loss, take_profit=take_profit, tag=tag, trade=trade)

        self.orders.append(order)
        return order

    def get_ticker_details(self, ticker: str) -> Optional[pd.Series]:
        return self._data.get_ticker_data(ticker)

    def get_ticker_execution_price(self, ticker: str, is_buy_order: bool) -> Optional[float]:
        """
        Determines execution price for an option.
        Simple logic: if limit, use limit if marketable. Else, use 'Last' or 'Mid'.
        A real system uses NBBO and considers liquidity.
        """
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None:
            # warnings.warn(f"No data for option {ticker} in current chain. Order cannot fill.")
            return None

        # Prioritize 'Last', then 'Ask' for buy, 'Bid' for sell, then calculated Mid
        fill_price = None
        last_price = ticker_data.get('close')
        ask_price = ticker_data.get('Ask')
        bid_price = ticker_data.get('Bid')

        if is_buy_order:
            market_price = None
            if pd.notna(ask_price) and ask_price > 0: market_price = ask_price
            elif pd.notna(last_price) and last_price > 0 : market_price = last_price
            elif pd.notna(bid_price) and bid_price > 0: market_price = bid_price # Less likely fill for buy
            fill_price = market_price
        else: # Sell order
            market_price = None
            if pd.notna(bid_price) and bid_price > 0: market_price = bid_price
            elif pd.notna(last_price) and last_price > 0: market_price = last_price
            elif pd.notna(ask_price) and ask_price > 0: market_price = ask_price # Less likely fill for sell
            fill_price = market_price
        
        return fill_price if pd.notna(fill_price) and fill_price > 0 else None


    def get_ticker_last_price(self, ticker: str) -> Optional[float]:
        """ Gets the last known price for an option, typically for MTM P&L. """
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None: return None
        
        last = ticker_data.get('close')
        if pd.notna(last) and last > 0: return last

        bid = ticker_data.get('Bid')
        ask = ticker_data.get('Ask')
        if pd.notna(bid) and bid > 0 and pd.notna(ask) and ask > 0:
            return (bid + ask) / 2
        if pd.notna(bid) and bid > 0: return bid
        if pd.notna(ask) and ask > 0: return ask
        return None


    def equity(self, ticker: str = None) -> float: # MTM equity
        if ticker:
            return sum(trade.value for trade in self.trades.get(ticker, []))
            # This seems simmilar to self.positions[ticker].pl
        else:
            mtm_value_of_open_positions = sum(trade.pl for trade in self.active_trades)
            return self._cash + mtm_value_of_open_positions

    @property
    def margin_available(self) -> float:
        # Simplified: Equity minus initial margin for any short positions (if allowed).
        # For now, assume all cash is available if no complex margin.
        # If selling options: self.equity() - margin_used_for_shorts

        # This is not standard margin, but represents cash tied up if we assume full payment for longs.
        # True margin_available is more complex. For now, let's use cash as a proxy for buying power.
        return self._cash # Simplification: assumes buying power is current cash.

    @property
    def active_trades(self) -> List[Trade]:
        return [trade for trades_list in self.trades.values() for trade in trades_list]

    @property
    def time(self): # Current spot data timestamp
        return self._data._time
    
    @property
    def spot(self) -> float:
        return self._data._spot

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
            # This requires ticker to be parsable or options_data to have expiry_date
            contract_details = self.get_ticker_details(ticker)
            if 'expiry_date' not in contract_details:
                warnings.warn(f"Cannot determine expiry for {ticker}. Skipping expiration check.")
                continue

            try:
                expiry_date = contract_details['expiry_date']
                if isinstance(expiry_date, (pd.Timestamp, datetime)):
                    expiry_date = expiry_date.date()
                else:
                    expiry_date = pd.to_datetime(expiry_date).date()
            except Exception as e:
                warnings.warn(f"Could not parse expiry for {ticker}: {e}. Skipping.")
                continue

            if current_date >= expiry_date:
                # Option expired or is expiring today
                # print(f"Option {ticker} expiring on {expiry_date} (current_date: {current_date})")

                for trade in list(trade_list): # Iterate copy of trade_list
                    
                    # Close the trade at this exit_price
                    self._close_trade(trade, self.get_ticker_execution_price(trade.ticker), tag="Expired")

                if not self.trades.get(ticker): # If all trades for this symbol are closed
                    symbols_to_remove_from_trades.append(ticker)
        
        for sym in symbols_to_remove_from_trades:
            if sym in self.trades and not self.trades[sym]: # double check if empty
                del self.trades[sym]

    def finalize(self): # Close all open positions at last available price
        for ticker in list(self.trades.keys()):
            for trade in list(self.trades.get(ticker, [])): # Iterate copy
                last_price = self.get_ticker_last_price(ticker)

                print(f"Finalizing trade for {ticker} at price {last_price}")
                self._close_trade(trade, last_price, tag="EndOfTest")

    # @profile
    def next(self): # Called for each spot bar
        self._process_orders()

        # Log equity
        self._equity[self.time] = self.equity()

        if self.equity() <= 0:
            # If equity is negative, set all to 0 and stop the simulation
            # For options, equity can go negative with short positions.
            # The definition of "out of money" might be different (e.g., margin call).
            # For simplicity, we'll keep the "stop if equity <= 0" rule.
            if self._equity[self.time] <=0:
                warnings.warn("Equity is zero or negative. Stopping simulation.", UserWarning)
                self.finalize() # Close all positions
                self._cash = 0
                raise _OutOfMoneyError
        

    def _process_orders(self):

        for order in list(self.orders): # Iterate a copy
            if order not in self.orders: continue # Already processed/canceled

            # Option orders are typically GTD (Good 'Til Day) or GTC.
            # For simplicity, assume orders persist until filled/canceled or end of day.
            # More realistically, unfilled day orders would be canceled by EOD.
            # Here, we process against the current options_data snapshot.
            exec_price = self.get_ticker_execution_price(order.ticker, order.is_long)
            print(f"exec_price: {exec_price}, order: {order.ticker}, size: {order.size}")

            if exec_price is None:
                # print(f"Order for {order.ticker} ({order.size} @ LMT {order.limit}) could not fill at current market.")
                # In a real system, unfilled limit orders might persist.
                # For this backtester, if it doesn't fill on this "tick" (daily snapshot), it waits.
                # User might need to cancel it via strategy logic.
                continue

            # Order can be filled
            # 1. Cost/Proceeds
            trade_value = exec_price * order.size * self._option_multiplier
            commission_cost = self._commission_per_contract * abs(order.size)

            # 2. Cash Check (Simplified)
            if order.is_long: # Buying an option
                required_cash = abs(trade_value) + commission_cost
                if self._cash < required_cash:
                    warnings.warn(f"Not enough cash for {order}. Has {self._cash:.2f}, needs {required_cash:.2f}. Order skipped.")
                    self.orders.remove(order)
                    continue
            # else: Shorting an option - margin would be checked here. Skipped for now.
            
            if order.size != 0: # If any part of the order remains to be opened
                if order.trade == None: # If not part of an existing trade
                    # Open a new trade
                    self._open_trade(order, exec_price)
                else: # If part of an existing trade, update the trade
                    self._reduce_trade(order, exec_price)

    def _reduce_trade(self, order: Order, price: float):
        # size_change is the amount by which the trade's size is changing.
        # e.g., trade.size = 10 (long), size_change = -5 (closing 5 contracts) -> new size = 5
        # e.g., trade.size = 10 (long), size_change = -10 (closing all) -> new size = 0
        trade = order.trade
        size_change = order.size # This is the amount to reduce the trade by
        assert trade.size * size_change <= 0, "size_change must be opposite or reduce existing trade size"
        assert abs(trade.size) >= abs(size_change)

        size_left = trade.size + size_change # e.g. 10 + (-5) = 5
        
        # Create a "closing" trade record for the portion being closed
        closed_portion_trade = trade._copy(
            size=-size_change, # The amount that was actually transacted to reduce
            exit_price=price,
            exit_datetime=self.time,
            exit_spot=self.spot,
            exit_tag=order.tag or "Partial Close" # Use order tag if available
        )
        # Update P&L for broker's cash based on this realized portion

        if size_left == 0: # Trade is fully closed
            self._close_trade(trade, price, order.tag)
        else: # Trade is partially closed
            trade._replace(size=size_left)
            # Add the record of the closed portion
            self.closed_trades.append(closed_portion_trade)
            trade_value = closed_portion_trade.size * price * self._option_multiplier
            commission_cost = self._commission_per_contract * abs(closed_portion_trade.size)
            self._cash += (trade_value - commission_cost) # Add P&L to cash
        
        self.orders.remove(order) # Order processed

    def _close_trade(self, trade: Trade, exec_price: float, tag: str = "Closed"):
        if trade in self.trades.get(trade.ticker, []):
            self.trades[trade.ticker].remove(trade)
            if not self.trades[trade.ticker]: # List is now empty
                del self.trades[trade.ticker]

        trade._replace(exit_price=exec_price, exit_datetime=self.time, exit_spot=self.spot, exit_tag=tag)
        self.closed_trades.append(trade)
        # Update cash based on the realized P&L of this trade
        trade_value = trade.size * exec_price * self._option_multiplier
        commission_cost = self._commission_per_contract * abs(trade.size)
        self._cash += (trade_value - commission_cost) # Add P&L to cash

    def _open_trade(self, order: Order, exec_price: float):
        trade = Trade(self, order.strategy_id, order.position_id, order.leg_id,
                      order.ticker, order.size, exec_price, self.time, self.spot,
                      stop_loss=order.stop_loss, take_profit=order.take_profit,
                      entry_tag=order.tag)
        if order.ticker not in self.trades:
            self.trades[order.ticker] = []
        self.trades[order.ticker].append(trade)
        trade_value = trade.size * exec_price * self._option_multiplier
        commission_cost = self._commission_per_contract * abs(trade.size)
        self._cash -= (trade_value + commission_cost)

        if order.ticker not in self.positions:
            self.positions[order.ticker] = Position(self, order.ticker)
        # Position object will query self.trades for its size/pl dynamically
        self.orders.remove(order) # Order processed

import time
class Backtest:
    def __init__(self,
                 db_path: str,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 100000,
                 commission_per_contract: float = 0.65, # New commission model
                 option_multiplier: int = 75, # New
                 ):

        self.db_path = db_path 

        self._broker_factory = partial(
            _Broker, cash=cash, commission_per_contract=commission_per_contract,
            option_multiplier=option_multiplier)
        self._strategy = strategy
        self._results: Optional[pd.Series] = None

    def run(self, **kwargs) -> pd.Series:
        data = _Data(self.db_path)
        data._table_names = data._table_names[90:703] # For testing, load only a few tables
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
        break_outer_loop = False
        for table in data._table_names:
            data.load_table(table)

            for row in data._spot_table.itertuples():
                # Process each row in the spot data
                data._time = row.Index
                data._spot = row.spot_price
                # Broker actions: expirations, process orders, update equity
                try:
                    # Only handle expirations on the last row of data._spot
                    if row.Index == data._spot_table.index[-1]:
                        print(row.Index.date())
                        broker.handle_expirations(row.Index.date())  # Pass date part
                    
                    # start = time.time()

                    broker.next()  # This will call _process_orders
                    # print(f"Processed {table} in {time.time() - start:.2f} seconds at {row.Index}")

                except _OutOfMoneyError:
                    print('Strategy ran out of money.')
                    progress_bar.close()
                    break_outer_loop = True
                    break
                except Exception as e:
                    print(f"Error in broker.next() or handle_expirations() on {table}: {e}")
                    traceback.print_exc()
                    progress_bar.close()
                    break_outer_loop = True
                    break  # Critical error
                # Strategy decision making
                try:
                    strategy.next()
                    processed_orders.extend(broker.orders)
                except Exception as e:
                    print(f"Error in strategy.next() on {table}: {e}")
                    traceback.print_exc()
                    progress_bar.close()
                    break_outer_loop = True
                    break # Critical error
                # Log processed orders from this step (optional, for detailed analysis)
                # processed_orders_during_run.extend(broker.orders) # Careful, broker.orders is dynamic

            progress_bar.update(1)
            if break_outer_loop == True:
                break


        else: # Loop completed without break
            final_positions = ({t: p.size for t, p in broker.positions.items()}
                                   | {'Cash': int(broker.margin_available)})
        
            broker.finalize() # Close any open positions at the end
            data.close()
        
        progress_bar.close()

        return processed_orders, final_positions, broker.closed_trades, broker.orders
    



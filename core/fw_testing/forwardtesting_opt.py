import traceback
import warnings
from datetime import datetime, date as DateObject # Added DateObject

from abc import ABC, abstractmethod
import pandas as pd
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from io import BytesIO
from functools import partial

import aiohttp
import asyncio
import re
import atexit
import threading

from live_data_fetcher import _Data, Endpoint
from sync_broker import _Broker
from core.backtesting_opt import Order,Trade,Position

class Strategy(ABC):
    def __init__(self, 
                 _broker: _Broker, 
                 _data: _Data, 
                 params: dict):
        self._broker: _Broker = _broker
        self._data: _Data = _data
        self._params = self._check_params(params)

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'
        
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
        """Process the next data point. This method is called on each new data point.
        """
        pass

    def buy(self, *,
        strategy_id: str,
        position_id: str,
        leg_id: str,
        ticker: str,
        quantity: float, # Number of lots
        stop_loss: float, take_profit: float, tag: str):
        assert quantity > 0, "Quantity for buying options must be positive"
        return self._broker.new_order(strategy_id, position_id, leg_id, ticker, quantity, stop_loss, take_profit, tag)

    def sell(self, *,
        strategy_id: str,
        position_id: str,
        leg_id: str,
        ticker: str,
        quantity: float, # Number of lots
        stop_loss: float, take_profit: float, tag: str):
        # Negative quantity for selling (to open short or close long)
        assert quantity > 0, "Quantity for selling options must be positive (use negative for broker call)"
        return self._broker.new_order(strategy_id, position_id, leg_id, ticker, -quantity, stop_loss, take_profit, tag)

    @property
    def time(self): # Current spot data timestamp
        return pd.Timestamp.now()
    
    @property
    def spot(self) -> float:
        return self.get_ticker_data(ticker="NIFTY50")  # Assuming NIFTY is the spot ticker

    @property
    def tte_to_expiry(self):
        raise NotImplementedError("Subclasses should implement tte_to_expiry mapping")
        # return self._data._tte_to_expiry

    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a specific ticker contract."""
        return self._data.get_ticker_data(ticker)

    @property
    def equity(self) -> float:          # MTM of all positions
        return self._broker._equity
    
    @property
    def cash(self) -> float:
        """Get the current cash balance in the broker account."""
        return self._broker._cash

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


class AlgoRunner:
    """Main class to run the strategy with real-time data"""
    def __init__(self, 
                 strategy: Type[Strategy],
                 *,
                 endpoint: Endpoint,
                 broker_adapter,
                 option_multiplier: int = 75,
                 update_interval: float = 1.0,
                 end_time: pd.Timestamp = pd.Timestamp('15:31:00')):
        self.endpoint = endpoint
        self._strategy = strategy
        self.broker_adapter = broker_adapter
        self.option_multiplier = option_multiplier
        self.update_interval = update_interval
        self.end_time = end_time

    def run(self, **strategy_kwargs):
        """Run the strategy in real-time"""
        print("Starting forward testing...")
        
        # Initialize components
        self._data = _Data(self.endpoint)
        self._broker = _Broker(data=self._data, broker_adapter=self.broker_adapter, option_multiplier=self.option_multiplier)
        self._strategy = self._strategy(self._broker, self._data, strategy_kwargs)
        processed_orders: List[Order] = []
        equity_curve = pd.Series(dtype=float)
        
        try:
            # Initialize strategy
            self._strategy.init()
            print("Strategy initialized successfully")
        except Exception as e:
            print(f'Strategy initialization failed: {e}')
            traceback.print_exc()
            return

        self._is_running = True
        iteration_count = 0
        
        print("Starting main trading loop...")
        while self._is_running:
            try:
                iteration_count += 1
                loop_start_time = time.time()

                # Update broker state (positions, orders, etc.)
                self._broker.next()
                equity_curve[loop_start_time] = self._broker.equity()

                # Call strategy next() method
                try:
                    self._strategy.next()
                    processed_orders.extend(self._broker.orders)
                except Exception as e:
                    print(f"Error in strategy.next(): {e}")
                    traceback.print_exc()

                # Check if end time reached
                if self.end_time and pd.Timestamp.now() >= self.end_time:
                    print(f"End time reached: {self.end_time}")
                    self._is_running = False

                # Performance monitoring
                loop_duration = time.time() - loop_start_time
                if iteration_count % 10 == 0:  # Log every 10 iterations
                    print(f"Iteration {iteration_count}: {pd.Timestamp.now()}, "
                          f"Loop time: {loop_duration:.3f}s")

                # Sleep for remaining time
                sleep_time = max(0, self.update_interval - loop_duration)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("Received interrupt signal, stopping...")
                self._is_running = False
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(self.update_interval)

        print("Forward testing completed")
        self.close()
        equity_curve = equity_curve.sort_index()

        return processed_orders, self._broker.closed_trades, equity_curve

    def stop(self):
        """Stop the algorithm"""
        self._is_running = False

    def close(self):
        """Cleanup resources"""
        if self._data:
            self._data.close()

        if self._broker:
            self._broker.disconnect()
        print("Cleanup completed")


''' Future development on AlgoRunner:
- Manage positions, orders, and track performance in real-time.
- Support restarting from a previous backtest state.
- Handle different markets and timezones.
- Persist results and equity curves for monitoring and analysis.
- Control features like termination, pausing, and resuming, squaring off spreads/positions and squaring off current positions.
'''


'''
1. How to get current algo updates
2. What things to test?
3. How to implement tte to expiry mapping?
4. How is blocking implemented?
'''
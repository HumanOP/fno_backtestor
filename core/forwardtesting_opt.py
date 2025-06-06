import traceback
import warnings
from datetime import datetime, date as DateObject # Added DateObject

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, List
import pandas as pd
import time

import time
from io import BytesIO

import aiohttp
import asyncio
import time
import re
import atexit
import threading

from .live_data_fetcher import _Data, Endpoint

class Strategy(ABC):
    def __init__(self, broker: '_Broker', _data: _Data, params: dict):
        self._broker: _Broker = broker
        self._data: _Data = _data
        self._params = self._check_params(params)
        self._records = {}
        self._start_on_day = 0


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
        """
        Called for each spot data bar AFTER options_data for the current day is loaded.
        Access `self.spot_data` for current spot bar.
        Access `self.options_data` for current day's option chain.
        """
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

    @property
    def time(self):
        return time.time()
    
    @property
    def spot(self):
        return self._data._spot         # Needs redefinition to getting underlying's data

    @property
    def tte_to_expiry(self):
        return self._data._tte_to_expiry

    @property
    def equity(self) -> float:          # MTM of all positions
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


class _Broker():
    def __init__():
        pass

''' Needed features/interfaces for _Broker:
- Order placement 
- MTM
- Position
- Tradebook (update_trades)
- Positions (update_positions)
- Orderbook (update_orderbook)
- Margin calculator
- Account balance
'''
        

class AlgoRunner:
    """Main class to run the strategy with real-time data"""
    def __init__(self, 
                 endpoint: Endpoint, 
                 strategy: Strategy, 
                 broker: '_Broker',
                 broker_creds: Optional[Dict[str, str]] = None,
                 update_interval: float = 1.0,
                 end_time: pd.Timestamp = None):
        self.endpoint = endpoint
        self._strategy_class = strategy
        self._broker_factory = broker
        self.broker_creds = broker_creds
        self.update_interval = update_interval
        self.end_time = end_time

    def run(self, **strategy_params):
        """Run the strategy in real-time"""
        print("Starting forward testing...")
        
        # Initialize components
        self._data = _Data(self.endpoint)
        self._broker = self._broker_factory(self.broker_creds)
        self._strategy = self._strategy_class(self._broker, self._data, strategy_params)
        
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
                self._broker.update_positions()
                self._broker.update_orders()

                # Call strategy next() method
                try:
                    self._strategy.next()
                except Exception as e:
                    print(f"Error in strategy.next(): {e}")
                    traceback.print_exc()

                # Check if end time reached
                if self.end_time and time.time() >= self.end_time:
                    print(f"End time reached: {self.end_time}")
                    self._is_running = False
                    break

                # Performance monitoring
                loop_duration = time.time() - loop_start_time
                if iteration_count % 10 == 0:  # Log every 10 iterations
                    print(f"Iteration {iteration_count}: {self._data._time}, "
                          f"Loop time: {loop_duration:.3f}s")

                # Sleep for remaining time
                sleep_time = max(0, self.update_interval - loop_duration)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("Received interrupt signal, stopping...")
                self._is_running = False
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(self.update_interval)

        print("Forward testing completed")
        self.cleanup()

    def stop(self):
        """Stop the algorithm"""
        self._is_running = False

    def cleanup(self):
        """Cleanup resources"""
        if self._data:
            self._data.cleanup()
        print("Cleanup completed")


''' Future development on AlgoRunner:
- Manage positions, orders, and track performance in real-time.
- Support restarting from a previous backtest state.
- Handle different markets and timezones.
- Persist results and equity curves for monitoring and analysis.
- Control features like termination, pausing, and resuming, squaring off spreads/positions and squaring off current positions.
'''

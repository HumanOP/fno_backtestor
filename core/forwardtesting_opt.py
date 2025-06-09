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
from backtesting_opt import Trade, Order, Position

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

    # @property
    # def equity(self) -> float:          # MTM of all positions
    #     return self._broker.equity()

    # def position(self, ticker: str) -> 'Position': # Now takes ticker
    #     return self._broker.positions.get(ticker, Position(self._broker, ticker, 0)) # Return empty if not found

    # @property
    # def orders(self) -> 'List[Order]':
    #     return self._broker.orders

    # # trades() and closed_trades() now refer to option trades
    # def trades(self, ticker: str = None) -> 'Tuple[Trade, ...]':
    #     if ticker:
    #         return tuple(self._broker.trades.get(ticker, []))
    #     return tuple(trade for trades_list in self._broker.trades.values() for trade in trades_list)
    
    # @property
    # def active_trades(self) -> 'Tuple[Trade, ...]':
    #     return tuple(trade for trades_list in self._broker.trades.values() for trade in trades_list)


    # @property
    # def closed_trades(self) -> 'Tuple[Trade, ...]':
    #     return tuple(self._broker.closed_trades)


class _Broker():
    def __init__(self, *,
                 data: _Data,
                 cash: float,
                 commission_per_contract: float,
                 option_multiplier: int,
                 margin_requirement: float = 0.2):  # Added margin requirement parameter
        """Initialize the broker with data source and trading parameters."""
        assert cash > 0, f"cash should be > 0, is {cash}"
        assert commission_per_contract >= 0, "commission_per_contract should be >= 0"
        assert option_multiplier > 0, "option_multiplier must be positive"
        assert 0 < margin_requirement <= 1, "margin_requirement must be between 0 and 1"

        self._data = data
        self._cash = cash
        self._commission_per_contract = commission_per_contract
        self._option_multiplier = option_multiplier
        self._margin_requirement = margin_requirement

        # Trading state
        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self._equity: Dict[pd.Timestamp, float] = {}
        self._margin_used: float = 0.0

    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self._cash:.2f}, Equity={self.equity():.2f} ({active_trades} open trades)>'

    def new_order(self, strategy_id: str, position_id: str, leg_id: str,
                  ticker: str, size: float, stop_loss: float = None, 
                  take_profit: float = None, tag: object = None, 
                  trade: Trade = None) -> Order:
        """Create and queue a new order."""
        assert size != 0, "Order size must be non-zero"
        order = Order(self, strategy_id, position_id, leg_id, ticker, size,
                     stop_loss=stop_loss, take_profit=take_profit, 
                     tag=tag, trade=trade)
        self.orders.append(order)
        return order

    def get_ticker_details(self, ticker: str) -> Optional[pd.Series]:
        """Get current market data for a ticker."""
        return self._data.get_ticker_data(ticker)

    def get_ticker_execution_price(self, ticker: str, is_buy_order: bool) -> Optional[float]:
        """Get execution price for an order based on market data."""
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None:
            return None

        last_price = ticker_data.get('close')
        ask_price = ticker_data.get('Ask')
        bid_price = ticker_data.get('Bid')

        if is_buy_order:
            if pd.notna(ask_price) and ask_price > 0:
                return ask_price
            elif pd.notna(last_price) and last_price > 0:
                return last_price
            elif pd.notna(bid_price) and bid_price > 0:
                return bid_price
        else:  # Sell order
            if pd.notna(bid_price) and bid_price > 0:
                return bid_price
            elif pd.notna(last_price) and last_price > 0:
                return last_price
            elif pd.notna(ask_price) and ask_price > 0:
                return ask_price
        return None

    def get_ticker_last_price(self, ticker: str) -> Optional[float]:
        """Get last known price for a ticker."""
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None:
            return None
        last = ticker_data.get('close')
        if pd.notna(last) and last > 0:
            return last
        return None

    def equity(self) -> float:
        """Calculate current equity (cash + unrealized P&L)."""
        unrealized_pl = sum(trade.pl for trades in self.trades.values() 
                          for trade in trades)
        return self._cash + unrealized_pl

    def update_positions(self):
        """Update position states based on current trades."""
        for ticker, trades in self.trades.items():
            if ticker not in self.positions:
                self.positions[ticker] = Position(self, ticker)
            position = self.positions[ticker]
            # Position size and P&L are calculated on-demand through properties

    def update_orders(self):
        """Process pending orders."""
        for order in self.orders[:]:  
            if self._process_order(order):
                self.orders.remove(order)

    def calculate_margin_requirement(self, ticker: str, size: float) -> float:
        """Calculate margin requirement for a position."""
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None:
            return 0.0
        
        # Get current price
        price = self.get_ticker_last_price(ticker)
        if price is None:
            return 0.0
            
        # Calculate position value
        position_value = abs(size) * price * self._option_multiplier
        
        # Calculate margin requirement
        margin = position_value * self._margin_requirement
        
        # Add additional margin for short positions
        if size < 0:
            margin *= 1.5  # Higher margin requirement for short positions
            
        return margin

    def get_available_margin(self) -> float:
        """Get available margin for new positions."""
        return self._cash - self._margin_used

    def update_margin(self):
        """Update margin usage based on current positions."""
        total_margin = 0.0
        for ticker, trades in self.trades.items():
            position_size = sum(trade.size for trade in trades)
            if position_size != 0:
                total_margin += self.calculate_margin_requirement(ticker, position_size)
        self._margin_used = total_margin

    def can_open_position(self, ticker: str, size: float) -> bool:
        """Check if we can open a new position based on margin requirements."""
        required_margin = self.calculate_margin_requirement(ticker, size)
        return required_margin <= self.get_available_margin()

    def _process_order(self, order: Order) -> bool:
        """Process a single order and return True if filled."""
        execution_price = self.get_ticker_execution_price(order.ticker, order.size > 0)
        if execution_price is None:
            return False

        # Calculate commission
        commission = abs(order.size) * self._commission_per_contract
        cost = order.size * execution_price * self._option_multiplier + commission

        # Check margin requirements for new positions
        if order.trade is None and not self.can_open_position(order.ticker, order.size):
            return False

        # Check if we have enough cash
        if cost > self._cash:
            return False

        # Create or update trade
        if order.trade is None:  # New position
            trade = Trade(self, order.strategy_id, order.position_id, 
                         order.leg_id, order.ticker, order.size,
                         execution_price, self._data._time, self._data._spot,
                         order.stop_loss, order.take_profit, order.tag)
            if order.ticker not in self.trades:
                self.trades[order.ticker] = []
            self.trades[order.ticker].append(trade)
        else:  # Closing position
            order.trade.close(abs(order.size), order.tag)
            if order.trade.size == 0:  # Position fully closed
                self.trades[order.ticker].remove(order.trade)
                self.closed_trades.append(order.trade)

        # Update cash and margin
        self._cash -= cost
        self.update_margin()
        return True

    def handle_expirations(self, current_date: DateObject):
        """Handle option expirations for the current date."""
        for ticker, trades in list(self.trades.items()):
            for trade in trades[:]:  # Copy list to allow modification
                if self._is_expired(trade.ticker, current_date):
                    # Close expired position
                    trade.close(abs(trade.size), "EXPIRED", finalize=True)
                    self.trades[ticker].remove(trade)
                    self.closed_trades.append(trade)

    def _is_expired(self, ticker: str, current_date: DateObject) -> bool:
        """Check if an option has expired."""
        ticker_data = self.get_ticker_details(ticker)
        if ticker_data is None:
            return False
        expiry_date = ticker_data.get('expiry_date')
        return expiry_date is not None and expiry_date <= current_date

    def next(self):
        """Process next market data update."""
        self.update_positions()
        self.update_orders()
        self.update_margin()  # Update margin usage
        # Record equity
        self._equity[self._data._time] = self.equity()

    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions including size, value, and P&L."""
        summary = {}
        for ticker, position in self.positions.items():
            trades = self.trades.get(ticker, [])
            if not trades:
                continue
                
            size = sum(trade.size for trade in trades)
            value = sum(trade.value for trade in trades)
            pl = sum(trade.pl for trade in trades)
            
            summary[ticker] = {
                'size': size,
                'value': value,
                'pl': pl,
                'margin_required': self.calculate_margin_requirement(ticker, size)
            }
        return summary

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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')))
import traceback
import warnings
from typing import Dict, Optional, List, Any
import logging
from math import copysign
from datetime import datetime, date as DateObject
import numpy as np
import pandas as pd
import pytz

from live_data_fetcher import _Data, Endpoint, QuestDBClient
from sync_ibkr_br_adapter import IBKRBrokerAdapter
from core.backtesting_opt import Order,Trade,Position


class _Broker:
    """Synchronous broker class for easier usage."""   
    def __init__(self, *,  data: _Data, broker_adapter, option_multiplier: int):
        """Initialize the broker with broker adapter only."""
        assert option_multiplier > 0, "option_multiplier must be positive"
        assert broker_adapter is not None, "broker adapter must be provided"

        logging.basicConfig(filename='broker_log.txt', level=logging.INFO)
        self._adapter = broker_adapter
        self._option_multiplier = option_multiplier
        self._data = data

        # These are the adapter books
        self.orderbook: List[Dict[str, Any]] = []
        self.tradebook: List[Dict[str, Any]] = []
        self.positionbook: List[Dict[str, Any]] = []

        # These are the internal books
        self.orders: List[Order] = [] # These are loose orders by STRATEGY
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {} # Updated as trades occur

        self.active_order_ids={}

        self.equity: float = 0.0
        self.cash: float = 0.0

        # Connect to broker
        if not self._adapter.connect():
            logging.error("Failed to connect to broker")
            raise ConnectionError("Failed to connect to broker")

        # Set order fill callback
        self._adapter.set_order_fill_callback(self.on_order_fill)
        
        # Initialize account info
        self.update_account_info()

    @property
    def time(self): # Current spot data timestamp
        return pd.Timestamp.now()
    
    @property
    def spot(self) -> float:
        return self._data._spot
    
    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self.cash:.2f}, Equity={self.equity:.2f} ({active_trades} open trades)>'

    def update_account_info(self) -> dict:
        """Fetch comprehensive account information from the broker adapter."""
        try:
            self.cash, self.equity = self._adapter.get_account_info()
            logging.info(f"Account info updated: Cash={self.cash:.2f}, Equity={self.equity:.2f}")
        except Exception as e:
            logging.error(f"Error updating account info: {str(e)}")
            return {}

    def update_positions(self):
        """Update positions from broker adapter."""
        try:
            self.positionbook = self._adapter.get_positions()
            logging.info("Positions updated")
        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")

    def update_orders(self):
        """Update orders from broker adapter."""
        try:
            self.orderbook = self._adapter.get_orders()
            logging.info("Orders updated")
        except Exception as e:
            logging.error(f"Error updating orders: {str(e)}")

    def update_trades(self):
        """Update trades from broker adapter."""
        try:
            self.tradebook = self._adapter.get_trades()
            logging.info("Trades updated")
        except Exception as e:
            logging.error(f"Error updating trades: {str(e)}")

    def next(self):
        """Process next market data update by fetching latest broker data."""
        self.on_order_fill()
        self._process_orders()
        self._process_trades()

        # self.update_account_info()
        self.update_positions()
        self.update_orders()
        self.update_trades()

    def _process_orders(self):
        for order in list(self.orders): # Iterate a copy
            print(f"Processing order: {order}")
            action = "BUY" if order.size > 0 else "SELL"
            required_margin = self.margin_impact(order.ticker, action, order.size)
            if self.cash < required_margin:
                warnings.warn(f"Not enough cash for {order}. Has {self.cash:.2f}, needs {required_margin:.2f}. Order skipped.")
                self.orders.remove(order)
                continue
            
            if order.size != 0: # If any part of the order remains to be opened
                order_id = self._adapter.place_order(order.ticker, action, order.size, "MARKET", order.tag)
                self.active_order_ids[order_id] = order

                self.orders.remove(order) # Order sent to broker, remove from internal list

        return self.active_order_ids
        
    def _open_trade(self, order: Order, entry_price: Optional[float], entry_datetime: Optional[pd.Timestamp]):
        trade = Trade(
            self, order.strategy_id, order.position_id, order.leg_id, order.ticker, order.size,
            entry_price, entry_datetime, None, order.stop_loss, order.take_profit, order.tag)
        if order.ticker not in self.trades:
            self.trades[order.ticker] = []
        self.trades[order.ticker].append(trade)
        order.trade=trade
        # Cash will be updated automatically broker.next()

        if order.ticker not in self.positions:
            self.positions[order.ticker] = Position(self, order.ticker)

    def _reduce_trade(self, order: Order, exit_price: Optional[float] = None, exit_datetime: Optional[pd.Timestamp] = None):
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
            exit_price=exit_price,
            exit_datetime=exit_datetime,
            exit_spot=self.spot,
            exit_tag=order.tag or "Partial Close" # Use order tag if available
        )
        # Update P&L for broker's cash based on this realized portion

        if size_left == 0: # Trade is fully closed
            self._close_trade(trade, exit_price=exit_price, exit_datetime=exit_datetime, tag=order.tag)
        else: # Trade is partially closed
            trade._replace(size=size_left)
            # Add the record of the closed portion
            self.closed_trades.append(closed_portion_trade)
            # trade_value = closed_portion_trade.size * price * self._option_multiplier
            # commission_cost = 0.65 * abs(closed_portion_trade.size) # commision is fixed
            # self._cash += (trade_value - commission_cost) # Add P&L to cash
        
        self.orders.remove(order) # Order processed
    
    def _close_trade(self, trade: Trade, exit_price: Optional[float] = None, exit_datetime: Optional[pd.Timestamp] = None, tag: str = "Closed"):
        if trade in self.trades.get(trade.ticker, []):
            self.trades[trade.ticker].remove(trade)
            if not self.trades[trade.ticker]: # List is now empty
                del self.trades[trade.ticker]

        trade._replace(exit_price=exit_price, exit_datetime=exit_datetime, exit_spot=self.spot, exit_tag=tag)
        self.closed_trades.append(trade)
        # Update cash based on the realized P&L of this trade
        # trade_value = trade.size * exit_price * self._option_multiplier
        # commission_cost = 0.65 * abs(trade.size) # commision is fixed fornow
        # self._cash += (trade_value - commission_cost) # Add P&L to cash

    def get_ticker_last_price(self,ticker):
        ticker_data = self._data.get_ticker_data(ticker)
        if ticker_data is None:
            # warnings.warn(f"No data for option {ticker} in current chain. Order cannot fill.")
            return None

        last_price = ticker_data.get('close')
        if last_price!=None:
            return last_price

    def _process_trades(self):
        for trade in self.trades:
            current_ticker_price=None
            ''' for pnl '''
            if trade.exit_price is None: # Mark-to-market P&L for open trade
                current_ticker_price = self.get_ticker_last_price(trade.ticker)
                if current_ticker_price is None: # Not in current chain / no price
                    trade.set_pl(0)
                    trade.set_pl_pct(0)
                    trade.set_value(0)
                    continue
                price_diff = current_ticker_price - trade.entry_price
            else: # Realized P&L for closed trade
                price_diff = trade.exit_price - trade.entry_price

            trade.set_pl(trade.size * price_diff * self._option_multiplier)

            # this is for pnl percentage
            initial_value_per_contract = trade.entry_price
            pnl_per_contract = price_diff * copysign(1, trade.size) # To align with P&L direction
            trade.set_pl_pct(pnl_per_contract / initial_value_per_contract if initial_value_per_contract else np.nan)

            #this for trade value
            trade.set_value(trade.size * current_ticker_price * self._option_multiplier)

    def is_market_open(self) -> bool:
        """Check if the market is open (NSE hours)."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

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
    
    def margin_impact(self, ticker: str, action: str, quantity: float) -> Optional[float]:
        """Calculate margin impact of an order."""
        margin_impact = self._adapter.get_margin_impact(ticker, action, quantity)
        return margin_impact

    def on_order_fill(self) -> None:
        for order_id, order in self.active_order_ids.items():
            # price = self.get_ticker_last_price(ticker=order.ticker)  # NOT CORRECT ----- ( this price we will get from ibkr's some function)
            print(f"Checking trades: {self._adapter.get_trades()}")
            for trade in self._adapter.get_trades():
                if trade["orderStatus"].orderId == order_id and trade["orderStatus"].status == 'Filled': 
                    if order.trade is None:
                        self._open_trade(       # set new internal trade object
                            order,
                            entry_price=trade['fills'][0].execution.price,
                            entry_datetime=trade['fills'][0].execution.time
                        )
                        self.active_order_ids.pop(order_id)

                    else: 
                        self._reduce_trade(     # reduce and set new internal trade object
                            order,
                            exit_price=trade['fills'][0].execution.price,
                            exit_datetime=trade['fills'][0].execution.time
                        )
                        self.active_order_ids.pop(order_id)

            
    def cancel_all_orders(self) -> None:
        """Cancel all open orders via the broker adapter."""
        try:
            self._adapter.cancel_all_orders()
        except Exception as e:
            logging.error(f"Error cancelling orders: {str(e)}")

    def disconnect(self) -> None:
        """Disconnect from the broker via the adapter."""
        try:
            self._adapter.disconnect()
            logging.info("Broker disconnected")
        except Exception as e:
            logging.error(f"Error disconnecting broker: {str(e)}")


from io import BytesIO
# Usage example
if __name__ == "__main__":
    endpoint = Endpoint(
        host='qdb3.twocc.in', 
        https=True, 
        username='2Cents', 
        password='2Cents1012cc'
    )
    
    fetcher = _Data(endpoint)
    # table_names_data = QuestDBClient(endpoint).execute_query("SHOW TABLES")
    # df = pd.read_csv(BytesIO(table_names_data))
    # table_names = df["table_name"].tolist()
    # print(table_names)


    # Initialize the IBKR adapter
    ibkr_adapter = IBKRBrokerAdapter(host='localhost', port=7497, client_id=1)
    try:
        t = 10
        import time
        broker = _Broker(option_multiplier=1, broker_adapter=ibkr_adapter,data=fetcher)
        ticker = "NIFTY10JUL2525400CE"
        order = broker.new_order("1","1","1", ticker, 75)
        margin_impact = broker.margin_impact(ticker, action="BUY", quantity=75)
        print(f"Margin Impact for {ticker}: {margin_impact}")
        while t>0:
            t = t -1
            broker.next()  # Process the order
            
            time.sleep(1)

        # whatif = ibkr_adapter.ib.whatIfOrder(ib_contract, order)
        # print(whatif)
        # return float(whatif.initMarginChange) if whatif and whatif.initMarginChange else None
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if 'broker' in locals():
            broker.disconnect()
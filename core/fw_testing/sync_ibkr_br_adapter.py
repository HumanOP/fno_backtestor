import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')))
import traceback
import warnings
from datetime import datetime, date as DateObject
import pandas as pd
from typing import Dict, Optional, List, Any
import time
import logging
from math import copysign
import numpy as np
# from core.backtesting_opt import Trade, Order, Position
from live_data_fetcher import _Data, Endpoint, QuestDBClient
from ib_async import IB, Contract, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Order, Fill
from core.backtesting_opt import Order,Trade,Position
import pytz

class IBKRBrokerAdapter:
    """Interactive Brokers-specific broker adapter implementation - Synchronous version."""
    
    def __init__(self, host: str = 'localhost', port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 10
        self._order_fill_callback = None

    def connect(self) -> bool:
        """Establish connection to IBKR."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            logging.info("Connected to IBKR successfully")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            self.is_connected = False
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect to IBKR."""
        self.reconnect_attempts = 0
        while self.reconnect_attempts < self.max_reconnect_attempts:
            if self.is_connected:
                logging.info("Already connected to IBKR.")
                return True
            logging.warning(f"Attempting reconnect ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
            try:
                self.ib = IB()
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.is_connected = True
                logging.info("Reconnected to IBKR.")
                return True
            except Exception as e:
                logging.error(f"Reconnection failed: {str(e)}")
                self.reconnect_attempts += 1
                time.sleep(self.reconnect_delay)
        logging.error("Max reconnection attempts reached.")
        self.is_connected = False
        return False

    def check_connection(self) -> bool:
        """Check if connected to IBKR."""
        try:
            return self.ib.isConnected()
        except Exception:
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """Fetch comprehensive account information from IBKR."""
        try:
            account_summary = self.ib.accountSummary()
            logging.info(f"Raw account summary: {account_summary}")
            summary = {tag.tag: float(tag.value) if tag.value.replace('.', '', 1).isdigit() else tag.value 
                      for tag in account_summary}
            cash = summary.get('AvailableFunds', 0.0)
            equity = summary.get('OptionMarketValue', 0.0) # Here more values can be added, cuurent ok since we are doing options only
            return cash, equity
        except Exception as e:
            logging.error(f"Error fetching account info from IBKR: {str(e)}")
            return 0.0, 0.0

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch current positions from IBKR."""
        try:
            portfolio = self.ib.portfolio()
            return [
                {
                    'ticker': pos.contract.localSymbol,
                    'contract': pos.contract,
                    'position': pos.position,
                    'marketPrice': pos.marketPrice,
                    'marketValue': pos.marketValue,
                    'averageCost': pos.averageCost,
                    'unrealizedPNL': pos.unrealizedPNL,
                    'realizedPNL': pos.realizedPNL,
                    'account': pos.account
                } for pos in portfolio
            ]
        except Exception as e:
            logging.error(f"Error fetching positions from IBKR: {str(e)}")
            return {}
        
    def get_orders(self) -> List[Dict[str, Any]]:
        """Fetch current orders from IBKR."""
        try:
            orders = self.ib.orders()
            return [
                {
                    'permId': order.permId,
                    'action': order.action,
                    'orderType': order.orderType,
                    'lmtPrice': order.lmtPrice,
                    'auxPrice': order.auxPrice,
                    'tif': order.tif,
                    'ocaType': order.ocaType,
                    'displaySize': order.displaySize,
                    'rule80A': order.rule80A,
                    'openClose': order.openClose,
                    'volatilityType': order.volatilityType,
                    'deltaNeutralOrderType': order.deltaNeutralOrderType,
                    'referencePriceType': order.referencePriceType,
                    'account': order.account,
                    'clearingIntent': order.clearingIntent,
                    'cashQty': order.cashQty,
                    'dontUseAutoPriceForHedge': order.dontUseAutoPriceForHedge,
                    'filledQuantity': order.filledQuantity,
                    'refFuturesConId': order.refFuturesConId,
                    'shareholder': order.shareholder
                } for order in orders
            ]
        except Exception as e:
            logging.error(f"Error fetching orders from IBKR: {str(e)}")
            return []

    def get_trades(self) -> List[Dict[str, Any]]:
        """Fetch current trades from IBKR."""
        try:
            trades = self.ib.trades()
            return [
                {
                    'contract': trade.contract,
                    'order': trade.order,
                    'orderStatus': trade.orderStatus,
                    'fills': trade.fills,
                    'log': trade.log,
                    'advancedError': trade.advancedError
                } for trade in trades
            ]
        except Exception as e:
            logging.error(f"Error fetching trades from IBKR: {str(e)}")
            return []

    def qualify_contract(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Qualify a ticker into an IBKR contract."""
        try:
            contract = Contract()
            contract.localSymbol = ticker
            contract.secType = 'OPT'  # Assuming options trading
            contract.exchange = 'NSE'
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                logging.error(f"Failed to qualify contract for ticker {ticker}")
                return None
            return {
                'ticker': qualified[0].localSymbol,
                'conId': qualified[0].conId,
                'contract': qualified[0]
            }
        except Exception as e:
            logging.error(f"Error qualifying contract {ticker}: {str(e)}")
            return None

    def place_order(self, ticker: str, action: str, quantity: float, order_type: str, order_params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Place an order with IBKR."""
        contract = self.qualify_contract(ticker)
        if contract is None:
            return None
        ib_contract = contract.get('contract')
        if not ib_contract:
            logging.error("No valid IBKR contract provided")
            return None
        try:
            if order_type == 'MARKET':
                order = MarketOrder(action, quantity)
            elif order_type == 'LIMIT':
                order = LimitOrder(action, quantity, order_params.get('limit_price'))
            elif order_type == 'STOP':
                order = StopOrder(action, quantity, order_params.get('stop_price'))
            elif order_type == 'STOP_LIMIT':
                order = StopLimitOrder(action, quantity, order_params.get('limit_price'), 
                                     order_params.get('stop_price'))
            else:
                logging.error(f"Unsupported order type: {order_type}")
                return None
            order.transmit = True  # This makes the order placement production-ready
            trade = self.ib.placeOrder(ib_contract, order)
            logging.info(f"Order placed: {contract.get('ticker')} {order_type} {action} {quantity}")
            return trade.order.orderId if trade else None
        except Exception as e:
            logging.error(f"Error placing {order_type} order for {contract.get('ticker')}: {str(e)}")
            return None

    def get_margin_impact(self, ticker: str, action: str, quantity: float) -> Optional[float]:
        """Calculate margin impact of an order with IBKR."""
        contract = self.qualify_contract(ticker)
        if contract is None:
            return None
        ib_contract = contract.get('contract')
        if not ib_contract:
            logging.error("No valid IBKR contract provided")
            return None
        try:
            order = MarketOrder(action, quantity)
            whatif = self.ib.whatIfOrder(ib_contract, order)
            return float(whatif.initMarginChange) if whatif and whatif.initMarginChange else None
        except Exception as e:
            logging.error(f"Error calculating margin for {contract.get('ticker')}: {str(e)}")
            return None

    def cancel_all_orders(self) -> None:
        """Cancel all open orders with IBKR."""
        try:
            self.ib.cancelAllOrders()
            logging.info("All orders cancelled")
        except Exception as e:
            logging.error(f"Error cancelling orders: {str(e)}")

    def close_positions(self, contract: Optional[Dict[str, Any]]) -> List[Any]:
        """Close all or specific positions with IBKR."""
        ib_contract = contract.get('contract') if contract else None
        try:
            if ib_contract:
                positions = [p for p in self.ib.portfolio() if p.contract.conId == ib_contract.conId]
            else:
                positions = self.ib.portfolio()
            trades = []
            for pos in positions:
                action = 'SELL' if pos.position > 0 else 'BUY'
                quantity = abs(pos.position)
                order = MarketOrder(action, quantity)
                trade = self.ib.placeOrder(pos.contract, order)
                trades.append(trade)
            logging.info(f"Closed {len(trades)} positions")
            return [trade.order.orderId for trade in trades]
        except Exception as e:
            logging.error(f"Error closing positions: {str(e)}")
            return []

    def set_order_fill_callback(self, callback):
        """Set callback for order fill events."""
        self._order_fill_callback = callback
        self.ib.execDetailsEvent += self._on_exec_details

    def _on_exec_details(self, trade: 'Trade', fill: 'Fill'):
        """Handle IBKR execution details and invoke callback."""
        if self._order_fill_callback and trade.orderStatus.status == 'Filled':
            self._order_fill_callback(
                order_id=trade.order.orderId,
                quantity=fill.execution.cumQty,
                price=fill.execution.price,
                ticker=trade.contract.localSymbol
            )

    def on_order_fill(self, order_id: Any, quantity: float, price: float, ticker: str) -> None:
        """Handle order fill events (called by Broker)."""
        if self._order_fill_callback:
            self._order_fill_callback(order_id, quantity, price, ticker)

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        try:
            self.ib.disconnect()
            self.is_connected = False
            logging.info("Disconnected from IBKR")
        except Exception as e:
            logging.error(f"Error disconnecting from IBKR: {str(e)}")


class _Broker:
    """Synchronous broker class for easier usage."""   
    def __init__(self, *, broker_adapter, option_multiplier: int,data:_Data):
        """Initialize the broker with broker adapter only."""
        assert option_multiplier > 0, "option_multiplier must be positive"
        assert broker_adapter is not None, "broker adapter must be provided"

        logging.basicConfig(filename='broker_log.txt', level=logging.INFO)
        self._adapter = broker_adapter
        self._option_multiplier = option_multiplier
        self._data=data

        # These are the adapter books
        self.orderbook: List[Dict[str, Any]] = []
        self.tradebook: List[Dict[str, Any]] = []
        self.positionbook: List[Dict[str, Any]] = []

        # These are the internal books
        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {} # Updated as trades occur

        self.equity: float = 0.0
        self.cash: float = 0.0
        self._broker_orders: Dict[Any, Order] = {}  # Map broker order ID to local Order

        # Connect to broker
        if not self._adapter.connect():
            logging.error("Failed to connect to broker")
            raise ConnectionError("Failed to connect to broker")

        # Set order fill callback
        self._adapter.set_order_fill_callback(self.on_order_fill)
        
        # Initialize account info
        self.update_account_info()
     
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
            self.positions = self._adapter.get_positions()
            logging.info("Positions updated")
        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")

    def update_orders(self):
        """Update orders from broker adapter."""
        try:
            self.orders = self._adapter.get_orders()
            logging.info("Orders updated")
        except Exception as e:
            logging.error(f"Error updating orders: {str(e)}")

    def update_trades(self):
        """Update trades from broker adapter."""
        try:
            self.trades = self._adapter.get_trades()
            logging.info("Trades updated")
        except Exception as e:
            logging.error(f"Error updating trades: {str(e)}")

    def next(self):
        """Process next market data update by fetching latest broker data."""
        self._process_orders()
        self._process_trades()

        self.update_account_info()
        self.update_positions()
        self.update_orders()
        self.update_trades()

    
    
    def get_ticker_execution_price(self, ticker: str, is_buy_order: bool) -> Optional[float]:
        """
        Determines execution price for an option.
        Simple logic: if limit, use limit if marketable. Else, use 'Last' or 'Mid'.
        A real system uses NBBO and considers liquidity.
        """
        ticker_data = self._data.get_ticker_data(ticker)
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
    
    def _process_orders(self):

        for order in list(self.orders): # Iterate a copy
            if order not in self.orders: continue # Already processed/canceled

            
            exec_price = self.get_ticker_execution_price(order.ticker, order.is_long)
            print(f"exec_price: {exec_price}, order: {order.ticker}, size: {order.size}")

            if exec_price is None:
           
                continue

        
            trade_value = exec_price * order.size * self._option_multiplier
            commission_cost = 0.65 * abs(order.size) # comission hardcoded

  
            if order.is_long: # Buying an option
                required_cash = abs(trade_value) + commission_cost
                if self._cash < required_cash:
                    warnings.warn(f"Not enough cash for {order}. Has {self._cash:.2f}, needs {required_cash:.2f}. Order skipped.")
                    self.orders.remove(order)
                    continue
        
            
            if order.size != 0: # If any part of the order remains to be opened
                if order.trade == None: # If not part of an existing trade
                    # Open a new trade
                    self._open_trade(order, exec_price)
                else: # If part of an existing trade, update the trade
                    self._reduce_trade(order, exec_price)

    def _open_trade(self, order: Order, exec_price: float):

        if exec_price is None:
            warnings.warn(f"exec_price is None for order {order.ticker}. Cannot open trade. Order skipped.")
            self.orders.remove(order)
            return

        trade = Trade(self, order.strategy_id, order.position_id, order.leg_id,
                      order.ticker, order.size, exec_price, self.time, self.spot,
                      stop_loss=order.stop_loss, take_profit=order.take_profit,
                      entry_tag=order.tag)
        if order.ticker not in self.trades:
            self.trades[order.ticker] = []
        self.trades[order.ticker].append(trade)
        trade_value = trade.size * exec_price * self._option_multiplier
        commission_cost = 0.65 * abs(trade.size) # commision is fixed for now
        self._cash -= (trade_value + commission_cost)

        if order.ticker not in self.positions:
            self.positions[order.ticker] = Position(self, order.ticker)
        # Position object will query self.trades for its size/pl dynamically
        self.orders.remove(order) # Order processed

    def _reduce_trade(self, order: Order, price: float):
        # size_change is the amount by which the trade's size is changing.
        # e.g., trade.size = 10 (long), size_change = -5 (closing 5 contracts) -> new size = 5
        # e.g., trade.size = 10 (long), size_change = -10 (closing all) -> new size = 0
        trade = order.trade
        size_change = order.size # This is the amount to reduce the trade by
        assert trade.size * size_change <= 0, "size_change must be opposite or reduce existing trade size"
        assert abs(trade.size) >= abs(size_change)

        # Safety check for price
        if price is None:
            price = trade.entry_price
            warnings.warn(f"price is None for trade {trade.ticker}. Using entry price {price} as fallback.")

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
            commission_cost = 0.65 * abs(closed_portion_trade.size) # commision is fixed
            self._cash += (trade_value - commission_cost) # Add P&L to cash
        
        self.orders.remove(order) # Order processed

    def _close_trade(self, trade: Trade, exec_price: float, tag: str = "Closed"):
        if trade in self.trades.get(trade.ticker, []):
            self.trades[trade.ticker].remove(trade)
            if not self.trades[trade.ticker]: # List is now empty
                del self.trades[trade.ticker]

        # Safety check for exec_price
        if exec_price is None:
            exec_price = trade.entry_price
            warnings.warn(f"exec_price is None for trade {trade.ticker}. Using entry price {exec_price} as fallback.")

        trade._replace(exit_price=exec_price, exit_datetime=self.time, exit_spot=self.spot, exit_tag=tag)
        self.closed_trades.append(trade)
        # Update cash based on the realized P&L of this trade
        trade_value = trade.size * exec_price * self._option_multiplier
        commission_cost = 0.65 * abs(trade.size) # commision is fixed fornow
        self._cash += (trade_value - commission_cost) # Add P&L to cash

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

    def new_order(self, ticker: str, action: str, quantity: float, order_type: str = "MARKET", order_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Place an order with the broker."""
        order_id = self._adapter.place_order(ticker, action, quantity, order_type, order_params)
        # assert size != 0, "Order size must be non-zero"

        # order = Order(self, strategy_id, position_id, leg_id, ticker, size,
        #               stop_loss=stop_loss, take_profit=take_profit, tag=tag, trade=trade)

        # self.orders.append(order)
        return order_id
    
    def margin_impact(self, ticker: str, action: str, quantity: float) -> Optional[float]:
        """Calculate margin impact of an order."""
        margin_impact = self._adapter.get_margin_impact(ticker, action, quantity)
        return margin_impact
    
    def close_positions(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Close all or specific positions."""
        contract = None
        if ticker:
            contract = self._adapter.qualify_contract(ticker)
        try:
            broker_order_ids = self._adapter.close_positions(contract)
            trades = []
            for order_id in broker_order_ids:
                ticker = ticker or 'unknown'
                price = self.get_ticker_last_price(ticker)
                if not price:
                    continue
                trade_obj = Trade(
                    self, 'default', 'default', 'default', ticker, 0,
                    price, pd.Timestamp.now(), None, None, None, "CLOSE"
                )
                if ticker not in self.trades:
                    self.trades[ticker] = []
                self.trades[ticker].append(trade_obj)
                trades.append(trade_obj)
            logging.info(f"Closed positions for {ticker if ticker else 'all'}: {len(trades)} trades")
            return trades
        except Exception as e:
            logging.error(f"Error closing positions: {str(e)}")
            return []

    def on_order_fill(self, order_id: Any, quantity: float, price: float, ticker: str) -> None:
        """Handle order fill events."""
        order = self._broker_orders.get(order_id)
        if not order:
            logging.warning(f"No local order found for broker order ID {order_id}")
            return
        try:
            if order.trade is None:
                trade = Trade(
                    self, order.strategy_id, order.position_id, order.leg_id, ticker, order.size,
                    price, pd.Timestamp.now(), None, order.stop_loss, order.take_profit, order.tag
                )
                if ticker not in self.trades:
                    self.trades[ticker] = []
                self.trades[ticker].append(trade)
            else:
                order.trade._replace(
                    exit_price=price, exit_datetime=pd.Timestamp.now(), exit_spot=None, exit_tag="FILLED"
                )
                if order.trade.size == 0:
                    self.trades[ticker].remove(order.trade)
                    self.closed_trades.append(order.trade)
            self.orders.remove(order)
            del self._broker_orders[order_id]
            logging.info(f"Order filled: {ticker}, Quantity: {quantity}, Price: {price}, Order ID: {order_id}")
        except Exception as e:
            logging.error(f"Error processing order fill for {ticker}: {str(e)}")

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
    table_names_data = QuestDBClient(endpoint).execute_query("SHOW TABLES")
    df = pd.read_csv(BytesIO(table_names_data))
    table_names = df["table_name"].tolist()
    print(table_names)


    # Initialize the IBKR adapter
    ibkr_adapter = IBKRBrokerAdapter(host='localhost', port=7497, client_id=1)
    try:
        broker = _Broker(option_multiplier=1, broker_adapter=ibkr_adapter,data=fetcher)
        
        broker.next()

        # print(f"pos: {broker.positions}\n\norders: {broker.orders}\n\ntrades: {broker.trades}")
        print(f"Current Equity: {broker.equity}, Cash: {broker.cash}")
        
        # Get latest price for a ticker
        ticker="NIFTY25JUN24800CE"
        order_id = broker.new_order("NIFTY25JUN24800CE", "BUY", 75)
        margin_impact = broker.margin_impact(ticker, "BUY", 75)
        print(f"Latest order ID: {order_id}")
        print(f"Margin Impact: {margin_impact}")

        # ticker_data = fetcher.get_ticker_data(ticker=ticker)
        # data = pd.read_csv(BytesIO(ticker_data))
        # print(ticker, data['ts'].values[0],data['systemTime'].values[0],datetime.now(pytz.timezone('UTC')))

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if 'broker' in locals():
            broker.disconnect()

'''
Trade(
        contract=Option(conId=786919186, symbol='NIFTY50', lastTradeDateOrContractMonth='20250703', strike=25100.0, right='P', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY2570325100PE', tradingClass='NIFTY'), 
        order=Order(permId=269849725, action='BUY', orderType='MKT', lmtPrice=0.0, auxPrice=0.0, tif='DAY', ocaType=3, displaySize=2147483647, rule80A='0', openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DUE158901', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=75.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder'), 
        orderStatus=OrderStatus(orderId=0, status='Filled', filled=0.0, remaining=0.0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', mktCapPrice=0.0), 
        fills=[Fill(contract=Option(conId=786919186, symbol='NIFTY50', lastTradeDateOrContractMonth='20250703', strike=25100.0, right='P', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY2570325100PE', tradingClass='NIFTY'), execution=Execution(execId='0000f0e1.684a0265.01.01', time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), acctNumber='DUE158901', exchange='NSE', side='BOT', shares=75.0, price=280.65, permId=269849725, clientId=16, orderId=276, liquidation=0, cumQty=75.0, avgPrice=280.65, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1, pendingPriceRevision=False), commissionReport=CommissionReport(execId='0000f0e1.684a0265.01.01', commission=33.062129, currency='INR', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc))], 
        log=[TradeLogEntry(time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), status='Filled', message='Fill 75.0@280.65', errorCode=0)], 
        advancedError=''), 
Trade(contract=Option(conId=786919171, symbol='NIFTY50', lastTradeDateOrContractMonth='20250703', strike=25100.0, right='C', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY2570325100CE', tradingClass='NIFTY'), order=Order(permId=269849724, action='BUY', orderType='MKT', lmtPrice=0.0, auxPrice=0.0, tif='DAY', ocaType=3, displaySize=2147483647, rule80A='0', openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DUE158901', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=75.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder'), orderStatus=OrderStatus(orderId=0, status='Filled', filled=0.0, remaining=0.0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', mktCapPrice=0.0), fills=[Fill(contract=Option(conId=786919171, symbol='NIFTY50', lastTradeDateOrContractMonth='20250703', strike=25100.0, right='C', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY2570325100CE', tradingClass='NIFTY'), execution=Execution(execId='0000f0e1.684a0266.01.01', time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), acctNumber='DUE158901', exchange='NSE', side='BOT', shares=75.0, price=366.05, permId=269849724, clientId=16, orderId=275, liquidation=0, cumQty=75.0, avgPrice=366.05, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1, pendingPriceRevision=False), commissionReport=CommissionReport(execId='0000f0e1.684a0266.01.01', commission=35.941394, currency='INR', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc))], log=[TradeLogEntry(time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), status='Filled', message='Fill 75.0@366.05', errorCode=0)], advancedError=''), Trade(contract=Option(conId=772157067, symbol='NIFTY50', lastTradeDateOrContractMonth='20250626', strike=25100.0, right='P', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY25JUN25100PE', tradingClass='NIFTY'), order=Order(permId=269849723, action='SELL', orderType='MKT', lmtPrice=0.0, auxPrice=0.0, tif='DAY', ocaType=3, displaySize=2147483647, rule80A='0', openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DUE158901', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=75.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder'), orderStatus=OrderStatus(orderId=0, status='Filled', filled=0.0, remaining=0.0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', mktCapPrice=0.0), fills=[Fill(contract=Option(conId=772157067, symbol='NIFTY50', lastTradeDateOrContractMonth='20250626', strike=25100.0, right='P', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY25JUN25100PE', tradingClass='NIFTY'), execution=Execution(execId='0000f0e1.684a0264.01.01', time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), acctNumber='DUE158901', exchange='NSE', side='SLD', shares=75.0, price=228.3, permId=269849723, clientId=16, orderId=274, liquidation=0, cumQty=75.0, avgPrice=228.3, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1, pendingPriceRevision=False), commissionReport=CommissionReport(execId='0000f0e1.684a0264.01.01', commission=47.905971, currency='INR', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc))], log=[TradeLogEntry(time=datetime.datetime(2025, 6, 12, 5, 55, 34, tzinfo=datetime.timezone.utc), status='Filled', message='Fill 75.0@228.3', errorCode=0)], advancedError=''), Trade(contract=Option(conId=772157060, symbol='NIFTY50', lastTradeDateOrContractMonth='20250626', strike=25100.0, right='C', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY25JUN25100CE', tradingClass='NIFTY'), order=Order(permId=269849722, action='SELL', orderType='MKT', lmtPrice=0.0, auxPrice=0.0, tif='DAY', ocaType=3, displaySize=2147483647, rule80A='0', openClose='', volatilityType=0, deltaNeutralOrderType='None', referencePriceType=0, account='DUE158901', clearingIntent='IB', cashQty=0.0, dontUseAutoPriceForHedge=True, filledQuantity=75.0, refFuturesConId=2147483647, shareholder='Not an insider or substantial shareholder'), orderStatus=OrderStatus(orderId=0, status='Filled', filled=0.0, remaining=0.0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', mktCapPrice=0.0), fills=[Fill(contract=Option(conId=772157060, symbol='NIFTY50', lastTradeDateOrContractMonth='20250626', strike=25100.0, right='C', multiplier='1', exchange='NSE', currency='INR', localSymbol='NIFTY25JUN25100CE', tradingClass='NIFTY'), execution=Execution(execId='0000f0e1.684a0263.01.01', time=datetime.datetime(2025, 6, 12, 5, 55, 33, tzinfo=datetime.timezone.utc), acctNumber='DUE158901', exchange='NSE', side='SLD', shares=75.0, price=284.55, permId=269849722, clientId=16, orderId=273, liquidation=0, cumQty=75.0, avgPrice=284.55, orderRef='', evRule='', evMultiplier=0.0, modelCode='', lastLiquidity=1, pendingPriceRevision=False), commissionReport=CommissionReport(execId='0000f0e1.684a0263.01.01', commission=53.89463, currency='INR', realizedPNL=0.0, yield_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2025, 6, 12, 5, 55, 33, tzinfo=datetime.timezone.utc))], log=[TradeLogEntry(time=datetime.datetime(2025, 6, 12, 5, 55, 33, tzinfo=datetime.timezone.utc), status='Filled', message='Fill 75.0@284.55', errorCode=0)], advancedError='')]
'''
# ibkr_broker_adapter.py
import traceback
import warnings
from datetime import datetime, date as DateObject
import pandas as pd
from typing import Dict, Optional, List, Any
import time
import aiohttp
import asyncio
import re
import atexit
import threading
import logging
from backtesting_opt import Trade, Order, Position
from live_data_fetcher import _Data, Endpoint
from ib_async import Contract, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Order, Fill
from ib_insync import IB
import pytz

class IBKRBrokerAdapter:
    """Interactive Brokers-specific broker adapter implementation."""
    
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

    async def connect(self) -> bool:
        """Establish connection to IBKR."""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            return True
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            self.is_connected = False
            return False

    async def reconnect(self) -> bool:
        """Attempt to reconnect to IBKR."""
        self.reconnect_attempts = 0
        while self.reconnect_attempts < self.max_reconnect_attempts:
            if self.is_connected:
                logging.info("Already connected to IBKR.")
                return True
            logging.warning(f"Attempting reconnect ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
            try:
                self.ib = IB()
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
                self.is_connected = True
                logging.info("Reconnected to IBKR.")
                return True
            except Exception as e:
                logging.error(f"Reconnection failed: {str(e)}")
                self.reconnect_attempts += 1
                await asyncio.sleep(self.reconnect_delay)
        logging.error("Max reconnection attempts reached.")
        self.is_connected = False
        return False

    def check_connection(self) -> bool:
        """Check if connected to IBKR."""
        try:
            return self.ib.isConnected()
        except Exception:
            return False

    def is_market_open(self) -> bool:
        """Check if the market is open (NSE hours)."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    async def qualify_contract(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Qualify a ticker into an IBKR contract."""
        try:
            contract = Contract()
            contract.localSymbol = ticker
            contract.secType = 'OPT'  # Assuming options trading
            contract.exchange = 'NSE'
            qualified = await self.ib.qualifyContractsAsync(contract)
            if not qualified:
                logging.error(f"Failed to qualify contract for ticker {ticker}")
                return None
            return {
                'symbol': qualified[0].localSymbol,
                'conId': qualified[0].conId,
                'contract': qualified[0]
            }
        except Exception as e:
            logging.error(f"Error qualifying contract {ticker}: {str(e)}")
            return None

    async def place_order(self, contract: Dict[str, Any], action: str, quantity: float, 
                         order_type: str, order_params: Dict[str, Any]) -> Optional[Any]:
        """Place an order with IBKR."""
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
            trade = self.ib.placeOrder(ib_contract, order)
            return trade.order.orderId if trade else None
        except Exception as e:
            logging.error(f"Error placing {order_type} order for {contract.get('symbol')}: {str(e)}")
            return None

    async def get_margin_impact(self, contract: Dict[str, Any], action: str, quantity: float) -> Optional[float]:
        """Calculate margin impact of an order with IBKR."""
        ib_contract = contract.get('contract')
        if not ib_contract:
            logging.error("No valid IBKR contract provided")
            return None
        try:
            order = MarketOrder(action, quantity)
            whatif = await self.ib.whatIfOrderAsync(ib_contract, order)
            return float(whatif.initMarginChange) if whatif and whatif.initMarginChange else None
        except Exception as e:
            logging.error(f"Error calculating margin for {contract.get('symbol')}: {str(e)}")
            return None

    async def get_account_info(self) -> Dict[str, Any]:
        """Fetch comprehensive account information from IBKR."""
        try:
            account_summary = await self.ib.accountSummaryAsync()
            logging.info(f"Raw account summary: {account_summary}")
            summary = {tag.tag: float(tag.value) if tag.value.replace('.', '', 1).isdigit() else tag.value 
                      for tag in account_summary}
            portfolio = self.ib.portfolio()
            orders = self.ib.openOrders()
            positions = [
                {
                    'ticker': pos.contract.localSymbol,
                    'size': pos.position,
                    'avg_cost': pos.averageCost,  # Fixed: Changed from avgCost to averageCost
                    'contract': pos.contract
                } for pos in portfolio
            ]
            open_orders = [
                {
                    'ticker': trade.contract.localSymbol,
                    'size': trade.order.totalQuantity if trade.order.action == 'BUY' else -trade.order.totalQuantity,
                    'order_id': trade.order.orderId,
                    'strategy_id': trade.order.ocaGroup or 'default',
                    'tag': None
                } for trade in orders if trade.orderStatus.status in ['PendingSubmit', 'PreSubmitted', 'Submitted']
            ]
            return {
                'summary': summary,
                'positions': positions,
                'orders': open_orders
            }
        except Exception as e:
            logging.error(f"Error fetching account info from IBKR: {str(e)}")
            return {'summary': {}, 'positions': [], 'orders': []}

    async def get_latest_price(self, contract: Dict[str, Any]) -> Optional[float]:
        """Fetch the latest price for a contract from IBKR."""
        ib_contract = contract.get('contract')
        if not ib_contract:
            logging.error("No valid IBKR contract provided")
            return None
        try:
            ticker = self.ib.reqMktData(ib_contract, '', True, False)
            await asyncio.sleep(0.5)  # Wait for data to arrive
            price = ticker.last if ticker.last and pd.notna(ticker.last) else None
            self.ib.cancelMktData(ib_contract)
            if price is None or price <= 0:
                logging.warning(f"No valid price data for {contract.get('symbol')}")
                return None
            return price
        except Exception as e:
            logging.error(f"Error fetching price for {contract.get('symbol')}: {str(e)}")
            return None

    def cancel_all_orders(self) -> None:
        """Cancel all open orders with IBKR."""
        try:
            self.ib.cancelAllOrders()
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
        """Handle order fill events (called by _Broker)."""
        if self._order_fill_callback:
            self._order_fill_callback(order_id, quantity, price, ticker)

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        try:
            self.ib.disconnect()
        except Exception as e:
            logging.error(f"Error disconnecting from IBKR: {str(e)}")

class Broker:
    def __init__(self, *, option_multiplier: int, broker_adapter):
        """Initialize the broker with broker adapter only."""
        assert option_multiplier > 0, "option_multiplier must be positive"
        assert broker_adapter is not None, "broker adapter must be provided"

        logging.basicConfig(filename='broker_log.txt', level=logging.INFO)
        self._adapter = broker_adapter
        self._option_multiplier = option_multiplier
        self.orders: List[Order] = []
        self.trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self._equity: Dict[pd.Timestamp, float] = {}
        self._cash: float = 0.0
        self._margin_used: float = 0.0
        self._broker_orders: Dict[Any, Order] = {}  # Map broker order ID to local Order

        if not self._adapter.check_connection():
            logging.error("Failed to initialize broker connection")
            raise ConnectionError("Failed to connect to broker")

        # Set order fill callback
        self._adapter.set_order_fill_callback(self.on_order_fill)

    async def initialize_account(self):
        """Asynchronously initialize account info."""
        account_info = await self._adapter.get_account_info()
        self._cash = account_info.get('summary', {}).get('AvailableFunds', 0.0)
        self._margin_used = account_info.get('summary', {}).get('InitMarginReq', 0.0)
        if self._cash <= 0:
            logging.warning(f"Invalid initial cash value: {self._cash}")

    async def update_positions(self):
        """Update positions from broker adapter."""
        try:
            account_info = await self._adapter.get_account_info()
            self.positions = {pos['ticker']: Position(ticker=pos['ticker'], size=pos['size'], 
                                                     avg_price=pos['avg_cost']) 
                             for pos in account_info['positions']}
            logging.info("Positions updated")
        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")

    async def equity(self) -> float:
        """Calculate current equity."""
        try:
            account_info = await self._adapter.get_account_info()
            return account_info.get('summary', {}).get('NetLiquidation', 0.0)
        except Exception as e:
            logging.error(f"Error calculating equity: {str(e)}")
            return 0.0

    def __repr__(self):
        active_trades = sum(len(ts) for ts in self.trades.values())
        return f'<Broker: Cash={self._cash:.2f}, Equity={self.equity():.2f} ({active_trades} open trades)>'

    async def get_ticker_last_price(self, ticker: str) -> Optional[float]:
        contract = await self._adapter.qualify_contract(ticker)
        if not contract:
            logging.error(f"Failed to qualify contract for ticker {ticker}")
            return None
        try:
            price = await self._adapter.get_latest_price(contract)
            if price is not None and price > 0:
                return price
            logging.warning(f"No valid price data for {ticker}")
            return None
        except Exception as e:
            logging.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    async def next(self):
        """Process next market data update by fetching latest broker data."""
        await self.update_positions()
        # self.update_orders()  # Commented out: Not implemented
        self._equity[pd.Timestamp.now()] = await self.equity()
        logging.info(f"Broker state updated: Equity={self.equity():.2f}, Cash={self._cash:.2f}")

    async def close_positions(self, ticker: Optional[str] = None) -> List[Trade]:
        contract = None
        if ticker:
            contract = await self._adapter.qualify_contract(ticker)
        try:
            broker_order_ids = self._adapter.close_positions(contract)
            trades = []
            for order_id in broker_order_ids:
                ticker = ticker or 'unknown'  # Fallback if ticker not specified
                price = await self.get_ticker_last_price(ticker)
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

    async def get_account_info(self) -> dict:
        """Fetch comprehensive account information from the broker adapter."""
        return await self._adapter.get_account_info()
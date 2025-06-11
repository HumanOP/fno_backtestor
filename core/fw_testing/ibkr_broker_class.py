'''This module is not in use. This is just a agregated class for Interactive Brokers (IBKR) trading operations, 
including connection, order placement and account management.'''


import asyncio
import logging
from ib_async import IB, Contract, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Trade, Fill, Order
from typing import List, Dict, Optional
import pytz
import datetime
import uuid

class Broker:
    """
    Handles connection to Interactive Brokers and order placement for options trading.
    Supports market, limit, stop, stop-limit, trailing stop, bracket, OCO, GTC, and combo orders.
    """

    def __init__(self, host: str = 'localhost', port: int = 7497, client_id: int = 1):
        """
        Initialize the Broker with IBKR connection parameters.
        
        Args:
            host (str): IBKR host address (default: 'localhost')
            port (int): IBKR port (default: 7497 for TWS, 7496 for Gateway)
            client_id (int): Unique client ID for API connection
        """
        logging.basicConfig(filename='broker.log', level=logging.INFO)
        logging.info("Initializing Broker...")
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 10
        self.connect()
        self.ib.execDetailsEvent += self.on_exec_details
        self.ib.errorEvent += self.on_error

    def connect(self) -> bool:
        """
        Establish connection to IBKR.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            logging.info(f"Connected to IBKR at {self.host}:{self.port}, Client ID: {self.client_id}")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            self.is_connected = False
            return False

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to IBKR if connection is lost.
        
        Returns:
            bool: True if reconnected, False after max attempts
        """
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
                await asyncio.sleep(self.reconnect_delay)
        logging.error("Max reconnection attempts reached.")
        self.is_connected = False
        return False

    def check_connection(self) -> bool:
        """
        Check if connected to IBKR.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            return self.ib.isConnected()
        except Exception:
            return False

    def _is_market_open(self) -> bool:
        """
        Check if the market is open based on NSE trading hours (9:15 AM to 3:30 PM IST).
        
        Returns:
            bool: True if market is open, False otherwise
        """
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    async def qualify_contracts(self, contracts: List[Contract]) -> List[Contract]:
        """
        Qualify a list of contracts with IBKR.
        
        Args:
            contracts (List[Contract]): List of contracts to qualify
            
        Returns:
            List[Contract]: Qualified contracts
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot qualify contracts.")
                return []
        try:
            qualified = self.ib.qualifyContracts(*contracts)
            if len(qualified) != len(contracts):
                logging.error(f"Failed to qualify all contracts. Expected {len(contracts)}, got {len(qualified)}")
            for q in qualified:
                logging.info(f"Qualified contract: {q.localSymbol}")
            return qualified
        except Exception as e:
            logging.error(f"Error qualifying contracts: {str(e)}")
            return []

    async def place_market_order(self, contract: Contract, action: str, quantity: float) -> Optional[Trade]:
        """
        Place a market order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place order.")
            return None
        try:
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed market order: {action} {quantity} {contract.localSymbol}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing market order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_limit_order(self, contract: Contract, action: str, quantity: float, limit_price: float) -> Optional[Trade]:
        """
        Place a limit order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            limit_price (float): Limit price for the order
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place order.")
            return None
        try:
            order = LimitOrder(action, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed limit order: {action} {quantity} {contract.localSymbol} at {limit_price}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing limit order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_stop_order(self, contract: Contract, action: str, quantity: float, stop_price: float) -> Optional[Trade]:
        """
        Place a stop order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            stop_price (float): Stop price for the order
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place stop order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place stop order.")
            return None
        try:
            order = StopOrder(action, quantity, stop_price)
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed stop order: {action} {quantity} {contract.localSymbol} at stop {stop_price}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing stop order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_stop_limit_order(self, contract: Contract, action: str, quantity: float, limit_price: float, stop_price: float) -> Optional[Trade]:
        """
        Place a stop-limit order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            limit_price (float): Limit price for the order
            stop_price (float): Stop price for the order
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place stop-limit order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place stop-limit order.")
            return None
        try:
            order = StopLimitOrder(action, quantity, limit_price, stop_price)
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed stop-limit order: {action} {quantity} {contract.localSymbol} at limit {limit_price}, stop {stop_price}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing stop-limit order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_trailing_stop_order(self, contract: Contract, action: str, quantity: float, trailing_amount: float, trail_type: str = 'absolute') -> Optional[Trade]:
        """
        Place a trailing stop order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            trailing_amount (float): Trailing amount (absolute or percentage)
            trail_type (str): 'absolute' for fixed amount or 'percentage' for percent
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place trailing stop order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place trailing stop order.")
            return None
        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'TRAIL'
            if trail_type == 'absolute':
                order.trailingPercent = 0
                order.auxPrice = trailing_amount
            elif trail_type == 'percentage':
                order.trailingPercent = trailing_amount
                order.auxPrice = 0
            else:
                logging.error(f"Invalid trail_type: {trail_type}. Use 'absolute' or 'percentage'.")
                return None
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed trailing stop order: {action} {quantity} {contract.localSymbol}, trailing {trailing_amount} ({trail_type}), Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing trailing stop order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_bracket_order(self, contract: Contract, action: str, quantity: float, limit_price: float, take_profit_price: float, stop_loss_price: float) -> List[Trade]:
        """
        Place a bracket order (parent limit order with take-profit and stop-loss child orders).
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            limit_price (float): Limit price for the parent order
            take_profit_price (float): Limit price for take-profit order
            stop_loss_price (float): Stop price for stop-loss order
            
        Returns:
            List[Trade]: List of trades (parent, take-profit, stop-loss) or empty list if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place bracket order.")
                return []
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place bracket order.")
            return []
        try:
            # Parent limit order
            parent = LimitOrder(action, quantity, limit_price)
            parent.orderId = self.ib.client.getReqId()
            parent.transmit = False  # Do not transmit until all orders are set

            # Take-profit order (opposite action)
            take_profit_action = 'SELL' if action == 'BUY' else 'BUY'
            take_profit = LimitOrder(take_profit_action, quantity, take_profit_price)
            take_profit.orderId = self.ib.client.getReqId()
            take_profit.parentId = parent.orderId
            take_profit.transmit = False

            # Stop-loss order (opposite action)
            stop_loss = StopOrder(take_profit_action, quantity, stop_loss_price)
            stop_loss.orderId = self.ib.client.getReqId()
            stop_loss.parentId = parent.orderId
            stop_loss.transmit = True  # Transmit all orders

            trades = []
            trades.append(self.ib.placeOrder(contract, parent))
            trades.append(self.ib.placeOrder(contract, take_profit))
            trades.append(self.ib.placeOrder(contract, stop_loss))
            logging.info(f"Placed bracket order: {action} {quantity} {contract.localSymbol}, Parent ID: {parent.orderId}, Take-Profit: {take_profit_price}, Stop-Loss: {stop_loss_price}")
            return trades
        except Exception as e:
            logging.error(f"Error placing bracket order for {contract.localSymbol}: {str(e)}")
            return []

    async def place_oco_order(self, contract: Contract, action: str, quantity: float, limit_price: float, stop_price: float) -> List[Trade]:
        """
        Place a One-Cancels-the-Other (OCO) order consisting of a limit order and a stop order.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            limit_price (float): Limit price for the limit order
            stop_price (float): Stop price for the stop order
            
        Returns:
            List[Trade]: List of trades (limit, stop) or empty list if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place OCO order.")
                return []
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place OCO order.")
            return []
        try:
            oca_group = str(uuid.uuid4())  # Unique identifier for OCO group

            # Limit order
            limit_order = LimitOrder(action, quantity, limit_price)
            limit_order.orderId = self.ib.client.getReqId()
            limit_order.ocaGroup = oca_group
            limit_order.ocaType = 1  # 1 = Cancel all remaining orders with block

            # Stop order
            stop_order = StopOrder(action, quantity, stop_price)
            stop_order.orderId = self.ib.client.getReqId()
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1

            trades = []
            trades.append(self.ib.placeOrder(contract, limit_order))
            trades.append(self.ib.placeOrder(contract, stop_order))
            logging.info(f"Placed OCO order: {action} {quantity} {contract.localSymbol}, Limit: {limit_price}, Stop: {stop_price}, OCA Group: {oca_group}")
            return trades
        except Exception as e:
            logging.error(f"Error placing OCO order for {contract.localSymbol}: {str(e)}")
            return []

    async def place_gtc_order(self, contract: Contract, action: str, quantity: float, order_type: str, limit_price: Optional[float] = None, stop_price: Optional[float] = None) -> Optional[Trade]:
        """
        Place a Good-Till-Cancelled (GTC) order for a contract.
        
        Args:
            contract (Contract): Contract to trade
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            order_type (str): 'MARKET', 'LIMIT', or 'STOP'
            limit_price (Optional[float]): Limit price for LIMIT orders
            stop_price (Optional[float]): Stop price for STOP orders
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place GTC order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place GTC order.")
            return None
        try:
            if order_type == 'MARKET':
                order = MarketOrder(action, quantity)
            elif order_type == 'LIMIT':
                if limit_price is None:
                    logging.error("Limit price required for LIMIT order.")
                    return None
                order = LimitOrder(action, quantity, limit_price)
            elif order_type == 'STOP':
                if stop_price is None:
                    logging.error("Stop price required for STOP order.")
                    return None
                order = StopOrder(action, quantity, stop_price)
            else:
                logging.error(f"Invalid order_type: {order_type}. Use 'MARKET', 'LIMIT', or 'STOP'.")
                return None

            order.tif = 'GTC'  # Set Time in Force to Good-Till-Cancelled
            trade = self.ib.placeOrder(contract, order)
            logging.info(f"Placed GTC {order_type} order: {action} {quantity} {contract.localSymbol}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing GTC {order_type} order for {contract.localSymbol}: {str(e)}")
            return None

    async def place_combo_order(self, combo_contract: Contract, action: str, quantity: float, limit_price: Optional[float] = None) -> Optional[Trade]:
        """
        Place an order for a combination contract (e.g., calendar spread).
        
        Args:
            combo_contract (Contract): Combination contract
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            limit_price (Optional[float]): Limit price, None for market order
            
        Returns:
            Optional[Trade]: Trade object or None if failed
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot place combo order.")
                return None
        if not self._is_market_open():
            logging.warning("Market is closed. Cannot place combo order.")
            return None
        try:
            order = LimitOrder(action, quantity, limit_price) if limit_price is not None else MarketOrder(action, quantity)
            trade = self.ib.placeOrder(combo_contract, order)
            logging.info(f"Placed combo order: {action} {quantity} {combo_contract.localSymbol}, Order ID: {trade.order.orderId}")
            return trade
        except Exception as e:
            logging.error(f"Error placing combo order for {combo_contract.localSymbol}: {str(e)}")
            return None

    async def get_margin_impact(self, contract: Contract, action: str, quantity: float) -> Optional[float]:
        """
        Calculate the margin impact of an order.
        
        Args:
            contract (Contract): Contract to evaluate
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of contracts
            
        Returns:
            Optional[float]: Margin impact or None if unavailable
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot calculate margin.")
                return None
        try:
            order = MarketOrder(action, quantity)
            whatif_result = self.ib.whatIfOrder(contract, order)
            if isinstance(whatif_result, list):
                if not whatif_result:
                    logging.warning(f"No margin data for {contract.localSymbol}")
                    return None
                whatif = whatif_result[0]
            else:
                whatif = whatif_result
            margin = float(whatif.initMarginChange) if whatif.initMarginChange else None
            logging.info(f"Margin impact for {action} {quantity} {contract.localSymbol}: {margin}")
            return margin
        except Exception as e:
            logging.error(f"Error calculating margin for {contract.localSymbol}: {str(e)}")
            return None

    async def get_account_summary(self) -> Dict[str, float]:
        """
        Fetch account summary (e.g., available funds, margin).
        
        Returns:
            Dict[str, float]: Dictionary of account metrics
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            if not await self.reconnect():
                logging.error("Failed to reconnect. Cannot fetch account summary.")
                return {}
        try:
            summary = {}
            for av in self.ib.accountSummary():
                summary[av.tag] = float(av.value) if av.value else 0.0
            logging.info(f"Account summary: {summary}")
            return summary
        except Exception as e:
            logging.error(f"Error fetching account summary: {str(e)}")
            return {}

    def cancel_all_orders(self) -> None:
        """
        Cancel all open orders.
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            loop = asyncio.get_event_loop()
            if not loop.run_until_complete(self.reconnect()):
                logging.error("Failed to reconnect. Cannot cancel orders.")
                return
        try:
            self.ib.reqGlobalCancel()
            logging.info("All open orders cancelled.")
        except Exception as e:
            logging.error(f"Error cancelling orders: {str(e)}")

    def close_positions(self, contract: Optional[Contract] = None) -> List[Trade]:
        """
        Close all positions or a specific contract's position.
        
        Args:
            contract (Optional[Contract]): Specific contract to close, or None for all
            
        Returns:
            List[Trade]: List of trades placed to close positions
        """
        if not self.check_connection():
            logging.warning("Not connected. Attempting to reconnect...")
            loop = asyncio.get_event_loop()
            if not loop.run_until_complete(self.reconnect()):
                logging.error("Failed to reconnect. Cannot close positions.")
                return []
        try:
            portfolio = self.ib.portfolio()
            trades = []
            for position in portfolio:
                if contract is None or position.contract.conId == contract.conId:
                    qty = position.position
                    action = 'SELL' if qty > 0 else 'BUY'
                    order = MarketOrder(action, abs(qty))
                    trade = self.ib.placeOrder(position.contract, order)
                    trades.append(trade)
                    logging.info(f"Placed {action} order to close {position.contract.localSymbol}, Order ID: {trade.order.orderId}")
            return trades
        except Exception as e:
            logging.error(f"Error closing positions: {str(e)}")
            return []

    def on_exec_details(self, trade: Trade, fill: Fill) -> None:
        """
        Handle execution details for placed orders.
        
        Args:
            trade (Trade): Trade object
            fill (Fill): Fill details
        """
        logging.info(f"Order filled: {trade.contract.localSymbol}, Filled: {fill.execution.cumQty}/{trade.order.totalQuantity}, Price: {fill.execution.price}")
        if trade.orderStatus.status == 'Filled':
            logging.info(f"Order fully filled: {trade.contract.localSymbol}, Order ID: {trade.order.orderId}")

    def on_error(self, reqId: int, errorCode: int, errorString: str, contract: Optional[Contract] = None) -> None:
        """
        Handle API errors.
        
        Args:
            reqId (int): Request ID
            errorCode (int): Error code
            errorString (str): Error message
            contract (Optional[Contract]): Related contract
        """
        logging.error(f"API Error: ReqId: {reqId}, Code: {errorCode}, Message: {errorString}, Contract: {contract}")
        if errorCode in [200, 201, 203, 162]:
            logging.warning(f"Contract-related error for {contract.localSymbol if contract else 'unknown'}: {errorString}")

    def disconnect(self) -> None:
        """
        Disconnect from IBKR.
        """
        try:
            self.ib.disconnect()
            self.is_connected = False
            logging.info("Disconnected from IBKR.")
        except Exception as e:
            logging.error(f"Error disconnecting from IBKR: {str(e)}")
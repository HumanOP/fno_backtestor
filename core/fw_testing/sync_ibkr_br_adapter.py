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
from ib_async import IB, Contract, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, Order, Trade, Fill

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

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        try:
            self.ib.disconnect()
            self.is_connected = False
            logging.info("Disconnected from IBKR")
        except Exception as e:
            logging.error(f"Error disconnecting from IBKR: {str(e)}")

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
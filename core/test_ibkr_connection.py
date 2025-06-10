# test_ibkr_connection.py
import asyncio
import logging
import pandas as pd
from ibkr_broker_adapter import BrokerAdapter, Broker

async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize BrokerAdapter and connect
    adapter = BrokerAdapter(host='localhost', port=7497, client_id=1)
    connected = await adapter.connect()
    if not connected:
        logging.error("Failed to connect to IBKR via BrokerAdapter")
        return
    logging.info("Successfully connected to IBKR via BrokerAdapter")
    await asyncio.sleep(1.0)  # Wait for connection to stabilize

    # Initialize Broker and account
    broker = Broker(option_multiplier=1, broker_adapter=adapter)
    await broker.initialize_account()  # Asynchronously initialize account info

    # --- Account Summary Section ---
    account_tags = [
        'AccountType', 'NetLiquidation', 'TotalCashValue', 'SettledCash', 'AccruedCash', 'BuyingPower',
        'EquityWithLoanValue', 'PreviousDayEquityWithLoanValue', 'GrossPositionValue', 'RegTEquity',
        'RegTMargin', 'SMA', 'InitMarginReq', 'MaintMarginReq', 'AvailableFunds', 'ExcessLiquidity',
        'Cushion', 'FullInitMarginReq', 'FullMaintMarginReq'
    ]
    account_info = await adapter.get_account_info()
    summary = account_info.get('summary', {})
    summary_dict = {tag: summary.get(tag, None) for tag in account_tags}
    print("\n=== Account Summary ===")
    for tag in account_tags:
        print(f"{tag}: {summary_dict[tag]}")
    pd.DataFrame([summary_dict]).to_csv('account_summary.csv', index=False)

    # --- Activity Summary Section ---
    await broker.update_positions()  # Changed to async
    activity_rows = []
    print("\n=== Activity Summary ===")
    print(f"{'Symbol':<20}{'Net':<8}")
    for symbol, position in broker.positions.items():
        net = position.size
        print(f"{symbol:<20}{net:<8}")
        activity_rows.append({'Symbol': symbol, 'Net': net})
    pd.DataFrame(activity_rows).to_csv('activity_summary.csv', index=False)

    # --- Trades Section ---
    trades_rows = []
    print("\n=== Trades ===")
    for symbol, trades in broker.trades.items():
        for trade in trades:
            row = {
                'Symbol': symbol,
                'Size': trade.size,
                'Entry Price': getattr(trade, 'entry_price', None),
                'Entry Datetime': getattr(trade, 'entry_datetime', None),
                'Exit Price': getattr(trade, 'exit_price', None),
                'Exit Datetime': getattr(trade, 'exit_datetime', None),
                'Tag': getattr(trade, 'tag', None)
            }
            print(row)
            trades_rows.append(row)
    pd.DataFrame(trades_rows).to_csv('trades.csv', index=False)

    # --- Orders Section ---
    orders_rows = []
    print("\n=== Orders ===")
    for order in broker.orders:
        row = {
            'Symbol': order.ticker,
            'Size': order.size,
            'Order ID': getattr(order, 'broker_order_id', None),
            'Stop Loss': getattr(order, 'stop_loss', None),
            'Take Profit': getattr(order, 'take_profit', None),
            'Tag': getattr(order, 'tag', None)
        }
        print(row)
        orders_rows.append(row)
    pd.DataFrame(orders_rows).to_csv('orders.csv', index=False)

    # Disconnect
    adapter.disconnect()
    await asyncio.sleep(0.1)
    logging.info("Disconnected from IBKR via BrokerAdapter")

def run_main():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    run_main()
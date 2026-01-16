import asyncio
import nest_asyncio
nest_asyncio.apply()    # Quick fix for the script to run

import logging
import random
import requests
import pytz
import os
import pandas as pd
from datetime import datetime, timezone
from ib_insync import IB, Index, Option
from questdb.ingress import Sender, IngressError, TimestampNanos
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("option_chain_listener.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Global data storage
option_data = []
underlying_data = []
created_tables: List[str] = []
table_lock = threading.Lock()

class OptionChainListener:
    def __init__(
        self,
        ib_client: IB,
        *,
        underlying_symbol: str = "NIFTY50",
        exchange: str = "NSE",
        currency: str = "INR",
        strike_range: float = 200,
        https: bool = True,
        questdb_host: str = None,
        questdb_port: str = None,
        questdb_username: str = None,
        questdb_password: str = None,
    ):
        self.ib = ib_client
        self.underlying_symbol = underlying_symbol
        self.exchange = exchange
        self.currency = currency
        self.strike_range = strike_range
        self.option_data = option_data
        self.underlying_data = underlying_data
        self.tickers = []
        self.underlying_ticker = None
        self.pending_option_rows = deque()  # Thread-safe deque for option ticks
        self.pending_underlying_rows = deque()  # Thread-safe deque for underlying ticks
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single worker for uploads
        self.print_thread = None
        self.upload_thread = None
        self.sender = None  # Persistent QuestDB sender

        # QuestDB connection settings
        self.questdb_protocol = 'https' if https else 'http'
        self.questdb_host = questdb_host or os.getenv("QUESTDB_HOST", "localhost")
        self.questdb_port = questdb_port or os.getenv("QUESTDB_PORT", "9000")
        self.questdb_username = questdb_username or os.getenv("QUESTDB_USERNAME", "admin")
        self.questdb_password = questdb_password or os.getenv("QUESTDB_PASSWORD", "quest")
        self.questdb_conf = (
            f"{self.questdb_protocol}::addr={self.questdb_host}:{self.questdb_port};"
            f"username={self.questdb_username};"
            f"password={self.questdb_password};"
            f"connect_timeout=5000;"
        )

    def ticker_builder(self, underlying, expiry_date, strike, right):
        """
        Build option ticker string from components.
        Example output: NIFTY25JUL2525400CE
        """

        # Map underlying back to original format
        if underlying == "NIFTY50":
            underlying = "NIFTY"

        # Expiry date: YYYYMMDD -> DDMMMYY
        year = expiry_date[:4]
        month = expiry_date[4:6]
        day = expiry_date[6:8]
        month_map = {
            '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR',
            '05': 'MAY', '06': 'JUN', '07': 'JUL', '08': 'AUG',
            '09': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC'
        }
        month_abbr = month_map[month]
        year_short = year[2:]

        expiry_str = f"{day}{month_abbr}{year_short}"

        # Strike (avoid .0 for integer values)
        strike_str = f"{int(strike)}"

        # Option type
        option_type = "CE" if right.upper() == "C" else "PE"

        # Final ticker
        ticker = f"{underlying}{expiry_str}{strike_str}{option_type}"
        return ticker


    def create_table_for_symbol(self, symbol: str, is_underlying: bool = False):
        """Create a QuestDB table for a symbol (option or underlying)."""
        if symbol in created_tables:
            return

        with table_lock:
            if symbol in created_tables:
                return
            url = f"https://{self.questdb_host}:{self.questdb_port}/exec"
            if is_underlying:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{symbol}" (
                    last DOUBLE,
                    bid DOUBLE,
                    ask DOUBLE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    ts TIMESTAMP
                ) TIMESTAMP(ts) PARTITION BY DAY;
                '''.strip()
            else:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{symbol}" (
                    strike DOUBLE,
                    right SYMBOL,
                    expiry STRING,
                    last DOUBLE,
                    bid DOUBLE,
                    ask DOUBLE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    call_volume DOUBLE,
                    put_volume DOUBLE,
                    call_open_interest DOUBLE,
                    put_open_interest DOUBLE,
                    model_price DOUBLE,
                    delta DOUBLE,
                    gamma DOUBLE,
                    vega DOUBLE,
                    theta DOUBLE,
                    implied_vol DOUBLE,
                    ts TIMESTAMP,
                    systemTime TIMESTAMP
                ) TIMESTAMP(ts) PARTITION BY DAY;
                '''.strip()

            try:
                response = requests.get(
                    url,
                    params={"query": ddl},
                    auth=(self.questdb_username, self.questdb_password),
                    timeout=5
                )
                if response.status_code == 200:
                    created_tables.append(symbol)
                    logger.info(f"Created table {symbol}")
                else:
                    logger.error(f"DDL error for {symbol}: {response.status_code} / {response.text}")
            except requests.RequestException as e:
                logger.error(f"Failed to create table {symbol}: {e}")

    async def ensure_sender(self):
        """Ensure Sender is connected, retry if necessary."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to QuestDB (attempt {attempt + 1}/{max_retries})")
                with Sender.from_conf(self.questdb_conf) as sender:
                    sender.flush()
                    self.running = True
                    logger.info("Connected to QuestDB sender")
                    return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        logger.error("Failed to connect to QuestDB after retries")
        self.running = False

    def upload_to_qdb_sync(self):
        """Synchronous function to upload batched rows to QuestDB."""
        while self.running:
            start = time.perf_counter()
            if  not self.running:
                logger.warning("Skipping upload: sender not connected")
                time.sleep(0.2)
                continue

            # Process option rows
            option_rows = []
            while self.pending_option_rows:
                option_rows.append(self.pending_option_rows.popleft())
                if len(option_rows) >= 1000:  # Max batch size
                    break

            # Process underlying rows
            underlying_rows = []
            while self.pending_underlying_rows:
                underlying_rows.append(self.pending_underlying_rows.popleft())
                if len(underlying_rows) >= 1000:  # Max batch size
                    break

            try:
                with Sender.from_conf(self.questdb_conf) as sender:
                    if option_rows:
                        for row in option_rows:
                            symbol = self.ticker_builder(row["Underlying"], row["Expiration"], row["Strike"], row["Right"])
                            if symbol not in created_tables:
                                self.create_table_for_symbol(symbol, is_underlying=False)
                            sender.row(
                                symbol,
                                symbols={"right": row["Right"]},
                                columns={
                                    "symbol": row["Symbol"],
                                    "strike": row["Strike"],
                                    "expiry": row["Expiration"],
                                    "last": row["Last"],
                                    "bid": row["Bid"],
                                    "ask": row["Ask"],
                                    "open": row["Open"],
                                    "high": row["High"],
                                    "low": row["Low"],
                                    "close": row["Close"],
                                    "volume": row["Volume"],
                                    "call_volume": row["CallVolume"],
                                    "put_volume": row["PutVolume"],
                                    "call_open_interest": row["CallOpenInterest"],
                                    "put_open_interest": row["PutOpenInterest"],
                                    "model_price": row["ModelPrice"],
                                    "delta": row["Delta"],
                                    "gamma": row["Gamma"],
                                    "vega": row["Vega"],
                                    "theta": row["Theta"],
                                    "implied_vol": row["ImpliedVol"],
                                    "systemTime": row["systemTime"]
                                },
                                at=row["ts"]
                                )
                        sender.flush()
                        logger.info(f"Uploaded {len(option_rows)} option rows to QuestDB")

                    if underlying_rows:
                        for row in underlying_rows:
                            symbol = row["Symbol"]
                            if symbol not in created_tables:
                                self.create_table_for_symbol(symbol, is_underlying=True)
                            sender.row(
                                symbol,
                                columns={
                                    "symbol": row["Symbol"],
                                    "last": row["Last"],
                                    "bid": row["Bid"],
                                    "ask": row["Ask"],
                                    "open": row["Open"],
                                    "high": row["High"],
                                    "low": row["Low"],
                                    "close": row["Close"],
                                    "volume": row["Volume"],
                                },
                                at=row["Time"]
                            )
                        sender.flush()
                        logger.info(f"Uploaded {len(underlying_rows)} underlying rows to QuestDB")

            except IngressError as e:
                logger.error(f"Upload error: {e}", exc_info=True)
                self.running = False
               
            except Exception as e:
                logger.error(f"Unexpected error uploading: {e}")
            finally:
                end = time.perf_counter()
                logger.info(f"Upload time: {end - start:.2f} seconds")
                option_rows.clear()
                underlying_rows.clear()
            # time.sleep(0.2)  # Flush every 0.2 seconds

    def start_upload_thread(self):
        """Start a dedicated thread for QuestDB uploads."""
        self.upload_thread = threading.Thread(target=self.upload_to_qdb_sync, daemon=True)
        self.upload_thread.start()
        logger.info("Started QuestDB upload thread")

    def on_tick(self, ticker):
        try:
            greeks = ticker.modelGreeks or type("G", (), {})()
            now = datetime.now(pytz.timezone('UTC'))
            is_call = ticker.contract.right == "C"
            logger.info(f"ticker: {ticker}")
            row = {
                "ts": ticker.time,
                "systemTime": now,
                "Symbol": ticker.contract.localSymbol,
                "Underlying": ticker.contract.symbol,
                "Strike": ticker.contract.strike,
                "Right": ticker.contract.right,
                "Expiration": ticker.contract.lastTradeDateOrContractMonth,
                "Last": ticker.last,
                "Bid": ticker.bid,
                "Ask": ticker.ask,
                "Open": ticker.open,
                "High": ticker.high,
                "Low": ticker.low,
                "Close": ticker.close,
                "Volume": ticker.volume,
                "CallVolume": ticker.callVolume if is_call else None,
                "PutVolume": ticker.putVolume if not is_call else None,
                "CallOpenInterest": ticker.callOpenInterest if is_call else None,
                "PutOpenInterest": ticker.putOpenInterest if not is_call else None,
                "ModelPrice": getattr(greeks, "optPrice", None),
                "Delta": getattr(greeks, "delta", None),
                "Gamma": getattr(greeks, "gamma", None),
                "Vega": getattr(greeks, "vega", None),
                "Theta": getattr(greeks, "theta", None),
                "ImpliedVol": getattr(greeks, "impliedVol", None),
            }
            logger.info(f"on_tick: ticker={ticker.contract.localSymbol} ticker_last={ticker.last} ticker_bid={ticker.bid} ticker_ask={ticker.ask} ticker_time={ticker.time}, systemTime={now}")
            if row["ts"] is None:
                logger.warning(f"Skipping tick with missing timestamp: {row['Symbol']}")
                return
            self.pending_option_rows.append(row)
            self.option_data.append(row)  # For printing/display purposes
        except Exception as e:
            logger.error(f"on_tick error: {e}")

    def on_underlying_tick(self, ticker):
        try:
            now = datetime.utcnow()
            row = {
                "Time": ticker.time,
                "Symbol": ticker.contract.symbol,
                "Last": ticker.last,
                "Bid": ticker.bid,
                "Ask": ticker.ask,
                "Open": ticker.open,
                "High": ticker.high,
                "Low": ticker.low,
                "Close": ticker.close,
                "Volume": ticker.volume
            }
            if row["Time"] is None:
                logger.warning(f"Skipping underlying tick with missing timestamp: {row['Symbol']}")
                return
            self.pending_underlying_rows.append(row)
            self.underlying_data.append(row)  # For printing/display purposes
        except Exception as e:
            logger.error(f"on_underlying_tick error: {e}")
    async def fetch_current_price(self, underlying: Index) -> float:
        ticker = self.ib.reqMktData(underlying)
        await asyncio.sleep(1)
        price = ticker.last or ticker.close
        logger.info(f"Underlying {self.underlying_symbol} spot = {price}")
        return price
    
    async def start_listening(self, client_id: int = 15, max_retries: int = 3, retry_delay: int = 2):
        """Main connection loop with retry logic."""
        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                logger.info(f"Connecting to TWS (clientId={client_id})")
                self.ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=10)
                logger.info("Connected to TWS")

                await self.ensure_sender()
                if not self.running:
                    raise Exception("Failed to connect to QuestDB")

                # Start upload thread
                self.start_upload_thread()

                underlying = Index(self.underlying_symbol, self.exchange, self.currency)
                self.ib.qualifyContracts(underlying)

                # Subscribe to underlying asset
                self.underlying_ticker = self.ib.reqMktData(underlying, genericTickList="106,100,101")
                self.underlying_ticker.updateEvent += self.on_underlying_tick
                logger.info(f"Subscribed to underlying {self.underlying_symbol}")

                current_price = await self.fetch_current_price(underlying)
                atm_strike = round(current_price / 50) * 50

                opt_params = self.ib.reqSecDefOptParams(
                    underlying.symbol, "", underlying.secType, underlying.conId
                )
                if not opt_params:
                    logger.error("Failed to retrieve option chain parameters")
                    raise Exception("No option chain parameters")

                chain_info = opt_params[0]
                expirations = sorted(chain_info.expirations)[:2]
                strikes = sorted(chain_info.strikes)
                selected_strikes = [s for s in strikes if abs(s - atm_strike) <= self.strike_range]

                option_contracts = [
                    Option(self.underlying_symbol, exp, strike, right, self.exchange, currency=self.currency)
                    for exp in expirations
                    for strike in selected_strikes
                    for right in ["C", "P"]
                ]

                self.ib.qualifyContracts(*option_contracts)
                logger.info(f"Qualified {len(option_contracts)} option contracts")

                for contract in option_contracts:
                    if not contract.localSymbol:
                        logger.error(f"Failed to qualify contract: {contract}")
                    else:
                        symbol = self.ticker_builder(contract.symbol,  contract.lastTradeDateOrContractMonth, contract.strike, contract.right)
                        self.create_table_for_symbol(symbol, is_underlying=False)
                        # time.sleep(0.1)
                        logger.info(f"Created table for option symbol: {symbol}")

                self.tickers = [self.ib.reqMktData(contract, genericTickList="106,100,101") for contract in option_contracts]
                for ticker in self.tickers:
                    ticker.updateEvent += self.on_tick

                logger.info(f"Subscribed to {len(self.tickers)} option contracts")

                # Start printing thread
                # self.start_printing()

                end = time.perf_counter()
                logger.info(f"Listener startup time: {end - start:.2f} seconds")

                # Keep running
                while True:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Stream error: {type(e).__name__}: {str(e)}")
                if "clientid" in str(e).lower() and attempt < max_retries - 1:
                    client_id = random.randint(1, 100)
                    await asyncio.sleep(retry_delay)
                    continue
                raise

    def stop(self):
        """Clean up resources."""
        self.running = False
        if self.sender:
            try:
                self.sender.close()
                logger.info("QuestDB sender closed")
            except Exception as e:
                logger.error(f"Error closing QuestDB sender: {type(e).__name__}: {str(e)}")
        self.sender = None
        if self.ib.isConnected():
            if self.underlying_ticker:
                self.ib.cancelMktData(self.underlying_ticker)
                logger.info("Unsubscribed from underlying ticker")
            for ticker in self.tickers:
                self.ib.cancelMktData(ticker)
            self.ib.disconnect()
            logger.info("TWS disconnected")
        self.executor.shutdown(wait=True)
        logger.info("Thread pool shut down")
        if self.print_thread and self.print_thread.is_alive():
            logger.info("Waiting for print thread to terminate")
        if self.upload_thread and self.upload_thread.is_alive():
            logger.info("Waiting for upload thread to terminate")

async def main():
    ib = IB()
    listener = OptionChainListener(
        ib_client=ib,
        # questdb_host="localhost",
        # questdb_port="9000",
        # questdb_username="admin",
        # questdb_password="quest",
        https=True,             # True for remote qdb and False for local dockerised qdb
        questdb_host="qdb6.twocc.in",
        questdb_port="443",
        questdb_username="2Cents",
        questdb_password="2Cents1012cc"
    )
    try:
        await listener.start_listening()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping")
        listener.stop()

if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.get_event_loop().run_until_complete(main())
import asyncio
import nest_asyncio
import logging
import random
import requests
import os
import pandas as pd
from datetime import datetime,timezone
from ib_insync import IB, Index, Option
from questdb.ingress import Sender, IngressError, TimestampNanos
import time
import pytz
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Set, List, Dict
import redis,json
from queue import Queue

# Patch asyncio for Jupyter-style environments
nest_asyncio.apply()

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

# Store data globally
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
        self.pending_rows: List[Dict] = []
        self.pending_underlying_rows: List[Dict] = []
        self.last_flush = None
        self.last_underlying_flush = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.print_thread = None
        # self.opt_que = Queue()
        # self.und_que = Queue()
        # Redis
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.opt_redis_key = "ib:opt_ticks"
        self.und_redis_key = "ib:und_ticks"

        # QuestDB connection settings
        self.questdb_host = questdb_host or os.getenv("QUESTDB_HOST", "localhost")
        self.questdb_port = questdb_port or os.getenv("QUESTDB_PORT", "9009")
        self.questdb_username = questdb_username or os.getenv("QUESTDB_USERNAME", "admin")
        self.questdb_password = questdb_password or os.getenv("QUESTDB_PASSWORD", "quest")
        self.questdb_conf = (
            f"https::addr={self.questdb_host}:{self.questdb_port};"
            f"username={self.questdb_username};"
            f"password={self.questdb_password};"
            f"connect_timeout=5000;"
        )

    def create_table_for_symbol(self, symbol: str, is_underlying: bool = False):
        """Create a QuestDB table for a symbol (option or underlying)."""
        if symbol in created_tables:
            return

        with table_lock:
            if symbol in created_tables:
                return
            url = f"https://{self.questdb_host}:443/exec"
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

    async def fetch_current_price(self, underlying: Index) -> float:
        ticker = self.ib.reqMktData(underlying)
        await asyncio.sleep(1)
        price = ticker.last or ticker.close
        logger.info(f"Underlying {self.underlying_symbol} spot = {price}")
        return price

    def upload_to_qdb_sync(self, rows: List[Dict], is_underlying: bool = False):
        """Synchronous function to upload rows to QuestDB in a thread."""
        start = time.perf_counter()
        if not self.running:
            logger.warning("Skipping upload: sender not connected")
            return

        try:
            with Sender.from_conf(self.questdb_conf) as sender:
                for row in rows:
                    symbol = row["Symbol"]
                    if symbol not in created_tables:
                        self.create_table_for_symbol(symbol, is_underlying=is_underlying)
                    if is_underlying:
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
                    else:
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
                                "systemTime":row['systemTime']
                            },
                            at=row["ts"]
                        )
                sender.flush()
                logger.info(f"Uploaded {len(rows)} {'underlying' if is_underlying else 'option'} rows to QuestDB")
        except IngressError as e:
            logger.error(f"Upload error: {e} -->",exc_info=True)
            self.running = False
        except Exception as e:
            logger.error(f"Unexpected error uploading: {e}")
        finally:
            end = time.perf_counter()
            logger.info(f"Upload time: {end - start:.2f} seconds")

    async def _upload_to_qdb(self, rows: List[Dict], is_underlying: bool = False):
        """Upload a batch of rows to QuestDB in a thread."""
        if not rows:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.upload_to_qdb_sync, rows, is_underlying)

    def print_data_sync(self):
        """Synchronous function to print data in a thread."""
        while self.running:
            if self.option_data or self.underlying_data:
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                pd.set_option('display.float_format', '{:.2f}'.format)
                if self.option_data:
                    df_options = pd.DataFrame(self.option_data)
                    logger.info("\n=== Option Chain Data ===")
                    print(df_options.tail(30)[[
                        'Time', 'Symbol', 'Strike', 'Right', 'Expiration', 'Bid', 'Ask', 'Last',
                        'Volume', 'CallVolume', 'PutVolume', 'CallOpenInterest', 'PutOpenInterest',
                        'Delta', 'Gamma', 'Theta', 'ImpliedVol'
                    ]])
                    logger.info("========================\n")
                if self.underlying_data:
                    df_underlying = pd.DataFrame(self.underlying_data)
                    logger.info("\n=== Underlying Asset Data ===")
                    print(df_underlying.tail(30)[[
                        'Time', 'Symbol', 'Last', 'Bid', 'Ask', 'Open', 'High', 'Low', 'Close', 'Volume'
                    ]])
                    logger.info("========================\n")
            time.sleep(10)

    def start_printing(self):
        """Start printing data in a separate thread."""
        self.print_thread = threading.Thread(target=self.print_data_sync, daemon=True)
        self.print_thread.start()
        logger.info("Started data printing thread")

    def on_tick(self, ticker):
        try:
            greeks = ticker.modelGreeks or type("G", (), {})()
            # now = datetime.now(pytz.UTC)
            now_ns = time.time_ns()
            now = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc)
            is_call = ticker.contract.right == "C"
            row = {
                "ts": ticker.time.isoformat(timespec='microseconds') if ticker.time else None,
                "systemTime": now.isoformat(timespec='microseconds'),
                "Symbol": ticker.contract.localSymbol,
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
            key = f"{self.opt_redis_key}:{row['Symbol']}"
            self.redis.set(key, json.dumps(row))
            asyncio.create_task(self.upload_from_redis(key, is_underlying=False))
        except Exception as e:
            logger.error(f"on_tick error: {e}")

    def on_underlying_tick(self, ticker):
        try:
            now = datetime.utcnow()
            row = {
                "Time": ticker.time.isoformat() if ticker.time else None,
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
            key = f"{self.und_redis_key}:{row['Symbol']}"
            self.redis.set(key, json.dumps(row))
            asyncio.create_task(self.upload_from_redis(key, is_underlying=True))
        except Exception as e:
            logger.error(f"on_underlying_tick error: {e}")

    async def upload_from_redis(self, key, is_underlying=False):
        try:
            row_data = self.redis.get(key)
            if row_data:
                row = json.loads(row_data)
                if is_underlying:
                    row['Time'] = datetime.fromisoformat(row['Time'])
                else:
                    row['ts'] = datetime.fromisoformat(row['ts'])
                    row['systemTime'] = datetime.fromisoformat(row['systemTime'])

                await self._upload_to_qdb([row], is_underlying=is_underlying)
                self.redis.delete(key)
        except Exception as e:
            logger.error(f"upload_from_redis error: {e}")

    async def start_listening(self, client_id: int = 15, max_retries: int = 3, retry_delay: int = 2):
        """Main connection loop with retry logic."""
        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                logger.info(f"Connecting to TWS (clientId={client_id})")
                self.ib.connect("127.0.0.1", 7496, clientId=client_id, timeout=10)
                logger.info("Connected to TWS")

                # Start queue processing tasks
                # asyncio.create_task(self.process_opt())
                # asyncio.create_task(self.process_und())

                await self.ensure_sender()
                if not self.running:
                    raise Exception("Failed to connect to QuestDB")

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
                        symbol = contract.localSymbol
                        self.create_table_for_symbol(symbol, is_underlying=False)
                        time.sleep(0.1)
                        # created_tables.append(symbol)
                        logger.info(f"Created table for option symbol: {symbol}")

                self.tickers = [self.ib.reqMktData(contract, genericTickList="106,100,101") for contract in option_contracts]
                for ticker in self.tickers:
                    ticker.updateEvent += self.on_tick

                logger.info(f"Subscribed to {len(self.tickers)} option contracts")

                # Start printing thread
                self.start_printing()

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
        self.executor.shutdown(wait=True)
        logger.info("Thread pool shut down")
        if self.print_thread and self.print_thread.is_alive():
            logger.info("Waiting for print thread to terminate")
        if self.ib.isConnected():
            if self.underlying_ticker:
                self.ib.cancelMktData(self.underlying_ticker)
                logger.info("Unsubscribed from underlying ticker")
            for ticker in self.tickers:
                self.ib.cancelMktData(ticker)
            self.ib.disconnect()
            logger.info("TWS disconnected")

async def main():
    ib = IB()
    listener = OptionChainListener(
        ib_client=ib,
        questdb_host="qdb3.twocc.in",
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
    asyncio.run(main())

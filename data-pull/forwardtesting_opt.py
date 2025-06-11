from abc import ABC, abstractmethod
from functools import partial # lru_cache removed
from typing import  Dict, Optional
import re
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq

import pandas as pd
from typing import Dict, Optional
import pandas as pd
from questdb_query import Endpoint
import time
import asyncio
import time
import aiohttp
import pandas as pd
import asyncio
import time
import re
import atexit
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict

import aiohttp


class Endpoint:
    """HTTP connection parameters into QuestDB"""
    def __init__(self, host='127.0.0.1', port=None, https=False, 
                 username=None, password=None, token=None):
        self.host = host
        self.port = port or (443 if https else 9000)
        self.https = https
        self.username = username
        self.password = password
        self.token = token
        
        if ((self.username or self.password) and 
            not (self.username and self.password)):
            raise ValueError('Must provide both username and password or neither')
        if self.token and self.username:
            raise ValueError('Cannot use token with username and password')
        if token and not re.match(r'^[A-Za-z0-9-._~+/]+=*$', token):
            raise ValueError("Invalid characters in token")

    @property
    def url(self):
        protocol = 'https' if self.https else 'http'
        return f'{protocol}://{self.host}:{self.port}'

class QuestDBClient:
    """High-frequency async client for QuestDB operations"""
    def __init__(self, endpoint: Endpoint):
        self._endpoint = endpoint
        self._session = None
        self._loop = None
        self._loop_thread = None
        self._shutdown_event = None
        atexit.register(self.cleanup)

    def _start_event_loop(self):
        """Start event loop in a separate thread"""
        def run_loop():
            # Use SelectorEventLoop on Windows for aiohttp compatibility
            import platform
            if platform.system() == 'Windows':
                self._loop = asyncio.SelectorEventLoop()
            else:
                self._loop = asyncio.new_event_loop()
            
            asyncio.set_event_loop(self._loop)
            self._shutdown_event = asyncio.Event()
            self._loop.run_until_complete(self._shutdown_event.wait())
            self._loop.close()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        while self._loop is None:
            time.sleep(0.01)

    async def _create_session(self):
        """Create aiohttp session with proper authentication"""
        if self._session is not None and not self._session.closed:
            return  # Session exists and is healthy
            
        auth = None
        if self._endpoint.username:
            auth = aiohttp.BasicAuth(self._endpoint.username, self._endpoint.password)
        
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(auth=auth, timeout=timeout)

    async def _recreate_session(self):
        """Force recreate session (close old one and create new)"""
        # Close existing session if it exists
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Create new session
        self._session = None
        await self._create_session()
        print("New session created successfully")

    @staticmethod
    def _auth_headers(endpoint: Endpoint) -> Optional[Dict[str, str]]:
        """Generate authentication headers"""
        if endpoint.token:
            return {'Authorization': f'Bearer {endpoint.token}'}
        return None

    async def _execute_query_async(self, query: str, retry_count: int = 0) -> Optional[bytes]:
        """Execute a SQL query and return raw bytes with automatic session recovery"""
        if self._session is None or self._session.closed:
            await self._create_session()

        url = f'{self._endpoint.url}/exp'
        params = [('query', query)]
        headers = self._auth_headers(self._endpoint)

        try:
            async with self._session.get(url=url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise ValueError(f'QuestDB Error {resp.status}: {error_text}')
                return await resp.content.read()
                
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError, 
                aiohttp.ClientConnectionError, OSError) as e:
            # Session might be corrupted/closed - try to recover once
            if retry_count < 1:
                print(f"Session error detected, creating new session: {e}")
                await self._recreate_session()
                return await self._execute_query_async(query, retry_count + 1)
            else:
                raise RuntimeError(f"Session recovery failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

    def execute_query(self, query: str) -> Optional[bytes]:
        """Synchronous wrapper for query execution with session recovery"""
        if self._loop is None:
            self._start_event_loop()
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._execute_query_async(query), self._loop
            )
            return future.result(timeout=10)
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None

    async def _close_session(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None

    def cleanup(self):
        """Cleanup resources"""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._close_session(), self._loop
            ).result(timeout=1)
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=1)





class DataFetchStrategy(ABC):
    """Abstract base class for data fetching strategies"""
    @abstractmethod
    def fetch_data(self, client: QuestDBClient, **kwargs) -> Optional[bytes]:
        pass


class TickerDataStrategy(DataFetchStrategy):
    """Strategy for fetching ticker data"""
    def fetch_data(self, client: QuestDBClient, ticker: str, limit: int = 1) -> Optional[bytes]:
        # query = f"SELECT * FROM {ticker} ORDER BY Datetime DESC LIMIT {limit}"
        query = f"SELECT * FROM {ticker} ORDER BY Datetime"
        # query = f"SHOW TABLES"
        return client.execute_query(query)


class _Data:
    """Main class for fetching data with strategies"""
    def __init__(self, endpoint: Endpoint):
        self.client = QuestDBClient(endpoint)
        self.strategy: Optional[DataFetchStrategy] = None

    def set_strategy(self, strategy: DataFetchStrategy):
        self.strategy = strategy

    def fetch_data(self, **kwargs) -> Optional[bytes]:
        if not self.strategy:
            raise ValueError("No strategy set")
        
        try:
            return self.strategy.fetch_data(self.client, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def cleanup(self):
        self.client.cleanup()



# # Usage example
if __name__ == "__main__":
    # endpoint = Endpoint(
    #     host='localhost', 
    #     https=False, 
    #     username='admin', 
    #     password='quest'
    # )
    endpoint = Endpoint(
        host='qdb3.twocc.in', 
        https=True, 
        username='2Cents', 
        password='2Cents1012cc'
    )
    
    fetcher = _Data(endpoint)
    fetcher.set_strategy(TickerDataStrategy())
    ip=0
    storedict={}
    for j in range(38):
        for i in range(10):
            start =  time.time()
            
            ticker_data = fetcher.fetch_data(ticker=f'data0')
            # print(ticker_data)
            if ticker_data:
                # print(ticker_data)
                print(f"Iteration {i+1}: Fetched {len(ticker_data)} bytes")
            else:
                print(f"Iteration {i+1}: No data")
            end = time.time()
            print(f"Time taken: {end - start:.2f} seconds")
            ip+=1
            storedict[ip]=round((end - start),7)*1000
                # time.sleep(1)
    print(storedict)
    df = pd.DataFrame.from_dict(storedict, orient='index', columns=['time_taken_seconds'])
    df.index.name = 'iteration'
    df.reset_index(inplace=True)

    print(df)
    df.to_csv("remotenew.csv")

    #multi-table
    # for i in range(10):
    #     start =  time.time()
    #     for j in range(2,6):
    #         ticker_data = fetcher.fetch_data(ticker=f'data{j}')
    #         if ticker_data:
    #             print(ticker_data)
    #             print(f"Iteration {i+1}: Fetched {len(ticker_data)} bytes")
    #         else:
    #             print(f"Iteration {i+1}: No data")
    #         end = time.time()
    #         print(f"Time taken: {end - start:.2f} seconds")
    #         # time.sleep(1)

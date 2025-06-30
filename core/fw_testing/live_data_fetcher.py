import traceback
import warnings
from datetime import datetime, date as DateObject # Added DateObject

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
            return future.result(timeout=30)
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
            ).result(timeout=5)
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5)

class _Data:
    """Main class for fetching data with strategies"""
    def __init__(self, endpoint: Endpoint):
        self.consumer = QuestDBClient(endpoint)
        
    def get_ticker_data(self, ticker: str, limit: int = 1) -> Optional[bytes]:
        query = f"SELECT * FROM {ticker} ORDER BY ts DESC LIMIT {limit}"
        try:
            return self.consumer.execute_query(query)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def cleanup(self):
        self.consumer.cleanup()


# Usage example
if __name__ == "__main__":
    endpoint = Endpoint(
        host='qdb3.twocc.in', 
        https=True, 
        username='2Cents', 
        password='2Cents1012cc'
    )
    
    fetcher = _Data(endpoint)
    # Get all table names
    table_names_data = QuestDBClient(endpoint).execute_query("SHOW TABLES")
    df = pd.read_csv(BytesIO(table_names_data))
    table_names = df["table_name"].tolist()
    print(table_names)

    for ticker in table_names:
        start =  time.time()
        ticker_data = fetcher.get_ticker_data(ticker=ticker)
        if ticker_data:    
            end = time.time()
            print(f"Time taken: {end - start:.5f} seconds")

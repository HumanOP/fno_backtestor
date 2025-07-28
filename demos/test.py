import pandas as pd
from backtesting import Backtest, Strategy
from datetime import time

# Load data
df = pd.read_csv("banknifty_data_1m.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)

# Required columns: Open, High, Low, Close
df = df[["Open", "High", "Low", "Close"]]

class IntradayBreakout(Strategy):
    def init(self):
        self.current_day = None
        self.breakout_taken = False
        self.high = 0
        self.low = 0
        self.first = 0

    def next(self):
        dt = self.data.index[-1]
        current_time = dt.time()
        current_day = dt.date()

        # Reset at the beginning of a new day
        if self.current_day != current_day:
            self.current_day = current_day
            self.breakout_taken = False
            self.high = 0
            self.low = 0
            self.first = 0

        # Capture first 5 minutes' range
        if current_time <= time(9, 20):
            if self.first == 0:
                self.high = self.data.High[-1]
                self.low = self.data.Low[-1]
                self.first += 1
            else:
                self.high = max(self.high, self.data.High[-1])
                self.low = min(self.low, self.data.Low[-1])
            return  # Wait for breakout time

        # Don't take new trades if already in one
        if self.breakout_taken or self.position:
            return

        # Candle and breakout logic
        open_ = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        close = self.data.Close[-1]

        body = abs(close - open_)
        range_ = high - low
        body_ratio = body / range_ if range_ > 0 else 0

        # Bullish breakout
        if close > self.high and close > open_ and body_ratio > 0.7:
            sl = low
            risk = close - sl
            target = close + 3 * risk
            self.buy(size=1, sl=sl, tp=target)
            self.breakout_taken = True

        # Bearish breakout
        elif close < self.low and close < open_ and body_ratio > 0.7:
            sl = high
            risk = sl - close
            target = close - 3 * risk
            self.sell(size=1, sl=sl, tp=target)
            self.breakout_taken = True

        # Square off before market close
        if self.position and current_time >= time(15, 15):
            self.position.close()

        
     
            

# Run the backtest
bt = Backtest(df, IntradayBreakout, cash=100_000, commission=0.002)
stats = bt.run()
print(stats)
bt.plot()

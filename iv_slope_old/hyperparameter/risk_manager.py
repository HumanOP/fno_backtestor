import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from datetime import datetime
from typing import List, Tuple

class FixedLossPositionSizing:
    """Calculates position size based on a fixed percentage of account loss."""
    def __init__(self, account_value: float, margin: float, max_loss_percentage: float = 0.0, use_fixed_loss_sizing: bool = True):
        self.account_value = account_value
        self.margin = margin
        self.max_loss_percentage = max_loss_percentage
        self.use_fixed_loss_sizing = use_fixed_loss_sizing

    def get_timestamp(self) -> str:
        """Return the current timestamp as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_max_position(self) -> int:
        """Calculate maximum position size based on account value and margin."""
        return int(np.floor(self.account_value / self.margin))

    def size(self) -> int:
        """
        Calculate position size based on a fixed percentage of account value as max loss.
        """
        if not self.use_fixed_loss_sizing:
            return 0

        max_loss_dollars = self.account_value * (self.max_loss_percentage / 100.0)
        position_size = int(np.floor(max_loss_dollars / self.margin))
        max_position_size = self.calculate_max_position()
        position_size = min(position_size, max_position_size)

        print(f"{self.get_timestamp()} Fixed loss position sizing calculated...\n"
              f"{self.get_timestamp()} Max Loss Percentage: {self.max_loss_percentage:.2f}%\n"
              f"{self.get_timestamp()} Max Loss in Dollars: ${max_loss_dollars:.2f}\n"
              f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size


class IVBasedPositionSizing:
    """Calculates position size based on implied volatility (IV) ranges."""
    def __init__(
        self,
        account_value: float,
        margin: float,
        current_iv: float,
        use_vix_position_sizing: bool = True,
        iv_ranges: List[Tuple[float, float]] = [
            (0.40, 0.50),
            (0.30, 0.40),
            (0.20, 0.35),
            (0.15, 0.30),
            (0.10, 0.25),
        ]
    ):
        self.account_value = account_value
        self.margin = margin
        self.current_iv = current_iv
        self.use_vix_position_sizing = use_vix_position_sizing
        self.iv_ranges = iv_ranges

    def get_timestamp(self) -> str:
        """Return the current timestamp as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_max_position(self) -> int:
        """Calculate maximum position size based on account value and margin."""
        return int(np.floor(self.account_value / self.margin))

    def size(self) -> int:
        """
        Calculate position size based on implied volatility (IV) ranges.
        """
        if not self.use_vix_position_sizing or self.current_iv is None:
            return 0

        position_size = 0
        range_msg = f"below: {self.iv_ranges[-1][0] * 100:.0f}%"
        allocation_msg = "0%"
        for lower_bound, allocation in self.iv_ranges:
            if lower_bound <= self.current_iv:
                position_size = int(np.floor(self.account_value * allocation / self.margin))
                range_msg = f">= {lower_bound:.2f}"
                allocation_msg = f"{allocation * 100:.0f}%"
                break

        position_size = min(position_size, self.calculate_max_position())

        print(f"{self.get_timestamp()} IV-based position sizing calculated...\n"
              f"{self.get_timestamp()} IV is {range_msg} (Current IV: {self.current_iv:.2f})\n"
              f"{self.get_timestamp()} Position size is {allocation_msg} of account value\n"
              f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size
    
class KellyPositionSizing:
    """Calculates position size using Fractional Kelly Criterion."""
    def __init__(
        self,
        account_value: float,
        margin: float,
        win_probability: float = None,
        win_loss_ratio: float = None,
        use_kelly_sizing: bool = True,
        fraction: float = 1.0
    ):
        self.account_value = account_value
        self.margin = margin
        self.win_probability = win_probability
        self.win_loss_ratio = win_loss_ratio
        self.use_kelly_sizing = use_kelly_sizing
        self.fraction = fraction

    def get_timestamp(self) -> str:
        """Return the current timestamp as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_max_position(self) -> int:
        """Calculate maximum position size based on account value and margin."""
        return int(np.floor(self.account_value / self.margin))

    def size(self, pop: float = None, win_loss_ratio: float = None) -> int:
        """
        Calculate position size using Fractional Kelly Criterion.
        Parameters:
        - pop: Probability of profit (optional, defaults to self.win_probability)
        - win_loss_ratio: Win/loss ratio (optional, defaults to self.win_loss_ratio)
        Returns:
        - Position size (number of contracts)
        """
        if not self.use_kelly_sizing:
            return 0

        # Use provided parameters or instance variables
        p = pop if pop is not None else self.win_probability
        b = win_loss_ratio if win_loss_ratio is not None else self.win_loss_ratio

        if p is None or b is None:
            print(f"{self.get_timestamp()} Error: PoP or win/loss ratio not set.")
            return 0

        q = 1 - p
        raw_kelly = (b * p - q) / b if b != 0 else 0
        fractional_kelly = max(min(raw_kelly * self.fraction, 0.25), 0)  # Clip to 25% max to avoid overbetting
        position_size = int(np.floor(self.account_value * fractional_kelly / self.margin))
        position_size = min(position_size, self.calculate_max_position())

        print(f"{self.get_timestamp()} Kelly Criterion calculated...\n"
              f"{self.get_timestamp()} Win Probability: {p:.2f}, Win/Loss Ratio: {b:.2f}\n"
              f"{self.get_timestamp()} Raw Kelly Fraction: {raw_kelly:.2f}, Fractional Kelly Applied: {self.fraction:.2f}\n"
              f"{self.get_timestamp()} Final Kelly Fraction Used: {fractional_kelly:.2f}\n"
              f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size
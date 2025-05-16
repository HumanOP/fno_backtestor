import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

class RiskManagement:
    def __init__(self, account_value, margin, current_iv=None, win_probability=None, win_loss_ratio=None):
        self.account_value = account_value
        self.margin = margin
        self.current_iv = current_iv
        self.win_probability = win_probability
        self.win_loss_ratio = win_loss_ratio

    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_pop(self, S, T, sigma, r, legs):
        """
        Calculate Probability of Profit and win/loss ratio for any multi-legged options strategy.
        Parameters:
        - S: Current underlying price
        - T: Time to expiration (in years)
        - sigma: Implied volatility
        - r: Risk-free rate
        - legs: List of dictionaries, each containing:
            - type: 'call' or 'put'
            - strike: Strike price
            - premium: Premium paid (positive for long, negative for short)
            - quantity: Number of contracts (positive for long, negative for short)
        Returns:
        - Tuple (pop, win_loss_ratio): Probability of profit and win/loss ratio
        """
        # Calculate net premium
        net_premium = sum(leg["premium"] * leg["quantity"] for leg in legs)

        # Define payoff function
        def payoff(S_T):
            total_payoff = 0
            for leg in legs:
                if leg["type"] == "call":
                    intrinsic = max(0, S_T - leg["strike"])
                elif leg["type"] == "put":
                    intrinsic = max(0, leg["strike"] - S_T)
                else:
                    raise ValueError(f"Invalid option type: {leg['type']}")
                total_payoff += leg["quantity"] * (intrinsic - leg["premium"])
            return total_payoff + net_premium

        # Log-normal PDF for underlying price at expiration
        def lognormal_pdf(S_T):
            mu = np.log(S) + (r - 0.5 * sigma**2) * T
            sigma_T = sigma * np.sqrt(T)
            if S_T <= 0:
                return 0
            return (1 / (S_T * sigma_T * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((np.log(S_T) - mu) / sigma_T)**2
            )

        # Evaluate payoff over a price range to find profit/loss regions
        price_range = np.linspace(S * 0.5, S * 1.5, 1000)
        payoffs = [payoff(p) for p in price_range]
        
        # Identify profit regions (payoff > 0)
        profit_regions = []
        start = None
        for i in range(len(price_range)):
            if payoffs[i] > 0 and start is None:
                start = price_range[i]
            elif (payoffs[i] <= 0 or i == len(price_range) - 1) and start is not None:
                end = price_range[i - 1] if i < len(price_range) - 1 else price_range[i]
                profit_regions.append((start, end))
                start = None

        # Calculate PoP by integrating over profit regions
        pop = 0
        for start, end in profit_regions:
            pop += quad(lognormal_pdf, start, end)[0]

        if not profit_regions:
            print(f"{self.get_timestamp()} Warning: No profit regions found.")
            pop = 0

        # Calculate max profit and max loss for win/loss ratio
        max_profit = max(payoffs) if payoffs else 0
        max_loss = -min(payoffs) if payoffs else 0
        self.win_probability = pop
        self.win_loss_ratio = max_profit / max_loss if max_loss > 0 else 1.0

        print(f"{self.get_timestamp()} PoP calculated for strategy: {pop:.2%}")
        print(f"{self.get_timestamp()} Max Profit: ${max_profit:.2f}, Max Loss: ${max_loss:.2f}")
        print(f"{self.get_timestamp()} Win/Loss Ratio: {self.win_loss_ratio:.2f}")

        return pop, self.win_loss_ratio

    def kelly_criterion_position_sizing(self, pop=None, win_loss_ratio=None, use_kelly_sizing=True):
        """
        Calculate position size using Kelly Criterion.
        Parameters:
        - pop: Probability of profit (optional, defaults to self.win_probability)
        - win_loss_ratio: Win/loss ratio (optional, defaults to self.win_loss_ratio)
        - use_kelly_sizing: Whether to perform sizing
        """
        if not use_kelly_sizing:
            return 0

        # Use provided parameters or instance variables
        p = pop if pop is not None else self.win_probability
        b = win_loss_ratio if win_loss_ratio is not None else self.win_loss_ratio

        if p is None or b is None:
            print(f"{self.get_timestamp()} Error: PoP or win/loss ratio not set.")
            return 0

        q = 1 - p
        kelly_fraction = (b * p - q) / b if b != 0 else 0
        kelly_fraction = min(max(kelly_fraction, 0), 0.25)
        position_size = int(np.floor(self.account_value * kelly_fraction / self.margin))

        print(f"{self.get_timestamp()} Kelly Criterion calculated...\n"
              f"{self.get_timestamp()} Win Probability: {p:.2f}, Win/Loss Ratio: {b:.2f}\n"
              f"{self.get_timestamp()} Kelly Fraction: {kelly_fraction:.2f}\n"
              f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size

    def fixed_loss_position_sizing(self, max_loss_percentage, use_fixed_loss_sizing=True):
        """
        Calculate position size based on a fixed percentage of account value as max loss.
        """
        if not use_fixed_loss_sizing:
            return 0

        max_loss_dollars = self.account_value * (max_loss_percentage / 100.0)
        position_size = int(np.floor(max_loss_dollars / self.margin))
        max_position_size = int(np.floor(self.account_value / self.margin))
        position_size = min(position_size, max_position_size)

        print(f"{self.get_timestamp()} Fixed loss position sizing calculated...\n"
              f"{self.get_timestamp()} Max Loss Percentage: {max_loss_percentage:.2f}%\n"
              f"{self.get_timestamp()} Max Loss in Dollars: ${max_loss_dollars:.2f}\n"
              f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size

    def iv_based_position_sizing(self, use_vix_position_sizing=True):
        """
        Calculate position size based on implied volatility (IV) ranges.
        """
        if not use_vix_position_sizing or self.current_iv is None:
            return 0

        position_size = 0
        if 0.10 <= self.current_iv < 0.15:
            position_size = int(np.floor(self.account_value * 0.25 / self.margin))
            print(f"{self.get_timestamp()} IV is between 10 and 15...\n"
                  f"{self.get_timestamp()} Position size is 25% of account value\n"
                  f"{self.get_timestamp()} Trading {position_size} contracts")
        elif 0.15 <= self.current_iv < 0.20:
            position_size = int(np.floor(self.account_value * 0.30 / self.margin))
            print(f"{self.get_timestamp()} IV is between 15 and 20...\n"
                  f"{self.get_timestamp()} Position size is 30% of account value\n"
                  f"{self.get_timestamp()} Trading {position_size} contracts")
        elif 0.20 <= self.current_iv < 0.30:
            position_size = int(np.floor(self.account_value * 0.35 / self.margin))
            print(f"{self.get_timestamp()} IV is between 20 and 30...\n"
                  f"{self.get_timestamp()} Position size is 35% of account value\n"
                  f"{self.get_timestamp()} Trading {position_size} contracts")
        elif 0.30 <= self.current_iv < 0.40:
            position_size = int(np.floor(self.account_value * 0.40 / self.margin))
            print(f"{self.get_timestamp()} IV is between 30 and 40...\n"
                  f"{self.get_timestamp()} Position size is 40% of account value\n"
                  f"{self.get_timestamp()} Trading {position_size} contracts")
        elif self.current_iv >= 0.40:
            position_size = int(np.floor(self.account_value * 0.50 / self.margin))
            print(f"{self.get_timestamp()} IV is greater than 40...\n"
                  f"{self.get_timestamp()} Position size is 50% of account value\n"
                  f"{self.get_timestamp()} Trading {position_size} contracts")

        return position_size
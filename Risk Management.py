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
        from numpy import linspace

        net_premium = sum(leg["premium"] * leg["quantity"] for leg in legs)

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

        def lognormal_pdf(S_T):
            mu = np.log(S) + (r - 0.5 * sigma ** 2) * T
            sigma_T = sigma * np.sqrt(T)
            if S_T <= 0:
                return 0
            return (1 / (S_T * sigma_T * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((np.log(S_T) - mu) / sigma_T) ** 2
            )

        price_range = linspace(S * 0.5, S * 1.5, 1000)
        dS = price_range[1] - price_range[0]
        payoffs = np.array([payoff(p) for p in price_range])
        pdfs = np.array([lognormal_pdf(p) for p in price_range])

        # Probability of profit (PoP)
        profit_mask = payoffs > 0
        loss_mask = payoffs < 0

        pop = np.sum(pdfs[profit_mask] * dS)

        # Expected win/loss
        expected_win = np.sum(payoffs[profit_mask] * pdfs[profit_mask] * dS)
        expected_loss = -np.sum(payoffs[loss_mask] * pdfs[loss_mask] * dS)

        # Avoid division by zero
        win_loss_ratio = expected_win / expected_loss if expected_loss > 0 else 1.0

        self.win_probability = pop
        self.win_loss_ratio = win_loss_ratio

        print(f"{self.get_timestamp()} PoP calculated for strategy: {pop:.2%}")
        print(f"{self.get_timestamp()} Expected Win: ${expected_win:.2f}, Expected Loss: ${expected_loss:.2f}")
        print(f"{self.get_timestamp()} Win/Loss Ratio (expected): {win_loss_ratio:.2f}")

        return pop, win_loss_ratio


    def kelly_criterion_position_sizing(self, pop=None, win_loss_ratio=None, use_kelly_sizing=True, fraction=1.0):
        """
        Calculate position size using Fractional Kelly Criterion.

        Parameters:
        - pop: Probability of profit (optional, defaults to self.win_probability)
        - win_loss_ratio: Win/loss ratio (optional, defaults to self.win_loss_ratio)
        - use_kelly_sizing: Whether to perform sizing
        - fraction: Fraction of full Kelly (1.0 = full Kelly, 0.5 = half Kelly, etc.)
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
        raw_kelly = (b * p - q) / b if b != 0 else 0
        fractional_kelly = max(min(raw_kelly * fraction, 0.25), 0)  # Clip to 25% max to avoid overbetting
        position_size = int(np.floor(self.account_value * fractional_kelly / self.margin))

        print(f"{self.get_timestamp()} Kelly Criterion calculated...\n"
            f"{self.get_timestamp()} Win Probability: {p:.2f}, Win/Loss Ratio: {b:.2f}\n"
            f"{self.get_timestamp()} Raw Kelly Fraction: {raw_kelly:.2f}, Fractional Kelly Applied: {fraction:.2f}\n"
            f"{self.get_timestamp()} Final Kelly Fraction Used: {fractional_kelly:.2f}\n"
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
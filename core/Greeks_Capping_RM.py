from datetime import datetime

class RiskManagement:
    def __init__(self):
        self.initial_greeks = {}  # Store initial Greeks and IV for each leg
        self.default_thresholds = {
            'delta': 0.5,  # Absolute delta threshold
            'gamma': 0.1,    # Gamma threshold
            'theta': -0.05,  # Theta threshold (negative for decay)
            'vega': 0.2,     # Vega threshold
            'iv': 0.3        # Implied volatility threshold
        }

    def get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def greeks_capping(self, legs, custom_default_thresholds=None):
        """
        Track initial Greeks and IV, calculate percentage change from initial to current values,
        and exit legs if current values exceed their specific thresholds.
        Parameters:
        - legs: List of dictionaries, each containing:
            - type: 'call' or 'put'
            - strike: Strike price
            - leg_id: Unique identifier for the leg
            - greeks: Dictionary with current 'delta', 'gamma', 'theta', 'vega'
            - iv: Current implied volatility
            - thresholds: Dictionary with per-leg thresholds for 'delta', 'gamma', 'theta', 'vega', 'iv'
        - custom_default_thresholds: Optional dictionary to override default class-level thresholds
        Returns:
        - List of leg_ids to exit
        """
        # Cache timestamp to avoid multiple calls
        timestamp = self.get_timestamp()

        # Update default thresholds if provided
        if custom_default_thresholds:
            self.default_thresholds.update(custom_default_thresholds)
        default_thresholds = self.default_thresholds
        required_greeks = {'delta', 'gamma', 'theta', 'vega'}

        # Initialize Greeks and IV if not set
        if not self.initial_greeks:
            self.initial_greeks = {}
            log_lines = []
            for leg in legs:
                leg_id = leg.get('leg_id')
                greeks = leg.get('greeks')
                iv = leg.get('iv')
                if not leg_id or not greeks or not all(k in greeks for k in required_greeks) or iv is None:
                    log_lines.append(f"{timestamp} Error: Missing or incomplete Greeks/IV data for leg {leg_id}")
                    continue
                self.initial_greeks[leg_id] = {'greeks': greeks, 'iv': iv}
                log_lines.append(
                    f"{timestamp} Initial values for leg {leg_id} ({leg['type']} K={leg['strike']:.2f}): "
                    f"Delta={greeks['delta']:.4f}, Gamma={greeks['gamma']:.4f}, "
                    f"Theta={greeks['theta']:.4f}, Vega={greeks['vega']:.4f}, IV={iv:.4f}"
                )
            if log_lines:
                print("\n".join(log_lines))

        # Process legs for percentage changes and threshold checks
        legs_to_exit = []
        log_lines = []
        for leg in legs:
            leg_id = leg.get('leg_id')
            greeks = leg.get('greeks')
            iv = leg.get('iv')
            if not leg_id or not greeks or not all(k in greeks for k in required_greeks) or iv is None:
                log_lines.append(f"{timestamp} Error: Missing or incomplete Greeks/IV data for leg {leg_id}")
                continue

            initial = self.initial_greeks.get(leg_id, {'greeks': greeks, 'iv': iv})
            thresholds = leg.get('thresholds', default_thresholds)
            if not all(k in thresholds for k in required_greeks | {'iv'}):
                log_lines.append(f"{timestamp} Warning: Incomplete thresholds for leg {leg_id}, using defaults")
                thresholds = default_thresholds

            # Calculate percentage changes
            percent_changes = {}
            for greek in required_greeks:
                initial_val = initial['greeks'][greek]
                current_val = greeks[greek]
                percent_changes[greek] = ((current_val - initial_val) / abs(initial_val) * 100) if initial_val != 0 else 0.0
            initial_iv = initial['iv']
            percent_changes['iv'] = ((iv - initial_iv) / initial_iv * 100) if initial_iv != 0 else 0.0

            # Log current values, percentage changes, and thresholds
            log_lines.append(
                f"{timestamp} Current values for leg {leg_id} ({leg['type']} K={leg['strike']:.2f}): "
                f"Delta={greeks['delta']:.4f}, Gamma={greeks['gamma']:.4f}, "
                f"Theta={greeks['theta']:.4f}, Vega={greeks['vega']:.4f}, IV={iv:.4f}"
            )
            log_lines.append(
                f"{timestamp} Percentage change from initial for leg {leg_id}: "
                f"Delta={percent_changes['delta']:.2f}%, Gamma={percent_changes['gamma']:.2f}%, "
                f"Theta={percent_changes['theta']:.2f}%, Vega={percent_changes['vega']:.2f}%, IV={percent_changes['iv']:.2f}%"
            )
            log_lines.append(
                f"{timestamp} Thresholds for leg {leg_id}: "
                f"Delta={thresholds['delta']:.4f}, Gamma={thresholds['gamma']:.4f}, "
                f"Theta={thresholds['theta']:.4f}, Vega={thresholds['vega']:.4f}, IV={thresholds['iv']:.4f}"
            )

            # Check thresholds
            if (abs(greeks['delta']) > thresholds['delta'] or
                greeks['gamma'] > thresholds['gamma'] or
                greeks['theta'] < thresholds['theta'] or
                greeks['vega'] > thresholds['vega'] or
                iv > thresholds['iv']):
                log_lines.append(f"{timestamp} Exit recommended for leg {leg_id} due to Greeks or IV exceeding thresholds.")
                legs_to_exit.append(leg_id)

        if log_lines:
            print("\n".join(log_lines))

        return legs_to_exit
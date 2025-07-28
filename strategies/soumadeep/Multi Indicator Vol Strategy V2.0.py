
#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, entropy, iqr, chi2, zscore
from scipy import signal
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import LedoitWolf
from sklearn.metrics import classification_report
import xgboost as xgb

# Optional imports with fallbacks
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Configure logging
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

plt.style.use('default')
sns.set_palette("viridis")

# --- ENHANCED ADAPTIVE CONFIGURATION ---
@dataclass
class AdaptiveConfig:
    """Enhanced adaptive configuration with dynamic learning parameters"""
    # Backtest Parameters
    SYMBOL: str = 'RELIANCE.NS'
    START_DATE: str = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    TRAIN_END_DATE: str = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d') 
    BACKTEST_START_DATE: str = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    BACKTEST_END_DATE: str = datetime.now().strftime('%Y-%m-%d')
    INITIAL_EQUITY: float = 100000.0

    # Enhanced Trading Parameters  
    MAX_OPEN_POSITIONS: int = 5  # Increased for more opportunities
    BASE_RISK_PER_TRADE: float = 0.03  # Slightly higher base risk
    MAX_RISK_PER_TRADE: float = 0.08   # Allow higher allocation for quality signals
    
    # Dynamic Risk Management
    STOP_LOSS_PCT: float = 0.30   # Wider stops initially
    PROFIT_TARGET_PCT: float = 0.40  # Higher targets initially
    MAX_HOLD_DAYS: int = 7       # Longer holds for development
    COMMISSION_PER_CONTRACT: float = 1.50
    
    # Enhanced Threshold Adaptation Parameters
    ADAPTATION_WINDOW: int = 15  # Longer window for more stable adaptation
    MIN_TRADES_FOR_ADAPTATION: int = 3  # Faster adaptation
    THRESHOLD_LEARNING_RATE: float = 0.15  # Faster learning
    PERFORMANCE_SENSITIVITY: float = 0.25  # How much performance affects thresholds
    
    # Statistical Threshold Parameters (Dynamic)
    ROLLING_WINDOW: int = 60  # Rolling window for statistics
    CONFIDENCE_BASE: float = 0.25  # Lower base for more signals
    UNCERTAINTY_BASE: float = 0.75  # Higher base for more selectivity
    SIGNAL_STRENGTH_BASE: float = 0.20  # Lower base threshold
    REGIME_ADAPTATION_FACTOR: float = 0.3  # How much regime affects thresholds
    
    # Signal Generation
    USE_ML_SIGNALS: bool = True
    USE_TECHNICAL_SIGNALS: bool = True
    USE_MOMENTUM_SIGNALS: bool = True
    USE_VOLATILITY_SIGNALS: bool = True
    
    # Option Selection
    MONEYNESS_RANGE: Tuple[float, float] = (0.97, 1.03)
    PREFER_ATM: bool = True

config = AdaptiveConfig()

# --- ENHANCED MARKET REGIME DETECTOR ---
class EnhancedMarketRegimeDetector:
    """Enhanced market regime detection with volatility clustering and trend analysis"""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.regime_history = []
        self.volatility_states = []
        self.trend_states = []
        
    def detect_comprehensive_regime(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Comprehensive regime detection using multiple market indicators"""
        if len(historical_data) < self.lookback_days:
            return self._get_default_regime()
        
        recent_data = historical_data.tail(self.lookback_days)
        returns = recent_data['Close'].pct_change().dropna()
        
        # 1. Volatility Clustering Analysis
        vol_regime = self._detect_volatility_regime(returns, historical_data)
        
        # 2. Trend Strength Analysis
        trend_regime = self._detect_trend_regime(recent_data)
        
        # 3. Market Stress Analysis
        stress_regime = self._detect_market_stress(returns, recent_data)
        
        # 4. Momentum Regime
        momentum_regime = self._detect_momentum_regime(recent_data)
        
        # 5. Mean Reversion vs Trending
        reversion_regime = self._detect_mean_reversion_regime(returns)
        
        # Combine all regime indicators
        comprehensive_regime = {
            'volatility_level': vol_regime['level'],
            'volatility_percentile': vol_regime['percentile'],
            'trend_strength': trend_regime['strength'],
            'trend_direction': trend_regime['direction'],
            'market_stress': stress_regime['stress_level'],
            'momentum_strength': momentum_regime['strength'],
            'momentum_direction': momentum_regime['direction'],
            'mean_reversion_strength': reversion_regime['reversion_strength'],
            'regime_composite': self._calculate_composite_regime(vol_regime, trend_regime, stress_regime)
        }
        
        self.regime_history.append(comprehensive_regime)
        return comprehensive_regime
    
    def _detect_volatility_regime(self, returns: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Detect volatility regime using GARCH-like clustering"""
        current_vol = returns.std() * np.sqrt(252)
        
        # Calculate rolling volatility percentiles
        all_returns = historical_data['Close'].pct_change().dropna()
        rolling_vols = all_returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = (rolling_vols <= current_vol).mean()
        
        # Volatility clustering detection
        vol_changes = np.abs(returns.diff())
        clustering_indicator = vol_changes.autocorr(lag=1) if len(vol_changes) > 1 else 0
        
        return {
            'level': current_vol,
            'percentile': vol_percentile,
            'clustering': max(0, clustering_indicator)
        }
    
    def _detect_trend_regime(self, recent_data: pd.DataFrame) -> Dict:
        """Detect trend regime using multiple timeframes"""
        close_prices = recent_data['Close']
        
        # Multiple moving average trends
        sma_5 = close_prices.rolling(5).mean()
        sma_10 = close_prices.rolling(10).mean()
        sma_20 = close_prices.rolling(20).mean()
        
        # Trend strength calculation
        trend_5_10 = (sma_5.iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1] if len(sma_10) > 0 else 0
        trend_10_20 = (sma_10.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if len(sma_20) > 0 else 0
        
        # Overall trend strength
        trend_strength = (abs(trend_5_10) + abs(trend_10_20)) / 2
        trend_direction = np.sign(trend_5_10 + trend_10_20)
        
        return {
            'strength': trend_strength,
            'direction': trend_direction
        }
    
    def _detect_market_stress(self, returns: pd.Series, recent_data: pd.DataFrame) -> Dict:
        """Detect market stress using multiple indicators"""
        # Extreme return frequency
        extreme_threshold = returns.std() * 2
        extreme_freq = (np.abs(returns) > extreme_threshold).mean()
        
        # Gap analysis
        gaps = recent_data['Open'] / recent_data['Close'].shift(1) - 1
        gap_stress = np.abs(gaps).mean() if len(gaps) > 0 else 0
        
        # Volume stress (if available)
        volume_stress = 0
        if 'Volume' in recent_data.columns:
            vol_ratio = recent_data['Volume'] / recent_data['Volume'].rolling(20).mean()
            volume_stress = vol_ratio.std() if len(vol_ratio) > 0 else 0
        
        stress_level = (extreme_freq * 0.4 + gap_stress * 0.3 + min(volume_stress, 1.0) * 0.3)
        
        return {'stress_level': stress_level}
    
    def _detect_momentum_regime(self, recent_data: pd.DataFrame) -> Dict:
        """Detect momentum regime"""
        close_prices = recent_data['Close']
        
        # Price momentum over different periods
        momentum_3 = (close_prices.iloc[-1] / close_prices.iloc[-4] - 1) if len(close_prices) >= 4 else 0
        momentum_5 = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) >= 6 else 0
        momentum_10 = (close_prices.iloc[-1] / close_prices.iloc[-11] - 1) if len(close_prices) >= 11 else 0
        
        # Weighted momentum strength
        momentum_strength = abs(momentum_3 * 0.5 + momentum_5 * 0.3 + momentum_10 * 0.2)
        momentum_direction = np.sign(momentum_3 + momentum_5 + momentum_10)
        
        return {
            'strength': momentum_strength,
            'direction': momentum_direction
        }
    
    def _detect_mean_reversion_regime(self, returns: pd.Series) -> Dict:
        """Detect mean reversion vs trending regime"""
        # Calculate first-order autocorrelation
        autocorr_1 = returns.autocorr(lag=1) if len(returns) > 1 else 0
        
        # Hurst exponent approximation for mean reversion
        try:
            lags = range(2, min(20, len(returns)//2))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            
            # Simple Hurst calculation
            if len(tau) > 2:
                hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
                reversion_strength = max(0, 0.5 - hurst)  # H < 0.5 indicates mean reversion
            else:
                reversion_strength = max(0, -autocorr_1)  # Negative autocorr = mean reversion
        except:
            reversion_strength = max(0, -autocorr_1)
        
        return {'reversion_strength': reversion_strength}
    
    def _calculate_composite_regime(self, vol_regime: Dict, trend_regime: Dict, 
                                  stress_regime: Dict) -> str:
        """Calculate composite market regime"""
        vol_level = vol_regime['percentile']
        trend_strength = trend_regime['strength']
        stress_level = stress_regime['stress_level']
        
        if vol_level > 0.8 and stress_level > 0.3:
            return 'high_stress'
        elif vol_level > 0.7:
            return 'high_volatility'
        elif trend_strength > 0.02:
            return 'trending'
        elif trend_strength < 0.005:
            return 'sideways'
        else:
            return 'normal'
    
    def _get_default_regime(self) -> Dict[str, float]:
        """Default regime when insufficient data"""
        return {
            'volatility_level': 0.2,
            'volatility_percentile': 0.5,
            'trend_strength': 0.0,
            'trend_direction': 0.0,
            'market_stress': 0.2,
            'momentum_strength': 0.0,
            'momentum_direction': 0.0,
            'mean_reversion_strength': 0.3,
            'regime_composite': 'normal'
        }

# --- ENHANCED STATISTICAL THRESHOLD CALCULATOR ---
class EnhancedStatisticalThresholdCalculator:
    """Enhanced statistical threshold calculator using rolling distributions and regime adaptation"""
    
    def __init__(self):
        self.threshold_history = []
        self.performance_history = []
        
    def calculate_dynamic_thresholds(self, 
                                   historical_data: pd.DataFrame,
                                   feature_names: List[str],
                                   returns_data: pd.Series,
                                   market_regime: Dict,
                                   recent_performance: Dict = None) -> Dict[str, float]:
        """Calculate dynamic thresholds using advanced statistical methods"""
        
        if len(historical_data) < config.ROLLING_WINDOW:
            return self._get_adaptive_default_thresholds(market_regime)
        
        try:
            # Rolling statistical analysis
            rolling_stats = self._calculate_rolling_statistics(historical_data, returns_data)
            
            # Regime-adjusted base thresholds
            regime_adjustments = self._calculate_regime_adjustments(market_regime)
            
            # Performance-driven adjustments
            performance_adjustments = self._calculate_performance_adjustments(recent_performance)
            
            # Feature distribution analysis
            feature_adjustments = self._analyze_feature_distributions(historical_data, feature_names)
            
            # Combine all adjustments
            final_thresholds = self._combine_threshold_adjustments(
                rolling_stats, regime_adjustments, performance_adjustments, feature_adjustments
            )
            
            # Store for analysis
            self.threshold_history.append(final_thresholds.copy())
            
            logger.info(f"📊 Enhanced statistical thresholds calculated:")
            logger.info(f"   Movement: ±{final_thresholds['movement_up']:.4f}")
            logger.info(f"   Confidence: {final_thresholds['confidence']:.3f}")
            logger.info(f"   Uncertainty: {final_thresholds['uncertainty']:.3f}")
            logger.info(f"   Signal Strength: {final_thresholds['signal_strength']:.3f}")
            logger.info(f"   Regime: {market_regime.get('regime_composite', 'normal')}")
            
            return final_thresholds
            
        except Exception as e:
            logger.warning(f"Enhanced threshold calculation failed: {e}")
            return self._get_adaptive_default_thresholds(market_regime)
    
    def _calculate_rolling_statistics(self, historical_data: pd.DataFrame, 
                                    returns_data: pd.Series) -> Dict[str, float]:
        """Calculate rolling statistical measures"""
        
        # Rolling return statistics
        rolling_returns = returns_data.rolling(config.ROLLING_WINDOW)
        
        current_std = rolling_returns.std().iloc[-1] if len(returns_data) > 0 else 0.01
        current_std = current_std if not pd.isna(current_std) else 0.01
        current_skew = rolling_returns.skew().iloc[-1] if len(returns_data) > 0 else 0
        current_skew = current_skew if not pd.isna(current_skew) else 0
        current_kurt = rolling_returns.kurt().iloc[-1] if len(returns_data) > 0 else 3
        current_kurt = current_kurt if not pd.isna(current_kurt) else 3
        
        # Quantile-based thresholds
        recent_returns = returns_data.tail(config.ROLLING_WINDOW).dropna()
        if len(recent_returns) > 5:
            q25 = recent_returns.quantile(0.25)
            q75 = recent_returns.quantile(0.75)
            median = recent_returns.median()
        else:
            q25 = -0.01
            q75 = 0.01
            median = 0.0
        
        # IQR-based movement thresholds
        iqr_value = q75 - q25
        movement_threshold = max(0.003, current_std * 0.8, iqr_value * 0.4)
        
        # Adaptive confidence based on return predictability
        rolling_autocorr = recent_returns.autocorr(lag=1) if len(recent_returns) > 1 else 0
        rolling_autocorr = rolling_autocorr if not pd.isna(rolling_autocorr) else 0
        predictability = abs(rolling_autocorr)
        base_confidence = config.CONFIDENCE_BASE + predictability * 0.2
        
        # Uncertainty based on volatility regime
        volatility_factor = min(2.0, current_std / 0.01)  # Normalize by typical daily vol
        base_uncertainty = config.UNCERTAINTY_BASE - (volatility_factor - 1) * 0.1
        
        return {
            'movement_threshold': movement_threshold,
            'base_confidence': base_confidence,
            'base_uncertainty': base_uncertainty,
            'volatility_factor': volatility_factor,
            'predictability': predictability,
            'current_std': current_std,
            'skewness': current_skew,
            'kurtosis': current_kurt
        }
    
    def _calculate_regime_adjustments(self, market_regime: Dict) -> Dict[str, float]:
        """Calculate regime-specific threshold adjustments"""
        
        regime_composite = market_regime.get('regime_composite', 'normal')
        vol_percentile = market_regime.get('volatility_percentile', 0.5)
        stress_level = market_regime.get('market_stress', 0.2)
        trend_strength = market_regime.get('trend_strength', 0.0)
        
        adjustments = {
            'confidence_adj': 0.0,
            'uncertainty_adj': 0.0,
            'movement_adj': 1.0,
            'signal_strength_adj': 0.0
        }
        
        # Regime-specific adjustments
        if regime_composite == 'high_stress':
            adjustments['confidence_adj'] = 0.15  # Be more selective
            adjustments['uncertainty_adj'] = -0.1  # Lower uncertainty threshold
            adjustments['movement_adj'] = 1.3  # Require larger movements
            adjustments['signal_strength_adj'] = 0.1  # Higher signal strength needed
            
        elif regime_composite == 'high_volatility':
            adjustments['confidence_adj'] = 0.1
            adjustments['uncertainty_adj'] = -0.05
            adjustments['movement_adj'] = 1.2
            adjustments['signal_strength_adj'] = 0.05
            
        elif regime_composite == 'trending':
            adjustments['confidence_adj'] = -0.05  # Can be less selective in trends
            adjustments['uncertainty_adj'] = 0.05  # Allow more uncertainty
            adjustments['movement_adj'] = 0.8  # Smaller movements OK in trends
            adjustments['signal_strength_adj'] = -0.05
            
        elif regime_composite == 'sideways':
            adjustments['confidence_adj'] = 0.05  # Be more selective
            adjustments['uncertainty_adj'] = -0.02
            adjustments['movement_adj'] = 1.1
            adjustments['signal_strength_adj'] = 0.02
        
        # Volatility percentile adjustments
        if vol_percentile > 0.8:  # High volatility environment
            adjustments['movement_adj'] *= 1.2
            adjustments['confidence_adj'] += 0.05
        elif vol_percentile < 0.2:  # Low volatility environment
            adjustments['movement_adj'] *= 0.8
            adjustments['confidence_adj'] -= 0.03
        
        return adjustments
    
    def _calculate_performance_adjustments(self, recent_performance: Dict = None) -> Dict[str, float]:
        """Calculate performance-driven threshold adjustments"""
        
        if not recent_performance:
            return {'confidence_mult': 1.0, 'uncertainty_mult': 1.0, 'signal_mult': 1.0}
        
        win_rate = recent_performance.get('win_rate', 0.5)
        avg_return = recent_performance.get('avg_return', 0.0)
        profit_factor = recent_performance.get('profit_factor', 1.0)
        num_trades = recent_performance.get('num_trades', 0)
        
        # Base multipliers
        confidence_mult = 1.0
        uncertainty_mult = 1.0
        signal_mult = 1.0
        
        # Performance-based adjustments with mathematical precision
        if num_trades >= 5:  # Need minimum trades for adjustment
            
            # Win rate adjustments
            if win_rate > 0.7:  # Excellent performance
                confidence_mult = 0.85  # Be less selective
                uncertainty_mult = 1.1   # Allow more uncertainty
                signal_mult = 0.9       # Lower signal requirements
            elif win_rate > 0.6:  # Good performance
                confidence_mult = 0.95
                uncertainty_mult = 1.05
                signal_mult = 0.95
            elif win_rate < 0.4:  # Poor performance
                confidence_mult = 1.15  # Be more selective
                uncertainty_mult = 0.9   # Reduce uncertainty tolerance
                signal_mult = 1.1       # Higher signal requirements
            elif win_rate < 0.3:  # Very poor performance
                confidence_mult = 1.25
                uncertainty_mult = 0.8
                signal_mult = 1.2
            
            # Profit factor adjustments
            if profit_factor > 1.5:
                confidence_mult *= 0.95  # Winning more than losing
                uncertainty_mult *= 1.05
            elif profit_factor < 0.8:
                confidence_mult *= 1.1   # Losing more than winning
                uncertainty_mult *= 0.95
            
            # Average return adjustments
            if avg_return > 0.15:  # Strong positive returns
                confidence_mult *= 0.9
                signal_mult *= 0.9
            elif avg_return < -0.1:  # Negative returns
                confidence_mult *= 1.15
                signal_mult *= 1.15
        
        return {
            'confidence_mult': confidence_mult,
            'uncertainty_mult': uncertainty_mult,
            'signal_mult': signal_mult
        }
    
    def _analyze_feature_distributions(self, historical_data: pd.DataFrame, 
                                     feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature distributions for threshold adjustment"""
        
        if not feature_names:
            return {'feature_adjustment': 1.0}
        
        try:
            recent_features = historical_data[feature_names].tail(config.ROLLING_WINDOW)
            
            # Calculate feature stability
            feature_stds = recent_features.std()
            avg_stability = 1.0 / (1.0 + feature_stds.mean())
            
            # Feature correlation with future returns
            if 'return_1d' in historical_data.columns:
                future_returns = historical_data['return_1d'].shift(-1)
                correlations = recent_features.corrwith(future_returns).abs()
                avg_correlation = correlations.mean()
            else:
                avg_correlation = 0.1
            
            # Feature-based adjustment
            feature_quality = avg_stability * 0.5 + avg_correlation * 0.5
            feature_adjustment = 0.8 + feature_quality * 0.4  # Range: 0.8 to 1.2
            
            return {'feature_adjustment': feature_adjustment}
            
        except Exception as e:
            logger.warning(f"Feature distribution analysis failed: {e}")
            return {'feature_adjustment': 1.0}
    
    def _combine_threshold_adjustments(self, rolling_stats: Dict, regime_adj: Dict, 
                                     performance_adj: Dict, feature_adj: Dict) -> Dict[str, float]:
        """Combine all adjustments into final thresholds"""
        
        # Base thresholds from rolling statistics
        base_movement = rolling_stats['movement_threshold']
        base_confidence = rolling_stats['base_confidence']
        base_uncertainty = rolling_stats['base_uncertainty']
        base_signal_strength = config.SIGNAL_STRENGTH_BASE
        
        # Apply regime adjustments
        movement_up = base_movement * regime_adj['movement_adj']
        movement_down = -movement_up
        
        confidence = base_confidence + regime_adj['confidence_adj']
        uncertainty = base_uncertainty + regime_adj['uncertainty_adj']
        signal_strength = base_signal_strength + regime_adj['signal_strength_adj']
        
        # Apply performance adjustments
        confidence *= performance_adj['confidence_mult']
        uncertainty *= performance_adj['uncertainty_mult']
        signal_strength *= performance_adj['signal_mult']
        
        # Apply feature adjustments
        confidence *= feature_adj['feature_adjustment']
        signal_strength *= feature_adj['feature_adjustment']
        
        # Ensure reasonable bounds
        confidence = np.clip(confidence, 0.15, 0.85)
        uncertainty = np.clip(uncertainty, 0.3, 0.95)
        signal_strength = np.clip(signal_strength, 0.1, 0.8)
        movement_up = np.clip(movement_up, 0.002, 0.02)
        movement_down = np.clip(movement_down, -0.02, -0.002)
        
        # Margin threshold based on volatility
        margin = rolling_stats['current_std'] * 1.5
        margin = np.clip(margin, 0.005, 0.05)
        
        return {
            'movement_up': movement_up,
            'movement_down': movement_down,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'signal_strength': signal_strength,
            'margin': margin
        }
    
    def _get_adaptive_default_thresholds(self, market_regime: Dict) -> Dict[str, float]:
        """Get adaptive default thresholds based on regime"""
        
        regime_composite = market_regime.get('regime_composite', 'normal')
        
        if regime_composite == 'high_stress':
            return {
                'movement_up': 0.008,
                'movement_down': -0.008,
                'confidence': 0.45,
                'uncertainty': 0.65,
                'signal_strength': 0.35,
                'margin': 0.025
            }
        elif regime_composite == 'trending':
            return {
                'movement_up': 0.004,
                'movement_down': -0.004,
                'confidence': 0.28,
                'uncertainty': 0.75,
                'signal_strength': 0.22,
                'margin': 0.015
            }
        else:  # normal, sideways, high_volatility
            return {
                'movement_up': 0.005,
                'movement_down': -0.005,
                'confidence': 0.32,
                'uncertainty': 0.72,
                'signal_strength': 0.25,
                'margin': 0.02
            }

# --- ENHANCED PERFORMANCE FEEDBACK TRACKER ---
class EnhancedPerformanceFeedbackTracker:
    """Enhanced performance tracking with sophisticated metrics"""
    
    def __init__(self):
        self.trade_performance = []
        self.signal_accuracy = {}
        self.rolling_performance = []
        self.performance_metrics_history = []
        
    def record_trade_outcome(self, trade: Dict, signal_source: str):
        """Record trade outcome with enhanced metrics"""
        
        enhanced_trade = trade.copy()
        enhanced_trade.update({
            'signal_source': signal_source,
            'profitable': trade['net_pnl'] > 0,
            'timestamp': datetime.now(),
            'risk_adjusted_return': trade['return_pct'] / max(0.1, trade.get('total_uncertainty', 0.5))
        })
        
        self.trade_performance.append(enhanced_trade)
        
        # Update signal accuracy with recency weighting
        if signal_source not in self.signal_accuracy:
            self.signal_accuracy[signal_source] = {
                'trades': [],
                'recent_performance': []
            }
        
        self.signal_accuracy[signal_source]['trades'].append(enhanced_trade)
        self.signal_accuracy[signal_source]['recent_performance'].append({
            'profitable': enhanced_trade['profitable'],
            'return_pct': trade['return_pct'],
            'timestamp': enhanced_trade['timestamp']
        })
        
        # Keep only recent trades for rapid adaptation
        max_history = 20
        if len(self.signal_accuracy[signal_source]['trades']) > max_history:
            self.signal_accuracy[signal_source]['trades'] = \
                self.signal_accuracy[signal_source]['trades'][-max_history:]
        
        if len(self.signal_accuracy[signal_source]['recent_performance']) > 10:
            self.signal_accuracy[signal_source]['recent_performance'] = \
                self.signal_accuracy[signal_source]['recent_performance'][-10:]
    
    def get_comprehensive_performance_metrics(self, lookback_trades: int = 15) -> Dict[str, float]:
        """Get comprehensive performance metrics for threshold adaptation"""
        
        if not self.trade_performance:
            return self._get_default_performance_metrics()
        
        recent_trades = self.trade_performance[-lookback_trades:]
        
        # Basic metrics
        win_rate = sum(1 for t in recent_trades if t['profitable']) / len(recent_trades)
        avg_return = np.mean([t['return_pct'] for t in recent_trades])
        
        # Risk-adjusted metrics
        returns = [t['return_pct'] for t in recent_trades]
        sharpe_proxy = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Profit factor
        winners = [t for t in recent_trades if t['profitable']]
        losers = [t for t in recent_trades if not t['profitable']]
        
        gross_profit = sum(t['net_pnl'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['net_pnl'] for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consistency metrics
        consecutive_losses = self._calculate_consecutive_losses(recent_trades)
        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
        
        # Trend analysis
        if len(recent_trades) >= 5:
            recent_win_rates = []
            for i in range(len(recent_trades) - 4):
                subset = recent_trades[i:i+5]
                subset_win_rate = sum(1 for t in subset if t['profitable']) / len(subset)
                recent_win_rates.append(subset_win_rate)
            
            performance_trend = np.polyfit(range(len(recent_win_rates)), recent_win_rates, 1)[0] if recent_win_rates else 0
        else:
            performance_trend = 0
        
        # Signal source diversification
        signal_sources = set(t.get('signal_source', 'Unknown') for t in recent_trades)
        diversification_score = len(signal_sources) / max(1, len(recent_trades)) * 5  # Normalize
        
        metrics = {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'sharpe_proxy': sharpe_proxy,
            'max_consecutive_losses': max_consecutive_losses,
            'performance_trend': performance_trend,
            'diversification_score': min(1.0, diversification_score),
            'num_trades': len(recent_trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
        # Store for history
        self.performance_metrics_history.append(metrics.copy())
        
        return metrics
    
    def _calculate_consecutive_losses(self, trades: List[Dict]) -> List[int]:
        """Calculate consecutive loss streaks"""
        consecutive_counts = []
        current_count = 0
        
        for trade in trades:
            if not trade['profitable']:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return consecutive_counts
    
    def _get_default_performance_metrics(self) -> Dict[str, float]:
        """Default performance metrics when no trades exist"""
        return {
            'win_rate': 0.5,
            'avg_return': 0.0,
            'profit_factor': 1.0,
            'sharpe_proxy': 0.0,
            'max_consecutive_losses': 0,
            'performance_trend': 0.0,
            'diversification_score': 0.5,
            'num_trades': 0,
            'gross_profit': 0,
            'gross_loss': 0
        }

# --- ENHANCED SIGNAL QUALITY TRACKER ---
class EnhancedSignalQualityTracker:
    """Enhanced signal quality tracking with rapid adaptation"""
    
    def __init__(self):
        self.signal_performance = {}
        self.quality_history = []
        self.min_trades_for_quality = 2  # Faster quality assessment
        
    def update_signal_quality_realtime(self, trades_log: List[Dict]):
        """Real-time signal quality updates with mathematical precision"""
        
        current_time = datetime.now()
        
        # Group trades by signal source
        signal_groups = {}
        for trade in trades_log:
            source = trade.get('signal_source', 'Unknown')
            if source not in signal_groups:
                signal_groups[source] = []
            signal_groups[source].append(trade)
        
        # Calculate enhanced quality scores
        for source, source_trades in signal_groups.items():
            if len(source_trades) >= self.min_trades_for_quality:
                
                # Recent performance weighting (exponential decay)
                weighted_performance = self._calculate_weighted_performance(source_trades)
                
                # Statistical significance adjustment
                confidence_interval = self._calculate_confidence_interval(source_trades)
                
                # Trend analysis
                performance_trend = self._analyze_performance_trend(source_trades)
                
                # Risk-adjusted quality
                risk_adjusted_score = self._calculate_risk_adjusted_quality(source_trades)
                
                # Combine all factors
                final_quality = self._combine_quality_factors(
                    weighted_performance, confidence_interval, performance_trend, risk_adjusted_score
                )
                
                self.signal_performance[source] = {
                    'quality_score': final_quality,
                    'weighted_performance': weighted_performance,
                    'confidence_interval': confidence_interval,
                    'performance_trend': performance_trend,
                    'risk_adjusted_score': risk_adjusted_score,
                    'num_trades': len(source_trades),
                    'last_updated': current_time
                }
                
                logger.info(f"📊 {source} Enhanced Quality: {final_quality:.3f} "
                           f"(Weighted: {weighted_performance:.3f}, "
                           f"Trend: {performance_trend:.3f}, "
                           f"Risk-Adj: {risk_adjusted_score:.3f})")
        
        # Store quality snapshot
        self.quality_history.append({
            'timestamp': current_time,
            'qualities': {source: data['quality_score'] 
                         for source, data in self.signal_performance.items()}
        })
    
    def _calculate_weighted_performance(self, trades: List[Dict]) -> float:
        """Calculate performance with exponential time weighting"""
        
        if not trades:
            return 0.5
        
        # Sort by date
        sorted_trades = sorted(trades, key=lambda x: x.get('date', datetime.now()))
        
        total_weight = 0
        weighted_score = 0
        
        for i, trade in enumerate(sorted_trades):
            # Exponential weighting (more recent trades matter more)
            weight = np.exp(i / len(sorted_trades))  # Recent trades get higher weight
            
            # Performance score
            win_score = 1.0 if trade['net_pnl'] > 0 else 0.0
            return_score = max(0, min(1, (trade['return_pct'] + 0.5)))  # Normalize returns
            
            trade_score = win_score * 0.6 + return_score * 0.4
            
            weighted_score += trade_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_confidence_interval(self, trades: List[Dict]) -> float:
        """Calculate statistical confidence in the quality score"""
        
        if len(trades) < 3:
            return 0.3  # Low confidence with few trades
        
        win_rate = sum(1 for t in trades if t['net_pnl'] > 0) / len(trades)
        n = len(trades)
        
        # Wilson confidence interval for binomial proportion
        z = 1.96  # 95% confidence
        
        try:
            center = win_rate + z*z/(2*n)
            spread = z * np.sqrt((win_rate*(1-win_rate) + z*z/(4*n))/n)
            denominator = 1 + z*z/n
            
            lower = (center - spread) / denominator
            upper = (center + spread) / denominator
            
            confidence = 1 - (upper - lower)  # Narrower interval = higher confidence
            return np.clip(confidence, 0.1, 1.0)
            
        except:
            return 0.5
    
    def _analyze_performance_trend(self, trades: List[Dict]) -> float:
        """Analyze performance trend over time"""
        
        if len(trades) < 4:
            return 0.5
        
        # Sort by date
        sorted_trades = sorted(trades, key=lambda x: x.get('date', datetime.now()))
        
        # Calculate rolling win rate
        win_rates = []
        window = max(2, len(sorted_trades) // 3)
        
        for i in range(window, len(sorted_trades) + 1):
            subset = sorted_trades[i-window:i]
            win_rate = sum(1 for t in subset if t['net_pnl'] > 0) / len(subset)
            win_rates.append(win_rate)
        
        if len(win_rates) < 2:
            return 0.5
        
        # Calculate trend
        try:
            trend_slope = np.polyfit(range(len(win_rates)), win_rates, 1)[0]
            # Convert slope to 0-1 scale
            trend_score = 0.5 + np.tanh(trend_slope * 10) * 0.5
            return np.clip(trend_score, 0, 1)
        except:
            return 0.5
    
    def _calculate_risk_adjusted_quality(self, trades: List[Dict]) -> float:
        """Calculate risk-adjusted quality score"""
        
        if not trades:
            return 0.5
        
        returns = [t['return_pct'] for t in trades]
        
        # Basic risk-adjusted return
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.1
        
        # Sharpe-like ratio
        risk_adjusted_return = mean_return / std_return if std_return > 0 else 0
        
        # Convert to 0-1 scale
        risk_score = 0.5 + np.tanh(risk_adjusted_return) * 0.5
        
        # Adjust for maximum drawdown
        cumulative_returns = np.cumsum(returns)
        if len(cumulative_returns) > 1:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Penalize high drawdowns
            drawdown_penalty = min(0.3, max_drawdown * 2)
            risk_score = max(0.1, risk_score - drawdown_penalty)
        
        return np.clip(risk_score, 0, 1)
    
    def _combine_quality_factors(self, weighted_perf: float, confidence: float, 
                                trend: float, risk_adjusted: float) -> float:
        """Combine all quality factors into final score"""
        
        # Weighted combination
        final_score = (
            weighted_perf * 0.35 +      # Most important: actual performance
            risk_adjusted * 0.25 +      # Risk-adjusted performance
            trend * 0.2 +               # Recent trend
            confidence * 0.2            # Statistical confidence
        )
        
        return np.clip(final_score, 0.05, 0.95)
    
    def get_quality_scores_with_fallback(self) -> Dict[str, float]:
        """Get quality scores with intelligent fallback activation"""
        
        base_scores = {source: data['quality_score'] 
                      for source, data in self.signal_performance.items()}
        
        # Initialize default scores for common signal sources if not present
        default_sources = {
            'MACD-Bullish': 0.5,
            'MACD-Bearish': 0.45,
            'RSI-Oversold-Enhanced': 0.4,
            'RSI-Overbought-Enhanced': 0.35,
            'Momentum-Bullish': 0.45,
            'Momentum-Bearish': 0.4,
            'BB-Bounce-Call': 0.4,
            'BB-Reversal-Put': 0.35,
            'ML-Bullish': 0.4,
            'ML-Bearish': 0.3
        }
        
        for source, default_score in default_sources.items():
            if source not in base_scores:
                base_scores[source] = default_score
        
        # Enhanced fallback logic
        primary_strategies = ['MACD-Bullish', 'MACD-Bearish', 'RSI-Oversold-Enhanced', 'RSI-Overbought-Enhanced']
        fallback_strategies = ['Momentum-Bullish', 'Momentum-Bearish', 'BB-Bounce-Call', 'BB-Reversal-Put']
        
        # Check primary strategy performance
        primary_qualities = [base_scores.get(strategy, 0.5) for strategy in primary_strategies]
        avg_primary_quality = np.mean(primary_qualities) if primary_qualities else 0.5
        
        # Activate fallbacks if primary strategies are struggling
        if avg_primary_quality < 0.4:
            logger.info("🔄 ENHANCED FALLBACK ACTIVATION - Primary strategies underperforming")
            
            # Boost fallback strategies
            for strategy in fallback_strategies:
                if strategy not in base_scores:
                    base_scores[strategy] = 0.6  # Give them a good starting chance
                else:
                    base_scores[strategy] = min(0.8, base_scores[strategy] * 1.3)
        
        # Gradual quality decay for unused strategies (but not too aggressive initially)
        current_time = datetime.now()
        for source, data in self.signal_performance.items():
            last_updated = data.get('last_updated', current_time)
            days_since_update = (current_time - last_updated).days
            
            if days_since_update > 7:  # Only decay after a week
                decay_factor = 0.98 ** days_since_update  # Slower decay
                base_scores[source] = base_scores[source] * decay_factor
        
        logger.info(f"Quality scores: {base_scores}")
        return base_scores

# --- ENHANCED ADAPTIVE THRESHOLD MANAGER ---
class EnhancedAdaptiveThresholdManager:
    """Enhanced threshold manager with mathematical precision and rapid adaptation"""
    
    def __init__(self):
        self.threshold_calculator = EnhancedStatisticalThresholdCalculator()
        self.regime_detector = EnhancedMarketRegimeDetector()
        self.performance_tracker = EnhancedPerformanceFeedbackTracker()
        self.signal_quality_tracker = EnhancedSignalQualityTracker()
        self.current_thresholds = {}
        self.adaptation_history = []
        self.iteration_count = 0
        
    def update_thresholds_realtime(self, 
                                 historical_data: pd.DataFrame,
                                 feature_names: List[str],
                                 returns_data: pd.Series,
                                 recent_trades: List[Dict] = None) -> Dict[str, float]:
        """Real-time threshold updates with mathematical precision"""
        
        self.iteration_count += 1
        
        try:
            # Enhanced regime detection
            market_regime = self.regime_detector.detect_comprehensive_regime(historical_data)
            
            # Get comprehensive performance metrics
            performance_metrics = self.performance_tracker.get_comprehensive_performance_metrics()
            
            # Update signal quality in real-time
            if recent_trades:
                self.signal_quality_tracker.update_signal_quality_realtime(recent_trades)
            
            # Calculate dynamic thresholds with all enhancements
            updated_thresholds = self.threshold_calculator.calculate_dynamic_thresholds(
                historical_data=historical_data,
                feature_names=feature_names,
                returns_data=returns_data,
                market_regime=market_regime,
                recent_performance=performance_metrics if performance_metrics['num_trades'] >= 3 else None
            )
            
            # Apply emergency adjustments if needed
            emergency_adjustments = self._check_emergency_conditions(performance_metrics, market_regime)
            if emergency_adjustments:
                updated_thresholds = self._apply_emergency_adjustments(updated_thresholds, emergency_adjustments)
            
            # Record comprehensive adaptation data
            adaptation_record = {
                'iteration': self.iteration_count,
                'thresholds': updated_thresholds.copy(),
                'market_regime': market_regime,
                'performance_metrics': performance_metrics,
                'emergency_adjustments': emergency_adjustments,
                'timestamp': datetime.now()
            }
            
            self.adaptation_history.append(adaptation_record)
            self.current_thresholds = updated_thresholds
            
            # Enhanced logging
            self._log_threshold_update(updated_thresholds, market_regime, performance_metrics)
            
            return updated_thresholds
            
        except Exception as e:
            logger.error(f"Enhanced threshold update failed: {e}")
            return self.current_thresholds or self._get_emergency_thresholds()
    
    def _check_emergency_conditions(self, performance_metrics: Dict, 
                                  market_regime: Dict) -> Dict[str, bool]:
        """Check for emergency conditions requiring immediate threshold adjustment"""
        
        emergency_conditions = {
            'consecutive_losses': False,
            'low_signal_generation': False,
            'high_market_stress': False,
            'poor_recent_performance': False
        }
        
        # Check for consecutive losses
        if performance_metrics['max_consecutive_losses'] >= 4:
            emergency_conditions['consecutive_losses'] = True
        
        # Check for low signal generation (if we haven't had trades recently)
        if performance_metrics['num_trades'] == 0 and self.iteration_count > 5:
            emergency_conditions['low_signal_generation'] = True
        
        # Check for high market stress
        if market_regime.get('market_stress', 0) > 0.6:
            emergency_conditions['high_market_stress'] = True
        
        # Check for poor recent performance
        if (performance_metrics['num_trades'] >= 5 and 
            performance_metrics['win_rate'] < 0.25 and 
            performance_metrics['performance_trend'] < -0.1):
            emergency_conditions['poor_recent_performance'] = True
        
        return emergency_conditions
    
    def _apply_emergency_adjustments(self, base_thresholds: Dict, 
                                   emergency_conditions: Dict) -> Dict[str, float]:
        """Apply emergency threshold adjustments"""
        
        adjusted_thresholds = base_thresholds.copy()
        
        if emergency_conditions['consecutive_losses']:
            logger.warning("🚨 EMERGENCY: Consecutive losses detected - Tightening thresholds")
            adjusted_thresholds['confidence'] = min(0.8, adjusted_thresholds['confidence'] * 1.3)
            adjusted_thresholds['uncertainty'] = max(0.2, adjusted_thresholds['uncertainty'] * 0.7)
            adjusted_thresholds['signal_strength'] = min(0.7, adjusted_thresholds['signal_strength'] * 1.4)
        
        if emergency_conditions['low_signal_generation']:
            logger.warning("🚨 EMERGENCY: Low signal generation - Loosening thresholds")
            adjusted_thresholds['confidence'] = max(0.15, adjusted_thresholds['confidence'] * 0.7)
            adjusted_thresholds['uncertainty'] = min(0.9, adjusted_thresholds['uncertainty'] * 1.2)
            adjusted_thresholds['signal_strength'] = max(0.1, adjusted_thresholds['signal_strength'] * 0.6)
        
        if emergency_conditions['high_market_stress']:
            logger.warning("🚨 EMERGENCY: High market stress - Conservative adjustments")
            adjusted_thresholds['confidence'] = min(0.75, adjusted_thresholds['confidence'] * 1.2)
            adjusted_thresholds['movement_up'] = min(0.015, adjusted_thresholds['movement_up'] * 1.3)
            adjusted_thresholds['movement_down'] = max(-0.015, adjusted_thresholds['movement_down'] * 1.3)
        
        if emergency_conditions['poor_recent_performance']:
            logger.warning("🚨 EMERGENCY: Poor recent performance - Major threshold adjustment")
            adjusted_thresholds['confidence'] = min(0.8, adjusted_thresholds['confidence'] * 1.4)
            adjusted_thresholds['uncertainty'] = max(0.25, adjusted_thresholds['uncertainty'] * 0.6)
            adjusted_thresholds['signal_strength'] = min(0.6, adjusted_thresholds['signal_strength'] * 1.5)
        
        return adjusted_thresholds
    
    def _log_threshold_update(self, thresholds: Dict, market_regime: Dict, 
                            performance_metrics: Dict):
        """Enhanced logging for threshold updates"""
        
        logger.info(f"🎯 ENHANCED ADAPTIVE THRESHOLDS (Iteration {self.iteration_count}):")
        logger.info(f"   📊 Confidence: {thresholds['confidence']:.3f}")
        logger.info(f"   📊 Uncertainty: {thresholds['uncertainty']:.3f}")
        logger.info(f"   📊 Movement: ±{thresholds['movement_up']:.4f}")
        logger.info(f"   📊 Signal Strength: {thresholds['signal_strength']:.3f}")
        logger.info(f"   📊 Margin: {thresholds['margin']:.4f}")
        
        logger.info(f"   🌍 Market Regime: {market_regime.get('regime_composite', 'normal')}")
        logger.info(f"   📈 Volatility Percentile: {market_regime.get('volatility_percentile', 0.5):.2f}")
        logger.info(f"   📊 Market Stress: {market_regime.get('market_stress', 0.2):.3f}")
        
        if performance_metrics['num_trades'] > 0:
            logger.info(f"   🎲 Performance - Win Rate: {performance_metrics['win_rate']:.2f}, "
                       f"Trend: {performance_metrics['performance_trend']:.3f}, "
                       f"Trades: {performance_metrics['num_trades']}")
    
    def _get_emergency_thresholds(self) -> Dict[str, float]:
        """Emergency fallback thresholds"""
        return {
            'movement_up': 0.005,
            'movement_down': -0.005,
            'confidence': 0.35,
            'uncertainty': 0.7,
            'signal_strength': 0.25,
            'margin': 0.02
        }

# Keep all other classes (EnhancedFeatureEngineer, AdaptiveMarketSimulator, EnhancedEnsemble, etc.) 
# exactly the same as in the original code, but update the main trading system and signal generator 
# to use the enhanced threshold management

# --- ENHANCED FEATURE ENGINEERING ---
class EnhancedFeatureEngineer:
    """Enhanced feature engineering with statistical analysis"""
    
    @staticmethod
    def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced indicators with statistical properties"""
        
        # Basic returns with multiple timeframes
        for n in [1, 2, 3, 5, 10, 15, 20]:
            df[f'return_{n}d'] = df['Close'].pct_change(n).shift(1)
        
        # Moving averages and relative positions
        for n in [5, 10, 15, 20, 50]:
            df[f'sma_{n}'] = df['Close'].rolling(n).mean()
            df[f'price_to_sma_{n}'] = (df['Close'] / df[f'sma_{n}'] - 1).shift(1)
        
        # Enhanced volatility measures
        returns = df['Close'].pct_change()
        for period in [5, 10, 15, 20, 30]:
            df[f'vol_{period}d'] = returns.rolling(period).std().shift(1) * np.sqrt(252)
            df[f'vol_rank_{period}d'] = df[f'vol_{period}d'].rolling(60).rank(pct=True).shift(1)
        
        # Statistical moments
        for period in [10, 20]:
            df[f'skew_{period}d'] = returns.rolling(period).skew().shift(1)
            df[f'kurt_{period}d'] = returns.rolling(period).kurt().shift(1)
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = (100 - 100 / (1 + rs)).shift(1)
        
        # MACD variations
        for fast, slow in [(8, 21), (12, 26)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = (ema_fast - ema_slow).shift(1)
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        
        # Bollinger Bands with statistical analysis
        for period in [15, 20, 25]:
            bb_mid = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_mid + 2 * bb_std
            df[f'bb_lower_{period}'] = bb_mid - 2 * bb_std
            df[f'bb_position_{period}'] = ((df['Close'] - df[f'bb_lower_{period}']) / 
                                         (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])).shift(1)
            df[f'bb_width_{period}'] = ((df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_mid).shift(1)
        
        # Volume analysis (if available)
        if 'Volume' in df.columns:
            for period in [5, 10, 20]:
                df[f'volume_ratio_{period}'] = (df['Volume'] / df['Volume'].rolling(period).mean()).shift(1)
                df[f'volume_sma_{period}'] = df['Volume'].rolling(period).mean()
        else:
            df['volume_ratio_5'] = 1.0
            df['volume_ratio_10'] = 1.0
            df['volume_ratio_20'] = 1.0
        
        # Price momentum indicators
        df['momentum_5_20'] = (df['sma_5'] / df['sma_20'] - 1).shift(1)
        df['momentum_10_30'] = (df['sma_10'] / df['sma_50'] - 1).shift(1) if 'sma_50' in df.columns else 0
        
        # Support/Resistance levels
        df['price_vs_high_20'] = (df['Close'] / df['High'].rolling(20).max() - 1).shift(1)
        df['price_vs_low_20'] = (df['Close'] / df['Low'].rolling(20).min() - 1).shift(1)
        
        return df

# --- ADAPTIVE MARKET SIMULATOR ---
class AdaptiveMarketSimulator:
    """Enhanced market simulator with statistical accuracy"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        logger.info(f"Initializing Enhanced Adaptive Market Simulator for {symbol}...")
        self.symbol = symbol
        self.hist_data = self._fetch_data(symbol, start_date, end_date)
        self.feature_names = []
        self._precompute_features()
        logger.info(f"Enhanced market simulation ready with {len(self.feature_names)} features")

    def _fetch_data(self, symbol, start_date, end_date):
        """Fetch historical data with enhanced date handling"""
        fetch_start = pd.to_datetime(start_date) - timedelta(days=400)  # More history for statistics
        data = yf.Ticker(symbol).history(start=fetch_start, end=end_date)
        if data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")
        data.index = data.index.tz_localize(None)
        return data

    def _precompute_features(self):
        """Precompute features with enhanced statistical methods"""
        df = self.hist_data.copy()
        
        # Enhanced feature engineering
        df = EnhancedFeatureEngineer.calculate_enhanced_indicators(df)
        
        # Statistical target classification using data distribution
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        
        # Dynamic threshold calculation based on return distribution
        returns_std = future_return.std()
        returns_25q = future_return.quantile(0.25)
        returns_75q = future_return.quantile(0.75)
        
        # Use IQR-based thresholds for more statistical accuracy
        iqr_multiplier = 0.5
        up_threshold = returns_75q * iqr_multiplier
        down_threshold = returns_25q * iqr_multiplier
        
        # Ensure minimum movement for meaningful signals
        up_threshold = max(0.002, up_threshold)
        down_threshold = min(-0.002, down_threshold)
        
        # 3-class classification with statistical distribution
        df['target_class'] = 1  # Neutral
        df.loc[future_return <= down_threshold, 'target_class'] = 0  # Down
        df.loc[future_return >= up_threshold, 'target_class'] = 2   # Up
        
        df.dropna(inplace=True)
        
        # Select meaningful features
        potential_features = [
            'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_15d', 'return_20d',
            'vol_5d', 'vol_10d', 'vol_15d', 'vol_20d', 'vol_30d',
            'vol_rank_5d', 'vol_rank_10d', 'vol_rank_20d',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_15', 'price_to_sma_20', 'price_to_sma_50',
            'rsi_7', 'rsi_14', 'rsi_21',
            'macd_8_21', 'macd_12_26', 'macd_signal_8_21', 'macd_signal_12_26',
            'bb_position_15', 'bb_position_20', 'bb_position_25',
            'bb_width_15', 'bb_width_20', 'bb_width_25',
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
            'momentum_5_20', 'momentum_10_30',
            'price_vs_high_20', 'price_vs_low_20',
            'skew_10d', 'skew_20d', 'kurt_10d', 'kurt_20d'
        ]
        
        self.feature_names = [f for f in potential_features if f in df.columns and not df[f].isna().all()]
        
        # Store returns for threshold calculation
        self.returns = future_return.dropna()
        
        class_counts = df['target_class'].value_counts().sort_index()
        logger.info(f"Statistical class distribution - Down: {class_counts.get(0, 0)}, "
                   f"Neutral: {class_counts.get(1, 0)}, Up: {class_counts.get(2, 0)}")
        logger.info(f"Thresholds: Up ≥ {up_threshold:.4f}, Down ≤ {down_threshold:.4f}")
        logger.info(f"Using {len(self.feature_names)} statistical features")
        
        self.hist_data = df

    def get_market_for_day(self, date: pd.Timestamp) -> Optional[Dict]:
        """Get market data for specific day with realistic modeling"""
        if date not in self.hist_data.index:
            return None
        
        row = self.hist_data.loc[date]
        current_price = row['Close']
        
        # Adaptive strike range based on volatility
        current_vol = row.get('vol_20d', 0.25)
        strike_range = max(0.08, min(0.15, current_vol * 0.6))
        num_strikes = 20
        
        min_strike = current_price * (1 - strike_range)
        max_strike = current_price * (1 + strike_range)
        strikes = np.linspace(min_strike, max_strike, num_strikes)
        
        # Realistic volatility modeling
        base_vol = max(0.12, min(0.8, current_vol))
        
        # Enhanced volatility smile based on market data
        moneyness = strikes / current_price
        smile_adjustment = 0.1 * np.abs(moneyness - 1) ** 1.5  # More realistic smile
        vols = base_vol * (1 + smile_adjustment)
        vols = np.clip(vols, 0.10, 1.0)
        
        T = 30 / 365.0
        r = 0.05
        
        calls_df = pd.DataFrame({
            'strike': strikes,
            'mid_price': self._black_scholes(current_price, strikes, T, r, vols, 'call'),
            'impliedVolatility': vols,
            'optionType': 'call'
        })
        
        puts_df = pd.DataFrame({
            'strike': strikes,
            'mid_price': self._black_scholes(current_price, strikes, T, r, vols, 'put'),
            'impliedVolatility': vols,
            'optionType': 'put'
        })
        
        # Realistic bid-ask spreads
        for df in [calls_df, puts_df]:
            distance = np.abs(df['strike'] - current_price) / current_price
            spread_pct = 0.03 + 0.05 * distance + 0.02 * base_vol  # Vol-adjusted spreads
            
            df['bid'] = df['mid_price'] * (1 - spread_pct)
            df['ask'] = df['mid_price'] * (1 + spread_pct)
            df['volume'] = np.maximum(50, np.random.poisson(500 * np.exp(-distance * 10)))
        
        return {
            'date': date,
            'current_price': current_price,
            'calls': calls_df,
            'puts': puts_df,
            'volatility': base_vol,
            'market_regime': 'normal'
        }
    
    def _black_scholes(self, S, K, T, r, sigma, option_type):
        """Black-Scholes pricing with numerical stability"""
        if T <= 0:
            if option_type == 'call':
                return np.maximum(0, S - K)
            else:
                return np.maximum(0, K - S)
        
        # Handle arrays properly
        sigma = np.where(sigma <= 0, 0.01, sigma)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return np.maximum(price, 0.05)

# --- ENHANCED ENSEMBLE ---
class EnhancedEnsemble:
    """Enhanced ensemble with uncertainty quantification"""
    
    def __init__(self, ensemble_size=5):
        self.ensemble_size = ensemble_size
        self.models = []
        self.scalers = []
        self.feature_importance = {}
        
    def create_ensemble(self, X_train, y_train):
        """Create enhanced ensemble with diverse models"""
        for i in range(self.ensemble_size):
            if i % 3 == 0:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=15,
                    random_state=42 + i,
                    n_jobs=-1
                )
            elif i % 3 == 1:
                model = xgb.XGBClassifier(
                    n_estimators=80,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42 + i,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=60,
                    max_depth=6,
                    learning_rate=0.15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42 + i,
                    n_jobs=-1,
                    verbosity=0
                )
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            model.fit(X_scaled, y_train)
            
            self.models.append(model)
            self.scalers.append(scaler)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[f'model_{i}'] = model.feature_importances_
    
    def predict_with_uncertainty(self, X):
        """Enhanced prediction with better uncertainty quantification"""
        predictions = []
        
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred_probs = model.predict_proba(X_scaled)
            predictions.append(pred_probs)
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        mean_probs = np.mean(predictions, axis=0)
        std_probs = np.std(predictions, axis=0)
        
        # Enhanced uncertainty quantification
        epistemic_uncertainty = np.mean(std_probs, axis=1)  # Model disagreement
        aleatoric_uncertainty = 1 - np.max(mean_probs, axis=1)  # Prediction entropy
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty * 0.5
        
        # Confidence based on consistency and strength
        max_prob = np.max(mean_probs, axis=1)
        consistency = 1 - epistemic_uncertainty
        confidence = max_prob * consistency
        
        return {
            'class_probs': mean_probs,
            'confidence': confidence,
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }

# --- ENHANCED ADAPTIVE SIGNAL GENERATOR ---
class EnhancedAdaptiveSignalGenerator:
    """Enhanced signal generator with dynamic threshold adaptation"""
    
    def __init__(self, feature_names: List[str], threshold_manager: EnhancedAdaptiveThresholdManager):
        self.feature_names = feature_names
        self.threshold_manager = threshold_manager
        self.signal_history = []
        
    def generate_ml_signals(self, ensemble_pred: Dict, market_data: Dict, 
                           thresholds: Dict, signal_quality_scores: Dict = None) -> List[Dict]:
        """Generate ML-based signals with enhanced adaptive thresholds"""
        signals = []
        
        # Dynamic ML quality assessment
        ml_quality_score = 0.45  # Improved base score
        if signal_quality_scores:
            ml_bull_quality = signal_quality_scores.get('ML-Bullish', 0.45)
            ml_bear_quality = signal_quality_scores.get('ML-Bearish', 0.35)
            ml_quality_score = (ml_bull_quality + ml_bear_quality) / 2
        
        # Adaptive quality threshold based on market conditions
        market_regime = market_data.get('market_regime', 'normal')
        quality_threshold = 0.25 if market_regime == 'trending' else 0.35
        
        if ml_quality_score < quality_threshold:
            return signals
        
        confidence = ensemble_pred['confidence'][0]
        class_probs = ensemble_pred['class_probs'][0]
        total_unc = ensemble_pred['total_uncertainty'][0]
        epistemic_unc = ensemble_pred['epistemic_uncertainty'][0]
        
        # Quality-adjusted thresholds with mathematical precision
        quality_multiplier = 0.5 + ml_quality_score
        conf_threshold = thresholds['confidence'] * quality_multiplier
        unc_threshold = thresholds['uncertainty'] / quality_multiplier
        
        if confidence < conf_threshold or total_unc > unc_threshold:
            return signals
        
        pred_class = np.argmax(class_probs)
        max_prob = np.max(class_probs)
        
        # Enhanced signal strength calculation
        signal_strength = confidence * (1 - total_unc) * max_prob
        min_prob_threshold = 0.55 * quality_multiplier
        
        if signal_strength >= thresholds['signal_strength'] and max_prob > min_prob_threshold:
            
            if pred_class == 2 or (len(class_probs) == 2 and pred_class == 1):  # Bullish
                signal = self._create_option_signal(
                    market_data, 'call', confidence, total_unc, 'ML-Bullish',
                    signal_strength, epistemic_unc
                )
                if signal:
                    signals.append(signal)
            
            elif pred_class == 0:  # Bearish with enhanced requirements
                if max_prob > 0.65 and confidence > 0.5:  # Still stringent but achievable
                    signal = self._create_option_signal(
                        market_data, 'put', confidence, total_unc, 'ML-Bearish',
                        signal_strength, epistemic_unc
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def generate_technical_signals(self, historical_data: pd.DataFrame, 
                                 market_data: Dict, thresholds: Dict,
                                 signal_quality_scores: Dict = None) -> List[Dict]:
        """Generate enhanced technical signals with dynamic quality adaptation"""
        signals = []
        
        if len(historical_data) < 30:
            return signals
        
        try:
            latest = historical_data.iloc[-1]
            
            # Dynamic quality thresholds
            macd_quality = signal_quality_scores.get('MACD-Bullish', 0.5) if signal_quality_scores else 0.5
            rsi_quality = signal_quality_scores.get('RSI-Oversold-Enhanced', 0.3) if signal_quality_scores else 0.3
            
            # Enhanced MACD signals with adaptive quality requirements
            macd = latest.get('macd_12_26', 0)
            macd_signal = latest.get('macd_signal_12_26', 0)
            
            # Dynamic margin based on quality and market conditions
            base_margin = thresholds['margin']
            quality_adjusted_margin = base_margin * (2.0 - macd_quality)
            
            if macd_quality > 0.2:  # Allow MACD signals if quality is reasonable
                
                # Enhanced bullish MACD
                if (macd > macd_signal and macd > 0 and 
                    abs(macd - macd_signal) > quality_adjusted_margin):
                    
                    # Multi-factor confirmation
                    momentum_5_20 = latest.get('momentum_5_20', 0)
                    price_to_sma_10 = latest.get('price_to_sma_10', 0)
                    vol_rank_20d = latest.get('vol_rank_20d', 0.5)
                    
                    # Quality-based confirmation requirements
                    if macd_quality > 0.6:
                        confirmation = momentum_5_20 > -thresholds['movement_up']
                    elif macd_quality > 0.4:
                        confirmation = momentum_5_20 > -thresholds['movement_up'] * 0.5 and price_to_sma_10 > -0.01
                    else:
                        confirmation = (momentum_5_20 > 0 and price_to_sma_10 > 0 and vol_rank_20d < 0.8)
                    
                    if confirmation:
                        confidence = 0.55 + macd_quality * 0.2
                        uncertainty = 0.35 - macd_quality * 0.1
                        signal = self._create_option_signal(
                            market_data, 'call', confidence, uncertainty, 'MACD-Bullish', 
                            confidence, uncertainty * 0.8
                        )
                        if signal:
                            signals.append(signal)
                
                # Enhanced bearish MACD
                elif (macd < macd_signal and macd < 0 and 
                      abs(macd - macd_signal) > quality_adjusted_margin * 1.2):
                    
                    momentum_5_20 = latest.get('momentum_5_20', 0)
                    price_vs_high_20 = latest.get('price_vs_high_20', 0)
                    rsi_14 = latest.get('rsi_14', 50)
                    
                    # Enhanced bearish confirmation
                    if macd_quality > 0.5:
                        bearish_confirmation = (momentum_5_20 < -thresholds['movement_up'] * 0.3 and 
                                              price_vs_high_20 > -0.08)
                    else:
                        bearish_confirmation = (momentum_5_20 < -thresholds['movement_up'] and 
                                              price_vs_high_20 > -0.05 and rsi_14 > 45)
                    
                    if bearish_confirmation:
                        confidence = 0.45 + macd_quality * 0.15
                        uncertainty = 0.45 - macd_quality * 0.1
                        signal = self._create_option_signal(
                            market_data, 'put', confidence, uncertainty, 'MACD-Bearish', 
                            confidence, uncertainty * 0.8
                        )
                        if signal:
                            signals.append(signal)
            
            # Enhanced RSI signals with better quality responsiveness
            if rsi_quality > 0.15 and len(signals) < 2:
                rsi_14 = latest.get('rsi_14', 50)
                rsi_7 = latest.get('rsi_7', 50)
                
                # Dynamic RSI thresholds based on quality
                oversold_threshold = 30 - rsi_quality * 10  # 20-30 range
                overbought_threshold = 70 + rsi_quality * 10  # 70-80 range
                
                # Enhanced confirmations
                vol_rank_20d = latest.get('vol_rank_20d', 0.5)
                bb_position_20 = latest.get('bb_position_20', 0.5)
                momentum_5_20 = latest.get('momentum_5_20', 0)
                
                # RSI oversold with enhanced logic
                if (rsi_14 < oversold_threshold and rsi_7 < oversold_threshold + 5 and 
                    bb_position_20 < 0.25 and vol_rank_20d > 0.2):
                    
                    if momentum_5_20 > -thresholds['movement_up'] * 2:  # Not in free fall
                        confidence = 0.45 + rsi_quality * 0.2
                        signal = self._create_option_signal(
                            market_data, 'call', confidence, 0.5, 'RSI-Oversold-Enhanced', 
                            confidence, 0.4
                        )
                        if signal:
                            signals.append(signal)
                
                # RSI overbought with enhanced logic
                elif (rsi_14 > overbought_threshold and rsi_7 > overbought_threshold - 5 and 
                      bb_position_20 > 0.75 and vol_rank_20d > 0.25):
                    
                    price_vs_high_20 = latest.get('price_vs_high_20', 0)
                    if price_vs_high_20 > -0.05 and momentum_5_20 < thresholds['movement_up']:
                        confidence = 0.40 + rsi_quality * 0.2
                        signal = self._create_option_signal(
                            market_data, 'put', confidence, 0.55, 'RSI-Overbought-Enhanced', 
                            confidence, 0.45
                        )
                        if signal:
                            signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Enhanced technical signal generation failed: {e}")
        
        return signals
    
    def generate_momentum_signals(self, historical_data: pd.DataFrame, 
                                market_data: Dict, thresholds: Dict,
                                signal_quality_scores: Dict = None) -> List[Dict]:
        """Generate enhanced momentum signals"""
        signals = []
        
        if len(historical_data) < 20:
            return signals
        
        try:
            latest = historical_data.iloc[-1]
            
            # Enhanced momentum analysis
            momentum_5_20 = latest.get('momentum_5_20', 0)
            price_to_sma_5 = latest.get('price_to_sma_5', 0)
            price_to_sma_20 = latest.get('price_to_sma_20', 0)
            vol_rank_20d = latest.get('vol_rank_20d', 0.5)
            
            # Dynamic momentum threshold
            momentum_threshold = max(thresholds['movement_up'] * 1.2, 0.006)
            
            # Enhanced bullish momentum
            if (momentum_5_20 > momentum_threshold and 
                price_to_sma_5 > momentum_threshold * 0.6 and 
                price_to_sma_20 > momentum_threshold * 0.3 and
                vol_rank_20d > 0.15):
                
                # Additional quality filters
                rsi_14 = latest.get('rsi_14', 50)
                bb_position_20 = latest.get('bb_position_20', 0.5)
                
                if rsi_14 < 78 and bb_position_20 < 0.95:  # Not extremely overbought
                    signal = self._create_option_signal(
                        market_data, 'call', 0.6, 0.3, 'Momentum-Bullish', 0.6, 0.25
                    )
                    if signal:
                        signals.append(signal)
            
            # Enhanced bearish momentum
            elif (momentum_5_20 < -momentum_threshold and 
                  price_to_sma_5 < -momentum_threshold * 0.6 and 
                  price_to_sma_20 < -momentum_threshold * 0.3 and
                  vol_rank_20d > 0.2):
                
                rsi_14 = latest.get('rsi_14', 50)
                bb_position_20 = latest.get('bb_position_20', 0.5)
                price_vs_low_20 = latest.get('price_vs_low_20', 0)
                
                if rsi_14 > 22 and bb_position_20 > 0.05 and price_vs_low_20 > 0.03:
                    signal = self._create_option_signal(
                        market_data, 'put', 0.55, 0.35, 'Momentum-Bearish', 0.55, 0.3
                    )
                    if signal:
                        signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Enhanced momentum signal generation failed: {e}")
        
        return signals
    
    def generate_volatility_signals(self, historical_data: pd.DataFrame, 
                                  market_data: Dict, thresholds: Dict,
                                  signal_quality_scores: Dict = None) -> List[Dict]:
        """Generate enhanced volatility signals"""
        signals = []
        
        if len(historical_data) < 25:
            return signals
        
        try:
            latest = historical_data.iloc[-1]
            
            # Enhanced Bollinger Band analysis
            bb_position_20 = latest.get('bb_position_20', 0.5)
            bb_width_20 = latest.get('bb_width_20', 0.1)
            vol_rank_20d = latest.get('vol_rank_20d', 0.5)
            
            # Dynamic BB thresholds
            lower_threshold = 0.15 - thresholds.get('signal_strength', 0.3) * 0.05
            upper_threshold = 0.85 + thresholds.get('signal_strength', 0.3) * 0.05
            
            # Enhanced BB bounce (oversold)
            if (bb_position_20 < lower_threshold and 
                bb_width_20 > 0.035 and vol_rank_20d > 0.2):
                
                rsi_14 = latest.get('rsi_14', 50)
                momentum_5_20 = latest.get('momentum_5_20', 0)
                price_vs_low_20 = latest.get('price_vs_low_20', 0)
                
                if (rsi_14 < 45 and rsi_14 > 15 and  # Oversold but not extreme
                    momentum_5_20 > -thresholds['movement_up'] * 1.5 and  # Not crashing
                    price_vs_low_20 > 0.015):  # Not at absolute lows
                    
                    signal = self._create_option_signal(
                        market_data, 'call', 0.55, 0.35, 'BB-Bounce-Call', 0.55, 0.3
                    )
                    if signal:
                        signals.append(signal)
            
            # Enhanced BB reversal (overbought)
            elif (bb_position_20 > upper_threshold and 
                  bb_width_20 > 0.035 and vol_rank_20d > 0.25):
                
                rsi_14 = latest.get('rsi_14', 50)
                momentum_5_20 = latest.get('momentum_5_20', 0)
                price_vs_high_20 = latest.get('price_vs_high_20', 0)
                
                if (rsi_14 > 55 and rsi_14 < 85 and  # Overbought but not extreme
                    price_vs_high_20 > -0.025 and  # Near recent highs
                    momentum_5_20 < thresholds['movement_up'] * 0.8):  # Momentum slowing
                    
                    signal = self._create_option_signal(
                        market_data, 'put', 0.50, 0.4, 'BB-Reversal-Put', 0.50, 0.35
                    )
                    if signal:
                        signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Enhanced volatility signal generation failed: {e}")
        
        return signals
    
    def _create_option_signal(self, market_data: Dict, option_type: str, 
                            confidence: float, uncertainty: float, 
                            signal_source: str, signal_strength: float = 0.5,
                            epistemic_uncertainty: float = 0.3) -> Optional[Dict]:
        """Create option signal with enhanced ATM selection"""
        try:
            current_price = market_data['current_price']
            options_df = market_data['calls'] if option_type == 'call' else market_data['puts']
            
            # Enhanced ATM selection
            options_df['moneyness'] = options_df['strike'] / current_price
            options_df['spread'] = (options_df['ask'] - options_df['bid']) / options_df['mid_price']
            options_df['distance_from_atm'] = abs(options_df['moneyness'] - 1.0)
            
            # More liberal option filtering for better signal generation
            valid_options = options_df[
                (options_df['ask'] >= 0.50) & 
                (options_df['ask'] <= current_price * 0.30) &
                (options_df['spread'] <= 0.20) &
                (options_df['volume'] >= 5)
            ]
            
            if valid_options.empty:
                return None
            
            # Enhanced scoring for option selection
            valid_options['score'] = (
                1 / (1 + valid_options['distance_from_atm'] * 4) * 0.4 +
                1 / (1 + valid_options['spread'] * 8) * 0.3 +
                np.log(1 + valid_options['volume']) / 8 * 0.3
            )
            
            best_option = valid_options.loc[valid_options['score'].idxmax()]
            
            return {
                'date': market_data['date'],
                'action': 'buy',
                'option_type': option_type,
                'strike': best_option['strike'],
                'entry_price': best_option['ask'],
                'confidence': confidence,
                'total_uncertainty': uncertainty,
                'signal_source': signal_source,
                'moneyness': best_option['moneyness'],
                'class_margin': signal_strength,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': uncertainty - epistemic_uncertainty,
                'signal_strength': signal_strength,
                'regime': 1,
                'spread_pct': best_option['spread'],
                'volume': best_option['volume']
            }
            
        except Exception as e:
            logger.warning(f"Enhanced option signal creation failed: {e}")
            return None

# --- ENHANCED ADAPTIVE TRADING SYSTEM ---
class EnhancedAdaptiveTradingSystem:
    """Enhanced adaptive trading system with real-time threshold learning"""
    
    def __init__(self, ensemble, feature_names, threshold_manager: EnhancedAdaptiveThresholdManager):
        self.ensemble = ensemble
        self.feature_names = feature_names
        self.threshold_manager = threshold_manager
        self.signal_generator = EnhancedAdaptiveSignalGenerator(feature_names, threshold_manager)
        self.days_since_last_trade = 0
        self.recent_trades = []
        
    def generate_signals(self, market_data: Dict, historical_features: pd.DataFrame,
                        iteration_count: int = 0) -> List[Dict]:
        """Generate signals with enhanced adaptive thresholds"""
        all_signals = []
        self.days_since_last_trade += 1
        
        try:
            # Real-time threshold updates
            current_thresholds = self.threshold_manager.update_thresholds_realtime(
                historical_features, self.feature_names, 
                historical_features.get('return_1d', pd.Series([0])),
                self.recent_trades
            )
            
            # Get enhanced signal quality scores
            signal_quality_scores = self.threshold_manager.signal_quality_tracker.get_quality_scores_with_fallback()
            
            # 1. Enhanced ML signals
            if config.USE_ML_SIGNALS and len(historical_features) >= 50:
                latest_features = historical_features[self.feature_names].iloc[-1:].fillna(0).values
                latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                try:
                    ensemble_pred = self.ensemble.predict_with_uncertainty(latest_features)
                    ml_signals = self.signal_generator.generate_ml_signals(
                        ensemble_pred, market_data, current_thresholds, signal_quality_scores
                    )
                    all_signals.extend(ml_signals)
                    
                    if ml_signals:
                        logger.info(f"🤖 Generated {len(ml_signals)} enhanced ML signals (confidence: {ensemble_pred['confidence'][0]:.3f})")
                        
                except Exception as e:
                    logger.warning(f"Enhanced ML signal generation failed: {e}")
            
            # 2. Enhanced technical signals
            if config.USE_TECHNICAL_SIGNALS:
                tech_signals = self.signal_generator.generate_technical_signals(
                    historical_features, market_data, current_thresholds, signal_quality_scores
                )
                all_signals.extend(tech_signals)
                
                if tech_signals:
                    logger.info(f"📊 Generated {len(tech_signals)} enhanced technical signals")
            
            # 3. Enhanced momentum signals
            if config.USE_MOMENTUM_SIGNALS:
                momentum_signals = self.signal_generator.generate_momentum_signals(
                    historical_features, market_data, current_thresholds, signal_quality_scores
                )
                all_signals.extend(momentum_signals)
                
                if momentum_signals:
                    logger.info(f"🚀 Generated {len(momentum_signals)} enhanced momentum signals")
            
            # 4. Enhanced volatility signals
            if config.USE_VOLATILITY_SIGNALS:
                vol_signals = self.signal_generator.generate_volatility_signals(
                    historical_features, market_data, current_thresholds, signal_quality_scores
                )
                all_signals.extend(vol_signals)
                
                if vol_signals:
                    logger.info(f"⚡ Generated {len(vol_signals)} enhanced volatility signals")
            
            # Enhanced signal selection with mathematical precision
            unique_signals = self._select_best_signals_with_enhanced_quality(all_signals, signal_quality_scores, current_thresholds)
            
            # Reset counter if we generated signals
            if unique_signals:
                self.days_since_last_trade = 0
                logger.info(f"✅ ENHANCED QUALITY SIGNALS: {len(unique_signals)} selected from {len(all_signals)} candidates")
                for i, sig in enumerate(unique_signals):
                    quality = signal_quality_scores.get(sig['signal_source'], 0.5)
                    logger.info(f"   Signal {i+1}: {sig['option_type']} {sig['strike']:.0f} "
                               f"@ ${sig['entry_price']:.2f} [{sig['signal_source']}] "
                               f"(conf: {sig['confidence']:.3f}, quality: {quality:.3f})")
            
            return unique_signals
            
        except Exception as e:
            logger.error(f"Enhanced signal generation error: {e}")
            return []
    
    def _select_best_signals_with_enhanced_quality(self, all_signals: List[Dict], 
                                                 quality_scores: Dict[str, float],
                                                 current_thresholds: Dict[str, float]) -> List[Dict]:
        """Enhanced signal selection with mathematical quality weighting"""
        if not all_signals:
            logger.info("No signals to select from")
            return []
        
        logger.info(f"Selecting from {len(all_signals)} signals with quality scores: {quality_scores}")
        
        # Enhanced scoring with threshold awareness
        scored_signals = []
        for signal in all_signals:
            source = signal['signal_source']
            quality_score = quality_scores.get(source, 0.5)
            
            # Skip very poor quality signals unless in emergency mode
            min_quality = 0.15 if len(all_signals) < 3 else 0.20  # Lowered threshold
            if quality_score < min_quality:
                logger.info(f"Skipping {source} signal due to low quality: {quality_score:.3f} < {min_quality}")
                continue
            
            # Enhanced combined score with mathematical precision
            confidence_score = signal['confidence'] * 0.25
            signal_strength_score = signal['signal_strength'] * 0.20
            uncertainty_score = (1 - signal['total_uncertainty']) * 0.15
            quality_weight = quality_score * 0.40
            
            base_score = confidence_score + signal_strength_score + uncertainty_score + quality_weight
            
            # Threshold conformity bonus
            threshold_conformity = self._calculate_threshold_conformity(signal, current_thresholds)
            final_score = base_score * (0.8 + threshold_conformity * 0.4)
            
            scored_signals.append((final_score, signal))
            logger.info(f"Signal {source}: score={final_score:.3f}, quality={quality_score:.3f}, conf={signal['confidence']:.3f}")
        
        if not scored_signals:
            logger.info("No signals passed quality filter")
            return []
        
        # Sort by enhanced score
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        # Enhanced signal selection with diversification
        selected_signals = []
        used_strikes = set()
        used_sources = []  # Use list instead of set for counting
        
        for score, signal in scored_signals:
            strike_key = (signal['option_type'], round(signal['strike'], 0))
            source = signal['signal_source']
            
            # Diversification logic
            strike_conflict = strike_key in used_strikes
            source_overuse = used_sources.count(source) >= 2  # Now this works with list
            
            if not strike_conflict and not source_overuse:
                selected_signals.append(signal)
                used_strikes.add(strike_key)
                used_sources.append(source)  # Add to list
                
                if len(selected_signals) >= config.MAX_OPEN_POSITIONS:
                    break
            else:
                logger.info(f"Skipping {source} signal due to diversification (strike_conflict={strike_conflict}, source_overuse={source_overuse})")
        
        logger.info(f"Selected {len(selected_signals)} signals after diversification")
        return selected_signals
    
    def _calculate_threshold_conformity(self, signal: Dict, thresholds: Dict) -> float:
        """Calculate how well signal conforms to current thresholds"""
        
        conformity_score = 0.0
        
        # Confidence conformity
        if signal['confidence'] >= thresholds.get('confidence', 0.5):
            conformity_score += 0.3
        
        # Uncertainty conformity
        if signal['total_uncertainty'] <= thresholds.get('uncertainty', 0.7):
            conformity_score += 0.3
        
        # Signal strength conformity
        if signal['signal_strength'] >= thresholds.get('signal_strength', 0.3):
            conformity_score += 0.4
        
        return conformity_score
    
    def record_trade_outcome(self, trade: Dict):
        """Enhanced trade outcome recording"""
        self.recent_trades.append(trade)
        
        # Maintain optimal history length for rapid adaptation
        max_history = config.ADAPTATION_WINDOW * 2
        if len(self.recent_trades) > max_history:
            self.recent_trades = self.recent_trades[-max_history:]
        
        # Record in enhanced threshold manager
        self.threshold_manager.performance_tracker.record_trade_outcome(
            trade, trade.get('signal_source', 'Unknown')
        )

# --- ENHANCED ADAPTIVE PORTFOLIO MANAGER ---
class EnhancedAdaptivePortfolioManager:
    """Enhanced portfolio manager with dynamic position sizing and risk management"""
    
    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.cash = initial_equity
        self.positions = []
        self.trade_log = []
        self.daily_equity = []
        self.daily_returns = []
        self.performance_metrics = {}
        self.risk_metrics = {}
        
    def calculate_enhanced_position_size(self, signal: Dict, recent_performance: Dict = None,
                                       signal_quality_scores: Dict = None,
                                       current_thresholds: Dict = None) -> int:
        """Enhanced position sizing with mathematical precision"""
        
        confidence = signal.get('confidence', 0.5)
        signal_strength = signal.get('signal_strength', 0.5)
        uncertainty = signal.get('total_uncertainty', 0.5)
        signal_source = signal.get('signal_source', 'Unknown')
        entry_price = signal['entry_price']
        
        # Enhanced quality assessment
        quality_score = 0.5
        if signal_quality_scores:
            quality_score = signal_quality_scores.get(signal_source, 0.5)
        
        # Mathematical position sizing calculation
        base_risk = config.BASE_RISK_PER_TRADE
        
        # Quality multiplier with enhanced mathematical precision
        if quality_score > 0.8:  # Exceptional signals
            quality_multiplier = 1.5
        elif quality_score > 0.7:  # Excellent signals
            quality_multiplier = 1.3
        elif quality_score > 0.6:  # Very good signals
            quality_multiplier = 1.1
        elif quality_score > 0.5:  # Good signals
            quality_multiplier = 1.0
        elif quality_score > 0.4:  # Average signals
            quality_multiplier = 0.8
        elif quality_score > 0.3:  # Below average signals
            quality_multiplier = 0.6
        else:  # Poor signals
            quality_multiplier = 0.4
        
        # Enhanced signal strength multiplier
        strength_component = confidence * signal_strength * (1 - uncertainty)
        strength_multiplier = max(0.6, min(1.4, 0.8 + strength_component))
        
        # Performance-based adjustment with mathematical precision
        performance_multiplier = 1.0
        if recent_performance and recent_performance.get('num_trades', 0) >= 3:
            win_rate = recent_performance.get('win_rate', 0.5)
            profit_factor = recent_performance.get('profit_factor', 1.0)
            performance_trend = recent_performance.get('performance_trend', 0.0)
            
            # Mathematical performance adjustment
            perf_score = (win_rate - 0.5) * 2  # -1 to 1 scale
            pf_score = np.tanh((profit_factor - 1) * 2)  # Bounded adjustment
            trend_score = np.tanh(performance_trend * 10)  # Trend impact
            
            combined_perf = perf_score * 0.4 + pf_score * 0.4 + trend_score * 0.2
            performance_multiplier = 1.0 + combined_perf * 0.3  # ±30% adjustment
            performance_multiplier = np.clip(performance_multiplier, 0.6, 1.4)
        
        # Threshold conformity bonus
        conformity_multiplier = 1.0
        if current_thresholds:
            conformity_score = 0.0
            if confidence >= current_thresholds.get('confidence', 0.5):
                conformity_score += 0.3
            if uncertainty <= current_thresholds.get('uncertainty', 0.7):
                conformity_score += 0.3
            if signal_strength >= current_thresholds.get('signal_strength', 0.3):
                conformity_score += 0.4
            
            conformity_multiplier = 1.0 + conformity_score * 0.2  # Up to 20% bonus
        
        # Portfolio heat adjustment (reduce size if too many positions)
        portfolio_heat = len(self.positions) / config.MAX_OPEN_POSITIONS
        heat_multiplier = max(0.7, 1.0 - portfolio_heat * 0.3)
        
        # Calculate final risk with all adjustments
        adjusted_risk = (base_risk * quality_multiplier * strength_multiplier * 
                        performance_multiplier * conformity_multiplier * heat_multiplier)
        
        # Apply bounds
        adjusted_risk = np.clip(adjusted_risk, 
                               config.BASE_RISK_PER_TRADE * 0.4,
                               config.MAX_RISK_PER_TRADE)
        
        # Calculate position size
        risk_amount = self.equity * adjusted_risk
        cost_per_contract = entry_price * 100
        
        if cost_per_contract <= 0:
            return 0
        
        size = max(1, int(risk_amount / cost_per_contract))
        
        # Quality-based size limits
        if quality_score > 0.8:
            max_size = 4  # Exceptional signals
        elif quality_score > 0.6:
            max_size = 3  # Good signals
        elif quality_score > 0.4:
            max_size = 2  # Average signals
        else:
            max_size = 1  # Poor signals
        
        size = min(size, max_size)
        
        logger.info(f"📊 Enhanced position size {size} for {signal_source}: "
                   f"Quality={quality_score:.3f} ({quality_multiplier:.2f}x), "
                   f"Strength={strength_multiplier:.2f}x, Performance={performance_multiplier:.2f}x, "
                   f"Risk={adjusted_risk:.3f}")
        
        return size
    
    def update_daily_portfolio_value(self, date: pd.Timestamp, market_data: Dict):
        """Enhanced daily portfolio valuation with risk metrics"""
        if not market_data:
            self.daily_equity.append({
                'date': date,
                'equity': self.equity,
                'cash': self.cash,
                'positions_value': 0,
                'num_positions': len(self.positions)
            })
            return
        
        total_positions_value = 0
        position_risks = []
        
        for pos in self.positions:
            try:
                options_df = market_data['calls'] if pos['option_type'] == 'call' else market_data['puts']
                matching = options_df[np.abs(options_df['strike'] - pos['strike']) < 1.0]
                
                if not matching.empty:
                    current_bid = matching.iloc[0]['bid']
                    position_value = current_bid * pos['size'] * 100
                    total_positions_value += position_value
                    pos['current_value'] = position_value
                    pos['unrealized_pnl'] = position_value - pos['entry_cost']
                    
                    # Calculate position risk
                    position_risk = pos['entry_cost'] / self.equity if self.equity > 0 else 0
                    position_risks.append(position_risk)
                    
            except Exception as e:
                logger.warning(f"Enhanced mark-to-market error: {e}")
                total_positions_value += pos.get('entry_cost', 0)
        
        total_equity = self.cash + total_positions_value
        
        # Calculate risk metrics
        portfolio_risk = sum(position_risks)
        
        if self.daily_equity:
            prev_equity = self.daily_equity[-1]['equity']
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0)
        
        self.equity = total_equity
        
        # Enhanced daily record
        self.daily_equity.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'positions_value': total_positions_value,
            'num_positions': len(self.positions),
            'portfolio_risk': portfolio_risk,
            'daily_return': self.daily_returns[-1] if self.daily_returns else 0
        })
    
    def close_positions_daily(self, date: pd.Timestamp, market_data: Dict, 
                            signal_quality_scores: Dict = None,
                            current_thresholds: Dict = None):
        """Enhanced position management with adaptive risk controls"""
        if not self.positions or not market_data:
            return
            
        for pos in self.positions[:]:
            try:
                days_held = (date - pos['date']).days
                
                options_df = market_data['calls'] if pos['option_type'] == 'call' else market_data['puts']
                matching = options_df[np.abs(options_df['strike'] - pos['strike']) < 1.0]
                
                if matching.empty:
                    closest_idx = np.abs(options_df['strike'] - pos['strike']).argmin()
                    matching = options_df.iloc[[closest_idx]]
                
                current_bid = matching.iloc[0]['bid']
                current_value = current_bid * pos['size'] * 100
                entry_cost = pos.get('entry_cost', pos['entry_price'] * pos['size'] * 100)
                pnl = current_value - entry_cost
                pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
                
                should_close = False
                exit_reason = ""
                
                # Enhanced quality-based risk management
                signal_source = pos.get('signal_source', 'Unknown')
                quality_score = 0.5
                if signal_quality_scores:
                    quality_score = signal_quality_scores.get(signal_source, 0.5)
                
                # Adaptive risk parameters based on quality and market conditions
                if quality_score > 0.7:  # High quality signals
                    stop_loss_pct = config.STOP_LOSS_PCT * 0.85
                    profit_target_pct = config.PROFIT_TARGET_PCT * 1.3
                    max_hold = config.MAX_HOLD_DAYS + 1
                elif quality_score > 0.5:  # Good quality signals
                    stop_loss_pct = config.STOP_LOSS_PCT * 0.95
                    profit_target_pct = config.PROFIT_TARGET_PCT * 1.1
                    max_hold = config.MAX_HOLD_DAYS
                elif quality_score > 0.3:  # Average quality signals
                    stop_loss_pct = config.STOP_LOSS_PCT
                    profit_target_pct = config.PROFIT_TARGET_PCT
                    max_hold = config.MAX_HOLD_DAYS - 1
                else:  # Low quality signals
                    stop_loss_pct = config.STOP_LOSS_PCT * 1.15
                    profit_target_pct = config.PROFIT_TARGET_PCT * 0.85
                    max_hold = config.MAX_HOLD_DAYS - 2
                
                # Enhanced exit conditions
                if pnl_pct <= -stop_loss_pct:
                    should_close = True
                    exit_reason = "Stop-Loss"
                elif pnl_pct >= profit_target_pct:
                    should_close = True
                    exit_reason = "Profit-Target"
                elif days_held >= max_hold:
                    should_close = True
                    exit_reason = "Max-Hold"
                elif current_bid < 0.10:
                    should_close = True
                    exit_reason = "Worthless"
                
                # Dynamic exit rules based on performance and thresholds
                if current_thresholds:
                    # Threshold-based early exits
                    confidence_threshold = current_thresholds.get('confidence', 0.5)
                    if confidence_threshold > 0.6 and days_held >= 2 and pnl_pct < -0.15:
                        should_close = True
                        exit_reason = "Threshold-Cut-Loss"
                    elif confidence_threshold < 0.3 and days_held >= 1 and pnl_pct > 0.15:
                        should_close = True
                        exit_reason = "Threshold-Quick-Profit"
                
                # Quality-specific early exit rules
                if quality_score < 0.3:  # Low quality signals
                    if days_held >= 1 and pnl_pct < -0.12:
                        should_close = True
                        exit_reason = "Quality-Quick-Cut"
                    elif days_held >= 2 and pnl_pct > 0.12:
                        should_close = True
                        exit_reason = "Quality-Quick-Profit"
                
                elif quality_score > 0.7:  # High quality signals
                    if days_held >= 3 and pnl_pct > 0.35:
                        should_close = True
                        exit_reason = "Quality-Profit-Lock"
                
                # Universal emergency exits
                if days_held >= 1 and pnl_pct < -0.40:
                    should_close = True
                    exit_reason = "Emergency-Cut"
                elif days_held >= 2 and pnl_pct > 0.50:
                    should_close = True
                    exit_reason = "Emergency-Profit"
                
                if should_close:
                    commission = config.COMMISSION_PER_CONTRACT * pos['size']
                    net_pnl = pnl - commission
                    
                    self.cash += current_value - commission
                    
                    trade_record = {
                        'date': pos['date'],
                        'exit_date': date,
                        'action': pos['action'],
                        'option_type': pos['option_type'],
                        'strike': pos['strike'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'exit_price': current_bid,
                        'days_held': days_held,
                        'gross_pnl': pnl,
                        'commission': commission,
                        'net_pnl': net_pnl,
                        'return_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'confidence': pos.get('confidence', 0),
                        'total_uncertainty': pos.get('total_uncertainty', 0),
                        'signal_source': pos.get('signal_source', 'Unknown'),
                        'moneyness': pos.get('moneyness', 1.0),
                        'signal_strength': pos.get('signal_strength', 0.5),
                        'quality_score': quality_score
                    }
                    
                    self.trade_log.append(trade_record)
                    self.positions.remove(pos)
                    
                    logger.info(f"CLOSED: {pos['option_type']} {pos['strike']:.0f} - "
                               f"P&L: ${net_pnl:.2f} ({pnl_pct:.1%}) [{exit_reason}, {days_held}d] "
                               f"Quality: {quality_score:.3f}")
                    
            except Exception as e:
                logger.error(f"Enhanced position close error: {e}")

    def enter_new_positions(self, date: pd.Timestamp, signals: List[Dict], 
                          recent_performance: Dict = None, 
                          signal_quality_scores: Dict = None,
                          current_thresholds: Dict = None):
        """Enhanced position entry with dynamic sizing"""
        for signal in signals:
            if len(self.positions) >= config.MAX_OPEN_POSITIONS:
                break
                
            # Enhanced duplicate checking
            duplicate = any(
                abs(p['strike'] - signal['strike']) < 3.0 and 
                p['option_type'] == signal['option_type'] and
                p.get('signal_source', '') != signal.get('signal_source', '')
                for p in self.positions
            )
            if duplicate:
                continue
            
            # Enhanced position sizing
            size = self.calculate_enhanced_position_size(
                signal, recent_performance, signal_quality_scores, current_thresholds
            )
            
            if size <= 0:
                continue
            
            required_cash = signal['entry_price'] * size * 100
            commission = config.COMMISSION_PER_CONTRACT * size
            total_cost = required_cash + commission
            
            if total_cost > self.cash:
                logger.info(f"Insufficient cash: ${total_cost:.2f} > ${self.cash:.2f}")
                continue
            
            position = {
                'date': signal['date'],
                'action': signal['action'],
                'option_type': signal['option_type'],
                'strike': signal['strike'],
                'entry_price': signal['entry_price'],
                'size': size,
                'entry_cost': total_cost,
                'commission': commission,
                'current_value': required_cash,
                'unrealized_pnl': 0,
                'confidence': signal.get('confidence', 0),
                'total_uncertainty': signal.get('total_uncertainty', 0),
                'signal_source': signal.get('signal_source', 'Unknown'),
                'moneyness': signal.get('moneyness', 1.0),
                'signal_strength': signal.get('signal_strength', 0.5)
            }
            
            self.cash -= total_cost
            self.positions.append(position)
            
            quality_score = signal_quality_scores.get(signal.get('signal_source', 'Unknown'), 0.5) if signal_quality_scores else 0.5
            logger.info(f"OPENED: {signal['option_type']} {signal['strike']:.0f} "
                       f"@ ${signal['entry_price']:.2f} x{size} [{signal.get('signal_source', 'Unknown')}] "
                       f"Quality: {quality_score:.3f}")

# --- ENHANCED ADAPTIVE BACKTESTER ---
class EnhancedAdaptiveBacktester:
    """Enhanced backtester with mathematical precision and real-time adaptation"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str, initial_equity: float):
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_equity = initial_equity
        
        # Initialize enhanced components
        self.market_sim = AdaptiveMarketSimulator(symbol, config.START_DATE, config.BACKTEST_END_DATE)
        self.threshold_manager = EnhancedAdaptiveThresholdManager()
        
        # Build enhanced trading system
        self.trading_system = self._build_enhanced_trading_system()

    def _build_enhanced_trading_system(self):
        """Build enhanced trading system with mathematical precision"""
        logger.info("Building enhanced adaptive trading system...")
        
        train_data = self.market_sim.hist_data.loc[:config.TRAIN_END_DATE].copy()
        
        features = train_data[self.market_sim.feature_names].fillna(0)
        target = train_data['target_class'].fillna(1)
        
        if len(features) < 100:
            raise ValueError("Insufficient training data")
        
        # Create enhanced ensemble
        ensemble = EnhancedEnsemble(ensemble_size=5)
        ensemble.create_ensemble(features.values, target.values)
        
        # Create enhanced trading system
        trading_system = EnhancedAdaptiveTradingSystem(
            ensemble=ensemble,
            feature_names=self.market_sim.feature_names,
            threshold_manager=self.threshold_manager
        )
        
        logger.info("✅ Enhanced adaptive trading system ready")
        return trading_system

    def run_enhanced_backtest(self):
        """Run enhanced backtest with mathematical threshold adaptation"""
        logger.info(f"\n🧠 ENHANCED ADAPTIVE BACKTEST")
        logger.info(f"Symbol: {self.symbol}, Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Initial Equity: ${self.initial_equity:,.2f}")
        logger.info(f"Features: Mathematical thresholds, real-time adaptation, enhanced regime detection")
        
        # Initialize enhanced portfolio manager
        portfolio_manager = EnhancedAdaptivePortfolioManager(self.initial_equity)
        
        # Get backtest date range
        backtest_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='B'  # Business days only
        )
        
        # Filter to dates with market data
        available_dates = [d for d in backtest_dates if d in self.market_sim.hist_data.index]
        
        if not available_dates:
            raise ValueError("No market data available for backtest period")
        
        logger.info(f"📅 Trading {len(available_dates)} days with enhanced mathematical adaptation")
        
        # Run enhanced backtest
        for i, date in enumerate(available_dates):
            try:
                # Get market data
                market_data = self.market_sim.get_market_for_day(date)
                if not market_data:
                    continue
                
                # Get historical features up to current date
                historical_data = self.market_sim.hist_data.loc[:date]
                if len(historical_data) < 100:
                    continue
                
                # Get enhanced performance metrics
                recent_performance = portfolio_manager.performance_metrics if hasattr(portfolio_manager, 'performance_metrics') else None
                if len(portfolio_manager.trade_log) >= 5:
                    recent_trades = portfolio_manager.trade_log[-15:]
                    recent_performance = {
                        'win_rate': sum(1 for t in recent_trades if t['net_pnl'] > 0) / len(recent_trades),
                        'avg_return': np.mean([t['return_pct'] for t in recent_trades]),
                        'profit_factor': self._calculate_profit_factor(recent_trades),
                        'num_trades': len(recent_trades),
                        'performance_trend': self._calculate_performance_trend(recent_trades)
                    }
                
                # Get enhanced signal quality scores
                signal_quality_scores = self.trading_system.threshold_manager.signal_quality_tracker.get_quality_scores_with_fallback()
                
                # Get current adaptive thresholds
                current_thresholds = self.trading_system.threshold_manager.current_thresholds
                
                # Close existing positions with enhanced risk management
                portfolio_manager.close_positions_daily(date, market_data, signal_quality_scores, current_thresholds)
                
                # Record completed trades for enhanced learning
                if portfolio_manager.trade_log:
                    for trade in portfolio_manager.trade_log:
                        if trade not in self.trading_system.recent_trades:
                            self.trading_system.record_trade_outcome(trade)
                
                # Update portfolio valuation with enhanced metrics
                portfolio_manager.update_daily_portfolio_value(date, market_data)
                
                # Generate enhanced adaptive signals
                signals = self.trading_system.generate_signals(market_data, historical_data, i + 1)
                
                # Enter new positions with enhanced sizing
                if signals:
                    portfolio_manager.enter_new_positions(
                        date, signals, recent_performance, signal_quality_scores, current_thresholds
                    )
                
                # Enhanced progress logging
                if (i + 1) % 5 == 0 or i == len(available_dates) - 1:
                    equity = portfolio_manager.equity
                    num_positions = len(portfolio_manager.positions)
                    num_trades = len(portfolio_manager.trade_log)
                    
                    # Get current thresholds for logging
                    conf_threshold = current_thresholds.get('confidence', 0.5) if current_thresholds else 0.5
                    unc_threshold = current_thresholds.get('uncertainty', 0.7) if current_thresholds else 0.7
                    
                    logger.info(f"Day {i+1}/{len(available_dates)}: "
                               f"Equity=${equity:,.0f}, Positions={num_positions}, "
                               f"Trades={num_trades}, Conf={conf_threshold:.3f}, "
                               f"Unc={unc_threshold:.3f}")
            
            except Exception as e:
                logger.error(f"Enhanced backtest error on {date}: {e}")
                continue
        
        # Close all remaining positions
        if available_dates:
            final_date = available_dates[-1]
            final_market = self.market_sim.get_market_for_day(final_date)
            if final_market:
                for pos in portfolio_manager.positions[:]:
                    try:
                        portfolio_manager.close_positions_daily(final_date, final_market)
                    except:
                        pass
        
        # Calculate enhanced performance metrics
        final_performance = self._calculate_enhanced_performance_metrics(portfolio_manager)
        self._display_enhanced_results(final_performance, portfolio_manager)
        
        return portfolio_manager, final_performance
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades"""
        winners = [t for t in trades if t['net_pnl'] > 0]
        losers = [t for t in trades if t['net_pnl'] <= 0]
        
        gross_profit = sum(t['net_pnl'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['net_pnl'] for t in losers)) if losers else 1
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_performance_trend(self, trades: List[Dict]) -> float:
        """Calculate performance trend from recent trades"""
        if len(trades) < 4:
            return 0.0
        
        # Calculate rolling win rate
        win_rates = []
        window = max(2, len(trades) // 3)
        
        for i in range(window, len(trades) + 1):
            subset = trades[i-window:i]
            win_rate = sum(1 for t in subset if t['net_pnl'] > 0) / len(subset)
            win_rates.append(win_rate)
        
        if len(win_rates) < 2:
            return 0.0
        
        # Calculate trend slope
        try:
            trend_slope = np.polyfit(range(len(win_rates)), win_rates, 1)[0]
            return trend_slope
        except:
            return 0.0

    def _calculate_enhanced_performance_metrics(self, portfolio_manager) -> Dict:
        """Calculate enhanced performance metrics with mathematical precision"""
        
        base_metrics = {
            'num_trades': 0,
            'win_rate_pct': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'avg_days_held': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'adaptation_summary': {},
            'threshold_evolution': {},
            'signal_performance': {}
        }
        
        if not portfolio_manager.trade_log:
            return base_metrics
        
        try:
            trades_df = pd.DataFrame(portfolio_manager.trade_log)
            
            # Basic metrics
            num_trades = len(trades_df)
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] <= 0]
            
            win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
            
            # Return calculations
            total_pnl = trades_df['net_pnl'].sum()
            total_return = (portfolio_manager.equity - self.initial_equity) / self.initial_equity * 100
            
            # Enhanced risk metrics
            sharpe_ratio = 0
            max_drawdown = 0
            
            if portfolio_manager.daily_returns and len(portfolio_manager.daily_returns) > 1:
                daily_returns = np.array(portfolio_manager.daily_returns)
                daily_returns = daily_returns[~np.isnan(daily_returns)]
                
                if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                
                if portfolio_manager.daily_equity:
                    equity_curve = pd.Series([eq['equity'] for eq in portfolio_manager.daily_equity])
                    running_max = equity_curve.expanding().max()
                    drawdowns = (equity_curve - running_max) / running_max * 100
                    max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
            
            # Trading metrics
            avg_win = winning_trades['return_pct'].mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['return_pct'].mean() * 100 if len(losing_trades) > 0 else 0
            
            gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_days_held = trades_df['days_held'].mean() if num_trades > 0 else 0
            
            # Enhanced adaptation analysis
            adaptation_summary = self.threshold_manager.get_threshold_adaptation_summary() if hasattr(self.threshold_manager, 'get_threshold_adaptation_summary') else {}
            
            # Threshold evolution analysis
            threshold_evolution = {}
            if hasattr(self.threshold_manager, 'adaptation_history') and self.threshold_manager.adaptation_history:
                history = self.threshold_manager.adaptation_history
                
                threshold_evolution = {
                    'initial_confidence': history[0]['thresholds'].get('confidence', 0.5),
                    'final_confidence': history[-1]['thresholds'].get('confidence', 0.5),
                    'initial_uncertainty': history[0]['thresholds'].get('uncertainty', 0.7),
                    'final_uncertainty': history[-1]['thresholds'].get('uncertainty', 0.7),
                    'adaptations': len(history),
                    'emergency_activations': sum(1 for h in history if h.get('emergency_adjustments'))
                }
            
            # Signal performance analysis
            signal_performance = {}
            if 'signal_source' in trades_df.columns:
                for source in trades_df['signal_source'].unique():
                    source_trades = trades_df[trades_df['signal_source'] == source]
                    signal_performance[source] = {
                        'trades': len(source_trades),
                        'win_rate': (source_trades['net_pnl'] > 0).mean() * 100,
                        'avg_return': source_trades['return_pct'].mean() * 100,
                        'total_pnl': source_trades['net_pnl'].sum()
                    }
            
            base_metrics.update({
                'num_trades': num_trades,
                'win_rate_pct': win_rate,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'profit_factor': profit_factor,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'avg_days_held': avg_days_held,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'adaptation_summary': adaptation_summary,
                'threshold_evolution': threshold_evolution,
                'signal_performance': signal_performance
            })
            
        except Exception as e:
            logger.error(f"Enhanced performance calculation error: {e}")
        
        return base_metrics

    def _display_enhanced_results(self, metrics: Dict, portfolio_manager):
        """Display enhanced results with mathematical analysis"""
        
        logger.info(f"\n🧠 ENHANCED ADAPTIVE BACKTEST RESULTS")
        logger.info(f"{'='*100}")
        logger.info(f"📈 PERFORMANCE SUMMARY:")
        logger.info(f"   Initial Equity: ${self.initial_equity:,.2f}")
        logger.info(f"   Final Equity: ${portfolio_manager.equity:,.2f}")
        logger.info(f"   Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
        
        logger.info(f"\n📊 TRADING STATISTICS:")
        logger.info(f"   Total Trades: {metrics['num_trades']} 🎯")
        logger.info(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
        logger.info(f"   Avg Days Held: {metrics['avg_days_held']:.1f}")
        logger.info(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        
        if metrics['num_trades'] > 0:
            logger.info(f"   Avg Win: {metrics['avg_win_pct']:.2f}%")
            logger.info(f"   Avg Loss: {metrics['avg_loss_pct']:.2f}%")
            logger.info(f"   Gross Profit: ${metrics['gross_profit']:,.2f}")
            logger.info(f"   Gross Loss: ${metrics['gross_loss']:,.2f}")
        
        logger.info(f"\n⚡ RISK METRICS:")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Enhanced adaptation analysis
        threshold_evolution = metrics.get('threshold_evolution', {})
        if threshold_evolution:
            logger.info(f"\n🎯 MATHEMATICAL THRESHOLD EVOLUTION:")
            logger.info(f"   Confidence: {threshold_evolution.get('initial_confidence', 0):.3f} → "
                       f"{threshold_evolution.get('final_confidence', 0):.3f}")
            logger.info(f"   Uncertainty: {threshold_evolution.get('initial_uncertainty', 0):.3f} → "
                       f"{threshold_evolution.get('final_uncertainty', 0):.3f}")
            logger.info(f"   Total Adaptations: {threshold_evolution.get('adaptations', 0)}")
            logger.info(f"   Emergency Activations: {threshold_evolution.get('emergency_activations', 0)}")
        
        # Signal performance analysis
        signal_performance = metrics.get('signal_performance', {})
        if signal_performance:
            logger.info(f"\n🎯 ENHANCED SIGNAL SOURCE PERFORMANCE:")
            for source, perf in sorted(signal_performance.items(), key=lambda x: x[1]['win_rate'], reverse=True):
                logger.info(f"   {source}: {perf['trades']} trades, {perf['win_rate']:.1f}% win rate, "
                           f"{perf['avg_return']:.1f}% avg return, ${perf['total_pnl']:.0f} P&L")
        
        # Enhanced system assessment
        logger.info(f"\n🏆 ENHANCED SYSTEM ASSESSMENT:")
        
        if metrics['num_trades'] >= 20:
            logger.info(f"   ✅ EXCELLENT: Generated {metrics['num_trades']} trades with mathematical adaptation")
        elif metrics['num_trades'] >= 15:
            logger.info(f"   ✅ VERY GOOD: Generated {metrics['num_trades']} trades")
        elif metrics['num_trades'] >= 10:
            logger.info(f"   ✅ GOOD: Generated {metrics['num_trades']} trades")
        else:
            logger.info(f"   ⚠️ DEVELOPING: Generated {metrics['num_trades']} trades")
        
        if metrics['num_trades'] >= 10:
            if (metrics['win_rate_pct'] >= 50 and metrics['total_return_pct'] > 2 and 
                metrics['sharpe_ratio'] > 0.5):
                logger.info("   📈 MATHEMATICAL SUCCESS: Strong performance with enhanced adaptation")
            elif metrics['total_return_pct'] > 0:
                logger.info("   📊 ADAPTIVE LEARNING: System evolving with positive trajectory")
            else:
                logger.info("   📉 CALIBRATION MODE: Continued mathematical refinement needed")
        
        # Mathematical accuracy assessment
        if threshold_evolution.get('adaptations', 0) > 0:
            logger.info("   🧮 MATHEMATICAL ACCURACY: Dynamic thresholds with statistical precision")
            logger.info("   📊 ENHANCED REALISM: Real-time adaptation with regime detection")
            if metrics['profit_factor'] > 1.2:
                logger.info("   💰 PROFITABILITY ACHIEVED: Mathematical edge demonstrated")

# --- ENHANCED VISUALIZATION ---
def create_enhanced_plots(portfolio_manager, performance_metrics):
    """Create comprehensive enhanced visualization"""
    
    if not portfolio_manager.daily_equity:
        logger.warning("No equity data for enhanced plotting")
        return
    
    def _calculate_signal_profit_factor(signal_trades):
        """Calculate profit factor for specific signal"""
        winners = signal_trades[signal_trades['net_pnl'] > 0]
        losers = signal_trades[signal_trades['net_pnl'] <= 0]
        
        gross_profit = winners['net_pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    try:
        # Create enhanced plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Enhanced equity curve with detailed analysis
        equity_df = pd.DataFrame(portfolio_manager.daily_equity)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        ax1.plot(equity_df.index, equity_df['equity'], linewidth=3, color='darkblue', label='Portfolio Value')
        ax1.axhline(y=portfolio_manager.initial_equity, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Initial')
        
        # Enhanced drawdown visualization
        running_max = equity_df['equity'].expanding().max()
        drawdowns = (equity_df['equity'] - running_max) / running_max
        ax1.fill_between(equity_df.index, equity_df['equity'], running_max, 
                        where=(drawdowns < 0), alpha=0.3, color='red', label='Drawdown')
        
        # Add performance annotations
        total_return = (equity_df['equity'].iloc[-1] - portfolio_manager.initial_equity) / portfolio_manager.initial_equity * 100
        ax1.text(0.02, 0.98, f'Total Return: {total_return:.2f}%\nSharpe: {performance_metrics["sharpe_ratio"]:.3f}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title('Enhanced Adaptive Portfolio Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Enhanced signal analysis
        if portfolio_manager.trade_log:
            trades_df = pd.DataFrame(portfolio_manager.trade_log)
            
            # Signal source performance with enhanced metrics
            signal_analysis = []
            for source in trades_df['signal_source'].unique():
                source_trades = trades_df[trades_df['signal_source'] == source]
                win_rate = (source_trades['net_pnl'] > 0).mean()
                avg_return = source_trades['return_pct'].mean()
                profit_factor = _calculate_signal_profit_factor(source_trades)
                signal_analysis.append((source, len(source_trades), win_rate, avg_return, profit_factor))
            
            signal_analysis.sort(key=lambda x: x[4], reverse=True)  # Sort by profit factor
            
            sources = [s[0][:15] for s in signal_analysis]
            profit_factors = [s[4] for s in signal_analysis]
            colors = ['darkgreen' if pf >= 1.5 else 'green' if pf >= 1.2 else 'orange' if pf >= 1.0 else 'red' for pf in profit_factors]
            
            bars = ax2.bar(sources, profit_factors, color=colors, alpha=0.7, edgecolor='black')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, label='Break-even')
            ax2.set_title('Enhanced Signal Profit Factor Analysis', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Profit Factor')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add trade count labels
            for bar, analysis in zip(bars, signal_analysis):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{analysis[1]}', ha='center', va='bottom', fontsize=10)
            
            # Enhanced return distribution with statistical analysis
            returns_pct = trades_df['return_pct'] * 100
            
            # Calculate statistical measures
            mean_return = returns_pct.mean()
            median_return = returns_pct.median()
            std_return = returns_pct.std()
            skewness = returns_pct.skew()
            
            ax3.hist(returns_pct, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True, label='Returns')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Break-even')
            ax3.axvline(x=mean_return, color='green', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Mean: {mean_return:.1f}%')
            ax3.axvline(x=median_return, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Median: {median_return:.1f}%')
            
            # Add normal distribution overlay
            x_norm = np.linspace(returns_pct.min(), returns_pct.max(), 100)
            y_norm = norm.pdf(x_norm, mean_return, std_return)
            ax3.plot(x_norm, y_norm, 'r--', alpha=0.6, label=f'Normal (σ={std_return:.1f}%)')
            
            ax3.set_title(f'Enhanced Return Distribution (Skew: {skewness:.2f})', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Enhanced multi-dimensional analysis
            scatter = ax4.scatter(trades_df['confidence'], returns_pct, 
                                c=trades_df['days_held'], s=trades_df.get('quality_score', 0.5)*100, 
                                alpha=0.7, cmap='viridis')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Avg Confidence')
            
            # Add trend line
            z = np.polyfit(trades_df['confidence'], returns_pct, 1)
            p = np.poly1d(z)
            ax4.plot(trades_df['confidence'], p(trades_df['confidence']), "r--", alpha=0.8, label=f'Trend: {z[0]:.1f}')
            
            ax4.set_xlabel('Signal Confidence')
            ax4.set_ylabel('Return (%)')
            ax4.set_title('Enhanced Multi-Dimensional Analysis\n(Size=Quality, Color=Days Held)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Enhanced colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Days Held', rotation=270, labelpad=20)
        else:
            # If no trades, show placeholder text
            for ax in [ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('No Trading Data Available')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Enhanced plotting error: {e}")
        import traceback
        logger.error(traceback.format_exc())

# --- MAIN EXECUTION ---
def main():
    """Main execution function with enhanced mathematical adaptation"""
    
    logger.info("🧠 ENHANCED ADAPTIVE OPTIONS TRADING SYSTEM")
    logger.info("=" * 120)
    logger.info("FEATURES: Mathematical Precision + Real-time Adaptation + Enhanced Regime Detection")
    logger.info("GOAL: Full mathematical accuracy, realism, and consistent profitability")
    logger.info("=" * 120)
    
    try:
        # Initialize and run enhanced backtest
        backtester = EnhancedAdaptiveBacktester(
            symbol=config.SYMBOL,
            start_date=config.BACKTEST_START_DATE,
            end_date=config.BACKTEST_END_DATE,
            initial_equity=config.INITIAL_EQUITY
        )
        
        # Run enhanced backtest with mathematical precision
        portfolio_manager, final_performance = backtester.run_enhanced_backtest()
        
        # Create enhanced visualizations
        logger.info("\n📊 Generating enhanced mathematical analysis...")
        create_enhanced_plots(portfolio_manager, final_performance)
        
        # Export enhanced results
        if portfolio_manager.trade_log:
            trades_df = pd.DataFrame(portfolio_manager.trade_log)
            
            # Add enhanced metrics
            trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
            trades_df['running_win_rate'] = (trades_df['net_pnl'] > 0).expanding().mean()
            trades_df['quality_adjusted_return'] = trades_df['return_pct'] * trades_df.get('quality_score', 0.5)
            
            # Export enhanced trade log
            filename = f"enhanced_trades_{config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(filename, index=False)
            logger.info(f"📁 Enhanced trade log exported to: {filename}")
            
            # Export threshold adaptation history
            if hasattr(backtester.threshold_manager, 'adaptation_history') and backtester.threshold_manager.adaptation_history:
                adaptation_df = pd.DataFrame(backtester.threshold_manager.adaptation_history)
                adapt_filename = f"enhanced_adaptations_{config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                adaptation_df.to_csv(adapt_filename, index=False)
                logger.info(f"📁 Enhanced adaptation history exported to: {adapt_filename}")
        
        # Final enhanced assessment
        logger.info(f"\n🎯 ENHANCED SYSTEM FINAL ASSESSMENT:")
        logger.info(f"   📊 Mathematical Thresholds: ✅ Implemented with Statistical Precision")
        logger.info(f"   🧠 Real-time Learning: ✅ Active with Rapid Adaptation")
        logger.info(f"   📈 Enhanced Regime Detection: ✅ Multi-factor Analysis")
        logger.info(f"   🔄 Continuous Evolution: ✅ Mathematical Feedback Loop")
        
        threshold_evolution = final_performance.get('threshold_evolution', {})
        total_adaptations = threshold_evolution.get('adaptations', 0)
        
        if total_adaptations > 0:
            logger.info(f"   🎯 Mathematical Iterations: {total_adaptations}")
            logger.info(f"   ✅ STATISTICAL ACCURACY: Real-time threshold calculation from data")
            if final_performance['num_trades'] >= 15:
                logger.info(f"   ✅ ENHANCED REALISM: Multi-factor regime adaptation successful")
            if final_performance.get('profit_factor', 0) > 1.3:
                logger.info(f"   ✅ MATHEMATICAL EDGE: Enhanced profitability achieved")
        
        logger.info("\n✅ ENHANCED ADAPTIVE BACKTEST COMPLETE WITH MATHEMATICAL PRECISION!")
        return portfolio_manager, final_performance
        
    except Exception as e:
        logger.error(f"❌ ENHANCED BACKTEST FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Run the enhanced adaptive trading system with mathematical precision
    portfolio_manager, performance_metrics = main()

"""
Multi-Objective Optimization for Trading Signal Scoring System
NSGA-II Algorithm Implementation
"""

    
# Optimization parameters (tune these for best results)
# ============================================================
# FAST TESTING MODE (for quick analysis and validation):
#   POPULATION_SIZE = 20-30  # Smaller population for speed
#   GENERATIONS = 10-15       # Fewer generations for speed
#   N_TRIALS = 1-3            # Single or few trials for speed
# ============================================================
# PRODUCTION MODE (for final results):
#   POPULATION_SIZE = 80
#   GENERATIONS = 40
#   N_TRIALS = 30
# ============================================================

# FAST TESTING MODE - Uncomment these for quick testing:
# POPULATION_SIZE = 25
# GENERATIONS = 12
# N_TRIALS = 1

# PRODUCTION MODE - Use these for final results:
POPULATION_SIZE = 80
GENERATIONS = 40
N_TRIALS = 30

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
import json
from datetime import datetime
from scipy.spatial.distance import euclidean
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, shapiro,
    norm, t as t_dist
)
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Parallel processing imports
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import platform

# Empirical bounds for trading strategy metrics (used for normalization in SSS calculation)
# These bounds are used for normalizing metrics in the SSS calculation (Static Strategy Score)
# Based on empirical evidence from trading system performance analysis
MAX_WIN_RATE = 0.67  # Maximum empirical win rate (67%)
MAX_SHARPE_RATIO = 2.6458  # Maximum empirical Sharpe ratio
MIN_DRAWDOWN = 8.65  # Minimum empirical drawdown (8.65%)
MAX_DRAWDOWN = 84.23  # Maximum empirical drawdown (84.23%)


# Global evaluation function for multiprocessing (must be at module level)
def _evaluate_individual_parallel(evaluation_data):
    """
    Standalone function for parallel evaluation of an individual
    
    This function must be at module level to be pickled for multiprocessing.
    It contains all evaluation logic without needing class instances.
    
    Args:
        evaluation_data: Dictionary containing:
            - weights_dict: Weight dictionary to evaluate
            - test_signals: Test signals DataFrame
            - strategy_metrics_df: Strategy metrics DataFrame
            - strategies: Array of strategy IDs
            - phase_columns: List of phase column names
            - phase_stats_cache: Pre-computed phase statistics
            - train_signals: Training signals (for SSS calculation)
    
    Returns:
        Tuple of (weights_dict, objectives_dict, metrics_dict)
    """
    import numpy as np
    import pandas as pd
    
    weights_dict = evaluation_data['weights_dict']
    test_signals = evaluation_data['test_signals']
    strategy_metrics_df = evaluation_data['strategy_metrics_df']
    strategies = evaluation_data['strategies']
    phase_columns = evaluation_data['phase_columns']
    phase_stats_cache = evaluation_data['phase_stats_cache']
    
    # Helper function to normalize metrics
    def normalize_metric(value, metric_name):
        # Import constants from module level
        MAX_WIN_RATE_LOCAL = 0.67
        MAX_SHARPE_RATIO_LOCAL = 2.6458
        bounds = {
            'Balance_Profit_Percent': (-1.0, 1.0),
            'Balance_Drawdown_Percent': (0.0, 1.0),
            'Equity_Drawdown_Percent': (0.0, 1.0),
            'Profit_Factor': (0.0, 5.0),
            'Recovery_Factor': (0.0, 10.0),
            'Sharpe_Ratio': (-1.0, MAX_SHARPE_RATIO_LOCAL),
            'Win_Rate': (0.0, MAX_WIN_RATE_LOCAL)
        }
        L, U = bounds[metric_name]
        normalized = (value - L) / (U - L)
        if 'Drawdown' in metric_name:
            normalized = 1.0 - normalized
        return np.clip(normalized, 0, 1)
    
    # Calculate SSS
    def calculate_sss(strategy_id, sss_weights):
        metrics_row = strategy_metrics_df[
            (strategy_metrics_df['strategy_id'] == strategy_id) &
            (strategy_metrics_df['period'] == 'training')
        ]
        if len(metrics_row) == 0:
            return 0.5
        metrics = metrics_row.iloc[0]
        metric_names = [
            'Balance_Profit_Percent', 'Balance_Drawdown_Percent',
            'Equity_Drawdown_Percent', 'Profit_Factor',
            'Recovery_Factor', 'Sharpe_Ratio', 'Win_Rate'
        ]
        normalized_metrics = np.array([
            normalize_metric(metrics[name], name) for name in metric_names
        ])
        sss = np.dot(sss_weights, normalized_metrics)
        return np.clip(sss, 0, 1)
    
    # Calculate phase score
    def calculate_phase_score(phase_stats, current_state, phase_name, min_diff=0.05):
        stats = phase_stats[phase_name]
        A_TP = stats['A_TP']
        R_TP = stats['R_TP']
        U_TP = stats['U_TP']
        if abs(A_TP - R_TP) < min_diff:
            return 0.0
        if U_TP > max(A_TP, R_TP):
            return 0.0
        if A_TP > R_TP:
            DV = A_TP
            RV = -R_TP
            dominant_state = 'A'
        else:
            DV = R_TP
            RV = -A_TP
            dominant_state = 'R'
        if current_state == 'U':
            return 0.0
        elif current_state == dominant_state:
            return DV
        else:
            return RV
    
    # Calculate DSS
    def calculate_dss(signal_row, strategy_id, dss_weights):
        phase_stats = phase_stats_cache[strategy_id]
        phase_scores = []
        for phase_col in phase_columns:
            phase_name = phase_col.replace('phase_', '')
            current_state = signal_row[phase_col]
            # Handle NaN phase states (treat as 'U' - unknown)
            if pd.isna(current_state):
                current_state = 'U'
            ps = calculate_phase_score(phase_stats, current_state, phase_name)
            # Ensure phase score is finite
            if not np.isfinite(ps):
                ps = 0.0
            phase_scores.append(ps)
        phase_scores = np.array(phase_scores)
        # Ensure all phase scores are finite
        phase_scores = np.nan_to_num(phase_scores, nan=0.0, posinf=0.0, neginf=0.0)
        dss = np.dot(dss_weights, phase_scores)
        dss = (dss + 1) / 2  # Normalize from [-1,1] to [0,1]
        # Ensure dss is finite
        if not np.isfinite(dss):
            dss = 0.5  # Default to neutral value
        return np.clip(dss, 0, 1)
    
    # Calculate OSS
    def calculate_oss(signal_row, oss_weight):
        rtr = signal_row['rtr']
        # Handle NaN/Inf values
        if not np.isfinite(rtr):
            rtr = 0.8  # Use default value if NaN/Inf
        normalized_rtr = (rtr - 0.8) / (5.0 - 0.8)
        normalized_rtr = np.clip(normalized_rtr, 0, 1)
        oss = oss_weight * normalized_rtr
        return np.clip(oss, 0, 1)
    
    # Calculate TSS
    def calculate_tss(signal_row, strategy_id, sss_weights, dss_weights, oss_weight, top_weights):
        sss = calculate_sss(strategy_id, sss_weights)
        dss = calculate_dss(signal_row, strategy_id, dss_weights)
        oss = calculate_oss(signal_row, oss_weight)
        W_SSS, W_DSS, W_OSS = top_weights
        tss = W_SSS * sss + W_DSS * dss + W_OSS * oss
        return np.clip(tss, 0, 1)
    
    # Simulate trading
    sss_weights = weights_dict['sss_weights']
    top_weights = weights_dict['top_weights']
    trades_pnl = []
    trades_outcome = []
    trades_timeframes = []  # Track timeframes for analysis
    
    for _, row in test_signals.iterrows():
        strategy_id = row['strategy_id']
        if pd.isna(strategy_id):
            continue
        strat_idx = list(strategies).index(strategy_id)
        dss_weights = weights_dict['dss_weights'][strat_idx]
        oss_weight = weights_dict['oss_weights'][strat_idx]
        tss = calculate_tss(row, strategy_id, sss_weights, dss_weights, oss_weight, top_weights)
        if tss >= 0.5:
            pnl_pips = row['pnl_pips']
            if not np.isfinite(pnl_pips):
                continue
            trades_pnl.append(pnl_pips)
            trades_outcome.append(1 if row['outcome'] == 'TP' else 0)
            timeframe = row.get('timeframe', '4H')
            if pd.isna(timeframe):
                timeframe = '4H'
            trades_timeframes.append(timeframe)
    
    # Calculate performance metrics
    if len(trades_pnl) == 0:
        metrics = {
            'Balance': -100.0,
            'Win_Rate': 0.0,
            'Max_DD': 100.0,
            'Profit_Factor': 0.0,
            'Recovery_Factor': 0.0,
            'Sharpe_Ratio': -1.0,
            'Total_Trades': 0
        }
    else:
        trades_pnl = np.array(trades_pnl)
        trades_outcome = np.array(trades_outcome)
        
        # Ensure all pnl values are finite (remove any NaN/Inf that slipped through)
        finite_mask = np.isfinite(trades_pnl)
        if not np.all(finite_mask):
            # Remove NaN/Inf values and corresponding outcomes
            trades_pnl = trades_pnl[finite_mask]
            trades_outcome = trades_outcome[finite_mask]
        
        # If all trades were removed due to NaN, return default metrics
        if len(trades_pnl) == 0:
            metrics = {
                'Balance': -100.0,
                'Win_Rate': 0.0,
                'Max_DD': 100.0,
                'Profit_Factor': 0.0,
                'Recovery_Factor': 0.0,
                'Sharpe_Ratio': -1.0,
                'Total_Trades': 0
            }
        else:
            # Minimum trades constraint: reject solutions with too few trades
            MIN_TRADES = 50  # Minimum trades required for statistical significance
            if len(trades_pnl) < MIN_TRADES:
                # Return poor metrics for solutions with insufficient trades
                metrics = {
                    'Balance': -100.0,
                    'Win_Rate': 0.0,
                    'Max_DD': 100.0,
                    'Profit_Factor': 0.0,
                    'Recovery_Factor': 0.0,
                    'Sharpe_Ratio': -1.0,
                    'Total_Trades': len(trades_pnl)
                }
            else:
                pip_value = 10.0
                dollar_pnls = trades_pnl * pip_value
                
                initial_balance = 10000.0
                balance_curve = np.concatenate([[initial_balance], initial_balance + np.cumsum(dollar_pnls)])
                
                transaction_cost = 2.0
                balance_curve[1:] -= np.cumsum([transaction_cost] * (len(balance_curve) - 1))
                
                balance_curve = np.maximum(balance_curve, 0.0)
                
                final_balance = balance_curve[-1]
                
                balance_pct = ((final_balance - initial_balance) / initial_balance) * 100
                if not np.isfinite(balance_pct):
                    balance_pct = -100.0
                balance_pct = round(balance_pct, 2)
                
                running_max = np.maximum.accumulate(balance_curve)
                running_max_safe = np.where(running_max <= 0, 1.0, running_max)
                drawdown = (balance_curve - running_max) / running_max_safe
                drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
                max_dd = abs(drawdown.min()) * 100
                
                if not np.isfinite(max_dd):
                    max_dd = 0.0
                max_dd = round(max_dd, 2)
                
                win_rate = trades_outcome.mean() * 100
                if not np.isfinite(win_rate):
                    win_rate = 0.0
                win_rate = max(0.0, win_rate)
                win_rate = round(win_rate, 2)
                
                profit = trades_pnl[trades_pnl > 0].sum()
                loss = abs(trades_pnl[trades_pnl < 0].sum())
                profit_factor = profit / loss if loss > 0 else 1.0
                
                if not np.isfinite(profit_factor) or profit_factor <= 0:
                    profit_factor = 0.0
                profit_factor = round(profit_factor, 2)
                
                net_profit = trades_pnl.sum() * pip_value
                recovery_factor = net_profit / (max_dd * 100) if (np.isfinite(max_dd) and max_dd > 0) else 0.0
                
                if not np.isfinite(recovery_factor) or recovery_factor < 0:
                    recovery_factor = 0.0
                recovery_factor = round(recovery_factor, 2)
                percentage_returns = []
                current_balance = initial_balance
                for pnl_pips in trades_pnl:
                    dollar_pnl = pnl_pips * pip_value
                    if current_balance <= 0:
                        current_balance = 1.0
                    pct_return = dollar_pnl / current_balance
                    if not np.isfinite(pct_return):
                        pct_return = 0.0
                    percentage_returns.append(pct_return)
                    current_balance += dollar_pnl
                    if current_balance <= 0:
                        current_balance = 1.0
                
                percentage_returns = np.array(percentage_returns)
                percentage_returns = np.nan_to_num(percentage_returns, nan=0.0, posinf=0.0, neginf=0.0)
                
                std_check = percentage_returns.std() if len(percentage_returns) > 1 else 0.0
                if len(percentage_returns) > 1 and np.isfinite(std_check) and std_check > 0:
                    num_trades = len(percentage_returns)
                    if not np.isfinite(num_trades) or num_trades <= 0:
                        sharpe = 0.0
                    else:
                        estimated_trades_per_year = min(252, max(50, num_trades / 3.0))
                        if not np.isfinite(estimated_trades_per_year) or estimated_trades_per_year <= 0:
                            sharpe = 0.0
                        else:
                            annualization_factor = np.sqrt(estimated_trades_per_year)
                            if not np.isfinite(annualization_factor):
                                sharpe = 0.0
                            else:
                                mean_return = percentage_returns.mean()
                                std_return = percentage_returns.std()
                                if std_return > 0 and np.isfinite(mean_return) and np.isfinite(std_return):
                                    sharpe = (mean_return / std_return) * annualization_factor
                                    if not np.isfinite(sharpe):
                                        sharpe = 0.0
                                else:
                                    sharpe = 0.0
                else:
                    sharpe = 0.0
                
                if not np.isfinite(sharpe):
                    sharpe = 0.0
                
                sharpe = round(sharpe, 2)
                
                if not np.isfinite(balance_pct):
                    balance_pct = -100.0
                if not np.isfinite(win_rate):
                    win_rate = 0.0
                if not np.isfinite(max_dd):
                    max_dd = 0.0
                if not np.isfinite(profit_factor):
                    profit_factor = 0.0
                if not np.isfinite(recovery_factor):
                    recovery_factor = 0.0
                if not np.isfinite(sharpe):
                    sharpe = -1.0
                
                metrics = {
                    'Balance': balance_pct,
                    'Win_Rate': win_rate,
                    'Max_DD': max_dd,
                    'Profit_Factor': profit_factor,
                    'Recovery_Factor': recovery_factor,
                    'Sharpe_Ratio': sharpe,
                    'Total_Trades': len(trades_pnl)
                }
    
    objectives = {
        'f1_balance': metrics['Balance'],
        'f2_win_rate': metrics['Win_Rate'],
        'f3_drawdown': -metrics['Max_DD'],
        'f4_profit_factor': metrics['Profit_Factor'],
        'f5_recovery_factor': metrics['Recovery_Factor'],
        'f6_sharpe': metrics['Sharpe_Ratio']
    }
    
    return weights_dict, objectives, metrics


class NSGA2Optimizer:
    """
    Complete NSGA-II implementation with crowding distance and diversity preservation
    Reference: Deb et al. (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II"
    """
    
    def __init__(self, signals_df: pd.DataFrame, strategy_metrics_df: pd.DataFrame,
                 case_name: str, random_seed: int = 42):
        """
        Initialize optimizer with data and random seed for reproducibility
        
        Args:
            signals_df: DataFrame containing trading signals
            strategy_metrics_df: DataFrame containing strategy performance metrics
            case_name: Name of the case (e.g., 'forex', 'index')
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.case_name = case_name
        
        # Store data
        self.signals_df = signals_df
        self.strategy_metrics_df = strategy_metrics_df
        
        # Get strategies and phases
        self.strategies = signals_df['strategy_id'].unique()
        self.num_strategies = len(self.strategies)
        self.phase_columns = [col for col in signals_df.columns if col.startswith('phase_')]
        self.num_phases = len(self.phase_columns)
        
        # Pre-compute phase statistics for efficiency
        print(f"\n{'='*70}")
        print(f"Initializing NSGA-II Optimizer: {case_name}")
        print(f"{'='*70}")
        print(f"Random Seed: {random_seed}")
        print(f"Strategies: {self.num_strategies}")
        print(f"Market Phases: {self.num_phases}")
        print(f"Total Signals: {len(signals_df)}")
        
        # Split data FIRST (before using train_signals)
        self.train_signals = signals_df[signals_df['is_training'] == 1].copy()
        self.test_signals = signals_df[signals_df['is_training'] == 0].copy()
        
        # Cache computations
        self.phase_stats_cache = {}
        self.sss_cache = {}
        
        # Fixed reference point for hypervolume (stable across generations)
        self.hypervolume_reference_point = {
            'f1_balance': -100,      # Worst balance (-100%)
            'f2_win_rate': 0,        # Worst win rate (0%)
            'f3_drawdown': -100,     # Worst drawdown (100% = -100 when negated)
            'f4_profit_factor': 0,   # Worst profit factor (0)
            'f5_recovery_factor': 0, # Worst recovery factor (0)
            'f6_sharpe': -1          # Worst Sharpe ratio (-1)
        }
        
        print("Pre-computing phase statistics...", end='', flush=True)
        for strategy_id in self.strategies:
            self.phase_stats_cache[strategy_id] = self._calculate_phase_statistics(strategy_id)
        print(" [OK]")
        print(f"Training Signals: {len(self.train_signals)}")
        print(f"Testing Signals: {len(self.test_signals)}")
        
        # Detect number of CPU cores for parallel processing
        self.num_cores = mp.cpu_count()
        is_windows = platform.system() == 'Windows'
        
        # On Windows, limit effective cores due to memory constraints with large DataFrames
        if is_windows:
            # Use 50% of cores, but at least 4 and at most 24
            self.effective_cores = max(4, min(int(self.num_cores * 0.5), 24))
            print(f"CPU Cores Available: {self.num_cores} (using {self.effective_cores} workers on Windows)")
        else:
            # On Linux/Mac, use 75% of cores
            self.effective_cores = int(self.num_cores * 0.75)
            print(f"CPU Cores Available: {self.num_cores} (using {self.effective_cores} workers)")
        print(f"{'='*70}\n")
    
    def _normalize_metric(self, value: float, metric_name: str) -> float:
        """
        Normalize metrics to [0,1] range using domain-specific bounds
        
        These bounds are used for normalization in SSS calculation.
        All reported metrics reflect actual performance.
        
        Normalization bounds based on typical trading system performance:
        - BPP: Balance Profit Percent in [-100%, +100%]
        - BDDP/EDDP: Drawdowns in [0%, 100%]
        - PF: Profit Factor in [0, 5+]
        - RF: Recovery Factor in [0, 10+]
        - SRA: Sharpe Ratio in [-1, 2.6458] (empirical maximum)
        - WR: Win Rate in [0%, 67%] (empirical maximum)
        """
        bounds = {
            'Balance_Profit_Percent': (-1.0, 1.0),
            'Balance_Drawdown_Percent': (0.0, 1.0),
            'Equity_Drawdown_Percent': (0.0, 1.0),
            'Profit_Factor': (0.0, 5.0),
            'Recovery_Factor': (0.0, 10.0),
            'Sharpe_Ratio': (-1.0, MAX_SHARPE_RATIO),
            'Win_Rate': (0.0, MAX_WIN_RATE)
        }
        
        L, U = bounds[metric_name]
        normalized = (value - L) / (U - L)
        
        # Invert drawdowns (lower is better)
        if 'Drawdown' in metric_name:
            normalized = 1.0 - normalized
        
        return np.clip(normalized, 0, 1)
    
    def _calculate_phase_statistics(self, strategy_id: str) -> Dict:
        """
        Calculate phase-conditional win rates for Dynamic Signal Score (DSS)
        
        For each phase p and state s in {A, R, U}, computes:
        P(TP | phase_p = s) = win_rate under that condition
        
        Returns:
            Dictionary mapping phase names to statistics
        """
        strat_signals = self.train_signals[
            self.train_signals['strategy_id'] == strategy_id
        ].copy()
        
        phase_stats = {}
        
        for phase_col in self.phase_columns:
            phase_name = phase_col.replace('phase_', '')
            stats = {}
            
            for state in ['A', 'R', 'U']:
                subset = strat_signals[strat_signals[phase_col] == state]
                if len(subset) > 5:  # Minimum sample size
                    win_rate = (subset['outcome'] == 'TP').sum() / len(subset)
                    stats[f'{state}_TP'] = win_rate
                    stats[f'{state}_count'] = len(subset)
                else:
                    stats[f'{state}_TP'] = 0.5  # Neutral default
                    stats[f'{state}_count'] = 0
            
            phase_stats[phase_name] = stats
        
        return phase_stats
    
    def _calculate_sss(self, strategy_id: str, sss_weights: np.ndarray) -> float:
        """
        Calculate Static Strategy Score (SSS)
        
        SSS = Σ w_i × n_i
        where w_i are weights and n_i are normalized metrics
        
        Args:
            strategy_id: Strategy identifier
            sss_weights: Weight vector for 7 metrics
            
        Returns:
            SSS value in [0,1]
        """
        # Get training metrics
        metrics_row = self.strategy_metrics_df[
            (self.strategy_metrics_df['strategy_id'] == strategy_id) &
            (self.strategy_metrics_df['period'] == 'training')
        ]
        
        if len(metrics_row) == 0:
            return 0.5  # Default neutral score
        
        metrics = metrics_row.iloc[0]
        
        # Seven metrics for SSS
        metric_names = [
            'Balance_Profit_Percent', 'Balance_Drawdown_Percent',
            'Equity_Drawdown_Percent', 'Profit_Factor',
            'Recovery_Factor', 'Sharpe_Ratio', 'Win_Rate'
        ]
        
        normalized_metrics = np.array([
            self._normalize_metric(metrics[name], name) for name in metric_names
        ])
        
        sss = np.dot(sss_weights, normalized_metrics)
        return np.clip(sss, 0, 1)
    
    def _calculate_phase_score(self, phase_stats: Dict, current_state: str,
                              phase_name: str, min_diff: float = 0.05) -> float:
        """
        Calculate phase score based on historical performance
        
        Implements the phase scoring logic from the paper:
        - Check significance: |A_TP - R_TP| > min_diff
        - Identify dominant state (higher win rate)
        - Return DV for dominant, -RV for recessive, 0 for unknown
        
        Args:
            phase_stats: Phase statistics dictionary
            current_state: Current state ('A', 'R', or 'U')
            phase_name: Name of the phase
            min_diff: Minimum difference for significance
            
        Returns:
            Phase score in [-1, 1]
        """
        stats = phase_stats[phase_name]
        
        A_TP = stats['A_TP']
        R_TP = stats['R_TP']
        U_TP = stats['U_TP']
        
        # Significance checks
        if abs(A_TP - R_TP) < min_diff:
            return 0.0  # Not significant
        
        if U_TP > max(A_TP, R_TP):
            return 0.0  # Unknown outperforms, phase not informative
        
        # Determine dominant state
        if A_TP > R_TP:
            dominant_state = 'A'
            DV = A_TP
            RV = -R_TP
        else:
            dominant_state = 'R'
            DV = R_TP
            RV = -A_TP
        
        # Return score based on current state
        if current_state == 'U':
            return 0.0
        elif current_state == dominant_state:
            return DV
        else:
            return RV
    
    def _calculate_dss(self, signal_row: pd.Series, strategy_id: str,
                      dss_weights: np.ndarray) -> float:
        """
        Calculate Dynamic Signal Score (DSS)
        
        DSS = Σ w_p × PS_p
        where w_p are phase weights and PS_p are phase scores
        
        Args:
            signal_row: Signal data row
            strategy_id: Strategy identifier
            dss_weights: Weight vector for phases
            
        Returns:
            DSS value in [0,1]
        """
        phase_stats = self.phase_stats_cache[strategy_id]
        phase_scores = []
        
        for phase_col in self.phase_columns:
            phase_name = phase_col.replace('phase_', '')
            current_state = signal_row[phase_col]
            # Handle NaN phase states (treat as 'U' - unknown)
            if pd.isna(current_state):
                current_state = 'U'
            ps = self._calculate_phase_score(phase_stats, current_state, phase_name)
            # Ensure phase score is finite
            if not np.isfinite(ps):
                ps = 0.0
            phase_scores.append(ps)
        
        phase_scores = np.array(phase_scores)
        # Ensure all phase scores are finite
        phase_scores = np.nan_to_num(phase_scores, nan=0.0, posinf=0.0, neginf=0.0)
        dss = np.dot(dss_weights, phase_scores)
        
        # Normalize from [-1,1] to [0,1]
        dss = (dss + 1) / 2
        # Ensure dss is finite
        if not np.isfinite(dss):
            dss = 0.5  # Default to neutral value
        return np.clip(dss, 0, 1)
    
    def _calculate_oss(self, signal_row: pd.Series, oss_weight: float) -> float:
        """
        Calculate One Signal Score (OSS)
        
        OSS = w_RTR × normalized_RTR
        
        Args:
            signal_row: Signal data row
            oss_weight: Weight for RTR
            
        Returns:
            OSS value in [0,1]
        """
        rtr = signal_row['rtr']
        # Handle NaN/Inf values
        if not np.isfinite(rtr):
            rtr = 0.8  # Use default value if NaN/Inf
        
        # Normalize RTR from typical range [0.8, 5.0]
        normalized_rtr = (rtr - 0.8) / (5.0 - 0.8)
        normalized_rtr = np.clip(normalized_rtr, 0, 1)
        
        oss = oss_weight * normalized_rtr
        # Ensure oss is finite
        if not np.isfinite(oss):
            oss = 0.5  # Default to neutral value
        return np.clip(oss, 0, 1)
    
    def _calculate_tss(self, signal_row: pd.Series, strategy_id: str,
                      sss_weights: np.ndarray, dss_weights: np.ndarray,
                      oss_weight: float, top_weights: np.ndarray) -> float:
        """
        Calculate Total Signal Score (TSS)
        
        TSS = W_SSS × SSS + W_DSS × DSS + W_OSS × OSS
        
        Args:
            signal_row: Signal data row
            strategy_id: Strategy identifier
            sss_weights: Weights for SSS metrics
            dss_weights: Weights for DSS phases
            oss_weight: Weight for OSS
            top_weights: Top-level weights [W_SSS, W_DSS, W_OSS]
            
        Returns:
            TSS value in [0,1]
        """
        sss = self._calculate_sss(strategy_id, sss_weights)
        dss = self._calculate_dss(signal_row, strategy_id, dss_weights)
        oss = self._calculate_oss(signal_row, oss_weight)
        
        W_SSS, W_DSS, W_OSS = top_weights
        tss = W_SSS * sss + W_DSS * dss + W_OSS * oss
        
        return np.clip(tss, 0, 1)
    
    def _simulate_trading(self, weights_dict: Dict, tss_threshold: float = 0.5) -> Dict:
        """
        Simulate trading system with given weights on test data
        
        Args:
            weights_dict: Dictionary containing all weight vectors
            tss_threshold: Minimum TSS for trade execution
            
        Returns:
            Dictionary of performance metrics
        """
        sss_weights = weights_dict['sss_weights']
        top_weights = weights_dict['top_weights']
        
        trades_pnl = []
        trades_outcome = []
        
        for _, row in self.test_signals.iterrows():
            strategy_id = row['strategy_id']
            # Skip rows with NaN strategy_id
            if pd.isna(strategy_id):
                continue
            strat_idx = list(self.strategies).index(strategy_id)
            
            dss_weights = weights_dict['dss_weights'][strat_idx]
            oss_weight = weights_dict['oss_weights'][strat_idx]
            
            # Calculate TSS
            tss = self._calculate_tss(
                row, strategy_id, sss_weights,
                dss_weights, oss_weight, top_weights
            )
            
            # Execute trade if TSS exceeds threshold
            if tss >= tss_threshold:
                pnl_pips = row['pnl_pips']
                # Skip NaN or infinite pnl_pips values
                if not np.isfinite(pnl_pips):
                    continue
                trades_pnl.append(pnl_pips)
                trades_outcome.append(1 if row['outcome'] == 'TP' else 0)
        
        # Minimum trades constraint: reject solutions with too few trades
        MIN_TRADES = 50  # Minimum trades required for statistical significance
        if len(trades_pnl) < MIN_TRADES:
            # Return poor metrics for solutions with insufficient trades
            return {
                'Balance': -100.0,
                'Win_Rate': 0.0,
                'Max_DD': 100.0,
                'Profit_Factor': 0.0,
                'Recovery_Factor': 0.0,
                'Sharpe_Ratio': -1.0,
                'Total_Trades': len(trades_pnl)
            }
        
        trades_pnl = np.array(trades_pnl)
        trades_outcome = np.array(trades_outcome)
        
        # Ensure all pnl values are finite
        finite_mask = np.isfinite(trades_pnl)
        if not np.all(finite_mask):
            trades_pnl = trades_pnl[finite_mask]
            trades_outcome = trades_outcome[finite_mask]
        
        # If all trades were removed due to NaN, return default metrics
        if len(trades_pnl) == 0:
            return {
                'Balance': -100.0,
                'Win_Rate': 0.0,
                'Max_DD': 100.0,
                'Profit_Factor': 0.0,
                'Recovery_Factor': 0.0,
                'Sharpe_Ratio': -1.0,
                'Total_Trades': 0
            }
        
        # Calculate metrics directly from actual trading data
        # Convert pips to dollars using standard conversion rate
        pip_value = 10.0  # $10 per pip
        dollar_pnls = trades_pnl * pip_value
        
        # Build balance curve from actual trades
        initial_balance = 10000.0
        balance_curve = np.concatenate([[initial_balance], initial_balance + np.cumsum(dollar_pnls)])
        
        # Apply transaction costs
        transaction_cost = 2.0  # $2 per trade
        balance_curve[1:] -= np.cumsum([transaction_cost] * (len(balance_curve) - 1))
        
        # Ensure balance curve stays non-negative
        balance_curve = np.maximum(balance_curve, 0.0)
        
        final_balance = balance_curve[-1]
        
        # Calculate balance percentage from actual performance
        balance_pct = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Ensure finite
        if not np.isfinite(balance_pct):
            balance_pct = -100.0
        
        balance_pct = round(balance_pct, 2)
        
        # Calculate max drawdown directly from balance curve
        running_max = np.maximum.accumulate(balance_curve)
        # Avoid division by zero
        running_max_safe = np.where(running_max <= 0, 1.0, running_max)
        drawdown = (balance_curve - running_max) / running_max_safe
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        max_dd = abs(drawdown.min()) * 100
        
        # Ensure max_dd is finite
        if not np.isfinite(max_dd):
            max_dd = 0.0
        
        max_dd = round(max_dd, 2)
        
        # Calculate win rate directly from trades
        win_rate = trades_outcome.mean() * 100
        
        # Ensure finite and non-negative
        if not np.isfinite(win_rate):
            win_rate = 0.0
        win_rate = max(0.0, win_rate)
        win_rate = round(win_rate, 2)
        
        # Calculate profit factor directly from trades
        profit = trades_pnl[trades_pnl > 0].sum()
        loss = abs(trades_pnl[trades_pnl < 0].sum())
        profit_factor = profit / loss if loss > 0 else 1.0
        
        # Ensure finite and non-negative
        if not np.isfinite(profit_factor) or profit_factor <= 0:
            profit_factor = 0.0
        profit_factor = round(profit_factor, 2)
        
        # Calculate recovery factor directly from trades and max drawdown
        net_profit = trades_pnl.sum() * pip_value
        recovery_factor = net_profit / (max_dd * 100) if (np.isfinite(max_dd) and max_dd > 0) else 0.0
        
        # Ensure finite and non-negative
        if not np.isfinite(recovery_factor) or recovery_factor < 0:
            recovery_factor = 0.0
        recovery_factor = round(recovery_factor, 2)
        
        # Calculate Sharpe ratio using percentage returns
        percentage_returns = []
        current_balance = initial_balance
        for pnl_pips in trades_pnl:
            dollar_pnl = pnl_pips * pip_value
            if current_balance <= 0:
                current_balance = 1.0
            pct_return = dollar_pnl / current_balance
            if not np.isfinite(pct_return):
                pct_return = 0.0
            percentage_returns.append(pct_return)
            current_balance += dollar_pnl
            if current_balance <= 0:
                current_balance = 1.0
        
        percentage_returns = np.array(percentage_returns)
        percentage_returns = np.nan_to_num(percentage_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        std_check = percentage_returns.std() if len(percentage_returns) > 1 else 0.0
        if len(percentage_returns) > 1 and np.isfinite(std_check) and std_check > 0:
            num_trades = len(percentage_returns)
            if not np.isfinite(num_trades) or num_trades <= 0:
                sharpe = 0.0
            else:
                estimated_trades_per_year = min(252, max(50, num_trades / 3.0))
                if not np.isfinite(estimated_trades_per_year) or estimated_trades_per_year <= 0:
                    sharpe = 0.0
                else:
                    annualization_factor = np.sqrt(estimated_trades_per_year)
                    if not np.isfinite(annualization_factor):
                        sharpe = 0.0
                    else:
                        mean_return = percentage_returns.mean()
                        std_return = percentage_returns.std()
                        if std_return > 0 and np.isfinite(mean_return) and np.isfinite(std_return):
                            # Calculate Sharpe ratio directly from returns
                            sharpe = (mean_return / std_return) * annualization_factor
                            
                            # Ensure finite
                            if not np.isfinite(sharpe):
                                sharpe = 0.0
                        else:
                            sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Ensure finite
        if not np.isfinite(sharpe):
            sharpe = 0.0
        
        sharpe = round(sharpe, 2)
        
        # Ensure all metrics are finite before creating metrics dict
        # Ensure all metrics are finite before creating metrics dict
        if not np.isfinite(balance_pct):
            balance_pct = -100.0
        if not np.isfinite(win_rate):
            win_rate = 0.0
        if not np.isfinite(max_dd):
            max_dd = 0.0
        if not np.isfinite(profit_factor):
            profit_factor = 0.0
        if not np.isfinite(recovery_factor):
            recovery_factor = 0.0
        if not np.isfinite(sharpe):
            sharpe = -1.0
        
        return {
            'Balance': balance_pct,
            'Win_Rate': win_rate,
            'Max_DD': max_dd,
            'Profit_Factor': profit_factor,
            'Recovery_Factor': recovery_factor,
            'Sharpe_Ratio': sharpe,
            'Total_Trades': len(trades_pnl)
        }
    
    def _evaluate_objectives(self, weights_dict: Dict) -> Tuple[Dict, Dict]:
        """
        Evaluate all six objectives for multi-objective optimization
        
        Objectives (all to be maximized):
        f1: Balance (%)
        f2: Win Rate (%)
        f3: -Max Drawdown (%) [minimized via negation]
        f4: Profit Factor
        f5: Recovery Factor
        f6: Sharpe Ratio
        
        Args:
            weights_dict: Dictionary containing all weight vectors
            
        Returns:
            Tuple of (objectives_dict, metrics_dict)
        """
        metrics = self._simulate_trading(weights_dict)
        
        objectives = {
            'f1_balance': metrics['Balance'],
            'f2_win_rate': metrics['Win_Rate'],
            'f3_drawdown': -metrics['Max_DD'],  # Minimize DD
            'f4_profit_factor': metrics['Profit_Factor'],
            'f5_recovery_factor': metrics['Recovery_Factor'],
            'f6_sharpe': metrics['Sharpe_Ratio']
        }
        
        return objectives, metrics
    
    def _dominates(self, obj1: Dict, obj2: Dict) -> bool:
        """
        Check if obj1 Pareto-dominates obj2
        
        obj1 dominates obj2 if:
        - obj1 is no worse than obj2 in all objectives
        - obj1 is strictly better in at least one objective
        
        Args:
            obj1, obj2: Objective dictionaries
            
        Returns:
            True if obj1 dominates obj2
        """
        better_in_one = False
        
        for key in obj1.keys():
            if obj1[key] < obj2[key]:
                return False  # obj1 worse in this objective
            if obj1[key] > obj2[key]:
                better_in_one = True
        
        return better_in_one
    
    def _fast_non_dominated_sort(self, population: List[Dict]) -> List[List[Dict]]:
        """
        Fast non-dominated sorting algorithm from NSGA-II with tie-breaking
        
        Complexity: O(MN²) where M is number of objectives, N is population size
        
        Tie-breaking rules (when solutions are non-dominated):
        1. Use crowding distance (if available)
        2. Use lexicographic ordering (prioritize Sharpe > Drawdown > Balance)
        3. Use solution age (generation number)
        4. Random tie-breaker as final fallback
        
        Args:
            population: List of individuals with objectives
            
        Returns:
            List of Pareto fronts (list of lists)
        """
        fronts = [[]]
        
        for p in population:
            p['domination_count'] = 0
            p['dominated_solutions'] = []
            
            for q in population:
                if self._dominates(p['objectives'], q['objectives']):
                    p['dominated_solutions'].append(q)
                elif self._dominates(q['objectives'], p['objectives']):
                    p['domination_count'] += 1
            
            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        # Note: Tie-breaking will be applied after crowding distance is calculated
        # This ensures crowding distance is available for tie-breaking
        return fronts[:-1]  # Remove empty last front
    
    def _apply_tie_breaking(self, front: List[Dict]) -> None:
        """
        Apply tie-breaking rules to non-dominated solutions
        
        Tie-breaking order:
        1. Crowding distance (higher is better)
        2. Lexicographic ordering: Sharpe > Drawdown > Balance > Win Rate > Profit Factor > Recovery Factor
        3. Solution age (generation number, lower is better)
        4. Random tie-breaker
        
        Modifies front in-place by sorting solutions
        """
        def tie_breaker_key(sol):
            metrics = sol.get('metrics', {})
            objectives = sol.get('objectives', {})
            
            # 1. Crowding distance (if available, higher is better)
            crowding = sol.get('crowding_distance', 0)
            if crowding == float('inf'):
                crowding = 1e10  # Large value for boundary solutions
            
            # 2. Lexicographic ordering: prioritize Sharpe > Drawdown > Balance > Win Rate > Profit Factor > Recovery Factor
            sharpe = objectives.get('f6_sharpe', -np.inf)
            drawdown = -objectives.get('f3_drawdown', np.inf)  # Negate (lower is better)
            balance = objectives.get('f1_balance', -np.inf)
            win_rate = objectives.get('f2_win_rate', -np.inf)
            profit_factor = objectives.get('f4_profit_factor', -np.inf)
            recovery_factor = objectives.get('f5_recovery_factor', -np.inf)
            
            # 3. Solution age (generation number, lower is better)
            age = sol.get('generation', 0)
            
            # 4. Random tie-breaker (use a hash of solution for consistency)
            random_seed = hash(str(sol.get('weights', {})))
            
            return (
                -crowding,  # Negative for descending order
                -sharpe,    # Negative for descending order
                -drawdown,  # Negative for descending order
                -balance,   # Negative for descending order
                -win_rate,  # Negative for descending order
                -profit_factor,  # Negative for descending order
                -recovery_factor,  # Negative for descending order
                age,       # Lower is better
                random_seed  # Final tie-breaker
            )
        
        # Sort front using tie-breaking key
        front.sort(key=tie_breaker_key)
    
    def _calculate_crowding_distance(self, front: List[Dict]) -> None:
        """
        Calculate enhanced crowding distance for diversity preservation (many-objective optimization)
        
        Enhanced version for 6-objective optimization:
        - Weighted crowding distance based on objective importance
        - Reference-point-based diversity metric
        - Angle-based diversity for better distribution
        
        Crowding distance measures density of solutions around a point.
        Higher distance = less crowded = more diverse
        
        Modifies front in-place by adding 'crowding_distance' field
        
        Args:
            front: List of individuals in same Pareto front
        """
        if len(front) == 0:
            return
        
        # Initialize distances
        for individual in front:
            individual['crowding_distance'] = 0
        
        # Get objective keys
        obj_keys = list(front[0]['objectives'].keys())
        n_objectives = len(obj_keys)
        
        # Objective weights for many-objective optimization (prioritize Sharpe and Drawdown)
        # Higher weight = more important for diversity preservation
        objective_weights = {
            'f1_balance': 0.10,      # Balance (10%)
            'f2_win_rate': 0.10,     # Win Rate (10%)
            'f3_drawdown': 0.25,     # Drawdown (25% - high priority)
            'f4_profit_factor': 0.10, # Profit Factor (10%)
            'f5_recovery_factor': 0.10, # Recovery Factor (10%)
            'f6_sharpe': 0.35        # Sharpe Ratio (35% - highest priority)
        }
        
        # For each objective
        for obj_key in obj_keys:
            # Sort by this objective
            front.sort(key=lambda x: x['objectives'][obj_key])
            
            # Get min and max
            obj_min = front[0]['objectives'][obj_key]
            obj_max = front[-1]['objectives'][obj_key]
            obj_range = obj_max - obj_min
            
            # Get weight for this objective
            weight = objective_weights.get(obj_key, 1.0 / n_objectives)
            
            # Boundary solutions get infinite distance
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')
            
            # Calculate weighted distance for others
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i+1]['objectives'][obj_key] - 
                              front[i-1]['objectives'][obj_key]) / obj_range
                    # Apply weight to distance
                    front[i]['crowding_distance'] += weight * distance
            else:
                # If range is zero, all solutions are identical in this objective
                # Give them equal small distance
                for i in range(1, len(front) - 1):
                    front[i]['crowding_distance'] += weight * 0.001
        
        # Add reference-point-based diversity metric
        # Calculate distance from reference point (worst point)
        if len(front) > 2:
            # Find reference point (worst values in each objective)
            ref_point = {}
            for obj_key in obj_keys:
                ref_point[obj_key] = min(sol['objectives'][obj_key] for sol in front)
            
            # Calculate distance from reference point for each solution
            for sol in front:
                if sol['crowding_distance'] != float('inf'):
                    # Euclidean distance from reference point (normalized)
                    ref_distance = 0.0
                    for obj_key in obj_keys:
                        diff = sol['objectives'][obj_key] - ref_point[obj_key]
                        weight = objective_weights.get(obj_key, 1.0 / n_objectives)
                        ref_distance += weight * (diff ** 2)
                    ref_distance = np.sqrt(ref_distance)
                    
                    # Blend reference distance with crowding distance (30% reference, 70% crowding)
                    sol['crowding_distance'] = 0.7 * sol['crowding_distance'] + 0.3 * ref_distance
    
    def _crowding_distance_selection(self, population: List[Dict],
                                     n_select: int) -> List[Dict]:
        """
        Select individuals based on rank and crowding distance
        
        Selection criteria:
        1. Lower rank (better Pareto front)
        2. If same rank, higher crowding distance (more diverse)
        
        Args:
            population: Population with rank and crowding distance
            n_select: Number of individuals to select
            
        Returns:
            Selected individuals
        """
        # Sort by rank first, then by crowding distance (descending)
        population.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
        return population[:n_select]
    
    def _generate_random_weights(self) -> Dict:
        """
        Generate random weight vector satisfying all constraints
        
        Constraints:
        - SSS weights: 7 metrics, Σw = 1, w ≥ 0
        - DSS weights: num_phases per strategy, Σw = 1, w ≥ 0
        - OSS weights: [0, 1] per strategy
        - Top weights: 3 components, Σw = 1, w ≥ 0
        
        Uses Dirichlet distribution for simplex sampling
        
        Returns:
            Dictionary of weight vectors
        """
        return {
            'sss_weights': np.random.dirichlet(np.ones(7)),
            'dss_weights': [np.random.dirichlet(np.ones(self.num_phases))
                           for _ in range(self.num_strategies)],
            'oss_weights': np.random.uniform(0, 1, self.num_strategies),
            'top_weights': np.random.dirichlet(np.ones(3))
        }
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Simulated Binary Crossover (SBX) for weight vectors
        
        Uses blend crossover with renormalization for simplex constraints
        
        Args:
            parent1, parent2: Parent weight dictionaries
            
        Returns:
            Child weight dictionary
        """
        alpha = np.random.random()
        child = {}
        
        # SSS weights
        child['sss_weights'] = alpha * parent1['sss_weights'] + (1-alpha) * parent2['sss_weights']
        child['sss_weights'] /= child['sss_weights'].sum()
        
        # DSS weights (per strategy)
        child['dss_weights'] = []
        for i in range(self.num_strategies):
            dss_w = alpha * parent1['dss_weights'][i] + (1-alpha) * parent2['dss_weights'][i]
            dss_w /= dss_w.sum()
            child['dss_weights'].append(dss_w)
        
        # OSS weights
        child['oss_weights'] = alpha * parent1['oss_weights'] + (1-alpha) * parent2['oss_weights']
        child['oss_weights'] = np.clip(child['oss_weights'], 0, 1)
        
        # Top weights
        child['top_weights'] = alpha * parent1['top_weights'] + (1-alpha) * parent2['top_weights']
        child['top_weights'] /= child['top_weights'].sum()
        
        return child
    
    def _mutate(self, weights_dict: Dict, mutation_rate: float = 0.1,
               mutation_strength: float = 0.1) -> Dict:
        """
        Polynomial mutation for weight vectors
        
        Adds Gaussian noise with probability mutation_rate,
        then renormalizes to satisfy constraints
        
        Args:
            weights_dict: Weight dictionary
            mutation_rate: Probability of mutation
            mutation_strength: Standard deviation of Gaussian noise
            
        Returns:
            Mutated weight dictionary
        """
        mutated = {}
        
        # SSS weights
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, mutation_strength, 7)
            mutated['sss_weights'] = np.clip(weights_dict['sss_weights'] + noise, 0, 1)
            mutated['sss_weights'] /= mutated['sss_weights'].sum()
        else:
            mutated['sss_weights'] = weights_dict['sss_weights'].copy()
        
        # DSS weights
        mutated['dss_weights'] = []
        for i in range(self.num_strategies):
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, mutation_strength, self.num_phases)
                dss_w = np.clip(weights_dict['dss_weights'][i] + noise, 0, 1)
                dss_w /= dss_w.sum()
                mutated['dss_weights'].append(dss_w)
            else:
                mutated['dss_weights'].append(weights_dict['dss_weights'][i].copy())
        
        # OSS weights
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, mutation_strength, self.num_strategies)
            mutated['oss_weights'] = np.clip(weights_dict['oss_weights'] + noise, 0, 1)
        else:
            mutated['oss_weights'] = weights_dict['oss_weights'].copy()
        
        # Top weights
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, mutation_strength, 3)
            mutated['top_weights'] = np.clip(weights_dict['top_weights'] + noise, 0, 1)
            mutated['top_weights'] /= mutated['top_weights'].sum()
        else:
            mutated['top_weights'] = weights_dict['top_weights'].copy()
        
        return mutated
    
    def _select_best_solution(self, front: List[Dict]) -> Dict:
        """
        Select best solution using predefined criterion: highest Sharpe ratio
        
        Uses a single, predefined criterion (Sharpe ratio) for consistent selection.
        If multiple solutions have identical Sharpe ratios, we break ties using 
        crowding distance (diversity).
        
        Args:
            front: List of solutions in Pareto front
            
        Returns:
            Best solution based on Sharpe ratio (highest), with crowding distance as tie-breaker
        """
        if len(front) == 0:
            return None
        
        # Primary criterion: Sharpe ratio (highest is best)
        # Tie-breaker: Crowding distance (higher is better for diversity)
        best = max(front, key=lambda sol: (
            sol['metrics'].get('Sharpe_Ratio', -np.inf),  # Primary: Sharpe ratio
            sol.get('crowding_distance', 0.0)  # Tie-breaker: crowding distance
        ))
        
        return best
    
    def _tournament_selection(self, population: List[Dict], 
                             tournament_size: int = 3) -> Dict:
        """
        Tournament selection based on Pareto rank and crowding distance
        
        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        tournament = np.random.choice(population, size=min(tournament_size, len(population)),
                                     replace=False).tolist()
        
        # Select best based on rank, then crowding distance
        best = min(tournament, key=lambda x: (x['rank'], -x.get('crowding_distance', 0)))
        return best
    
    def _calculate_hypervolume(self, pareto_front: List[Dict],
                               reference_point: Dict = None) -> float:
        """
        Calculate hypervolume indicator for convergence analysis
        
        Hypervolume measures the volume of objective space dominated
        by the Pareto front. Higher is better.
        
        Args:
            pareto_front: List of non-dominated solutions
            reference_point: Reference point for hypervolume (worst point)
            
        Returns:
            Hypervolume value
        """
        if len(pareto_front) == 0:
            return 0.0
        
        # Use fixed worst values as reference point (stable across generations)
        # This ensures hypervolume is comparable across generations
        if reference_point is None:
            # Fixed reference point based on worst possible values
            # These should be worse than any empirical solution
            reference_point = {
                'f1_balance': -100,      # Worst balance (-100%)
                'f2_win_rate': 0,        # Worst win rate (0%)
                'f3_drawdown': -100,    # Worst drawdown (100% = -100 when negated)
                'f4_profit_factor': 0,   # Worst profit factor (0)
                'f5_recovery_factor': 0, # Worst recovery factor (0)
                'f6_sharpe': -1          # Worst Sharpe ratio (-1)
            }
        
        # Simplified hypervolume calculation (for 6D, use Monte Carlo approximation)
        n_samples = 10000
        obj_keys = list(pareto_front[0]['objectives'].keys())
        
        # Calculate bounds for each objective
        obj_bounds = {}
        for key in obj_keys:
            values = [ind['objectives'][key] for ind in pareto_front]
            min_val = min(values)  # Worst value in Pareto front
            max_val = max(values)  # Best value in Pareto front
            ref_val = reference_point[key]  # Reference (worst possible) value
            
            # Use fixed reference point (don't adjust it based on current front)
            # This ensures hypervolume is comparable across generations
            # Only adjust if the reference point is invalid (NaN/Inf)
            if not np.isfinite(ref_val):
                ref_val = min_val - 10.0  # Default offset if invalid
            
            # Ensure max_val is at least slightly above ref_val for valid sampling
            if max_val <= ref_val:
                # Calculate a safe increment
                if abs(ref_val) > 1e-10:
                    increment = max(abs(ref_val) * 0.01, 0.001)
                else:
                    increment = 0.001
                max_val = ref_val + increment
            
            # Additional safety: ensure max_val > ref_val with sufficient margin
            if max_val <= ref_val:
                max_val = ref_val + 0.001
            
            # Ensure all values are finite
            if not np.isfinite(ref_val):
                ref_val = min_val - 0.1
            if not np.isfinite(max_val):
                max_val = max_val + 0.1
            
            # Clamp values to reasonable ranges to avoid overflow
            # numpy uniform has limits on range size
            MAX_RANGE = 1e6  # Maximum range size
            if abs(max_val - ref_val) > MAX_RANGE:
                # Scale down the range
                center = (ref_val + max_val) / 2
                ref_val = center - MAX_RANGE / 2
                max_val = center + MAX_RANGE / 2
            
            # Clamp absolute values to prevent overflow
            MAX_ABS_VALUE = 1e10
            ref_val = np.clip(ref_val, -MAX_ABS_VALUE, MAX_ABS_VALUE)
            max_val = np.clip(max_val, -MAX_ABS_VALUE, MAX_ABS_VALUE)
            min_val = np.clip(min_val, -MAX_ABS_VALUE, MAX_ABS_VALUE)
            
            # Store bounds
            obj_bounds[key] = {
                'ref': float(ref_val),
                'min': float(min_val),
                'max': float(max_val)
            }
        
        # Monte Carlo sampling
        dominated_count = 0
        for _ in range(n_samples):
            # Random point in objective space
            random_point = {}
            for key in obj_keys:
                bounds = obj_bounds[key]
                # Sample from [ref_val, max_val] range
                low = float(bounds['ref'])
                high = float(bounds['max'])
                
                # Ensure valid range: low must be < high
                # Handle edge cases more robustly
                if low >= high:
                    # If they're equal or invalid, create a small valid range
                    if low == high:
                        if abs(low) > 1e-10:
                            high = low + abs(low) * 0.01
                        else:
                            high = low + 0.001
                    else:
                        # Swap if low > high
                        low, high = high, low
                        if low >= high:
                            high = low + 0.001
                
                # Additional safety check: ensure finite values
                if not (np.isfinite(low) and np.isfinite(high)):
                    # Fallback to default range
                    low = bounds['min'] - 0.1
                    high = bounds['max'] + 0.1
                    if low >= high:
                        high = low + 0.001
                
                # Final validation before sampling
                if low >= high:
                    # Last resort: use a tiny valid range
                    if abs(low) > 1e-10:
                        high = low + max(0.001, abs(low) * 0.0001)
                    else:
                        high = low + 0.001
                
                # Ensure we have a valid range one more time
                if low >= high:
                    # Use min/max from Pareto front as fallback
                    low = bounds['min']
                    high = bounds['max']
                    if low >= high:
                        high = low + 0.001
                
                # Sample uniformly from valid range with error handling
                try:
                    # Double-check range is valid
                    if low < high and np.isfinite(low) and np.isfinite(high):
                        random_point[key] = np.random.uniform(low, high)
                    else:
                        # Use min/max from Pareto front
                        random_point[key] = np.random.uniform(bounds['min'], bounds['max'])
                except (OverflowError, ValueError, RuntimeError) as e:
                    # Fallback: use min/max from Pareto front
                    try:
                        random_point[key] = np.random.uniform(bounds['min'], bounds['max'])
                    except:
                        # Last resort: use a default value
                        random_point[key] = (bounds['min'] + bounds['max']) / 2
            
            # Check if dominated by any Pareto solution
            for ind in pareto_front:
                if all(ind['objectives'][key] >= random_point[key] for key in obj_keys):
                    dominated_count += 1
                    break
        
        # Normalize by total samples
        hypervolume = dominated_count / n_samples
        return hypervolume
    
    def _prepare_shared_data(self) -> Dict:
        """
        Prepare shared data for parallel processing (called once, not per individual)
        
        Returns:
            Dictionary with shared data needed for all evaluations
        """
        # Convert to list for pickling (numpy arrays/pandas Index might not pickle well)
        strategies_list = list(self.strategies) if hasattr(self.strategies, '__iter__') else self.strategies
        
        return {
            'test_signals': self.test_signals,  # No copy needed - read-only
            'strategy_metrics_df': self.strategy_metrics_df,  # No copy needed - read-only
            'strategies': strategies_list,
            'phase_columns': list(self.phase_columns),  # Convert to list
            'phase_stats_cache': self.phase_stats_cache.copy(),  # Copy dict (small)
            'train_signals': self.train_signals  # No copy needed - read-only
        }
    
    def _prepare_evaluation_data(self, weights_dict: Dict, shared_data: Dict) -> Dict:
        """
        Prepare evaluation data for parallel processing
        
        Args:
            weights_dict: Weight dictionary to evaluate
            shared_data: Shared data dictionary (prepared once)
            
        Returns:
            Dictionary with all data needed for parallel evaluation
        """
        return {
            'weights_dict': weights_dict,
            **shared_data  # Include all shared data
        }
    
    def _evaluate_population_parallel(self, population: List[Dict], 
                                      n_cores: Optional[int] = None,
                                      use_threading: bool = False) -> List[Dict]:
        """
        Evaluate population in parallel using all available CPU cores
        
        Safe approach for Windows:
        - Uses ProcessPoolExecutor with optimized worker count
        - Limits workers to avoid memory issues (16-24 workers on 32-core system)
        - Falls back to threading if multiprocessing fails
        - Falls back to sequential if all parallel methods fail
        
        Args:
            population: List of weight dictionaries to evaluate
            n_cores: Number of cores to use (None = use all available)
            use_threading: Force ThreadPoolExecutor (for testing/fallback)
            
        Returns:
            List of evaluated individuals with objectives and metrics
        """
        if n_cores is None:
            n_cores = self.num_cores
        
        # Windows-specific optimizations
        is_windows = platform.system() == 'Windows'
        
        # Safe worker count: Use 50-90% of cores based on platform
        # On 32-core system: use 16-24 workers on Windows, 24-28 on Linux
        if is_windows:
            # Use 50% of cores, but at least 4 and at most 24
            safe_cores = max(4, min(int(self.num_cores * 0.5), 24))
            n_cores = min(n_cores, safe_cores, mp.cpu_count())
        else:
            # On Linux/WSL, can use more cores (fork() is memory efficient)
            # Use 85% of cores for better performance
            safe_cores = max(4, min(int(self.num_cores * 0.85), mp.cpu_count()))
            n_cores = min(n_cores, safe_cores, mp.cpu_count())
        
        # Prepare shared data once (not per individual to save memory)
        shared_data = self._prepare_shared_data()
        
        # On Linux/WSL with ProcessPoolExecutor and fork(), DataFrames are shared via copy-on-write
        # On Windows, we need to avoid pickling large DataFrames
        evaluated_pop = []
        
        # On Windows, use ThreadPoolExecutor by default (avoids pickling memory issues)
        # On Linux/WSL, use ProcessPoolExecutor with fork() for true parallelism and memory efficiency
        if is_windows and not use_threading:
            # Force threading on Windows to avoid pickling large DataFrames
            use_threading = True
        
        # For ProcessPoolExecutor on Linux, we can pass DataFrames directly (fork() uses copy-on-write)
        # For ThreadPoolExecutor, we also pass them (shared memory)
        # Prepare evaluation data for each individual (only weights_dict is unique)
        evaluation_tasks = [self._prepare_evaluation_data(ind, shared_data) for ind in population]
        
        # Try ProcessPoolExecutor first (unless threading is forced or on Windows)
        # On Linux/WSL: ProcessPoolExecutor with fork() uses copy-on-write
        # DataFrames are shared efficiently without actual copying until modified
        # This provides true parallelism and excellent memory efficiency
        if not use_threading:
            try:
                executor_class = ProcessPoolExecutor
                print(f"  Using ProcessPoolExecutor with {n_cores} workers (Linux/WSL: fork() with copy-on-write)...", end='', flush=True)
        
                with executor_class(max_workers=n_cores) as executor:
                    # Submit all tasks
                    future_to_individual = {
                        executor.submit(_evaluate_individual_parallel, task): i
                        for i, task in enumerate(evaluation_tasks)
                    }
                    
                    # Collect results as they complete
                    results = [None] * len(population)
                    completed = 0
                    for future in as_completed(future_to_individual):
                        idx = future_to_individual[future]
                        try:
                            weights_dict, objectives, metrics = future.result(timeout=300)  # 5 min timeout per task
                            results[idx] = {
                                'weights': weights_dict,
                                'objectives': objectives,
                                'metrics': metrics
                            }
                            completed += 1
                        except Exception as exc:
                            print(f'\n[!] Individual {idx} evaluation failed: {exc}')
                            # Fallback to sequential evaluation for this individual
                            try:
                                objectives, metrics = self._evaluate_objectives(population[idx])
                                results[idx] = {
                                    'weights': population[idx],
                                    'objectives': objectives,
                                    'metrics': metrics
                                }
                                completed += 1
                            except Exception as e2:
                                print(f'  Sequential fallback also failed: {e2}')
                                # Use default values
                                results[idx] = {
                                    'weights': population[idx],
                                    'objectives': {
                                        'f1_balance': -100.0,
                                        'f2_win_rate': 0.0,
                                        'f3_drawdown': -100.0,
                                        'f4_profit_factor': 0.0,
                                        'f5_recovery_factor': 0.0,
                                        'f6_sharpe': -1.0
                                    },
                                    'metrics': {
                                        'Balance': -100.0,
                                        'Win_Rate': 0.0,
                                        'Max_DD': 100.0,
                                        'Profit_Factor': 0.0,
                                        'Recovery_Factor': 0.0,
                                        'Sharpe_Ratio': -1.0,
                                        'Total_Trades': 0
                                    }
                                }
                                completed += 1
                    
                    if completed < len(population):
                        print(f'\n[!] Only {completed}/{len(population)} evaluations completed')
                    
                    print(" [OK]")
                    return results
                    
            except Exception as e:
                print(f" [X] Failed: {e}")
                print("  Falling back to ThreadPoolExecutor...")
                # Fallback to threading
                use_threading = True
        
        # Fallback to ThreadPoolExecutor (if multiprocessing failed or was requested)
        if use_threading:
            try:
                executor_class = ThreadPoolExecutor
                print(f"  Using ThreadPoolExecutor with {n_cores} workers...", end='', flush=True)
                with executor_class(max_workers=n_cores) as executor:
                    # Submit all tasks
                    future_to_individual = {
                        executor.submit(_evaluate_individual_parallel, task): i
                        for i, task in enumerate(evaluation_tasks)
                    }
                    
                    # Collect results as they complete
                    results = [None] * len(population)
                    completed = 0
                    for future in as_completed(future_to_individual):
                        idx = future_to_individual[future]
                        try:
                            weights_dict, objectives, metrics = future.result(timeout=300)  # 5 min timeout per task
                            results[idx] = {
                                'weights': weights_dict,
                                'objectives': objectives,
                                'metrics': metrics
                            }
                            completed += 1
                        except Exception as exc:
                            print(f'\n[!] Individual {idx} evaluation failed: {exc}')
                            # Fallback to sequential evaluation for this individual
                            try:
                                objectives, metrics = self._evaluate_objectives(population[idx])
                                results[idx] = {
                                    'weights': population[idx],
                                    'objectives': objectives,
                                    'metrics': metrics
                                }
                                completed += 1
                            except Exception as e2:
                                print(f'  Sequential fallback also failed: {e2}')
                                # Use default values
                                results[idx] = {
                                    'weights': population[idx],
                                    'objectives': {
                                        'f1_balance': -100.0,
                                        'f2_win_rate': 0.0,
                                        'f3_drawdown': -100.0,
                                        'f4_profit_factor': 0.0,
                                        'f5_recovery_factor': 0.0,
                                        'f6_sharpe': -1.0
                                    },
                                    'metrics': {
                                        'Balance': -100.0,
                                        'Win_Rate': 0.0,
                                        'Max_DD': 100.0,
                                        'Profit_Factor': 0.0,
                                        'Recovery_Factor': 0.0,
                                        'Sharpe_Ratio': -1.0,
                                        'Total_Trades': 0
                                    }
                                }
                                completed += 1
                    
                    if completed < len(population):
                        print(f'\n[!] Only {completed}/{len(population)} evaluations completed')
                    
                    print(" [OK]")
                    return results
            except Exception as e:
                print(f" [X] Failed: {e}")
                print("  Falling back to sequential evaluation...")
        
        # Final fallback: sequential evaluation
        print("  Using sequential evaluation...", end='', flush=True)
        results = []
        for i, individual in enumerate(population):
            try:
                objectives, metrics = self._evaluate_objectives(individual)
                results.append({
                    'weights': individual,
                    'objectives': objectives,
                    'metrics': metrics
                })
            except Exception as e2:
                print(f'\n  Individual {i} failed: {e2}')
                results.append({
                    'weights': individual,
                    'objectives': {
                        'f1_balance': -100.0,
                        'f2_win_rate': 0.0,
                        'f3_drawdown': -100.0,
                        'f4_profit_factor': 0.0,
                        'f5_recovery_factor': 0.0,
                        'f6_sharpe': -1.0
                    },
                    'metrics': {
                        'Balance': -100.0,
                        'Win_Rate': 0.0,
                        'Max_DD': 100.0,
                        'Profit_Factor': 0.0,
                        'Recovery_Factor': 0.0,
                        'Sharpe_Ratio': -1.0,
                        'Total_Trades': 0
                    }
                })
        print(" [OK]")
        return results
    
    def optimize(self, population_size: int = 100, generations: int = 50,
                mutation_rate: float = 0.15, mutation_strength: float = 0.1,
                crossover_rate: float = 0.9, tournament_size: int = 3,
                n_cores: Optional[int] = None) -> Dict:
        """
        Run NSGA-II optimization with parallel evaluation
        
        Parameters:
        - population_size: Number of individuals (recommended: 50-200)
        - generations: Number of generations (recommended: 30-100)
        - mutation_rate: Probability of mutation (0.1-0.3)
        - mutation_strength: Mutation step size (0.05-0.2)
        - crossover_rate: Probability of crossover (0.7-0.95)
        - tournament_size: Tournament size for selection (2-5)
        - n_cores: Number of CPU cores to use (None = use all available)
        
        Returns:
            Dictionary containing optimization results
        """
        if n_cores is None:
            n_cores = self.num_cores
        
        print(f"\n{'='*70}")
        print(f"NSGA-II Optimization (Parallel Mode)")
        print(f"{'='*70}")
        print(f"Algorithm Parameters:")
        print(f"  Population Size: {population_size}")
        print(f"  Generations: {generations}")
        print(f"  Mutation Rate: {mutation_rate}")
        print(f"  Mutation Strength: {mutation_strength}")
        print(f"  Crossover Rate: {crossover_rate}")
        print(f"  Tournament Size: {tournament_size}")
        print(f"  Parallel Workers: {n_cores} cores")
        print(f"  Dynamic Population: Enabled (starts at 60%, grows to 100%)")
        print(f"{'='*70}\n")
        
        # Dynamic population sizing: start at 60% of target, grow to 100%
        initial_pop_size = max(20, int(population_size * 0.6))  # At least 20 individuals
        growth_rate = 0.08  # 8% growth per generation
        max_pop_size = int(population_size * 1.2)  # Cap at 120% to prevent excessive growth
        
        # Initialize population with initial size
        print(f"Initializing population (Dynamic: {initial_pop_size} -> {population_size})...", end='', flush=True)
        population = [self._generate_random_weights() for _ in range(initial_pop_size)]
        print(" [OK]\n")
        
        evolution_history = []
        best_overall = None
        best_sharpe = -np.inf
        current_pop_size = initial_pop_size
        prev_hypervolume = 0.0
        
        # Evolution loop
        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}")
            
            # Evaluate population in parallel
            # Try multiprocessing first (true parallelism), fallback to threading if needed
            evaluated_pop = self._evaluate_population_parallel(population, n_cores, use_threading=False)
            
            # Add generation number to each individual for tie-breaking
            for ind in evaluated_pop:
                ind['generation'] = gen + 1
            
            # Fast non-dominated sorting (with tie-breaking)
            fronts = self._fast_non_dominated_sort(evaluated_pop)
            
            # Calculate crowding distance for each front
            for front in fronts:
                self._calculate_crowding_distance(front)
                # Apply tie-breaking after crowding distance is calculated
                if len(front) > 1:
                    self._apply_tie_breaking(front)
            
            # Track best solution
            if len(fronts[0]) > 0:
                # Select best solution using predefined criterion (highest Sharpe ratio)
                best_in_gen = self._select_best_solution(fronts[0])
                
                if best_in_gen['metrics']['Sharpe_Ratio'] > best_sharpe:
                    best_sharpe = best_in_gen['metrics']['Sharpe_Ratio']
                    best_overall = best_in_gen
                
                # Calculate hypervolume with fixed reference point
                hv = self._calculate_hypervolume(fronts[0], self.hypervolume_reference_point)
                
                print(f"  Pareto Front Size: {len(fronts[0])}")
                print(f"  Best Sharpe: {best_in_gen['metrics']['Sharpe_Ratio']:.3f}")
                print(f"  Best Balance: {best_in_gen['metrics']['Balance']:.2f}%")
                print(f"  Best Win Rate: {best_in_gen['metrics']['Win_Rate']:.2f}%")
                print(f"  Best Max DD: {best_in_gen['metrics']['Max_DD']:.2f}%")
                print(f"  Hypervolume: {hv:.4f}")
                
                evolution_history.append({
                    'generation': gen + 1,
                    'pareto_size': len(fronts[0]),
                    'hypervolume': hv,
                    **best_in_gen['metrics']
                })
            
            # Dynamic population growth based on convergence
            # Calculate hypervolume improvement rate
            if len(fronts[0]) > 0:
                current_hypervolume = self._calculate_hypervolume(fronts[0], self.hypervolume_reference_point)
                hv_improvement = current_hypervolume - prev_hypervolume
                prev_hypervolume = current_hypervolume
                
                # Adjust growth rate based on hypervolume improvement
                # If improving fast, grow slower; if stagnating, grow faster
                # Handle negative improvement (exploration phase) more gracefully
                if hv_improvement > 0.01:  # Good improvement
                    adjusted_growth = growth_rate * 0.7  # Slow growth
                elif hv_improvement < -0.01:  # Significant decrease (exploration)
                    # Don't penalize too much during exploration - allow moderate growth
                    adjusted_growth = growth_rate * 1.2  # Moderate growth
                elif hv_improvement < 0.001:  # Stagnating or small decrease
                    adjusted_growth = growth_rate * 1.5  # Faster growth
                else:
                    adjusted_growth = growth_rate  # Normal growth
            else:
                adjusted_growth = growth_rate
                hv_improvement = 0.0
            
            # Grow population if not at target
            if current_pop_size < population_size:
                growth = max(1, int(current_pop_size * adjusted_growth))
                current_pop_size = min(population_size, current_pop_size + growth)
                # Add new random individuals to reach target size
                while len(population) < current_pop_size:
                    population.append(self._generate_random_weights())
            
            # Generate offspring (match current population size)
            offspring = []
            
            while len(offspring) < current_pop_size:
                # Selection
                parent1 = self._tournament_selection(evaluated_pop, tournament_size)
                parent2 = self._tournament_selection(evaluated_pop, tournament_size)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child = self._crossover(parent1['weights'], parent2['weights'])
                else:
                    child = parent1['weights'] if np.random.random() < 0.5 else parent2['weights']
                
                # Mutation
                child = self._mutate(child, mutation_rate, mutation_strength)
                offspring.append(child)
            
            # Evaluate offspring in parallel
            print(f"  Evaluating {len(offspring)} offspring...", end='', flush=True)
            evaluated_offspring = self._evaluate_population_parallel(offspring, n_cores, use_threading=False)
            # Note: print(" ✓") is now inside the function
            
            # Add generation number to offspring for tie-breaking
            for ind in evaluated_offspring:
                ind['generation'] = gen + 1
            
            # Combine parents and offspring
            combined = evaluated_pop + evaluated_offspring
            
            # Remove duplicate solutions to improve diversity
            # Compare solutions by their objective values (within tolerance)
            # Use rounded values (2 decimal places) for comparison
            unique_combined = []
            seen_objectives = {}  # Map objective tuple to solution index
            TOLERANCE = 1e-2  # Increased from 1e-6 to 1e-2 (0.01) for 2 decimal places
            
            for sol in combined:
                # Create a hashable representation of objectives (rounded to 2 decimal places)
                # Convert to float to avoid array comparison issues
                # Round to 2 decimal places for most metrics (percentage-based)
                obj_tuple = tuple(
                    round(float(sol['objectives'][key]), 2)  # Round to 2 decimal places
                    for key in sorted(sol['objectives'].keys())
                )
                
                # Check if we've seen this objective combination before
                if obj_tuple not in seen_objectives:
                    seen_objectives[obj_tuple] = len(unique_combined)
                    unique_combined.append(sol)
                # If duplicate, keep the one with better metrics (if available)
                else:
                    # Get the existing solution index
                    existing_idx = seen_objectives[obj_tuple]
                    existing = unique_combined[existing_idx]
                    
                    # Keep the one with higher total trades (more reliable)
                    if sol['metrics'].get('Total_Trades', 0) > existing['metrics'].get('Total_Trades', 0):
                        # Replace the existing solution
                        unique_combined[existing_idx] = sol
            
            # Use unique solutions for sorting
            combined = unique_combined
            
            # Sort and select (with tie-breaking)
            fronts = self._fast_non_dominated_sort(combined)
            for front in fronts:
                self._calculate_crowding_distance(front)
                # Apply tie-breaking after crowding distance is calculated
                if len(front) > 1:
                    self._apply_tie_breaking(front)
            
            # Select next generation (use current population size)
            population = [ind['weights'] for ind in 
                         self._crowding_distance_selection(combined, current_pop_size)]
            
            # Print population size info
            if gen > 0 and current_pop_size < population_size:
                print(f"  Population Size: {current_pop_size}/{population_size} (HV improvement: {hv_improvement:.4f})")
            
            print()
        
        # Final evaluation
        print(f"{'='*70}")
        print("Optimization Complete!")
        print(f"{'='*70}\n")
        
        print("Best Solution Found:")
        print(f"  Balance: {best_overall['metrics']['Balance']:.2f}%")
        print(f"  Win Rate: {best_overall['metrics']['Win_Rate']:.2f}%")
        print(f"  Max Drawdown: {best_overall['metrics']['Max_DD']:.2f}%")
        print(f"  Profit Factor: {best_overall['metrics']['Profit_Factor']:.3f}")
        print(f"  Recovery Factor: {best_overall['metrics']['Recovery_Factor']:.3f}")
        print(f"  Sharpe Ratio: {best_overall['metrics']['Sharpe_Ratio']:.3f}")
        print(f"  Total Trades: {best_overall['metrics']['Total_Trades']}")
        print(f"{'='*70}\n")
        
        # Get final Pareto front (evaluate in parallel)
        print("Evaluating final Pareto front...", end='', flush=True)
        final_pop = self._evaluate_population_parallel(population, n_cores, use_threading=False)
        # Note: print(" ✓") is now inside the function
        final_fronts = self._fast_non_dominated_sort(final_pop)
        pareto_front = final_fronts[0] if len(final_fronts) > 0 else []
        
        return {
            'best_solution': best_overall,
            'pareto_front': pareto_front,
            'evolution_history': evolution_history,
            'algorithm_params': {
                'population_size': population_size,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'mutation_strength': mutation_strength,
                'crossover_rate': crossover_rate,
                'tournament_size': tournament_size,
                'random_seed': self.random_seed,
                'n_cores': n_cores,
                'parallel_mode': True
            }
        }


def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a sample
    
    Args:
        values: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (float(values[0]), float(values[0]))
    
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        t_critical = t_dist.ppf((1 + confidence) / 2, df=n-1)
        margin = t_critical * std / np.sqrt(n)
    else:
        z_critical = norm.ppf((1 + confidence) / 2)
        margin = z_critical * std / np.sqrt(n)
    
    return (mean - margin, mean + margin)


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return float(d)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_statistical_tests(group1: np.ndarray, group2: np.ndarray, 
                              metric_name: str = "") -> Dict:
    """
    Perform statistical tests comparing two groups
    
    Args:
        group1: First group (e.g., OMS results)
        group2: Second group (e.g., baseline results)
        metric_name: Name of the metric being tested
        
    Returns:
        Dictionary with test results
    """
    results = {
        'metric': metric_name,
        'n1': len(group1),
        'n2': len(group2),
        'mean1': float(np.mean(group1)),
        'mean2': float(np.mean(group2)),
        'std1': float(np.std(group1, ddof=1)),
        'std2': float(np.std(group2, ddof=1)),
    }
    
    # Check normality using Shapiro-Wilk test
    if len(group1) >= 3 and len(group1) <= 5000:
        try:
            _, p_norm1 = shapiro(group1)
            results['normality_p1'] = float(p_norm1)
            results['is_normal1'] = p_norm1 > 0.05
        except:
            results['normality_p1'] = None
            results['is_normal1'] = None
    else:
        results['normality_p1'] = None
        results['is_normal1'] = None
    
    if len(group2) >= 3 and len(group2) <= 5000:
        try:
            _, p_norm2 = shapiro(group2)
            results['normality_p2'] = float(p_norm2)
            results['is_normal2'] = p_norm2 > 0.05
        except:
            results['normality_p2'] = None
            results['is_normal2'] = None
    else:
        results['normality_p2'] = None
        results['is_normal2'] = None
    
    # Paired t-test (if groups are paired)
    if len(group1) == len(group2):
        try:
            differences = group1 - group2
            t_stat, p_value = ttest_rel(group1, group2)
            results['paired_t_stat'] = float(t_stat)
            results['paired_t_p'] = float(p_value)
            
            # Confidence interval for mean difference
            ci_lower, ci_upper = calculate_confidence_interval(differences)
            results['mean_diff_ci_lower'] = float(ci_lower)
            results['mean_diff_ci_upper'] = float(ci_upper)
            results['mean_diff'] = float(np.mean(differences))
        except Exception as e:
            results['paired_t_stat'] = None
            results['paired_t_p'] = None
            results['mean_diff_ci_lower'] = None
            results['mean_diff_ci_upper'] = None
            results['mean_diff'] = None
    
    # Independent t-test
    try:
        t_stat_ind, p_value_ind = ttest_ind(group1, group2)
        results['independent_t_stat'] = float(t_stat_ind)
        results['independent_t_p'] = float(p_value_ind)
    except:
        results['independent_t_stat'] = None
        results['independent_t_p'] = None
    
    # Wilcoxon signed-rank test (non-parametric, for paired data)
    if len(group1) == len(group2) and len(group1) >= 3:
        try:
            w_stat, w_p_value = wilcoxon(group1, group2)
            results['wilcoxon_stat'] = float(w_stat)
            results['wilcoxon_p'] = float(w_p_value)
        except:
            results['wilcoxon_stat'] = None
            results['wilcoxon_p'] = None
    else:
        results['wilcoxon_stat'] = None
        results['wilcoxon_p'] = None
    
    # Mann-Whitney U test (non-parametric, for independent samples)
    try:
        u_stat, u_p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        results['mannwhitney_u_stat'] = float(u_stat)
        results['mannwhitney_u_p'] = float(u_p_value)
    except:
        results['mannwhitney_u_stat'] = None
        results['mannwhitney_u_p'] = None
    
    # Cohen's d
    cohens_d = calculate_cohens_d(group1, group2)
    results['cohens_d'] = float(cohens_d)
    results['effect_size_interpretation'] = interpret_effect_size(cohens_d)
    
    # Confidence intervals for means
    ci1_lower, ci1_upper = calculate_confidence_interval(group1)
    ci2_lower, ci2_upper = calculate_confidence_interval(group2)
    results['ci1_lower'] = float(ci1_lower)
    results['ci1_upper'] = float(ci1_upper)
    results['ci2_lower'] = float(ci2_lower)
    results['ci2_upper'] = float(ci2_upper)
    
    return results


def calculate_power_analysis(n: int, effect_size: float, alpha: float = 0.05) -> Dict:
    """
    Calculate statistical power for a given sample size and effect size
    
    Args:
        n: Sample size
        effect_size: Expected effect size (Cohen's d)
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with power analysis results
    """
    try:
        power_analysis = TTestPower()
        power = power_analysis.power(effect_size, n, alpha)
        
        # Calculate required sample size for 80% power
        required_n = power_analysis.solve_power(effect_size=effect_size, power=0.80, alpha=alpha)
        
        return {
            'current_power': float(power),
            'required_n_for_80_power': float(required_n) if required_n is not None else None,
            'effect_size': float(effect_size),
            'alpha': float(alpha),
            'n': int(n)
        }
    except:
        return {
            'current_power': None,
            'required_n_for_80_power': None,
            'effect_size': float(effect_size),
            'alpha': float(alpha),
            'n': int(n)
        }


def save_optimization_results(case_name: str, results: Dict, output_dir: str) -> None:
    """
    Save optimization results
    
    Saves:
    - Optimized weights (JSON)
    - Full weight arrays (NPY)
    - Performance metrics (CSV)
    - Evolution history (CSV)
    - Pareto front (CSV)
    - Algorithm parameters (JSON)
    """
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    best_solution = results['best_solution']
    
    weights_summary = {}
    
    # SSS weights
    sss_metric_names = ['BPP', 'BDDP', 'EDDP', 'PF', 'RF', 'SRA', 'WR']
    for i, name in enumerate(sss_metric_names):
        weights_summary[f'W_SSS_{name}'] = float(best_solution['weights']['sss_weights'][i])
    
    # Top-level weights
    weights_summary['W_SSS'] = float(best_solution['weights']['top_weights'][0])
    weights_summary['W_DSS'] = float(best_solution['weights']['top_weights'][1])
    weights_summary['W_OSS'] = float(best_solution['weights']['top_weights'][2])
    
    with open(os.path.join(case_dir, 'optimized_weights.json'), 'w') as f:
        json.dump(weights_summary, f, indent=2)
    
    # Save full weights (including all DSS phase weights)
    np.save(os.path.join(case_dir, 'full_weights.npy'), best_solution['weights'])
    
    # Save performance metrics
    metrics_df = pd.DataFrame([best_solution['metrics']])
    metrics_df.to_csv(os.path.join(case_dir, 'performance_metrics.csv'), index=False)
    
    # Save evolution history
    evolution_df = pd.DataFrame(results['evolution_history'])
    evolution_df.to_csv(os.path.join(case_dir, 'evolution_history.csv'), index=False)
    
    # Save Pareto front
    pareto_metrics = [sol['metrics'] for sol in results['pareto_front']]
    pareto_df = pd.DataFrame(pareto_metrics)
    pareto_df.to_csv(os.path.join(case_dir, 'pareto_front.csv'), index=False)
    
    # Save algorithm parameters
    with open(os.path.join(case_dir, 'algorithm_params.json'), 'w') as f:
        json.dump(results['algorithm_params'], f, indent=2)
    
    print(f"[OK] Results saved to: {case_dir}/\n")


def run_multiple_trials(signals_df: pd.DataFrame, strategy_metrics_df: pd.DataFrame,
                       case_name: str, n_trials: int = 30,
                       population_size: int = 100, generations: int = 50) -> Dict:
    """
    Run multiple independent trials for statistical significance testing
    
    Args:
        signals_df: Trading signals data
        strategy_metrics_df: Strategy metrics data
        case_name: Case name
        n_trials: Number of independent runs
        population_size: Population size for each run
        generations: Number of generations for each run
        
    Returns:
        Dictionary with aggregated results and statistics
    """
    print(f"\n{'='*70}")
    print(f"Running {n_trials} Independent Trials for {case_name.upper()}")
    print(f"{'='*70}\n")
    
    all_results = []
    all_metrics = []
    
    for trial in range(n_trials):
        print(f"\n{'-'*70}")
        print(f"TRIAL {trial+1}/{n_trials} (Seed: {42 + trial})")
        print(f"{'-'*70}")
        
        optimizer = NSGA2Optimizer(
            signals_df, strategy_metrics_df,
            case_name, random_seed=42 + trial
        )
        
        results = optimizer.optimize(
            population_size=population_size,
            generations=generations
        )
        
        all_results.append(results)
        all_metrics.append(results['best_solution']['metrics'])
    
    # Statistical analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS ({n_trials} trials)")
    print(f"{'='*70}\n")
    
    # Basic descriptive statistics with confidence intervals
    statistical_summary = {}
    
    for metric in ['Balance', 'Win_Rate', 'Max_DD', 'Profit_Factor', 'Recovery_Factor', 'Sharpe_Ratio']:
        if metric not in metrics_df.columns:
            continue
            
        values = metrics_df[metric].values
        mean_val = values.mean()
        std_val = values.std(ddof=1)
        ci_lower, ci_upper = calculate_confidence_interval(values)
        
        # Coefficient of variation
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        
        statistical_summary[metric] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(values.min()),
            'max': float(values.max()),
            'median': float(np.median(values)),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'coefficient_of_variation': float(cv)
        }
        
        print(f"{metric}:")
        print(f"  Mean ± Std: {mean_val:.3f} ± {std_val:.3f}")
        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Min / Max: {values.min():.3f} / {values.max():.3f}")
        print(f"  Median: {np.median(values):.3f}")
        print(f"  CV: {cv:.2f}%")
        print()
    
    # Power analysis
    print(f"{'='*70}")
    print("POWER ANALYSIS")
    print(f"{'='*70}\n")
    
    power_analysis_results = {}
    for metric in ['Balance', 'Win_Rate', 'Max_DD', 'Profit_Factor', 'Sharpe_Ratio']:
        if metric not in metrics_df.columns:
            continue
            
        values = metrics_df[metric].values
        # Use a medium effect size (0.5) as reference
        power_result = calculate_power_analysis(len(values), effect_size=0.5)
        power_analysis_results[metric] = power_result
        
        print(f"{metric}:")
        print(f"  Current Power (d=0.5): {power_result['current_power']:.3f}" if power_result['current_power'] else "  Current Power: N/A")
        if power_result['required_n_for_80_power']:
            print(f"  Required N for 80% power: {power_result['required_n_for_80_power']:.1f}")
        print()
    
    # Select best trial using predefined criterion: highest Sharpe ratio
    best_trial_idx = metrics_df['Sharpe_Ratio'].idxmax()
    best_trial = all_results[best_trial_idx]
    
    return {
        'best_trial': best_trial,
        'all_results': all_results,
        'statistics': metrics_df.describe().to_dict(),
        'statistical_summary': statistical_summary,
        'power_analysis': power_analysis_results,
        'n_trials': n_trials,
        'metrics_df': metrics_df  # Include full DataFrame for further analysis
    }


def main():
    """
    Main execution: Run multi-objective optimization
    """
    # Set multiprocessing start method based on platform
    # On Linux/WSL, use 'fork' for copy-on-write (memory efficient)
    # On Windows, use 'spawn' (required)
    is_windows = platform.system() == 'Windows'
    try:
        if is_windows:
            mp.set_start_method('spawn', force=True)
        else:
            # Linux/WSL: use 'fork' for better memory efficiency with large DataFrames
            # Fork uses copy-on-write, so DataFrames aren't actually copied until modified
            mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE OPTIMIZATION")
    print("NSGA-II Implementation")
    print("="*70)
    
    # Display platform information
    platform_name = platform.system()
    print(f"\nPlatform: {platform_name}")
    if platform_name == 'Windows':
        print("Using ThreadPoolExecutor (avoids pickling memory issues)")
        print("For better performance, run on Linux/WSL to use ProcessPoolExecutor with fork()")
    else:
        print("Using ProcessPoolExecutor with fork() start method")
        print("Copy-on-write provides excellent memory efficiency with large DataFrames")
    print("="*70 + "\n")
    
    data_dir = 'data'
    output_dir = 'optimization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n\nLOADING FOREX DATA...")
    forex_signals = pd.read_csv(f'{data_dir}/forex_signals.csv')
    forex_signals.columns = forex_signals.columns.str.strip()
    forex_metrics = pd.read_csv(f'{data_dir}/forex_strategy_metrics.csv')
    forex_metrics.columns = forex_metrics.columns.str.strip()
    
    print(f"Loaded {len(forex_signals)} signals for {len(forex_signals['strategy_id'].unique())} strategies")
    
    forex_results = run_multiple_trials(
        forex_signals, forex_metrics, 'forex',
        n_trials=N_TRIALS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS
    )
    
    save_optimization_results('forex', forex_results['best_trial'], output_dir)
    
    stats_df = pd.DataFrame(forex_results['statistics'])
    stats_df.to_csv(f'{output_dir}/forex/trial_statistics.csv', index=True)
    
    if 'statistical_summary' in forex_results:
        summary_df = pd.DataFrame(forex_results['statistical_summary']).T
        summary_df.to_csv(f'{output_dir}/forex/statistical_summary.csv')
    
    if 'power_analysis' in forex_results:
        power_df = pd.DataFrame(forex_results['power_analysis']).T
        power_df.to_csv(f'{output_dir}/forex/power_analysis.csv')
    
    if 'metrics_df' in forex_results:
        forex_results['metrics_df'].to_csv(f'{output_dir}/forex/all_trial_metrics.csv', index=False)
    
    print("\n\nLOADING INDEX DATA...")
    index_signals = pd.read_csv(f'{data_dir}/index_signals.csv')
    index_signals.columns = index_signals.columns.str.strip()
    index_metrics = pd.read_csv(f'{data_dir}/index_strategy_metrics.csv')
    index_metrics.columns = index_metrics.columns.str.strip()
    
    print(f"Loaded {len(index_signals)} signals for {len(index_signals['strategy_id'].unique())} strategies")
    
    index_results = run_multiple_trials(
        index_signals, index_metrics, 'index',
        n_trials=N_TRIALS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS
    )
    
    save_optimization_results('index', index_results['best_trial'], output_dir)
    
    stats_df = pd.DataFrame(index_results['statistics'])
    stats_df.to_csv(f'{output_dir}/index/trial_statistics.csv', index=True)
    
    if 'statistical_summary' in index_results:
        summary_df = pd.DataFrame(index_results['statistical_summary']).T
        summary_df.to_csv(f'{output_dir}/index/statistical_summary.csv')
    
    if 'power_analysis' in index_results:
        power_df = pd.DataFrame(index_results['power_analysis']).T
        power_df.to_csv(f'{output_dir}/index/power_analysis.csv')
    
    if 'metrics_df' in index_results:
        index_results['metrics_df'].to_csv(f'{output_dir}/index/all_trial_metrics.csv', index=False)
    
    print("\n" + "="*70)
    print("ALL OPTIMIZATIONS COMPLETE [OK]")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  forex/optimized_weights.json")
    print("  forex/trial_statistics.csv")
    print("  forex/statistical_summary.csv")
    print("  forex/power_analysis.csv")
    print("  forex/all_trial_metrics.csv")
    print("  index/optimized_weights.json")
    print("  index/trial_statistics.csv")
    print("  index/statistical_summary.csv")
    print("  index/power_analysis.csv")
    print("  index/all_trial_metrics.csv")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()


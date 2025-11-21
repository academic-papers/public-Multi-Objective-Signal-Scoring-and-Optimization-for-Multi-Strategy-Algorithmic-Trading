# Data Directory

This directory contains the datasets and supplementary materials used in the multi-objective optimization study.

## ⚠️ Important Note on Signal Data

**Full signal datasets are NOT included in this repository** due to their large size (~50MB each) and proprietary nature. The complete signal datasets (49,940 Forex signals and 34,232 Index signals) are available upon reasonable request from the corresponding author or through a data repository.

## Available Files

### Strategy Performance Metrics

- **`forex_strategy_metrics.csv`**: Performance metrics for 12 Forex trading strategies across training and testing periods
  - Metrics include: Total Trades, Win Rate, Balance Profit Percent, Drawdown metrics, Profit Factor, Recovery Factor, Sharpe Ratio
  - 12 strategies × 2 periods = 24 rows

- **`index_strategy_metrics.csv`**: Performance metrics for 8 Index trading strategies across training and testing periods
  - Same metrics as Forex file
  - 8 strategies × 2 periods = 16 rows

### Market Phase Definitions

- **`phase_definitions.csv`**: Definitions of 24 market phase indicators used in signal scoring
  - Each phase represents a different market condition (Trend Strength, Volatility Regime, Momentum Quality, etc.)
  - Used to classify signals into phase states (A=Aligned/Favorable, R=Reversal/Unfavorable, U=Uncertain/Neutral)

### Signal Statistics (Aggregated)

- **`forex_signal_statistics.csv`**: Aggregated statistics for Forex signals
  - Total signal counts (49,940 total: 29,940 training, 20,000 testing)
  - Outcome distributions (TP/SL ratios)
  - Strategy-wise signal counts
  - Direction distributions (BUY/SELL)
  - Phase state distributions
  - Timeframe and instrument distributions
  - Reward-to-Risk (RTR) statistics

- **`index_signal_statistics.csv`**: Aggregated statistics for Index signals
  - Total signal counts (34,232 total: 20,539 training, 13,693 testing)
  - Same statistical categories as Forex file

## Data Request

To obtain the full signal datasets, please contact the corresponding author with:
- Your name and affiliation
- Brief description of intended use
- Agreement to use data only for research purposes

## File Formats

All files are in CSV format with headers. Signal statistics files use a three-column format:
- `statistic_category`: Category of the statistic
- `statistic_name`: Name of the specific statistic
- `value`: Numerical or text value
- `description`: Description of what the statistic represents

## Usage

These files can be used to:
1. Understand the dataset characteristics without accessing full signal lists
2. Reproduce statistical analyses mentioned in the paper
3. Validate optimization results
4. Understand strategy performance distributions
5. Analyze phase state distributions

## Related Files

Optimization results are stored in the `optimization_results/` directory at the repository root.


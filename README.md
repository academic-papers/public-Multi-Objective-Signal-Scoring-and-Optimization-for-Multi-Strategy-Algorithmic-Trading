# Multi-Objective Signal Scoring and Optimization for Multi-Strategy Algorithmic Trading

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

**A comprehensive multi-objective optimization framework for trading signal scoring systems using NSGA-II algorithm**

[📄 Paper](#) • [📊 Results](#optimization-results) • [💻 Code](#code-availability) • [📦 Data](#data-availability)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Code Availability](#code-availability)
- [Data Availability](#data-availability)
  - [Data Preprocessing and Cleaning](#data-preprocessing-and-cleaning)
- [Optimization Results](#optimization-results)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Supplementary Materials](#supplementary-materials)
- [Citation](#citation)

---

## 🎯 Overview

This repository contains the complete implementation and results for a multi-objective optimization framework designed for trading signal scoring systems. The framework employs the NSGA-II (Non-dominated Sorting Genetic Algorithm II) to optimize weight distributions across multiple trading strategies, balancing competing objectives such as profitability, risk management, and performance consistency.

### Key Features

- ✅ **NSGA-II Implementation**: Full multi-objective optimization with crowding distance
- ✅ **Statistical Rigor**: 30 independent runs with comprehensive statistical analysis
- ✅ **Convergence Metrics**: Hypervolume and generational distance tracking
- ✅ **Parallel Processing**: Efficient evaluation using ProcessPoolExecutor/ThreadPoolExecutor
- ✅ **Reproducibility**: Complete parameter specifications and seed control
- ✅ **Dual Case Studies**: Results for both Forex and Index trading scenarios

---

## 📁 Repository Structure

```
public-Multi-Objective-Signal-Scoring-and-Optimization-for-Multi-Strategy-Algorithmic-Trading/
│
├── 📄 nsga2_optimization.py                      # Python implementation of NSGA-II optimization algorithm
├── 📄 requirements.txt                           # Python dependencies (numpy, pandas, scipy, scikit-learn, statsmodels)
├── 📄 README.md                                  # Documentation and usage instructions
│
├── 📂 data/                                       # Input datasets and statistics
│   ├── forex_strategy_metrics.csv                # Forex strategy performance metrics
│   ├── index_strategy_metrics.csv                # Index strategy performance metrics
│   ├── phase_definitions.csv                     # Market phase indicator definitions
│   ├── forex_signal_statistics.csv               # Aggregated Forex signal statistics
│   ├── index_signal_statistics.csv               # Aggregated Index signal statistics
│   └── README.md                                 # Data directory documentation
│
└── 📂 optimization_results/                      # All optimization outputs
    │
    ├── 📂 forex/                                  # Forex trading case results
    │   ├── algorithm_params.json                  # Complete parameter specifications and configuration files
    │   ├── optimized_weights.json                 # Optimized weight distributions
    │   ├── pareto_front.csv                      # Pareto front solutions
    │   ├── evolution_history.csv                 # Evolution history showing convergence over generations
    │   ├── trial_statistics.csv                  # Trial statistics across multiple independent runs
    │   ├── statistical_summary.csv               # Statistical summaries
    │   ├── power_analysis.csv                    # Power analysis
    │   ├── performance_metrics.csv               # Detailed performance metrics
    │   ├── all_trial_metrics.csv                 # Complete trial data
    │   └── full_weights.npy                      # NumPy array of all weight vectors
    │
    └── 📂 index/                                  # Index trading case results
        ├── algorithm_params.json                  # Complete parameter specifications and configuration files
        ├── optimized_weights.json                 # Optimized weight distributions
        ├── pareto_front.csv                      # Pareto front solutions
        ├── evolution_history.csv                 # Evolution history showing convergence over generations
        ├── trial_statistics.csv                  # Trial statistics across multiple independent runs
        ├── statistical_summary.csv               # Statistical summaries
        ├── power_analysis.csv                    # Power analysis
        ├── performance_metrics.csv               # Detailed performance metrics
        ├── all_trial_metrics.csv                 # Complete trial data
        └── full_weights.npy                      # NumPy array of all weight vectors
```

---

## 💻 Code Availability

### Main Implementation

#### `nsga2_optimization.py`

**Description**: The complete Python implementation of the NSGA-II optimization algorithm for multi-objective signal scoring optimization. This script includes signal scoring and evaluation modules, data preprocessing and analysis scripts, and complete parameter specifications.

**Key Components**:
- **NSGA-II Algorithm**: Full implementation with non-dominated sorting, crowding distance calculation, and elitism
- **Multi-Objective Evaluation**: Optimizes multiple objectives simultaneously (Balance, Win Rate, Max Drawdown, Profit Factor, Recovery Factor, Sharpe Ratio)
- **Parallel Processing**: Uses `ProcessPoolExecutor` and `ThreadPoolExecutor` for efficient parallel evaluation
- **Statistical Analysis**: Comprehensive statistical testing including:
  - Confidence intervals (95% CI)
  - Effect sizes (Cohen's d)
  - Power analysis
  - Multiple comparison corrections
- **Convergence Tracking**: Hypervolume and generational distance metrics
- **Reproducibility**: Fixed random seeds and complete parameter logging

**Key Parameters** (configurable in script):
```python
POPULATION_SIZE = 80      # Population size for genetic algorithm
GENERATIONS = 40         # Number of generations
N_TRIALS = 30            # Independent runs for statistical rigor
```

**Dependencies**:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical functions and distance metrics
- `scikit-learn` - Machine learning utilities
- `statsmodels` - Advanced statistical analysis
- `concurrent.futures` - Parallel processing

**Installation**:
```bash
pip install -r requirements.txt
```

**Usage**:
```bash
python nsga2_optimization.py
```

The script automatically:
1. Loads trading signals and strategy metrics from `data/` directory
2. Runs NSGA-II optimization for both Forex and Index cases
3. Performs 30 independent trials with statistical analysis
4. Generates all results in `optimization_results/` directory including:
   - Pareto front solutions
   - Evolution history showing convergence over generations
   - Optimized weight distributions
   - Trial statistics across multiple independent runs
   - Statistical summaries, power analysis, and algorithm parameter specifications

---

## 📊 Data Availability

### Dataset Information

The datasets used in this study include:

- **Forex Trading Signals**: 49,940 historical trading signals (29,940 training, 20,000 testing)
- **Index Trading Signals**: 34,232 historical trading signals (20,539 training, 13,693 testing)
- **Strategy Performance Metrics**: Training and testing period metrics for all strategies
- **Market Phase Indicators**: 24 phase indicators with state definitions

### ⚠️ Important Note on Signal Data

**Full signal datasets are NOT included in this repository** due to their large size (~50MB each) and proprietary nature. The complete signal datasets are available upon reasonable request from the corresponding author or through a data repository.

### Data Preprocessing and Cleaning

**Input signal data is cleaned and preprocessed before optimization** due to multiple data quality issues that can affect optimization performance and reliability. Raw trading signal data often contains:

- **Missing values**: Incomplete phase indicators, missing timestamps, or undefined strategy identifiers
- **Invalid entries**: Out-of-range values, negative trade counts, or impossible reward-to-risk ratios
- **Data inconsistencies**: Mismatched strategy IDs, duplicate signals, or temporal ordering issues
- **Outliers**: Extreme values that may represent data entry errors or exceptional market conditions
- **Format inconsistencies**: Mixed data types, encoding issues, or inconsistent date formats

#### Data Preparation Techniques

The following preprocessing techniques are applied to ensure data quality before optimization:

1. **Missing Value Handling**:
   - Phase indicators with missing values are treated as 'U' (Unknown/Uncertain) state
   - Missing strategy IDs are filtered out to prevent evaluation errors
   - NaN/Inf values in numerical fields (RTR, PnL) are replaced with default values or removed

2. **Data Validation**:
   - Strategy ID verification against known strategy lists
   - Range checking for numerical values (RTR, PnL, probabilities)
   - Temporal consistency checks (timestamp ordering, date ranges)

3. **Outlier Detection and Treatment**:
   - Statistical outlier detection using IQR (Interquartile Range) method
   - Extreme RTR values are clipped to reasonable ranges (typically 0.1 to 20.0)
   - Unrealistic PnL values are flagged and handled appropriately

4. **Data Type Standardization**:
   - Consistent date/time format conversion
   - String normalization (trimming whitespace, case standardization)
   - Numeric type conversion with error handling

5. **Train/Test Split Validation**:
   - Verification of proper temporal split (no data leakage)
   - Ensuring sufficient samples in both training and testing sets
   - Balance checking across strategies and time periods

6. **Feature Engineering**:
   - Phase state encoding (A/R/U categorical to numeric if needed)
   - Normalization of reward-to-risk ratios for consistent scaling
   - Creation of derived features (e.g., time-based features from timestamps)

7. **Data Integrity Checks**:
   - Duplicate signal detection and removal
   - Cross-field consistency validation (e.g., outcome matches PnL sign)
   - Referential integrity (strategy IDs exist in metrics file)

These preprocessing steps ensure that the optimization algorithm receives clean, consistent, and reliable data, which is essential for obtaining meaningful and reproducible optimization results.

### Available Data Files

The `data/` directory contains:

#### Strategy Performance Metrics
- **`forex_strategy_metrics.csv`**: Performance metrics for 12 Forex strategies (training & testing)
  - Metrics: Total Trades, Win Rate, Balance Profit %, Drawdown metrics, Profit Factor, Recovery Factor, Sharpe Ratio
- **`index_strategy_metrics.csv`**: Performance metrics for 8 Index strategies (training & testing)
  - Same metrics structure as Forex file

#### Market Phase Definitions
- **`phase_definitions.csv`**: Definitions of 24 market phase indicators
  - Includes phase names and descriptions (Trend Strength, Volatility Regime, Momentum Quality, etc.)
  - Used to classify signals into states: A (Aligned/Favorable), R (Reversal/Unfavorable), U (Uncertain/Neutral)

#### Signal Statistics (Aggregated)
- **`forex_signal_statistics.csv`**: Comprehensive aggregated statistics for Forex signals
  - Dataset info (total signals, date ranges, strategy/instrument counts)
  - Outcome distributions (TP/SL ratios)
  - Strategy-wise signal counts
  - Direction distributions (BUY/SELL)
  - Phase state distributions
  - Timeframe and instrument distributions
  - Reward-to-Risk (RTR) statistics (mean, std, percentiles, skewness, kurtosis)

- **`index_signal_statistics.csv`**: Comprehensive aggregated statistics for Index signals
  - Same statistical categories as Forex file
  - Dataset-specific values for Index trading case

These statistics files provide complete dataset characteristics without requiring access to the full signal lists, enabling reproducibility of analyses and validation of results.

**For detailed information about data files, see [`data/README.md`](data/README.md)**

### Optimization Results

All optimization results are included in this repository under `optimization_results/`:

#### For Each Case (Forex & Index):

| File | Description | Format |
|------|-------------|--------|
| `algorithm_params.json` | Complete parameter specifications and configuration files | JSON |
| `optimized_weights.json` | Optimized weight distributions | JSON |
| `pareto_front.csv` | Pareto front solutions | CSV |
| `evolution_history.csv` | Evolution history showing convergence over generations | CSV |
| `trial_statistics.csv` | Trial statistics across multiple independent runs | CSV |
| `statistical_summary.csv` | Statistical summaries | CSV |
| `power_analysis.csv` | Power analysis | CSV |
| `performance_metrics.csv` | Detailed performance metrics for each solution | CSV |
| `all_trial_metrics.csv` | Complete metrics for all trials and solutions | CSV |
| `full_weights.npy` | NumPy array containing all weight vectors | NumPy Binary |

---

## 📈 Optimization Results

### Result Files Explained

#### 1. **`algorithm_params.json`**
Contains the exact NSGA-II parameters used for optimization:
- `population_size`: Number of individuals in each generation (80)
- `generations`: Number of evolutionary generations (40)
- `mutation_rate`: Probability of mutation (0.15)
- `mutation_strength`: Magnitude of mutations (0.1)
- `crossover_rate`: Probability of crossover (0.9)
- `tournament_size`: Size for tournament selection (3)
- `random_seed`: Random seed for reproducibility
- `n_cores`: Number of CPU cores used for parallel processing
- `parallel_mode`: Whether parallel processing was enabled

#### 2. **`optimized_weights.json`**
The best weight configuration found, including:
- `W_SSS_*`: Strategy-Specific Score weights (BPP, BDDP, EDDP, PF, RF, SRA, WR)
- `W_SSS`: Overall Strategy-Specific Score weight
- `W_DSS`: Dynamic Strategy Score weight
- `W_OSS`: Overall Strategy Score weight

#### 3. **`pareto_front.csv`**
Pareto-optimal solutions representing the best trade-offs between objectives:
- Columns: Balance, Win_Rate, Max_DD, Profit_Factor, Recovery_Factor, Sharpe_Ratio, Total_Trades
- Each row represents a non-dominated solution
- Solutions cannot be improved in one objective without worsening another

#### 4. **`evolution_history.csv`**
Tracks convergence metrics across generations:
- `generation`: Generation number
- `pareto_size`: Number of solutions in Pareto front
- `hypervolume`: Hypervolume indicator (convergence metric)
- Objective values averaged across Pareto front

#### 5. **`trial_statistics.csv`**
Descriptive statistics from 30 independent runs:
- `count`: Number of observations (30)
- `mean`: Average value
- `std`: Standard deviation
- `min`, `25%`, `50%`, `75%`, `max`: Percentiles
- Metrics: Balance, Win_Rate, Max_DD, Profit_Factor, Recovery_Factor, Sharpe_Ratio, Total_Trades

#### 6. **`statistical_summary.csv`**
Comprehensive statistical summary:
- `mean`: Mean value
- `std`: Standard deviation
- `min`, `max`: Range
- `median`: Median value
- `ci_95_lower`, `ci_95_upper`: 95% confidence interval bounds
- `coefficient_of_variation`: Relative variability measure

#### 7. **`power_analysis.csv`**
Statistical power analysis results:
- Power calculations for detecting effect sizes
- Helps assess statistical significance of results

#### 8. **`performance_metrics.csv`**
Detailed performance metrics for each evaluated solution:
- Complete objective values
- Strategy-specific metrics
- Evaluation timestamps

#### 9. **`all_trial_metrics.csv`**
Complete dataset of all trials:
- All solutions evaluated across all 30 trials
- Full objective vectors
- Trial identifiers
- Useful for custom analysis

#### 10. **`full_weights.npy`**
NumPy binary file containing:
- All weight vectors from optimization
- Can be loaded with: `np.load('full_weights.npy')`
- Useful for weight distribution analysis

---

## 🔧 Installation & Requirements

### Python Version
Python 3.8 or higher is required.

### Dependencies

Install required packages using pip:

```bash
pip install numpy pandas scipy scikit-learn statsmodels
```

Or create a `requirements.txt` file:

```txt
numpy>=1.19.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
```

Install from requirements file:
```bash
pip install -r requirements.txt
```

### System Requirements

- **CPU**: Multi-core processor recommended for parallel processing
- **RAM**: Minimum 8GB (16GB+ recommended for large datasets)
- **Storage**: ~500MB for code and results

---

## 🚀 Usage

### Running the Optimization

1. **Prepare Data**: Ensure trading signals and strategy metrics are available (see Data Availability section). Note that input signal data should be cleaned and preprocessed before optimization (see [Data Preprocessing and Cleaning](#data-preprocessing-and-cleaning) section).

2. **Configure Parameters** (optional):
   Edit `nsga2_optimization.py` to adjust:
   ```python
   POPULATION_SIZE = 80   # Adjust population size
   GENERATIONS = 40       # Adjust number of generations
   N_TRIALS = 30         # Adjust number of independent runs
   ```

3. **Run Optimization**:
   ```bash
   python nsga2_optimization.py
   ```

4. **Results**: All results will be saved in `optimization_results/` directory

### Quick Testing Mode

For faster testing and validation, uncomment the FAST TESTING MODE parameters:
```python
POPULATION_SIZE = 25
GENERATIONS = 12
N_TRIALS = 1
```

### Analyzing Results

Load and analyze results using Python:

```python
import pandas as pd
import numpy as np
import json

# Load Pareto front
pareto = pd.read_csv('optimization_results/forex/pareto_front.csv')

# Load optimized weights
with open('optimization_results/forex/optimized_weights.json', 'r') as f:
    weights = json.load(f)

# Load evolution history
evolution = pd.read_csv('optimization_results/forex/evolution_history.csv')

# Load statistical summary
stats = pd.read_csv('optimization_results/forex/statistical_summary.csv', index_col=0)
```

---

## 📚 Supplementary Materials

The following supplementary materials are available in this repository:

### Optimization Results

Complete optimization results including:
- Pareto front solutions for both Forex and Index cases
- Evolution history showing convergence over generations
- Optimized weight distributions
- Trial statistics across multiple independent runs
- Algorithm parameter specifications

All optimization results are located in the `optimization_results/` directory.

### Signal Statistics

Detailed statistics on trading signals including:
- Signal counts per strategy and time period
- Outcome distributions (TP/SL ratios)
- Phase state distributions
- Reward-to-risk ratio (RTR) statistics

Signal statistics files are located in the `data/` directory:
- `forex_signal_statistics.csv`
- `index_signal_statistics.csv`

### Extended Results

Additional performance metrics and comparisons:
- Complete trial metrics (`all_trial_metrics.csv`)
- Detailed performance metrics (`performance_metrics.csv`)
- Statistical summaries (`statistical_summary.csv`)
- Power analysis results (`power_analysis.csv`)

### Implementation Details

- Complete source code with documentation (`nsga2_optimization.py`)
- Parameter specifications (`algorithm_params.json`)
- Reproducibility guarantees (fixed random seeds)
- Usage instructions (this README)

### Data Files

- Strategy performance metrics for training and testing periods (`forex_strategy_metrics.csv`, `index_strategy_metrics.csv`)
- Market phase indicator data (`phase_definitions.csv`)
- Signal statistics (aggregated) for both cases

**Note**: Full signal datasets (49,940 Forex and 34,232 Index signals) are not included due to size (~50MB each) but are available upon reasonable request from the corresponding author. The optimization results are included in the supplementary materials.

---

## 📝 Citation

If you use this code or results in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Multi-Objective Signal Scoring and Optimization for Multi-Strategy Algorithmic Trading},
  author={Your Name},
  journal={Journal Name},
  year={2024},
  note={Code available at: github.com/academic-papers/public-Multi-Objective-Signal-Scoring-and-Optimization-for-Multi-Strategy-Algorithmic-Trading}
}
```

---

## 📧 Contact

For questions, data requests, or collaboration inquiries, please contact the corresponding author.

---

## 📄 License

This repository is provided for academic and research purposes. Please refer to the paper for detailed methodology and results.

---

<div align="center">

**Repository**: [github.com/academic-papers/public-Multi-Objective-Signal-Scoring-and-Optimization-for-Multi-Strategy-Algorithmic-Trading](https://github.com/academic-papers/public-Multi-Objective-Signal-Scoring-and-Optimization-for-Multi-Strategy-Algorithmic-Trading)

Made with ❤️ for reproducible research

</div>

# FnO Trading Infrastructure

A powerful and optimized framework for professional-grade F&O (Futures and Options) trading, covering the entire lifecycle from strategy research to live execution.

## 🚀 Key Features

- **High-Performance Backtesting**: Optimized engine for complex option strategies with high-speed simulation.
- **Live & Paper Trading**: Native Interactive Brokers (IBKR) integration (Sync/Async adapters) for seamless transition to production.
- **Optimisation Module**: Robust Walk Forward Optimization (WFO) and automated Hyperparameter tuning to validate and refine strategies.
- **Advanced Risk Management**: Integrated controls including Greeks-based capping, exposure monitoring, and position tracking.
- **Data Engineering**: Comprehensive pipeline for real-time market data fetching and persistence (Redis, DuckDB for local, and QuestDB for global storage).
- **Reporting & Analytics**: Professional-grade performance reporting using `QuantStats` and custom statistical builders.

## 📂 Project Structure

```text
fno_backtestor/
├── core/                   # Infrastructure Core
│   ├── fw_testing/         # Live/Paper Trading & IBKR Adapters
│   ├── data_uploader/      # Data Pipelines & Persistence
│   ├── backtesting_opt.py  # Backtesting Engine
│   ├── risk_manager.py     # Advanced Risk Management
│   └── quantstats/         # Analytics & Reporting
├── strategies/             # Strategy Implementation & Research
├── demos/                  # End-to-end usage examples
└── README.md               # Infrastructure Documentation
```

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.8+
- [Interactive Brokers TWS/Gateway](https://www.interactivebrokers.com/en/trading/tws.php)

### 2. Installation
```bash
git clone https://github.com/HumanOP/fno_backtestor.git
pip install -r requirements.txt
```

## 🏗️ Contributing

For detailed guidelines on how to run demos, add new strategies, or extend the core infrastructure, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

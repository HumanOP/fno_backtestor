# Backtesting Framework
## Overview
This repository contains the backtesting framework. It provides tools and utilities for testing, analyzing, and reporting on trading strategies.

## Project Structure

```
backtestor/
├── core/                     # Core utilities and classes for backtesting
├── examples/                 # Examples of how to run code
├── <strategy name>/          # Strategy wise backtesting scripts and analysis
└── report/                   # Output results, charts, and reports
```

## Directory Overview

### Core
Contains common utility classes, data processors, strategy implementations, and other reusable components used across the backtesting framework.

### Scripts
Executable Python scripts for running backtest simulations. These scripts typically import from the core module and generate outputs stored in the report directory.

### Notebooks
Jupyter notebooks for data exploration, strategy analysis, and visualization of results. These are ideal for prototyping new ideas and generating visualizations.

### Report
Contains the output of backtests, including performance metrics, charts, logs, and other generated artifacts.

## Path Management

When working with this project, proper path management is required to ensure modules can be imported correctly.

### For Python Scripts

When creating a script in a nested directory, add this at the top of your file to ensure imports work correctly:

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from core
from core.some_module import SomeClass
```

### For Jupyter Notebooks

When working with Jupyter notebooks in a nested directory, add this at the beginning of your notebook:

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Now you can import from core
from core.some_module import SomeClass
```

## Getting Started

1. Clone the repository
2. Set up your Python environment (Python 3.x recommended)
3. Install required dependencies: `pip install -r requirements.txt` (if applicable)
4. Run backtests using scripts in the `scripts/` directory
5. Analyze results using notebooks or by examining outputs in the `report/` directory

## Contributing

When adding new components:

1. Place reusable code in the `core/` directory
2. Create executable backtests in the `scripts/` directory
3. Place exploratory analysis in the `notebooks/` directory
4. Store all results in the `report/` directory
5. Follow the path management guidelines above for imports

<!-- ## License

[Add appropriate license information] -->

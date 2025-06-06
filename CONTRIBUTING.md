# Contributing Guidelines

Welcome aboard!
Please follow these guidelines to help us maintain a clean, consistent, and high-quality codebase.

---

## 📁 Current Project Structure

```
repo-root/
│
├── core/               # Core engine (all reusable logic)
│   ├── __init__.py
│   ├── data.py
│   ├── broker.py
│   ├── risk_manager.py
│   ├── _utils_.py
│   └── ... (other core files)
├── demos/                  # Jupyter notebooks/ scripts for demos of core modules
├── strategies/             # All strategy-specific code (each strategy = 1 folder)
│   ├── iv_slope/
│   │   ├── iv_slope.py
│   │   ├── iv_slope.ipynb
│   │   └── iv_slope.md     # Strategy docs
│   ├── mean_reversion/
│   │   └── mean_reversion.py
│   └── ...
├── data/                   # Should be added to .gitignore
├── results/
│   ├── iv_slope/
│   │   ├── run_2024-06-06.csv
│   │   ├── run_2024-06-06_paramsA.json
│   │   └── summary_plot_2024-06-06.png
│   ├── mean_reversion/
│   │   └── run_2024-06-06.csv
│   └── ...
│
├── tests/                  # Tests for both core and strategies
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
└── setup.py
```

No other file or folder should be in the root-repo
---

## 📦 Results Storage Guidelines

- **All backtest outputs/results must be saved to the `results/` folder.**
- Organize results by strategy:  
  - Each strategy gets its own subfolder: `results/<strategy_name>/`
  - Use clear, timestamped filenames, e.g., `run_2024-06-06.csv`, `run_2024-06-06_paramsA.json`, `summary_plot_<date>.png`
- If you experiment with parameters, encode key params in the filename.
- Do **not** commit large results files to the repo; add large result files to `.gitignore`.
  
---

## 📜 Naming Conventions

- **Directories & Files:**  
  - Use lowercase and underscores (`snake_case`).  
    - Examples: `trade_manager.py`, `order_utils/`
  - Avoid spaces and special characters.
- **Python Classes:**  
  - Use `CamelCase` (e.g., `TradeManager`).
- **Functions/Variables:**  
  - Use `snake_case` (e.g., `run_live()`).
- **Constants:**  
  - Use `UPPER_CASE_SNAKE_CASE` (e.g., `MAX_TRADES`).
- **Branch Names:**  
  - Use `feature/`, `bugfix/`, or `docs/` prefix, e.g., `feature/order-matching-engine`

---

## ⚙️ Code Style

- Follow [PEP8](https://pep8.org/) for Python.
- Write clear, descriptive docstrings for modules, classes, and functions.
- Use type annotations where possible.
- Add comments for non-trivial logic.

---

## ✅ Commits & Pull Requests

### **Commit Message Format**

Please use clear, conventional commit messages.  
Follow the **Conventional Commits** standard:  

<type>(optional scope): <short description>

[optional longer description]
[optional references to issues/PRs]


**Allowed types:**
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation changes
- `test` — Adding or updating tests
- `refactor` — Code refactoring (no feature or bug fix)
- `chore` — Maintenance, build, config, dependency updates
- `style` — Formatting, missing semi colons, etc; **not** code changes

**Examples:**
- `feat: add order execution module`
- `fix(broker): handle empty OHLCV data`
- `docs: update usage examples in README`
- `refactor(strategy): modularize signal calculation`
- `test: add unit tests for trade manager`
- `chore: update dependencies`

### **Referencing Issues in Commits and PRs**

- To **link a commit/PR to an issue**, use keywords followed by `#issue-number`.
- GitHub will automatically connect and (if keyword is a closing keyword) close the issue when PR is merged.

**Closing keywords:**  
`close`, `closes`, `closed`, `fix`, `fixes`, `fixed`, `resolve`, `resolves`, `resolved`  

**Examples:**
- `fix: handle missing candle data (fixes #12)`
- `feat: add CSV export functionality (closes #34)`

**In PR descriptions, you can write:**  
- `Closes #56`  
- `Fixes #42 and #43`  
- This will automatically close those issues when the PR is merged.

**Referencing (without closing):**
- `See #23 for background on data loader design`
- `Related to #44`

---

### Pull Request Guidelines

- **Always create a new branch** for your feature, bugfix, or documentation update.
  - Branch off from `main`.
  - Use descriptive branch names, e.g., `feature/mean-reversion-signal`, `fix/broker-bug`.
- Make your commits to this branch.
- **Do not commit directly to `main`**
- When ready, **open a pull request** from your branch into `main` or `develop`.
- Ensure your branch is up-to-date with the target branch before creating a PR.
- Provide a clear description and purpose of your changes in the PR.
- Reference relevant issues in the PR description.

---

## 🧪 Testing

- Add or update unit tests for your code in the `tests/` directory.
- Use `pytest` as the test runner.
- Run tests locally before submitting a PR:  

---

## 📄 Documentation

- Update `README.md` or strategy documentation inside strategy folder as needed.
- Write clear instructions/examples for new features.

---

## 🚫 What Not to Commit

- Sensitive credentials, API keys, or `.env` files.
- Large raw data files or outputs (add to `.gitignore`).
- Personal IDE settings (e.g., `.vscode/`).

---

Thanks for helping make FnO-Synapse better!

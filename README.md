# **XGBoost Forecasting of NEPSE Index Log-Returns with Walk-Forward Validation**  
Sahaj Raj Malla, Shreeyash Kayastha, Rumi Suwal, Harish Chandra Bhandari, Rajendra Adhikari  

The study develops a robust, reproducible machine learning framework for one-step-ahead forecasting of daily log-returns in the Nepal Stock Exchange (NEPSE) Index using XGBoost, with comprehensive benchmarking against tuned Ridge regression and ARIMA models. Rigorous walk-forward validation (expanding and rolling windows) is employed to ensure out-of-sample evaluation free of lookahead bias.

Key findings include superior performance of the optimal XGBoost configuration (expanding window, 20 lags), achieving a log-return RMSE of 0.013450, MAE of 0.009814, and directional accuracy of 65.15%.

## Repository Structure

```
.
├── LICENSE                  # Project license (Apache)
├── README.md                # This file
├── data
│   └── nepse.json           # Raw historical NEPSE Index data (JSON format)
├── logs                     # Log files from model executions
├── results
│   ├── nepse_arima_final    # ARIMA model results
│   │   ├── archives         # Zipped complete results
│   │   ├── models           # Tuned ARIMA configurations
│   │   ├── plots            # Forecast comparison plots
│   │   ├── predictions      # Forecast CSVs
│   │   ├── processed        # Processed data
│   │   ├── raw_data         # Raw data copy
│   │   └── tables           # Performance metrics tables
│   ├── nepse_ridge_final_fixed
│   │   └── ...              # Analogous structure for Ridge regression
│   └── nepse_xgboost_final  # Primary XGBoost results
│       ├── archives
│       ├── feature_importance  # Feature importance CSVs and plots
│       ├── models              # Final trained models (.pkl) and best parameters (.json)
│       ├── optuna_studies      # Saved Optuna studies (.pkl)
│       ├── plots               # Forecast and feature importance visualizations
│       ├── predictions         # Detailed forecast CSVs
│       ├── processed           # Processed datasets per lag configuration
│       ├── raw_data            # Raw data copy
│       └── tables              # Performance metrics and summary tables
├── src
│   ├── nepse_arima.py       # ARIMA modeling and evaluation script
│   ├── nepse_xgboost.py     # Main XGBoost pipeline (feature engineering, tuning, walk-forward)
│   └── ridge.py             # Ridge regression pipeline
└── visualizations
    ├── feature_importance_expanding_lag20.png
    ├── forecast_comparison_expanding_lag20.png
    ├── log_returns_distribution.png
    ├── nepse_price_series.png
    └── viz.ipynb            # Notebook for additional exploratory visualizations
```

## Requirements

The codebase is implemented in Python 3 and depends on the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- optuna
- tqdm
- joblib

A standard scientific Python environment (e.g., via conda or virtualenv) is sufficient.

## Data

The raw dataset (`data/nepse.json`) contains daily NEPSE Index records (date, closing price, etc.) from July 20, 1997, to November 11, 2025, sourced from the public portal NepseAlpha (https://nepsealpha.com/nepse-data). The data is used exclusively for academic research purposes.

## Usage

1. Ensure the raw data file is placed at `data/nepse.json`.

2. Execute the modeling scripts from the repository root:

   ```bash
   python src/nepse_xgboost.py    # Primary XGBoost experiments (recommended)
   python src/ridge.py            # Ridge regression benchmark
   python src/nepse_arima.py      # ARIMA benchmark
   ```

Each script performs:
- Data preparation and feature engineering
- Hyperparameter optimization (Optuna for XGBoost and Ridge; AIC for ARIMA)
- Walk-forward validation across lag configurations (10, 20, 30 days) and window types (expanding, rolling)
- Computation of performance metrics
- Generation of predictions, plots, feature importance (XGBoost), and tables
- Archiving of complete results

Outputs are automatically organized in the corresponding subdirectory under `results/`.

## Reproducibility

- Random seed is fixed at 42 for consistent results.
- All trained models, Optuna studies, hyperparameters, and detailed predictions are preserved.
- Walk-forward validation strictly respects temporal order, eliminating lookahead bias.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.
<!-- 
## Citation

Please cite the associated manuscript if this repository is used in your research:

```
Malla, S. R., Kayastha, S., Suwal, R., Bhandari, H. C., & Adhikari, R. (2026). 
XGBoost Forecasting of NEPSE Index Log-Returns with Walk-Forward Validation.
``` -->

## Acknowledgments

The authors acknowledge the public availability of NEPSE historical data via NepseAlpha and express gratitude to the developers of XGBoost, Optuna, and the broader open-source Python ecosystem.

For inquiries regarding the code or manuscript, please contact the corresponding author.
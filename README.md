# Financial Stock Predictor with Gradient Boosting

This project loads multiple years of financial statement data, combines them into a single dataset, cleans and preprocesses the feature set, then trains a gradient boosting regressor to predict Revenue Growth.

## Project structure

- `Data/` — yearly CSV files containing financial data
- `Analysis.ipynb` — main notebook for data loading, cleaning, exploration, and modeling
- `README.md` — project overview and usage notes

## Data

The project reads five CSV files from the `Data/` directory, sorted by year, and concatenates them into one dataframe.

Each CSV contains financial metrics and company-level features, including:
- revenue and profitability metrics
- balance sheet values
- cash flow figures
- valuation ratios
- sector/classification fields
- target columns such as `Revenue Growth`

## Workflow

The notebook includes:
1. Loading and concatenating the yearly datasets
2. Data cleaning and preprocessing
3. Visualization of cleaned/normalized feature relationships
4. Training an XGBoost regressor to predict `Revenue Growth`
5. Checking overfitting using train vs validation MSE

## Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

## Run the notebook

Open `Analysis.ipynb` in Jupyter or VS Code and run the cells in order.

## Notes

- Rows with missing target values in `Revenue Growth` are removed before training.
- Features are cleaned for modeling, and categorical columns are one-hot encoded.
- The notebook uses MSE as the evaluation metric for model diagnostics and overfitting checks.

## Example goal

The model is intended to learn relationships between a company’s financial metrics and future revenue growth behavior.

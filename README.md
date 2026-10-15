# Gradient Boosting on Company Financial Statements

Can a company's financial statements for year *Y* rank stocks by their return over year *Y+1*? And does gradient boosting find **nonlinear** signal beyond what a simple linear model already captures?

Data: 22,077 company-years (about 5,000 US tickers, 2014-2018), each with 200+ income-statement, balance-sheet, cash-flow and valuation indicators, plus the stock's price change over the following year.

## Method (`Forward_Return_Model.ipynb`)

1. **Target = next-year return, ranked within each year.** Market-wide moves and inflation-like drift hit every stock at once and cannot be predicted from company features (the median return swings from -13% to +18% across these years). Ranking within year removes them exactly and tames raw-return outliers (max 2.4 million percent).
2. **Features ranked within year**, so nominal scale and inflation drop out of level variables.
3. **Two-stage model.** Ridge regression captures the linear cross-sectional signal; XGBoost is then trained **only on ridge's out-of-fold residuals**, so it can only be credited for structure the linear model missed. A linear second stage serves as a control.
4. **Walk-forward evaluation.** Train on earlier years, test on the next (2016, 2017, 2018). Hyperparameters are chosen on training years only.
5. **Metric:** rank information coefficient (Spearman correlation of predicted vs. realized return rank within a test year).

## Results

| Model | Mean rank IC |
|---|---|
| Ridge (linear) | 0.184 |
| Boosting alone | 0.173 |
| Ridge + boosting on residuals | 0.180 |

* Boosting does **not** improve on the linear model, and a linear second stage recovers about as much of the residual signal as boosting does. There is no consistent evidence of nonlinear structure.
* The signal is concentrated in small-cap (IC 0.30) and loss-making (0.24) companies. Excluding the smallest third by market cap cuts the IC to 0.07.

## Caveats

* Only three out-of-sample years, so no significance claim.
* An IC of 0.18 is far above realistic levels (typically 0.02-0.05). The universe is full of micro-caps, and the data has no filing dates, so year-*Y* statements may be published inside the return window (possible look-ahead, which cannot be verified here). Treat the numbers as a demonstration of method, not a tradeable signal.
* No transaction costs, delisting returns, or point-in-time prices.

## Files

- `Forward_Return_Model.ipynb`: the analysis above
- `Analysis.ipynb`: the earlier baseline (predicts same-year revenue growth; superseded)
- `Data/`: yearly CSVs
- `figures/`: figures from the notebook

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Forward_Return_Model.ipynb --inplace   # ~20 s
```

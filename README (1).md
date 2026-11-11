# Customer Retention and Revenue Intelligence

B.Tech CSE (Data Science) project by Mukul Rajput, Lovely Professional University.

Analyses the Online Retail II transaction data to understand revenue patterns, group customers into segments, and predict which customers will buy again in the final 90 days of the data.

## What the project does

1. **Cleans the data**: removes duplicates, fixes data types, handles missing descriptions, flags cancellations and returns, and calculates `Revenue = Quantity x Price`.
2. **Explores revenue**: monthly trend, top 10 products, top 10 countries, top 10 customers, and transaction revenue distribution.
3. **Segments customers**: RFM-style features (Recency, Frequency, Monetary, plus quantity, product variety and average order value), log-transformed, standardised, and grouped with K-Means (4 clusters).
4. **Predicts repeat purchase**: compares Logistic Regression, Gradient Boosting, Random Forest and Extra Trees, and picks the best by F1-score.
5. **Writes findings and recommendations** to `project_summary.txt`.

## Key results

| Item | Value |
|---|---|
| Valid sales rows | 1,007,913 |
| Total revenue | 20,476,260.45 |
| Period | 01 Dec 2009 to 09 Dec 2011 |
| Top country | United Kingdom (about 85% of revenue) |
| Customers identified | 5,878 |
| Best model | Logistic Regression (F1 0.702, ROC-AUC 0.795) |

## Project files

```
Project/
  code.py                    main script
  online_retail_II.xlsx      dataset (you provide this)
  README.md                  this file
  Customer_Retention_Results/   created when you run the script
```

## Requirements

- Windows, macOS or Linux
- Python 3.10 or newer
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl

## Dataset

Online Retail II, UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/502/online+retail+ii
(Creative Commons Attribution 4.0 licence). Download the Excel file and save it as `online_retail_II.xlsx` next to `code.py`.

## How to run (Windows)

1. Put `code.py` and `online_retail_II.xlsx` in the same folder.
2. Click the folder's address bar in File Explorer, type `cmd`, press Enter.
3. Install the libraries (first time only):
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
   ```
4. Run:
   ```
   python code.py
   ```
5. Wait about 5 to 10 minutes. Results appear in `Customer_Retention_Results`.

To use a dataset stored elsewhere:
```
python code.py "D:\data\online_retail_II.xlsx"
```
A `.csv` file with the same columns also works.

## Output files

| File | Content |
|---|---|
| `cleaned_retail_data.csv` | Cleaned transactions with Revenue and cancellation/return flags |
| `valid_sales_data.csv` | Positive, non-cancelled sales used for analysis |
| `customer_segments.csv` | Customer features and K-Means segment |
| `segment_summary.csv` | Customers, recency, frequency, monetary value per segment |
| `customer_prediction_features.csv` | Features and repeat-purchase target |
| `model_comparison.csv` | Accuracy, precision, recall, F1, ROC-AUC for 4 models |
| `project_summary.txt` | Findings and recommendations |
| `01_` to `09_*.png` | Nine charts (revenue, products, countries, customers, distribution, segments, confusion matrix, model comparison, feature influence) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick "Add Python to PATH", or use `py code.py` |
| `pip is not recognized` | Use `python -m pip install ...` |
| `dataset not found` | Check that `online_retail_II.xlsx` is in the same folder as `code.py` |
| `ModuleNotFoundError` | Run the `pip install` command again |

## Limitations

- Revenue is positive recorded sales, not audited net revenue after returns.
- The prediction target is purchase activity in the last 90 days, not a formal churn label.
- One random 80:20 split is used; no time-based validation or hyperparameter search.
- Results are descriptive and predictive, not causal.
- Exact Random Forest scores can vary slightly between scikit-learn versions.

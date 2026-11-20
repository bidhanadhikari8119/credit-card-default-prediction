# Credit Card Default Prediction

## The problem
A bank wants to know **which credit card customers will miss next month's payment**, so its collections team can call the riskiest customers first.

## The data
- **Source:** [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **30,000 customers** in Taiwan (April–September 2005)
- Includes credit limit, age, education, 6 months of payment history, bill amounts and payment amounts
- **22.1% of customers defaulted** the next month

## What I did
1. **Checked the data:** no missing values, 35 duplicate rows (kept, since there are no IDs to confirm they are true duplicates).
2. **Cleaned it:** grouped the undocumented education codes (0, 5, 6) into "others". **Removed sex and marital status**, because US fair-lending rules do not allow them in credit decisions.
3. **Explored what drives default:** default rate by payment status and by credit limit.
4. **Built 3 models** on 80% of the data and tested them on the other 20% (6,000 customers):
   - Baseline (guesses "no default" for everyone)
   - Logistic Regression
   - Random Forest
5. **Tested two cutoffs (0.5 and 0.3)** to see the trade-off between catching more defaulters and making more calls.

## Key findings
1. **Late payment last month is the strongest warning sign.** Customers who paid on time defaulted about 14% of the time. Customers 1 month late defaulted 34% of the time, and customers 2 or more months late defaulted **70% of the time**. That group is only 10% of customers but includes about **1 in 3 of all defaulters**.
2. **Lower credit limits mean higher risk.** Customers in the lowest credit-limit group defaulted **31.8%** of the time, compared with **13.8%** in the highest group.
3. **Accuracy is misleading here.** The baseline scores **78% accuracy** but catches **zero** defaulters. Recall (how many defaulters we catch) is the number that matters.
4. **Random Forest beats Logistic Regression.** At the standard 0.5 cutoff it catches **37%** of defaulters, versus **24%**.
5. **Lowering the cutoff to 0.3 catches far more defaulters.** Random Forest then catches **55%** of defaulters (730 of 1,327) by calling **25%** of customers (1,522 of 6,000). About half of the people called really are high risk (48% precision).

| Model | Cutoff | Defaulters caught (recall) | Calls that hit a real defaulter (precision) | Customers to call |
|---|---|---|---|---|
| Logistic Regression | 0.5 | 24% | 69% | 466 |
| Logistic Regression | 0.3 | 45% | 56% | 1,071 |
| Random Forest | 0.5 | 37% | 63% | 777 |
| **Random Forest** | **0.3** | **55%** | **48%** | **1,522** |

## Recommendation
**Use the Random Forest model with a 0.3 cutoff to build the monthly collections call list.** Calling the riskiest 25% of customers catches more than half of next month's defaulters, compared with about a third at the standard cutoff. **Put anyone who is 2 or more months late at the top of the list**, because 7 in 10 of them default.

## Limitations and next steps
- **Age ranked #2 in Random Forest importance but has almost no correlation with default (0.014).** Random Forest tends to overrate number columns with many different values. Next step: check with permutation importance, and review whether age should stay in a credit model.
- The data is from Taiwan in 2005, so results may not carry over to other markets today.
- The best cutoff depends on the cost of a call versus the cost of a missed default. With those costs, the cutoff could be chosen to save the most money.

## How to run
1. Download `default of credit card clients.xls` from the UCI link above and put it in the same folder as the script.
2. Install the libraries:
   ```
   pip install pandas scikit-learn matplotlib xlrd
   ```
3. Run:
   ```
   python credit_default_model.py
   ```
   The script prints every result and saves `feature_importance.png`.

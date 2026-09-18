# Predicting late deliveries

Analysis of 2,400 order lines from a pharmaceutical wholesaler (synthetic data).
From a messy CSV to a running batch process with monitoring.

## The question

Can we predict, at the moment an order comes in, whether it will be delivered late?

## Key findings

**The promise is wrong, the process is fine.** Cold chain is the slowest category
(4.2 days) and the most reliable one (8% late), because its 6-day promise matches
what the process delivers. OTC and Rx ship faster but promise 4 days, and miss
that in 13-14% of cases. The problem is not in the warehouse, it is on the order
confirmation.

**December deviates significantly** (binomial test, p = 0.0003) but contains only
9 orders. The 95% confidence interval runs from 35% to 88% — far too wide to base
capacity decisions on.

**Data leakage found.** The `klachtmelding` (complaint) column predicts almost
perfectly, but only comes into existence *after* a delivery is late. Removed from
the feature set, along with the column the target was derived from.

**The model is not worth building.** PR-AUC 0.23 ± 0.05 against a floor of 0.13.
After optimising the decision threshold on cost rather than accuracy, it saves
roughly €3,400 per year compared to inspecting every order manually — which does
not justify building and maintaining it. The recommendation is therefore not to
model, but to start recording carrier, warehouse load and stock status per order.

## Approach

1. **Cleaning** — 45 duplicate rows, 23 spelling variants reduced to 10 cities,
   3 date formats unified, 7 impossible order quantities
2. **Exploration** — distributions, delivery time per category, late rate per month
3. **Testing** — binomial test and confidence intervals for the December effect
4. **Modelling** — gradient boosting inside a scikit-learn Pipeline, evaluated with
   5-fold cross-validation on PR-AUC
5. **Threshold** — optimised on total cost (€200 per missed late delivery, €15 per
   unnecessary inspection) instead of on accuracy
6. **Production** — batch script that scores new orders, refuses incomplete input,
   and writes a sorted risk list

## Running it

'''
python - m venv.venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python src/score.py
'''

## Files

- `les1.ipynb` — the full analysis, from raw data to model
- `src/score.py` — batch process that scores new orders
- `RAPPORT.md` — report with findings and recommendations
- `rapport/` — charts
- `Data/leveringen.csv` — raw data

## What I learned

Four times a single number nearly led me to the wrong conclusion: a percentage
without its sample size, a model comparison based on one split, a cost difference
inside the noise margin, and a PR-AUC measured on training data. Cross-validation
and confidence intervals caught each one.
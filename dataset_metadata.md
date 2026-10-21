# Dataset metadata

- **Dataset:** IBM Telco Customer Churn
- **URL used by the source code:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **Collection date:** Record the date when you run the project and fetch the CSV.
- **License / usage conditions:** Verify the current repository and dataset license/usage terms before submission; this bundle does not assert a license that has not been verified.
- **Expected task:** Binary classification of `Churn` (`Yes` / `No`).
- **Cleaning in the app:** Converts `TotalCharges` to numeric with invalid values coerced to missing; removes rows with missing values and duplicate rows.
- **Sampling/split:** No separate sample is created by the app; it uses the cleaned dataset and an 80% training / 20% test split with stratification and `random_state=42`.

## Key fields
The IBM Telco dataset commonly includes customer identifiers, demographic/service attributes, contract/payment information, monthly and total charges, and churn label. Confirm exact columns from the downloaded CSV before final submission. `customerID` is excluded from model features; `Churn` is the target. `TotalCharges` is converted to numeric.

## Data quality report
Actual row counts, missing-value counts, duplicate counts, and class distribution are runtime-dependent and are not fabricated here. Run `run_evaluation.py` or the notebook to inspect and record these values.

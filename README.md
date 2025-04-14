# 🏦 Credit Risk Analysis

A machine learning model for predicting credit risk and identifying customers likely to experience financial distress.

---

## 📚 Overview

This project analyzes the **"Give Me Some Credit"** dataset from Kaggle to build predictive models that can identify borrowers who are likely to default on their loans. The system helps financial institutions make more informed lending decisions and improve their credit risk management strategies.

---

## 🔍 Dataset

- **Source**: [Give Me Some Credit – Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **File**: `cs-training.csv`
- **Target Variable**: `SeriousDlqin2yrs`  
  *(1 = customer experienced 90+ days past due delinquency, 0 = customer did not)*

### 🔑 Key Features:

- `RevolvingUtilizationOfUnsecuredLines`
- `Age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

---

## 🛠️ Methods

### 🧹 Data Preprocessing

- Missing value imputation using **median**
- Feature engineering:
  - `IncomePerDebt = MonthlyIncome / DebtRatio`
  - `TotalPastDue = NumberOfTime30-59DaysPastDueNotWorse + NumberOfTimes90DaysLate + NumberOfTime60-89DaysPastDueNotWorse`
- Feature scaling with **StandardScaler**
- Class imbalance handled using **SMOTE**

### 🤖 Models

Three different classification models were trained and evaluated:

- 🌲 Random Forest
- 🔺 Gradient Boosting
- ⚡ XGBoost

---

## 📈 Results

| Model            | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|------------------|----------|-----------|--------|----------|---------|
| **Random Forest**    | 94%      | 0.95      | 0.93   | 0.94     | 0.94    |
| **Gradient Boosting**| 87%      | 0.88      | 0.85   | 0.87     | 0.87    |
| **XGBoost**          | 93%      | 0.96      | 0.90   | 0.93     | 0.93    |

✅ **Best Model**: **Random Forest** with highest accuracy, precision, recall, and AUC.

---

## 💻 Technologies Used

- `Python 3.x`
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn` (for SMOTE)
- `matplotlib`, `seaborn`

---

## 🙌 Acknowledgments

- [Kaggle – Give Me Some Credit Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

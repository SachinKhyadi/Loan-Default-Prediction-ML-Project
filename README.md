# Loan Default Prediction — Machine Learning Pipeline

Predicting borrower default risk using advanced ML techniques, engineered credit-risk features, and recall-optimized threshold tuning.
Based on the full project report you provided .

# 🚀 Overview

This repository contains an end-to-end ML pipeline to identify high-risk loan defaulters using borrower attributes, loan details, credit activity indicators, and engineered financial-stress features.
The project addresses a highly imbalanced dataset (~90% non-default) and implements targeted strategies—SMOTE, SMOTE-Tomek, class weighting, and probability-threshold tuning—to improve minority-class detection.

The final model is an XGBoost classifier optimized for recall, enabling lenders to detect risk-prone borrowers earlier and reduce financial losses.

# 🧠 Key Objectives

Build a predictive model for loan default risk.

Engineer meaningful credit-risk features to enhance model learning.

Handle severe class imbalance using modern resampling techniques.

Compare models (Logistic Regression, Ridge, Lasso, Random Forest, XGBoost).

Tune thresholds for better real-world decision-making.

# 🛠️ Tech Stack

Languages & Libraries

Python 3.x

NumPy, pandas

scikit-learn (Logistic Regression, Sampling, Metrics)

imbalanced-learn (SMOTE, SMOTE-Tomek)

XGBoost

Matplotlib / Seaborn

ML Techniques Used

Label Encoding

Standardization (for linear models)

SMOTE + SMOTE-Tomek

Class Weighting

Feature Engineering

Threshold Tuning

# Project Structure
.
├─ Loan-Default-Prediction.ipynb            # Main ML notebook
├─ ProjectReport.docx                       # Full project report
├─ data/                                    # Dataset (not included)
├─ models/                                  # Saved models (optional)
├─ images/                                  # EDA plots & confusion matrices
└─ README.md                                # This file

# Dataset Summary

✔ 67,463 borrower records

✔ 35 borrower, credit, and loan-related features

✔ Target variable: Default (1) vs Fully Paid/Current (0)

# Key feature groups include:

Loan Characteristics (loan amount, interest rate, term, grade)

Borrower Info (employment duration, home ownership)

Credit Indicators (revolving balance, delinquency counts)

Verification & Application Details

# 🔧 Data Preparation Pipeline

Major steps from the project report:

## 1️⃣ Handling Missing Values

Dataset had no missing values, so no imputation required.

## 2️⃣ Encoding

Label Encoding for all categorical variables to preserve compact feature space.

## 3️⃣ Feature Engineering (Key Enhancements)

Revolving Utilization Ratio

Loan Stress Metric (Loan Amount × Interest Rate)

DTI Binned Quantiles

Interaction Features

High-Risk Flag

## 4️⃣ Class Imbalance Fix

SMOTE

SMOTE-Tomek Links

Class weights (for linear models)

## 5️⃣ Scaling

StandardScaler applied only to training data (to avoid leakage) for linear models.

# 🔍 Exploratory Data Analysis

Distribution plots for loan amount, interest rate, DTI, revolving balance

Default rate comparisons across loan grades, employment length, home ownership

Correlation heatmap (revealed low linear correlation → motivates tree-based modeling)

EDA confirms:
➡ Default risk correlates with high interest rates and high credit utilization

➡ Loan grade is one of the strongest categorical predictors

➡ Dataset is not linearly separable

# Modeling & Experiments
Models Tested
Model	Recall (Default=1)	Notes
Logistic Regression	~0.34	Baseline, low predictive power
Ridge / Lasso	~0.34	No improvement (confirms non-linear relationships)
Random Forest	0.00	Predicts all borrowers as non-default
XGBoost (Base)	~0.08	Best tree model before tuning
XGBoost (Threshold = 0.15)	0.49	⭐ Best performance
Why Random Forest failed

Even after SMOTE & tuning, RF predicted all cases as majority class.

Highlights limitations for rare-event problems.

Why XGBoost succeeded

Captures non-linear interactions between risk features

Threshold tuning boosted recall from 8% → 49%

# 🎯 Final Model: XGBoost + Threshold Tuning

The optimal threshold = 0.15, prioritizing recall (catching defaulters) over accuracy.

Recall improved dramatically

More realistic risk detection behavior

Ideal for lending where missing a defaulter is costlier than false positives

# 🧪 How to Run Locally
1) Clone the repo
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction

2) Create a virtual environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

3) Install dependencies
pip install -r requirements.txt

4) Run the notebook

Open in Jupyter / VS Code and execute all cells.

# 📝 Key Takeaways

Class imbalance must be handled aggressively.

Accuracy is misleading in imbalanced datasets—recall matters more.

XGBoost + threshold tuning gives the best business-aligned performance.

Engineered risk-focused features significantly improve learning.

# 🔮 Future Improvements

Evaluate LightGBM / CatBoost

Add SHAP interpretability

Use Bayesian optimization for hyperparameter tuning

Incorporate credit bureau & behavioral data

Deploy as an API for real-time underwriting

# 📚 References

https://www.kaggle.com/datasets/hemanthsai7/loandefault

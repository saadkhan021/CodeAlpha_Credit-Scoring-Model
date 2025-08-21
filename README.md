# CodeAlpha_Credit-Scoring-Model
What the project is about

This is a Credit Scoring / Credit Risk Prediction project.
It predicts whether a customer will default (not pay) their credit card bill in the next month, using their past financial history.

Banks and financial institutions use such models to decide:

Should they approve a loan or credit card?
How risky is the customer?
What credit limit should be given?

How the project works

Dataset
Uses the Taiwan Credit Card Default dataset.
Contains information like:
Credit limit (income proxy)
Age, sex, education, marriage
Past bill amounts & payments
Past payment delays (if late, how late)
Whether they defaulted or not (target).
Data Preparation & Feature Engineering
Cleans and standardizes column names.

Creates new features like:

Total & average bill amounts
Total & average payments
Payment-to-bill ratio
Debt ratio (bills ÷ credit limit)
Number of late payments
Age groups
Model Training
Splits data into train, validation, and test sets.
Handles imbalance using SMOTE (since far fewer people default).

Trains two models:

Logistic Regression (baseline)
Random Forest (main model, tuned with GridSearch).
Model Evaluation
Checks accuracy using multiple metrics:
Precision (how many predicted defaults were correct)
Recall (how many actual defaults the model caught)
F1-score (balance of precision & recall)
ROC-AUC (overall ranking quality).
Plots ROC curve.
Model Calibration & Tuning
Adjusts the prediction threshold for best F1-score.
Calibrates probabilities so they better reflect true risk.
Fairness & Explainability
Tests if the model is fair across gender and age groups.
Uses SHAP (optional) to explain which features influence predictions most.

Outputs

Final trained model (best_model.joblib).
Reports (classification results, fairness check, EDA).
Graphs (ROC curve, correlations, histograms).
Model card (summary of dataset, features, metrics, limitations).

Final Outcome

A working credit scoring system that:
Takes in a customer’s financial details.
Predicts the chance of default next month.

Produces metrics, reports, and a saved model ready for deployment.

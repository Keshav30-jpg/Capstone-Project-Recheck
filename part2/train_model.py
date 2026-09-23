import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Part 2 — academic regression / classification experiments
# These experiments are demonstrations, not the leakage-safe production model.

df = pd.read_csv("cleaned_data.csv")
if "customerID" in df.columns:
  df = df.drop(columns=["customerID"])

for col in df.select_dtypes(include=["object", "category"]).columns:
  df[col] = df[col].astype(str).str.strip().str.lower()

y_reg = df["MonthlyCharges"]
y_clf = (df["Churn"] == "yes").astype(int)


X_reg_academic = df.drop(columns=["MonthlyCharges", "Churn"])
X_reg_fair = df.drop(columns=["MonthlyCharges", "Churn", "TotalCharges"])

# Churn classification keeps MonthlyCharges as a legitimate billing predictor.
X_clf = df.drop(columns=["Churn"])

print(
    f"Academic regression features: {X_reg_academic.shape} | Fair regression"
    f" features: {X_reg_fair.shape} | Classification features: {X_clf.shape}"
)


def _encode(frame):
  categorical_cols = frame.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()
  encoded = pd.get_dummies(frame, columns=categorical_cols, drop_first=True)
  for col in encoded.columns:
    if encoded[col].dtype == "bool":
      encoded[col] = encoded[col].astype(int)
  return encoded


X_reg_academic_enc = _encode(X_reg_academic)
X_reg_fair_enc = _encode(X_reg_fair)
X_clf_enc = _encode(X_clf)

# split & Scale
X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
    X_reg_academic_enc, y_reg, test_size=0.2, random_state=42
)
X_train_reg_fair, X_test_reg_fair, y_reg_train_fair, y_reg_test_fair = (
    train_test_split(X_reg_fair_enc, y_reg, test_size=0.2, random_state=42)
)
X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X_clf_enc, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

scaler_reg_fair = StandardScaler()
X_train_reg_fair_scaled = scaler_reg_fair.fit_transform(X_train_reg_fair)
X_test_reg_fair_scaled = scaler_reg_fair.transform(X_test_reg_fair)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Regression Model
print("\n Academic regression (includes TotalCharges)")
lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_reg_train)
y_pred_reg = lr.predict(X_test_reg_scaled)
print(
    f"OLS Linear Regression -> MSE:"
    f" {mean_squared_error(y_reg_test, y_pred_reg):.4f} | R²:"
    f" {r2_score(y_reg_test, y_pred_reg):.4f}"
)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_reg_scaled)
print(
    f"Ridge Regression -> MSE:"
    f" {mean_squared_error(y_reg_test, y_pred_ridge):.4f} | R²:"
    f" {r2_score(y_reg_test, y_pred_ridge):.4f}"
)

print("\n Fair regression (TotalCharges excluded) ")
lr_fair = LinearRegression()
lr_fair.fit(X_train_reg_fair_scaled, y_reg_train_fair)
y_pred_reg_fair = lr_fair.predict(X_test_reg_fair_scaled)
print(
    f"OLS Linear Regression (fair) -> MSE:"
    f" {mean_squared_error(y_reg_test_fair, y_pred_reg_fair):.4f} | R²:"
    f" {r2_score(y_reg_test_fair, y_pred_reg_fair):.4f}"
)

ridge_fair = Ridge(alpha=1.0)
ridge_fair.fit(X_train_reg_fair_scaled, y_reg_train_fair)
y_pred_ridge_fair = ridge_fair.predict(X_test_reg_fair_scaled)
print(
    f"Ridge Regression (fair) -> MSE:"
    f" {mean_squared_error(y_reg_test_fair, y_pred_ridge_fair):.4f} | R²:"
    f" {r2_score(y_reg_test_fair, y_pred_ridge_fair):.4f}"
)

# Classification Model
print("\n Academic logistic regression (MonthlyCharges retained)")
clf_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)
clf_model.fit(X_train_clf_scaled, y_clf_train)
y_pred_clf = clf_model.predict(X_test_clf_scaled)
y_prob_clf = clf_model.predict_proba(X_test_clf_scaled)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_clf_test, y_pred_clf))
print("\nClassification Report:\n", classification_report(y_clf_test, y_pred_clf))

# ROC Curve
fpr, tpr, _ = roc_curve(y_clf_test, y_prob_clf)
auc_val = roc_auc_score(y_clf_test, y_prob_clf)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Decision threshold sensitivity
# Academic threshold experiment 
threshold_list = [0.30, 0.40, 0.50, 0.60, 0.70]
print(
    "\nAcademic threshold experiment"
)
print(
    f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
)
for t in threshold_list:
  preds = (y_prob_clf >= t).astype(int)
  p = precision_score(y_clf_test, preds, zero_division=0)
  r = recall_score(y_clf_test, preds, zero_division=0)
  f1 = f1_score(y_clf_test, preds, zero_division=0)
  print(f"{t:<10.2f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

# Regularization Experiment
print("\n Academic regularization experiment ")
clf_strong_reg = LogisticRegression(
    max_iter=1000, C=0.01, class_weight="balanced", random_state=42
)
clf_strong_reg.fit(X_train_clf_scaled, y_clf_train)
y_prob_strong_reg = clf_strong_reg.predict_proba(X_test_clf_scaled)[:, 1]
print(
    f"Baseline (C=1.0) AUC: {auc_val:.4f} | Stronger L2 (C=0.01) AUC:"
    f" {roc_auc_score(y_clf_test, y_prob_strong_reg):.4f}"
)
print(
    "C=0.01 is stronger regularization (it can reduce overfitting, but it sometimes causes underfitting)"
)

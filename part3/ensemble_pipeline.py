import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from project_config import (  
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_churn_pipeline,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ACADEMIC EXPERIMENTS
# Scaling: fit on academic X_train (not used for final model ranking)

print("ACADEMIC EXPERIMENTS")

df_academic = pd.read_csv("cleaned_data.csv")
for col in df_academic.select_dtypes(include=["object", "category"]).columns:
  df_academic[col] = df_academic[col].astype(str).str.strip().str.lower()

if ID_COL in df_academic.columns:
  df_academic = df_academic.drop(columns=[ID_COL])

# MonthlyCharges is a legitimate churn predictor and is retained.
X_academic = df_academic.drop(columns=[TARGET_COL])
y_academic = (df_academic[TARGET_COL] == "yes").astype(int)

X_encoded = pd.get_dummies(X_academic, drop_first=True)
for col in X_encoded.columns:
  if X_encoded[col].dtype == "bool":
    X_encoded[col] = X_encoded[col].astype(int)

X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X_encoded, y_academic, test_size=0.2, random_state=42, stratify=y_academic
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Academic Experiment — Unconstrained Decision Tree / Overfitting Demonstration
dt_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)
train_acc_dt1 = accuracy_score(
    y_clf_train, dt_unconstrained.predict(X_train_scaled)
)
test_acc_dt1 = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))
print(
    "Academic Experiment - Unconstrained Decision Tree / Overfitting"
    " Demonstration"
)
print(
    f"Unconstrained Tree -> Train Acc: {train_acc_dt1:.4f} | Test Acc:"
    f" {test_acc_dt1:.4f}"
)

# Controlled Decision Tree
dt_controlled = DecisionTreeClassifier(
    max_depth=5, min_samples_split=20, random_state=42
)
dt_controlled.fit(X_train_scaled, y_clf_train)
train_acc_dt2 = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
test_acc_dt2 = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print(
    "Academic Experiment - Controlled Decision Tree (max_depth=5,"
    " min_samples_split=20)"
)
print(
    f"Controlled Tree -> Train Acc: {train_acc_dt2:.4f} | Test Acc:"
    f" {test_acc_dt2:.4f}"
)

# Gini vs Entropy comparison
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
train_acc_gini = accuracy_score(y_clf_train, dt_gini.predict(X_train_scaled))
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))
dt_entropy = DecisionTreeClassifier(
    max_depth=5, criterion="entropy", random_state=42
)
dt_entropy.fit(X_train_scaled, y_clf_train)
train_acc_entropy = accuracy_score(
    y_clf_train, dt_entropy.predict(X_train_scaled)
)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))
print("Academic Experiment - Gini vs Entropy (max_depth=5)")
print(f"Gini  Train Acc: {train_acc_gini:.4f} | Test Acc: {test_acc_gini:.4f}")
print(
    f"Entropy  Train Acc: {train_acc_entropy:.4f} | Test Acc:"
    f" {test_acc_entropy:.4f}"
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
print("Academic Experiment - Random Forest (train/test split, not CV ranking)")
print(
    f"Random Forest Train & Test Acc:"
    f" {accuracy_score(y_clf_train, rf.predict(X_train_scaled)):.4f} |"
    f" {accuracy_score(y_clf_test, rf.predict(X_test_scaled)):.4f} | ROC-AUC:"
    f" {roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1]):.4f}"
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train_scaled, y_clf_train)
print("Academic Experiment - Gradient Boosting (train/test split, not CV ranking)")
print(
    f"Gradient Boosting Train & Test Acc:"
    f" {accuracy_score(y_clf_train, gb.predict(X_train_scaled)):.4f} |"
    f" {accuracy_score(y_clf_test, gb.predict(X_test_scaled)):.4f} | ROC-AUC:"
    f" {roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1]):.4f}"
)

# Academic CV on already-scaled X_train — kept so the original protocol is
# visible, but flagged as optimistic (scaler saw every training fold).
print(
    "\nAcademic 5-fold CV on pre-scaled X_train (mild preprocessing leakage;"
    " NOT used for final model selection):"
)
academic_cv_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Controlled Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}
for name, model in academic_cv_models.items():
  scores = cross_val_score(
      model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc"
  )
  print(
      f"{name:<25} -> 5-Fold CV AUC: {scores.mean():.4f} (Std:"
      f" {scores.std():.4f})"
  )

# FINAL PRODUCTION PIPELINE (leakage-safe)
# Data: modeling_data.csv (missing values still present)

print("FINAL PIPELINE - leakage-safe preprocessing + model selection")

df = pd.read_csv("modeling_data.csv")
for col in df.select_dtypes(include=["object", "category"]).columns:
  df[col] = df[col].astype(str).str.strip().str.lower()
  df[col] = df[col].replace({"nan": np.nan, "none": np.nan})

if ID_COL in df.columns:
  df = df.drop(columns=[ID_COL])

X = df[FEATURE_COLUMNS].copy()
for col in NUMERIC_FEATURES:
  X[col] = pd.to_numeric(X[col], errors="coerce")
y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Controlled Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}

print(
    "\nFair 5-fold CV (same protocol for every candidate, preprocessing fitted"
    " inside each fold):"
)
cv_means = {}
for name, estimator in candidates.items():
  pipe = build_churn_pipeline(estimator)
  scores = cross_val_score(
      pipe, X_train_raw, y_train, cv=cv, scoring="roc_auc"
  )
  cv_means[name] = float(scores.mean())
  print(
      f"{name:<25} -> 5-Fold CV AUC: {scores.mean():.4f} (Std:"
      f" {scores.std():.4f})"
  )

winner_name = max(cv_means, key=cv_means.get)
print(f"\nSelected by CV ROC-AUC: {winner_name}")

param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["liblinear", "lbfgs"],
    },
    "Controlled Decision Tree": {
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [10, 20, 50],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}

# Academic things:
print("\nAcademic hyperparameter tuning :")
logistic_search = GridSearchCV(
    estimator=build_churn_pipeline(
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
    ),
    param_grid=param_grids["Logistic Regression"],
    cv=cv,
    scoring="roc_auc",
    n_jobs=1,
)
logistic_search.fit(X_train_raw, y_train)
print(f"Logistic best parameters: {logistic_search.best_params_}")
print(f"Logistic best CV ROC-AUC: {logistic_search.best_score_:.4f}")

print(f"\nProduction GridSearchCV on CV winner ({winner_name}):")
grid_search = GridSearchCV(
    estimator=build_churn_pipeline(candidates[winner_name]),
    param_grid=param_grids[winner_name],
    cv=cv,
    scoring="roc_auc",
    n_jobs=1,
)
grid_search.fit(X_train_raw, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# Learning curve on TRAIN only (sklearn learning_curve + inner CV)
print("\nAcademic learning curve - sklearn.model_selection.learning_curve on X_train")
train_sizes, train_scores, val_scores = learning_curve(
    clone(best_pipeline),
    X_train_raw,
    y_train,
    cv=cv,
    scoring="roc_auc",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=1,
)
print(
    "Train sizes:",
    np.round(train_sizes, 0),
    "| Val AUC mean:",
    np.round(val_scores.mean(axis=1), 4),
)
plt.figure(figsize=(7, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train AUC")
plt.plot(
    train_sizes, val_scores.mean(axis=1), marker="o", label="CV validation AUC"
)
plt.xlabel("Training examples")
plt.ylabel("ROC-AUC")
plt.title("Learning Curve (final pipeline, train CV only)")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.close()

# Production threshold from out-of-fold train probabilities (not the test set)
oof_proba = cross_val_predict(
    clone(best_pipeline),
    X_train_raw,
    y_train,
    cv=cv,
    method="predict_proba",
)[:, 1]
threshold_candidates = np.round(np.arange(0.30, 0.71, 0.05), 2)
best_threshold = 0.50
best_oof_f1 = -1.0
print("\nProduction threshold search on out-of-fold training probabilities:")
print(
    f"{'Threshold':<10} | {'OOF Precision':<14} | {'OOF Recall':<12} |"
    f" {'OOF F1':<10}"
)
for t in threshold_candidates:
  preds = (oof_proba >= t).astype(int)
  f1 = f1_score(y_train, preds, zero_division=0)
  p = precision_score(y_train, preds, zero_division=0)
  r = recall_score(y_train, preds, zero_division=0)
  print(f" {t:<9.2f} | {p:<14.4f} | {r:<12.4f} | {f1:<10.4f}")
  if f1 > best_oof_f1:
    best_oof_f1 = f1
    best_threshold = float(t)
print(
    f"Frozen production threshold = {best_threshold:.2f} (selected by OOF F1,"
    f" {best_oof_f1:.4f})"
)
# Unbiased test evaluation of frozen pipeline + frozen threshold
test_proba = best_pipeline.predict_proba(X_test_raw)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)
test_auc = roc_auc_score(y_test, test_proba)
print("\nFrozen model + threshold evaluated once on the untouched test set")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Test Recall: {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Test F1: {f1_score(y_test, test_pred, zero_division=0):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, zero_division=0))

# GridSearchCV already refits the winner on full X_train; do not refit on test.
model_filepath = "best_model.pkl"
joblib.dump(best_pipeline, model_filepath)
threshold_payload = {
    "threshold": best_threshold,
    "selection_metric": "f1",
    "selection": "out-of-fold probabilities on training folds",
    "best_cv_auc": float(grid_search.best_score_),
    "test_roc_auc": float(test_auc),
    "model": winner_name,
    "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}
with open("decision_threshold.json", "w", encoding="utf-8") as handle:
  json.dump(threshold_payload, handle, indent=2)

print(f"Successfully saved best pipeline to '{model_filepath}'")
print("Saved production threshold to 'decision_threshold.json'")
print(
    "Part 4 must pass RAW feature columns into this pipeline (no get_dummies)."
)

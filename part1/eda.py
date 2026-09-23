import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Academic EDA / data-quality demonstrations (Part 1)
# Global median fills below are for the *academic cleaned table only*.
# They are NOT the production preprocessing statistics (see modeling_data.csv).

df = pd.read_csv("churnguard_data.csv")
print("First 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nShape:", df.shape)

# Null values evaluation
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame(
    {"Counts": null_counts, "Percentage": null_percentages}
)
print("\nNull Value Table:\n", null_table)

# columns above 20 percentage null rate
high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print("Columns with >20% nulls:", high_null_cols)
print(
    "Academic note: no column exceeds the 20% drop threshold; columns are kept."
)

# Snapshot taken BEFORE full-table median imputation so the final model can
df_modeling = df.copy()

# median imputation using the full dataset
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
  if null_percentages[col] <= 20:
    df[col] = df[col].fillna(df[col].median())
print(
    "Academic experiment - full-table median imputation"
)

# Duplicate removal
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()
df_modeling = df_modeling.drop_duplicates()
print(f"Removed {duplicate_count} duplicate rows.")

# type conversion & universal string cleaning
mem_before = df.memory_usage(deep=True).sum()
df.columns = df.columns.str.strip()
df_modeling.columns = df_modeling.columns.str.strip()


def _standardize_text(frame):
  cat_cols = frame.select_dtypes(include=["object", "category"]).columns
  for col in cat_cols:
    frame[col] = frame[col].astype(str).str.strip().str.lower()
    frame[col] = frame[col].replace(r"\s+", " ", regex=True)
  return frame


if "TotalCharges" in df.columns:
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
  # Academic demo: remaining TotalCharges nulls filled with the full-table median.
  # Exception kept explicit: tenure == 0 is a legitimate new-customer case (0 spend),
  # not a missing-value that should inherit the global median.
  new_customer = df["tenure"] == 0
  df.loc[new_customer, "TotalCharges"] = 0
  df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

if "TotalCharges" in df_modeling.columns:
  df_modeling["TotalCharges"] = pd.to_numeric(
      df_modeling["TotalCharges"], errors="coerce"
  )
  df_modeling.loc[df_modeling["tenure"] == 0, "TotalCharges"] = 0

df = _standardize_text(df)
df_modeling = _standardize_text(df_modeling)

# Value mapping on clean lowercased strings
internet_mapping = {
    "fiberoptic": "fiber optic",
    "fibre optic": "fiber optic",
}
contract_mapping = {
    "1 year": "one year",
    "2 year": "two year",
    "month to month": "month-to-month",
    "month-to-m": "month-to-month",
    "monthly": "month-to-month",
}

if "InternetService" in df.columns:
  df["InternetService"] = df["InternetService"].replace(internet_mapping)
  # Academic demonstration: string "nan" (from missing values) mapped to "no".
  # This is a *business-rule choice to discuss*, not a statistically fitted imputer.
  df["InternetService"] = df["InternetService"].replace({"nan": "no"})

if "InternetService" in df_modeling.columns:
  df_modeling["InternetService"] = df_modeling["InternetService"].replace(
      internet_mapping
  )
  df_modeling["InternetService"] = df_modeling["InternetService"].replace(
      {"nan": np.nan}
  )

if "Contract" in df.columns:
  df["Contract"] = df["Contract"].replace(contract_mapping)
if "Contract" in df_modeling.columns:
  df_modeling["Contract"] = df_modeling["Contract"].replace(contract_mapping)

# Restore true missing markers on the modeling table (astype(str) turned them into "nan")
for col in df_modeling.select_dtypes(include=["object"]).columns:
  if col == "customerID":
    continue
  df_modeling[col] = df_modeling[col].replace({"nan": np.nan, "none": np.nan})

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory Usage Before: {mem_before} bytes | After: {mem_after} bytes")

print(df.describe())

# outlier handling and skewness analysis
if "tenure" in df.columns:
  df.loc[df["tenure"] < 0, "tenure"] = df["tenure"].median()
if "MonthlyCharges" in df.columns:
  df.loc[df["MonthlyCharges"] > 200, "MonthlyCharges"] = df[
      "MonthlyCharges"
  ].median()

if "tenure" in df_modeling.columns:
  df_modeling.loc[df_modeling["tenure"] < 0, "tenure"] = np.nan
if "MonthlyCharges" in df_modeling.columns:
  df_modeling.loc[df_modeling["MonthlyCharges"] > 200, "MonthlyCharges"] = np.nan

all_numeric = [
    col
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]
    if col in df.columns
]
skews = {col: df[col].skew() for col in all_numeric}
for col, val in skews.items():
  print(f"Skewness of {col}: {val:.4f}")

highest_skew_col = max(skews, key=lambda k: abs(skews[k]))
print(f"Column with highest absolute skewness: {highest_skew_col}")

# IQR Outlier detection (academic: detect and report, do not silently drop)
for col in ["MonthlyCharges", "TotalCharges"]:
  if col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = df[
        (df[col] < lower_bound) | (df[col] > upper_bound)
    ].shape[0]
    print(
        f"{col} Outliers: {outliers_count} (Bounds: [{lower_bound:.2f},"
        f" {upper_bound:.2f}])"
    )
print(
    "IQR outlier detection only."
)

# Visualizations
plt.figure(figsize=(8, 4))
plt.plot(df.index[:100], df["MonthlyCharges"].iloc[:100], color="blue", alpha=0.7)
plt.title("Line Plot: Monthly Charges Trend (First 100 Rows)")
plt.xlabel("Row Index")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("line_plot.png")
plt.close()

plt.figure(figsize=(8, 4))
df.groupby("Contract", observed=False)["MonthlyCharges"].mean().plot(
    kind="bar", color="orange", edgecolor="black"
)
plt.title("Average Monthly Charges by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Mean Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("bar_plot.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(df[highest_skew_col], bins=20, kde=True, color="purple")
plt.title(f"Histogram: Distribution of {highest_skew_col}")
plt.xlabel(highest_skew_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="tenure",
    y="TotalCharges",
    hue="Churn" if "Churn" in df.columns else None,
    alpha=0.6,
)
plt.title("Scatter Plot: Tenure vs Total Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df, x="InternetService", y="MonthlyCharges", hue="InternetService"
)
plt.title("Box Plot: Monthly Charges Spread across Internet Services")
plt.xlabel("Internet Service Type")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("box_plot.png")
plt.close()

# correlation and aggregation
plt.figure(figsize=(6, 5))
pearson_matrix = df[all_numeric].corr(method="pearson")
sns.heatmap(
    pearson_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1
)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Task 10: Save cleaned dataset 
df.to_csv("cleaned_data.csv", index=False)
print("\nSuccessfully saved academic 'cleaned_data.csv' ")

df_modeling.to_csv("modeling_data.csv", index=False)
print(
    "Saved 'modeling_data.csv' with missing values preserved for the leakage-safe"
    " production pipeline."
)
print("Modeling-table remaining nulls:\n", df_modeling.isnull().sum())



This is the final course project for my AIML course; however, I little warp the project because the original project repo, Capstone Project, Have liitle bit Errors.

Customer Churn Prediction:

An end-to-end Machine Learning + LLM customer churn prediction system covering EDA, ML experiments, leakage-safe model selection, production inference, guardrails, and AI explanations.
The ML model makes the prediction. The LLM explains the prediction.

Project Workflow
Raw Data → Part 1 EDA → Part 2 ML Experiments → Part 3 Model Selection
→ best_model.pkl + decision_threshold.json → Part 4 Inference → Final Churn Assessment

Part 1 — EDA & Data Preparation:
eda.py performs dataset inspection, missing-value analysis, duplicate removal, category standardisation, TotalCharges conversion, skewness/IQR analysis, correlation analysis, and EDA visualization.
The raw dataset contains 1030 rows and 30 duplicate rows, leaving 1000 records after duplicate removal.
Outputs: cleaned_data.csv, modeling_data.csv, line_plot.png, bar_plot.png, histogram.png, scatter_plot.png, box_plot.png, correlation_heatmap.png.

Part 2 — Regression & Classification:
train_model.py performs academic experiments with Linear Regression, Ridge Regression, and Logistic Regression.
Classification includes StandardScaler, confusion matrix, Precision, Recall, F1, ROC curve/AUC, threshold sensitivity, and regularization.
Academic Logistic Regression: Accuracy 0.70 | ROC-AUC 0.7451.
The Part 2 threshold sweep is an academic experiment; the production threshold is selected separately in Part 3.

 Part 3 — Model Selection & Production Pipeline:
 ensemble_pipeline.py compares Logistic Regression, Controlled Decision Tree, Random Forest, and Gradient Boosting.
 Leakage-Safe 5-Fold CV
```text
 Model                            Mean CV AUC                Std Dev
 Logistic Regression               0.7161                     0.0251
 Controlled Decision Tree          0.6946                     0.0494
 Random Forest                     0.6810                     0.0476
 Gradient Boosting                 0.6805                     0.0344
Selected model: Logistic Regression.
```
GridSearchCV: C=10.0, solver=liblinear, best CV ROC-AUC 0.7172.
Production threshold: 0.45, selected from out-of-fold training probabilities with OOF F1 0.5928.
```text
Final Untouched Test:

Metric                   Result
ROC-AUC                  0.7437
Precision                0.5140
Recall                   0.8594
F1-Score                 0.6433
```
Confusion Matrix: [[84, 52], [9, 55]]
Artifacts: best_model.pkl, decision_threshold.json, learning_curve.png.

Part 4 — Production Inference & AI Explanation:

app.py accepts raw customer features and runs:
```text
Raw Features
 ↓
PII Detection → Normalization → Schema Validation
 ↓
Production ML Pipeline → Churn Probability
 ↓
Frozen Threshold 0.45 → Prediction + Confidence
 ↓
Model Drivers → LLM Explanation → JSON Validation
 ↓
Final Structured Response
```
Confidence Rules:
≥0.70 or ≤0.30 → High | ≥0.55 or ≤0.45 → Medium | otherwise → Low.
The LLM receives the already-computed prediction, probability, confidence, and model drivers. The application validates its JSON response and keeps the ML-derived prediction/confidence.

Guardrails:
PII detection checks for email addresses, phone numbers, customer IDs, SSN-style values, and long card/payment-number patterns.
Examples: PaymentMethod=user@example.com → Blocked (PII Detected) and tenure=-3 → Rejected (Invalid Input).
If the LLM is unavailable or returns invalid JSON, a model-based fallback explanation is used.

Tech Stack:
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Joblib · JSON Schema · Requests · python-dotenv · LLM API · Streamlit




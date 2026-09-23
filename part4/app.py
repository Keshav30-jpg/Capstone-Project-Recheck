import json
import os
import re
import sys
from pathlib import Path

import joblib
import jsonschema
import pandas as pd
import requests
from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from project_config import (  
    ALLOWED_CATEGORIES,
    CATEGORICAL_FEATURES,
    CONTRACT_MAPPING,
    FEATURE_COLUMNS,
    INPUT_SCHEMA,
    INTERNET_MAPPING,
    NUMERIC_FEATURES,
)

# 1. ENVIRONMENT & CONFIGURATION
load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("LLM_API_KEY")
API_URL = os.getenv(
    "LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step",
    ],
}

SYSTEM_PROMPT = (
    "You are an AI decision-support assistant for a customer churn machine"
    " learning pipeline.\nExplain the model's already-computed decision in"
    " plain English.\nDo not invent a different label, probability, or"
    " confidence.\nReturn valid JSON only — no Markdown."
)

USER_PROMPT_TEMPLATE = """
Explain this model-computed churn assessment. Do not change the label or confidence.

Assigned Prediction: {prediction_label}
Model Churn Probability: {predicted_prob:.4f}
Model Confidence: {confidence}
Top Mathematical Drivers: {top_drivers}

Return this exact JSON structure:
{{
  "prediction_label": "{prediction_label}",
  "confidence_level": "{confidence}",
  "top_reason": "Clear explanation of the strongest driver pushing toward this prediction",
  "second_reason": "Clear explanation of the secondary contributing driver",
  "next_step": "Recommended proactive business retention action"
}}
"""

test_features_list = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 2.0,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-Month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 150.0,
    },
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "tenure": 60.0,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card",
        "MonthlyCharges": 53.0,
        "TotalCharges": 3200.0,
    },
    {
        "gender": "Female",
        "SeniorCitizen": 1,
        "tenure": 24.0,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 1800.0,
    },
]


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
  if not API_KEY:
    return None

  headers = {
      "Authorization": f"Bearer {API_KEY}",
      "Content-Type": "application/json",
  }
  payload = {
      "model": MODEL_NAME,
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
      ],
      "temperature": temperature,
      "max_tokens": max_tokens,
  }
  try:
    response = requests.post(
        API_URL, headers=headers, json=payload, timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
  except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
    return None

# detect the personally identifiable information.
def has_pii(text: str) -> bool:
  patterns = [
      r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
      r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
      r"\bcust(?:omer)?[-\s]?\d+\b",
      r"\b\d{3}-\d{2}-\d{4}\b",
      r"\b(?:\d[ -]*?){13,19}\b",
  ]
  lowered = text.lower()
  return any(re.search(p, lowered, flags=re.IGNORECASE) for p in patterns)


def _blocked_payload(reason: str):
  return {
      "prediction_label": "blocked",
      "confidence_level": "low",
      "top_reason": reason,
      "second_reason": "The classifier was not run on this request.",
      "next_step": "Remove identifiers and resubmit required fields only.",
  }


def _error_payload(reason: str):
  return {
      "prediction_label": "invalid_input",
      "confidence_level": "low",
      "top_reason": reason,
      "second_reason": "Prediction skipped because input failed validation.",
      "next_step": "Correct the fields and retry.",
  }


def load_artifacts():
  here = Path(__file__).resolve().parent
  model_candidates = [
      here / "best_model.pkl",
      here.parent / "part3" / "best_model.pkl",
  ]
  threshold_candidates = [
      here / "decision_threshold.json",
      here.parent / "part3" / "decision_threshold.json",
  ]
  model_path = next((p for p in model_candidates if p.exists()), None)
  if model_path is None:
    raise FileNotFoundError(
        "best_model.pkl not found. Run part3/ensemble_pipeline.py first."
    )
  pipeline = joblib.load(model_path)
  threshold = 0.50     # Initially assumes for backup
  for tpath in threshold_candidates:
    if tpath.exists():
      with open(tpath, encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])
      break
  return pipeline, threshold, model_path


best_pipeline, DECISION_THRESHOLD, MODEL_PATH = load_artifacts()
print(f"Loaded production pipeline from '{MODEL_PATH}'")
print(f"Frozen decision threshold: {DECISION_THRESHOLD:.2f}")


def normalize_features(input_features: dict) -> dict:
  normalized = dict(input_features)
  for col in CATEGORICAL_FEATURES:
    if col not in normalized:
      continue
    value = str(normalized[col]).strip().lower()
    value = re.sub(r"\s+", " ", value)
    if col == "InternetService":
      value = INTERNET_MAPPING.get(value, value)
    if col == "Contract":
      value = CONTRACT_MAPPING.get(value, value)
    normalized[col] = value
  for col in NUMERIC_FEATURES:
    if col not in normalized:
      continue
    if col == "SeniorCitizen":
      normalized[col] = int(normalized[col])
    else:
      normalized[col] = float(normalized[col])
  return normalized


def validate_features(normalized: dict):
  validate(instance=normalized, schema=INPUT_SCHEMA)
  for col, allowed in ALLOWED_CATEGORIES.items():
    if normalized[col] not in allowed:
      raise ValidationError(
          f"{col}={normalized[col]!r} is not in allowed values {allowed}"
      )


def confidence_from_probability(pred_prob: float) -> str:
  if pred_prob >= 0.70 or pred_prob <= 0.30:
    return "high"
  if pred_prob >= 0.55 or pred_prob <= 0.45:
    return "medium"
  return "low"


def get_model_drivers(input_df: pd.DataFrame):
  """Feature contributions from the fitted pipeline (raw input frame)."""
  try:
    preprocess = best_pipeline.named_steps["preprocess"]
    model = best_pipeline.named_steps["model"]
    transformed = preprocess.transform(input_df)
    names = preprocess.get_feature_names_out()
    if hasattr(model, "coef_"):
      impacts = transformed[0] * model.coef_[0]
    elif hasattr(model, "feature_importances_"):
      impacts = model.feature_importances_
    else:
      return ["Tenure duration", "Contract duration"]
    ranked = sorted(zip(names, impacts), key=lambda x: abs(x[1]), reverse=True)
    drivers = []
    for name, impact in ranked[:2]:
      direction = (
          "pushes toward Churn" if impact > 0 else "pushes toward Retain"
      )
      if isinstance(model, LogisticRegression) or hasattr(model, "coef_"):
        drivers.append(f"{name} ({direction})")
      else:
        drivers.append(f"{name} (importance {impact:.4f})")
    return drivers or ["Tenure duration", "Contract duration"]
  except (KeyError, AttributeError, ValueError):
    return ["Tenure duration", "Contract duration"]


def process_track_c(input_features: dict, temp: float = 0.0):
  input_str = json.dumps(input_features)
  if has_pii(input_str):
    return _blocked_payload("Blocked because the payload contains PII."), (
        "Blocked (PII Detected)"
    )

  try:
    normalized = normalize_features(input_features)
    validate_features(normalized)
  except (ValidationError, ValueError, TypeError) as exc:
    return _error_payload(str(exc)), "Rejected (Invalid Input)"

  input_df = pd.DataFrame([normalized], columns=FEATURE_COLUMNS)
  pred_prob = float(best_pipeline.predict_proba(input_df)[0, 1])
  pred_class = 1 if pred_prob >= DECISION_THRESHOLD else 0
  label = "Likely Churn" if pred_class == 1 else "Likely Retained"
  confidence = confidence_from_probability(pred_prob)
  drivers = get_model_drivers(input_df)

  fallback = {
      "prediction_label": label,
      "confidence_level": confidence,
      "top_reason": f"Primary driver: {drivers[0] if drivers else 'Account Tenure'}",
      "second_reason": (
          f"Secondary driver: {drivers[1] if len(drivers) > 1 else 'Payment Type'}"
      ),
      "next_step": (
          "Offer a discounted 1-year contract extension."
          if pred_class == 1
          else "Maintain regular service check-ins."
      ),
  }

  formatted_user_prompt = USER_PROMPT_TEMPLATE.format(
      predicted_prob=pred_prob,
      prediction_label=label,
      confidence=confidence,
      top_drivers=", ".join(drivers),
  )
  raw_response = call_llm(
      SYSTEM_PROMPT, formatted_user_prompt, temperature=temp
  )
  if not raw_response:
    return fallback, "Pass (Model Default / No API)"

  json_match = re.search(r"\{[^{}]*\}", raw_response, re.DOTALL)
  if not json_match:
    return fallback, "Pass (Default Used)"

  try:
    parsed_json = json.loads(json_match.group(0))
    if "confidence_level" in parsed_json and isinstance(
        parsed_json["confidence_level"], str
    ):
      parsed_json["confidence_level"] = parsed_json["confidence_level"].lower()
    validate(instance=parsed_json, schema=EXPLANATION_SCHEMA)
    parsed_json["prediction_label"] = label
    parsed_json["confidence_level"] = confidence
    return parsed_json, "Pass (LLM Generated)"
  except (json.JSONDecodeError, jsonschema.ValidationError):
    return fallback, "Pass (Default Used)"


if __name__ == "__main__":
  print("\n Running production inference on RAW customer features ")
  for idx, features in enumerate(test_features_list, 1):
    explanation, status = process_track_c(features, temp=0.0)
    print(
        f"Case #{idx} | Status: {status} | Prediction:"
        f" {explanation.get('prediction_label')} | Confidence:"
        f" {explanation.get('confidence_level')}"
    )
    print(f"   Top Driver   : {explanation.get('top_reason')}")
    print(f"   Action Plan  : {explanation.get('next_step')}\n")

  print("Guardrail checks")
  pii_case = dict(test_features_list[0])
  pii_case["PaymentMethod"] = "user@example.com"
  blocked, blocked_status = process_track_c(pii_case)
  print(f"PII case: {blocked_status} | label={blocked.get('prediction_label')}")

  bad_case = dict(test_features_list[0])
  bad_case["tenure"] = -3
  rejected, rejected_status = process_track_c(bad_case)
  print(
      f"Invalid tenure: {rejected_status} | label={rejected.get('prediction_label')}"
  )

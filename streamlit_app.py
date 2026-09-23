"""Streamlit UI for the existing ChurnGuard capstone implementation.

This file is intentionally a presentation layer only.  Predictions, input
normalisation, guardrails, explanations, and model artefacts remain in
``part4.app`` and are used here without changing the project pipeline.
"""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from part4 import app as inference


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "part1" / "cleaned_data.csv"
ASSET_DIR = ROOT / "part1"


st.set_page_config(
    page_title="ChurnGuard | Customer Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply the approved dashboard palette and responsive presentation CSS."""
    st.markdown(
        """
        <style>
          :root {
            --navy: #0E2238;
            --sidebar-text: #A3B8CC;
            --coral: #E03616;
            --teal: #0B5370;
            --deep-teal: #103649;
            --canvas: #F8FAFC;
            --ink: #1A1D20;
            --muted: #5E6C84;
          }
          .stApp { background: var(--canvas); color: var(--ink); }
          [data-testid="stSidebar"] { background: var(--navy); }
          [data-testid="stSidebar"] * { color: var(--sidebar-text); }
          [data-testid="stSidebar"] [data-baseweb="radio"] label,
          [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--sidebar-text) !important; }
          /* Keep the selected-radio accent on its control only; do not paint
             the navigation label, which caused the red text overlay. */
          [data-testid="stSidebar"] [data-baseweb="radio"] label {
            background: transparent !important;
          }
          [data-testid="stSidebar"] hr { border-color: rgba(163, 184, 204, .25); }
          .hero {
            background: linear-gradient(115deg, var(--teal), var(--deep-teal));
            border-radius: 18px;
            color: #FFFFFF;
            margin: 0 0 1.35rem;
            padding: 2rem 2.25rem;
            box-shadow: 0 10px 24px rgba(14, 34, 56, .16);
          }
          .hero h1 { color: #FFFFFF; font-size: clamp(1.7rem, 4vw, 2.65rem); line-height: 1.1; margin: 0 0 .45rem; }
          .hero p { color: rgba(255,255,255,.86); font-size: 1.03rem; margin: 0; max-width: 50rem; }
          .eyebrow { color: #A3B8CC; font-size: .74rem; font-weight: 700; letter-spacing: .12em; margin-bottom: .45rem; text-transform: uppercase; }
          .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5EAF0;
            border-radius: 14px;
            box-shadow: 0 3px 12px rgba(14, 34, 56, .06);
            min-height: 112px;
            padding: 1rem 1.15rem;
          }
          .metric-label { color: var(--muted); font-size: .82rem; margin-bottom: .38rem; }
          .metric-value { color: var(--ink); font-size: 1.7rem; font-weight: 750; line-height: 1; }
          .metric-note { color: var(--muted); font-size: .75rem; margin-top: .55rem; }
          .section-title { color: var(--ink); font-size: 1.25rem; font-weight: 700; margin: 1.3rem 0 .15rem; }
          .section-copy { color: var(--muted); margin-bottom: .8rem; }
          .result-card {
            background: #FFFFFF;
            border-left: 5px solid var(--coral);
            border-radius: 10px;
            box-shadow: 0 3px 12px rgba(14, 34, 56, .07);
            padding: 1.1rem 1.25rem;
          }
          .result-card h3 { color: var(--ink); margin: 0 0 .45rem; }
          .result-card p { color: var(--muted); margin: .3rem 0; }
          .stButton > button {
            background: var(--coral); border: 1px solid var(--coral); border-radius: 8px;
            color: #FFFFFF; font-weight: 700; min-height: 2.7rem; width: 100%;
          }
          .stButton > button:hover { background: #C72E12; border-color: #C72E12; color: #FFFFFF; }
          [data-testid="stDataFrame"] { border: 1px solid #E5EAF0; border-radius: 10px; overflow: hidden; }
          @media (max-width: 700px) {
            .hero { border-radius: 12px; padding: 1.35rem 1.15rem; }
            .metric-card { min-height: 96px; padding: .85rem; }
            .metric-value { font-size: 1.4rem; }
            [data-testid="stSidebar"] { min-width: 0; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    """Read the existing cleaned dataset for display only."""
    return pd.read_csv(DATA_PATH)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<section class='hero'><div class='eyebrow'>ChurnGuard · Decision Support</div>"
        f"<h1>{title}</h1><p>{subtitle}</p></section>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{escape(str(label))}</div>"
        f"<div class='metric-value'>{escape(str(value))}</div>"
        f"<div class='metric-note'>{escape(str(note))}</div></div>"
    )


def overview_page(data: pd.DataFrame) -> None:
    hero("Customer Churn Intelligence", "A concise view of the completed ChurnGuard data, model, and retention decision support.")
    churn_yes = data["Churn"].astype(str).str.strip().str.lower().eq("yes")
    churn_rate = churn_yes.mean()
    monthly_avg = pd.to_numeric(data["MonthlyCharges"], errors="coerce").mean()
    threshold_details = inference.DECISION_THRESHOLD

    columns = st.columns(4)
    values = [
        ("Customer records", f"{len(data):,}", "Cleaned project dataset"),
        ("Observed churn rate", f"{churn_rate:.1%}", "Based on saved records"),
        ("Average monthly charge", f"${monthly_avg:,.2f}", "Across all customers"),
        ("Decision threshold", f"{threshold_details:.2f}", "Frozen Part 3 threshold"),
    ]
    for column, value in zip(columns, values):
        with column:
            st.markdown(metric_card(*value), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Portfolio snapshot</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>The dashboard reads the completed analysis outputs and production artifacts without retraining the model.</div>", unsafe_allow_html=True)
    left, right = st.columns((1.15, 1))
    with left:
        churn_counts = data["Churn"].astype(str).str.strip().str.title().value_counts()
        st.bar_chart(churn_counts, color="#0B5370")
    with right:
        by_contract = (
            data.assign(ChurnFlag=churn_yes)
            .groupby("Contract", dropna=False)["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )
        st.caption("Churn rate by contract type")
        st.bar_chart(by_contract, color="#E03616")

    st.info(
        f"Production artifact: {inference.MODEL_PATH.name}. "
        "The dashboard invokes the existing validation, PII guardrail, and inference logic in Part 4."
    )


def prediction_page() -> None:
    hero("Customer Risk Assessment", "Enter the approved customer attributes to run the existing production inference and explanation workflow.")
    st.caption("Required fields mirror the existing Part 4 input schema. No customer identifier is collected or sent to the model.")

    with st.form("churn_prediction_form"):
        profile, service = st.columns(2)
        with profile:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0, step=1.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with service:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
            total = st.number_input("Total charges ($)", min_value=0.0, value=840.0, step=1.0)
        submitted = st.form_submit_button("Assess churn risk")

    if not submitted:
        return

    features = {
        "gender": gender,
        "SeniorCitizen": senior,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    with st.spinner("Running the completed production pipeline..."):
        explanation, status = inference.process_track_c(features, temp=0.0)

    if explanation.get("prediction_label") in {"blocked", "invalid_input"}:
        st.error(explanation.get("top_reason", "The submitted input could not be processed."))
        st.caption(f"Workflow status: {status}")
        return

    normalized = inference.normalize_features(features)
    input_frame = pd.DataFrame([normalized], columns=inference.FEATURE_COLUMNS)
    probability = float(inference.best_pipeline.predict_proba(input_frame)[0, 1])
    predicted_label = explanation["prediction_label"]

    first, second, third = st.columns(3)
    with first:
        st.markdown(metric_card("Assessment", predicted_label, f"Threshold: {inference.DECISION_THRESHOLD:.2f}"), unsafe_allow_html=True)
    with second:
        st.markdown(metric_card("Churn probability", f"{probability:.1%}", "Existing production model output"), unsafe_allow_html=True)
    with third:
        st.markdown(metric_card("Confidence", explanation["confidence_level"].title(), status), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model explanation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='result-card'>"
        f"<h3>Recommended retention action</h3><p>{escape(explanation['next_step'])}</p>"
        f"<p><strong>Primary driver:</strong> {escape(explanation['top_reason'])}</p>"
        f"<p><strong>Secondary driver:</strong> {escape(explanation['second_reason'])}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def insights_page(data: pd.DataFrame) -> None:
    hero("Data & EDA Insights", "Explore the saved analysis visualisations and a compact, responsive data preview from Part 1.")
    image_specs = [
        ("bar_plot.png", "Average monthly charges by contract type"),
        ("histogram.png", "Distribution of the highest-skew numeric feature"),
        ("scatter_plot.png", "Tenure versus total charges, coloured by churn"),
        ("correlation_heatmap.png", "Pearson correlation matrix"),
        ("box_plot.png", "Monthly-charge spread by internet service"),
        ("line_plot.png", "Monthly charges across the first 100 rows"),
    ]
    for start in range(0, len(image_specs), 2):
        columns = st.columns(2)
        for column, (filename, caption) in zip(columns, image_specs[start:start + 2]):
            with column:
                image_path = ASSET_DIR / filename
                if image_path.exists():
                    st.image(str(image_path), caption=caption, use_container_width=True)

    st.markdown("<div class='section-title'>Dataset preview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Read-only preview of the completed Part 1 cleaned dataset.</div>", unsafe_allow_html=True)
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download full cleaned dataset",
        data=DATA_PATH.read_bytes(),
        file_name=DATA_PATH.name,
        mime="text/csv",
    )


def model_page() -> None:
    hero("Production Model", "The final production pipeline is preserved as implemented: leakage-safe preprocessing, model selection, and a frozen decision threshold.")
    metadata = {
        "Selected model": inference.best_pipeline.named_steps["model"].__class__.__name__,
        "Decision threshold": f"{inference.DECISION_THRESHOLD:.2f}",
        "Model artifact": str(inference.MODEL_PATH.relative_to(ROOT)),
        "Numeric features": ", ".join(inference.NUMERIC_FEATURES),
        "Categorical features": ", ".join(inference.CATEGORICAL_FEATURES),
    }
    left, right = st.columns(2)
    with left:
        st.subheader("Artifact summary")
        for label, value in metadata.items():
            st.markdown(f"**{label}**  \n{value}")
    with right:
        st.subheader("Existing workflow")
        st.markdown(
            "1. Raw approved fields enter the existing Part 4 validation and PII checks.  \n"
            "2. The saved Part 3 pipeline applies its fitted preprocessing.  \n"
            "3. The existing frozen threshold determines the classification.  \n"
            "4. Existing driver and explanation logic produces decision support."
        )
    curve = ROOT / "part3" / "learning_curve.png"
    if curve.exists():
        st.markdown("<div class='section-title'>Saved learning curve</div>", unsafe_allow_html=True)
        st.image(str(curve), caption="Final pipeline learning curve from Part 3", use_container_width=True)


def main() -> None:
    inject_styles()
    data = load_dashboard_data()
    with st.sidebar:
        st.markdown("## ◈ ChurnGuard")
        st.caption("Capstone Project Dashboard")
        st.divider()
        page = st.radio("Navigate", ["Overview", "Risk Assessment", "Data Insights", "Model Details"], label_visibility="collapsed")
        st.divider()
        st.caption("Responsive decision-support interface")

    if page == "Overview":
        overview_page(data)
    elif page == "Risk Assessment":
        prediction_page()
    elif page == "Data Insights":
        insights_page(data)
    else:
        model_page()


if __name__ == "__main__":
    main()

# ================================================================
# Stage 6: Streamlit Dashboard
#
# Run with:  streamlit run dashboard/app.py
#
# Three pages:
#   1. Overview      — model stats + global feature importance
#   2. Student Lookup — explain any student from the test set
#   3. Predict       — enter values manually, get live prediction
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
import joblib

matplotlib.use("Agg")

# ----------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------------------------
# Load everything once and cache it
# Streamlit reruns the whole script on every interaction.
# @st.cache_resource means "only load this once, reuse it."
# Without this, the model would reload every time you click
# anything — very slow.
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("data/processed/xgb_model.json")
    return model

@st.cache_resource
def load_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_test, y_test

@st.cache_resource
def load_shap_values(_model, _X_test):
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer(_X_test)
    return explainer, shap_values

model        = load_model()
X_test, y_test = load_data()
explainer, shap_values = load_shap_values(model, X_test)
feature_names = list(X_test.columns)


# ----------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/graduation-cap.png",
    width=60
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔍 Student Lookup", "🧪 Try It Yourself"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost  \n"
    "**ROC-AUC:** 0.933  \n"
    "**Dropout Recall:** 81.3%  \n"
    "**Dataset:** UCI — 4,424 students"
)


# ================================================================
# PAGE 1: Overview
# ================================================================
if page == "🏠 Overview":
    st.title("🎓 Student Dropout Risk Predictor")
    st.markdown(
        "This tool uses machine learning to identify students "
        "at risk of dropping out — and explains **why**, "
        "feature by feature."
    )

    # ── Metric cards ────────────────────────────────────────────
    st.markdown("### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",        "0.933", "↑ vs random (0.5)")
    c2.metric("Overall Accuracy","88.2%")
    c3.metric("Dropout Recall",  "81.3%", "of real dropouts caught")
    c4.metric("Dropout Precision","81.9%","of alerts were correct")

    st.markdown("---")

    # ── Global SHAP bar chart ───────────────────────────────────
    st.markdown("### What Drives Dropout Risk?")
    st.markdown(
        "The chart below shows which features have the biggest "
        "average impact across all students. "
        "**2nd semester approvals dominates by a wide margin.**"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_values, max_display=12, show=False,
                   ax=ax)
    ax.set_title("Mean |SHAP Value| per Feature", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Dataset breakdown ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Dataset Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Full dataset (4,424 students)**")
        st.markdown("- Graduate: 2,209 (49.9%)")
        st.markdown("- Dropout:  1,421 (32.1%)")
        st.markdown("- Enrolled:   794 (17.9%)")
    with col2:
        st.markdown("**After binary conversion**")
        st.markdown("- Not Dropout: 3,003 (67.9%)")
        st.markdown("- Dropout:     1,421 (32.1%)")
        st.markdown("- *Enrolled students treated as Not Dropout*")


# ================================================================
# PAGE 2: Student Lookup
# ================================================================
elif page == "🔍 Student Lookup":
    st.title("🔍 Student Lookup")
    st.markdown(
        "Select any student from the test set to see their "
        "predicted dropout risk and a full SHAP explanation."
    )

    # ── Student selector ────────────────────────────────────────
    probs = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns([1, 2])
    with col1:
        student_idx = st.selectbox(
            "Select student index",
            options=list(range(len(X_test))),
            index=90   # default to the highest risk student
        )

    prob   = probs[student_idx]
    actual = y_test.iloc[student_idx]

    # ── Risk score display ──────────────────────────────────────
    with col2:
        st.markdown("#### Risk Assessment")
        if prob >= 0.7:
            st.error(  f"🔴 HIGH RISK — {prob*100:.1f}% dropout probability")
        elif prob >= 0.4:
            st.warning(f"🟡 MEDIUM RISK — {prob*100:.1f}% dropout probability")
        else:
            st.success(f"🟢 LOW RISK — {prob*100:.1f}% dropout probability")

        outcome_label = "Dropout" if actual == 1 else "Not Dropout"
        correct = (prob >= 0.5) == (actual == 1)
        st.markdown(
            f"**Actual outcome:** {outcome_label}  |  "
            f"**Model was:** {'✅ Correct' if correct else '❌ Wrong'}"
        )

    st.markdown("---")

    # ── SHAP waterfall ──────────────────────────────────────────
    st.markdown("#### Why did the model give this score?")
    st.markdown(
        "Each bar shows one feature's contribution. "
        "🔴 Red bars push the risk **up**. "
        "🔵 Blue bars push the risk **down**."
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[student_idx], show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # ── Raw feature values ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### This Student's Feature Values")
    st.markdown("*Scaled values shown — original data was normalised*")

    student_data = X_test.iloc[student_idx]
    shap_vals    = shap_values.values[student_idx]

    feature_df = pd.DataFrame({
        "Feature":    feature_names,
        "Value":      student_data.values.round(3),
        "SHAP Impact": shap_vals.round(4)
    }).sort_values("SHAP Impact", ascending=False)

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# PAGE 3: Try It Yourself
# ================================================================
elif page == "🧪 Try It Yourself":
    st.title("🧪 Try It Yourself")
    st.markdown(
        "Enter a student's details below to get a live "
        "dropout risk prediction with SHAP explanation."
    )
    st.info(
        "💡 These are the most important features. "
        "All others are set to the dataset average."
    )

    # ── Input form using top features ──────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        sem2_approved = st.slider(
            "2nd Semester Units Approved",
            min_value=0, max_value=20, value=5,
            help="Number of curricular units passed in semester 2"
        )
        sem1_approved = st.slider(
            "1st Semester Units Approved",
            min_value=0, max_value=20, value=5,
            help="Number of curricular units passed in semester 1"
        )
        sem2_grade = st.slider(
            "2nd Semester Grade (0–20)",
            min_value=0.0, max_value=20.0, value=12.0, step=0.5
        )
        age = st.slider(
            "Age at Enrollment",
            min_value=17, max_value=60, value=20
        )

    with col2:
        tuition = st.selectbox(
            "Tuition Fees Up to Date?",
            options=[1, 0],
            format_func=lambda x: "Yes ✓" if x == 1 else "No ✗"
        )
        scholarship = st.selectbox(
            "Scholarship Holder?",
            options=[0, 1],
            format_func=lambda x: "Yes ✓" if x == 1 else "No ✗"
        )
        sem2_enrolled = st.slider(
            "2nd Semester Units Enrolled",
            min_value=0, max_value=20, value=6
        )
        admission_grade = st.slider(
            "Admission Grade (0–200)",
            min_value=0.0, max_value=200.0, value=130.0, step=1.0
        )

    # ── Build input row ─────────────────────────────────────────
    # Start with the mean of all features as defaults,
    # then overwrite the ones the user provided
    base_input = X_test.mean().to_dict()

    user_values = {
        "Curricular units 2nd sem (approved)": sem2_approved,
        "Curricular units 1st sem (approved)": sem1_approved,
        "Curricular units 2nd sem (grade)":    sem2_grade,
        "Age at enrollment":                   age,
        "Tuition fees up to date":             tuition,
        "Scholarship holder":                  scholarship,
        "Curricular units 2nd sem (enrolled)": sem2_enrolled,
        "Admission grade":                     admission_grade,
    }
    base_input.update(user_values)
    input_df = pd.DataFrame([base_input])[feature_names]

    # ── Predict & explain ───────────────────────────────────────
    if st.button("🔮 Predict Dropout Risk", type="primary"):

        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.markdown("### Result")

        # Risk gauge using progress bar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prob >= 0.7:
                st.error(f"## 🔴 {prob*100:.1f}% Dropout Risk")
                st.markdown("**HIGH RISK** — This student needs immediate attention.")
            elif prob >= 0.4:
                st.warning(f"## 🟡 {prob*100:.1f}% Dropout Risk")
                st.markdown("**MEDIUM RISK** — Consider a check-in with this student.")
            else:
                st.success(f"## 🟢 {prob*100:.1f}% Dropout Risk")
                st.markdown("**LOW RISK** — Student appears to be on track.")

            st.progress(float(prob))

        # SHAP waterfall for this custom input
        st.markdown("---")
        st.markdown("### Why this score?")

        shap_input = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(shap_input[0], show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
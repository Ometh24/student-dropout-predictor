# ================================================================
# Stage 5: SHAP Explainability
#
# We use SHAP (SHapley Additive exPlanations) to explain
# WHY our XGBoost model makes each prediction.
#
# We generate 4 types of explanation:
#   1. Global bar chart  — which features matter most overall
#   2. Beeswarm plot     — how each feature affects predictions
#   3. Waterfall plot    — why the model scored ONE student
#   4. Dependence plot   — how one feature's effect changes
#
# All charts saved to a new folder: outputs/shap_plots/
# ================================================================

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")  # non-interactive backend — saves to file

# ----------------------------------------------------------------
# STEP 1: Load model and test data
# ----------------------------------------------------------------
print("=" * 55)
print("STEP 1: Loading model and data")
print("=" * 55)

X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

model = xgb.XGBClassifier()
model.load_model("data/processed/xgb_model.json")

print(f"  Model loaded ✓")
print(f"  Test set: {X_test.shape[0]} students, "
      f"{X_test.shape[1]} features")

os.makedirs("outputs/shap_plots", exist_ok=True)


# ----------------------------------------------------------------
# STEP 2: Create the SHAP explainer
#
# TreeExplainer is the fast version made specifically for
# tree-based models (XGBoost, Random Forest, etc.)
# It calculates exact SHAP values — no approximation needed.
#
# explainer(X_test) computes a SHAP value for EVERY feature
# for EVERY student in the test set.
# Shape of shap_values: (885 students × 36 features)
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 2: Calculating SHAP values")
print("=" * 55)
print("  This may take 20–30 seconds...")

explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print(f"  ✓ SHAP values computed")
print(f"  Shape: {shap_values.values.shape} "
      f"← (students × features)")
print(f"  Each number = how much that feature contributed "
      f"to that student's prediction")


# ----------------------------------------------------------------
# STEP 3: Global Bar Chart — feature importance
#
# Shows the AVERAGE impact of each feature across all students.
# Calculated as: mean(|SHAP value|) per feature.
#
# This answers: "Overall, which features matter most
# for predicting dropout across all students?"
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 3: Global bar chart (overall feature importance)")
print("=" * 55)

plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Top 15 Features — Mean Impact on Dropout Risk",
          fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_plots/1_global_bar.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_plots/1_global_bar.png")


# ----------------------------------------------------------------
# STEP 4: Beeswarm Plot — the full picture
#
# Every dot = one student.
# Horizontal position = SHAP value (left = lowers risk,
#                                    right = raises risk)
# Dot colour = actual feature value (red = high, blue = low)
#
# This is the most information-dense SHAP chart.
# It shows BOTH which features matter AND how they matter.
# e.g. "low 2nd sem approval rate (blue) → high SHAP value
#        (right) → raises dropout risk"
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 4: Beeswarm plot (how features affect predictions)")
print("=" * 55)

plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("How Features Push Dropout Risk Up or Down",
          fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_plots/2_beeswarm.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_plots/2_beeswarm.png")


# ----------------------------------------------------------------
# STEP 5: Waterfall Plot — explain ONE student
#
# This is the most powerful chart for real-world use.
# It shows exactly why the model gave one specific student
# their risk score — feature by feature.
#
# We pick the student the model is MOST confident will
# drop out — the highest predicted dropout probability.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 5: Waterfall plot (explaining one student)")
print("=" * 55)

# Find the student with highest predicted dropout probability
probs = model.predict_proba(X_test)[:, 1]
highest_risk_idx = int(np.argmax(probs))
predicted_prob   = probs[highest_risk_idx]
actual_outcome   = y_test.iloc[highest_risk_idx]

print(f"  Highest risk student: index {highest_risk_idx}")
print(f"  Predicted dropout probability: {predicted_prob*100:.1f}%")
print(f"  Actual outcome: "
      f"{'Dropout ✓' if actual_outcome == 1 else 'Not Dropout ✗'}")

plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_values[highest_risk_idx], show=False)
plt.title(
    f"Why Student #{highest_risk_idx} Got "
    f"{predicted_prob*100:.0f}% Dropout Risk",
    fontsize=12, pad=15
)
plt.tight_layout()
plt.savefig("outputs/shap_plots/3_waterfall_high_risk.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_plots/3_waterfall_high_risk.png")


# ----------------------------------------------------------------
# STEP 6: Waterfall for a LOW risk student too
#
# Showing both high and low risk side by side in the
# dashboard makes the explanation much more compelling.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 6: Waterfall plot (low risk student)")
print("=" * 55)

lowest_risk_idx = int(np.argmin(probs))
lowest_prob     = probs[lowest_risk_idx]
actual_low      = y_test.iloc[lowest_risk_idx]

print(f"  Lowest risk student: index {lowest_risk_idx}")
print(f"  Predicted dropout probability: {lowest_prob*100:.1f}%")
print(f"  Actual outcome: "
      f"{'Dropout' if actual_low == 1 else 'Not Dropout ✓'}")

plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_values[lowest_risk_idx], show=False)
plt.title(
    f"Why Student #{lowest_risk_idx} Got "
    f"{lowest_prob*100:.0f}% Dropout Risk",
    fontsize=12, pad=15
)
plt.tight_layout()
plt.savefig("outputs/shap_plots/4_waterfall_low_risk.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_plots/4_waterfall_low_risk.png")


# ----------------------------------------------------------------
# STEP 7: Dependence Plot
#
# Shows how ONE feature's effect changes across its range.
# We use 2nd semester approval rate — usually the strongest
# predictor of dropout.
#
# X axis = actual feature value
# Y axis = SHAP value (contribution to dropout risk)
# Colour = a second feature SHAP auto-selects as interacting
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 7: Dependence plot")
print("=" * 55)

feature = "Curricular units 2nd sem (approved)"
feat_idx = list(X_test.columns).index(feature)

plt.figure(figsize=(9, 6))
shap.dependence_plot(
    feat_idx,
    shap_values.values,
    X_test,
    feature_names=list(X_test.columns),
    show=False
)
plt.title(
    "How 2nd Semester Approvals Affects Dropout Risk",
    fontsize=12, pad=15
)
plt.tight_layout()
plt.savefig("outputs/shap_plots/5_dependence.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_plots/5_dependence.png")


# ----------------------------------------------------------------
# STEP 8: Print the top 10 features in plain English
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 8: Top 10 most important features (plain English)")
print("=" * 55)

mean_shap = np.abs(shap_values.values).mean(axis=0)
importance = pd.Series(mean_shap, index=X_test.columns)
importance = importance.sort_values(ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>10}")
print("  " + "-" * 63)
for rank, (feat, val) in enumerate(
        importance.head(10).items(), 1):
    bar = "█" * int(val * 80)
    print(f"  {rank:<6} {feat:<45} {val:>10.4f}  {bar}")

print("\n✓ Stage 5 complete — all SHAP plots saved!")
print("  Check the outputs/shap_plots/ folder")
print("  Next: Stage 6 — Streamlit dashboard")
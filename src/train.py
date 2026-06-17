# ================================================================
# Stage 4: Model Training
#
# We train 3 models and compare them honestly.
# The winner goes forward to Stage 5 (SHAP explainability).
#
# Key metric we care about most: RECALL for the Dropout class.
# Why recall and not accuracy?
#   - Accuracy: "how often is the model right overall?"
#   - Recall:   "of all real dropouts, how many did we catch?"
#
# Missing a student who IS going to drop out is much worse
# than a false alarm. So we optimise for catching dropouts.
# ================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os

# ----------------------------------------------------------------
# STEP 1: Load the processed data from Stage 3
# ----------------------------------------------------------------
print("=" * 55)
print("STEP 1: Loading processed data")
print("=" * 55)

X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")


# ----------------------------------------------------------------
# STEP 2: Handle class imbalance
#
# Remember: 2.1 non-dropouts for every 1 dropout.
# We tell each model to pay MORE attention to dropout cases
# by giving them higher weight during training.
#
# compute_class_weight calculates the right weights
# automatically based on how imbalanced the classes are.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 2: Computing class weights for imbalance")
print("=" * 55)

classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"  Weight for Not Dropout (0): {weights[0]:.3f}")
print(f"  Weight for Dropout     (1): {weights[1]:.3f}")
print("  → Model will penalise missing a dropout more heavily")


# ----------------------------------------------------------------
# STEP 3: Define a helper function to evaluate any model
#
# Instead of repeating the same evaluation code 3 times,
# we write it once as a function and call it 3 times.
# This is good coding practice — DRY (Don't Repeat Yourself).
# ----------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    """Print a clear evaluation summary for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Not Dropout", "Dropout"],
        output_dict=True
    )

    dropout_recall    = report["Dropout"]["recall"]
    dropout_precision = report["Dropout"]["precision"]
    dropout_f1        = report["Dropout"]["f1-score"]
    overall_accuracy  = report["accuracy"]

    print(f"\n  ── {name} ──")
    print(f"  Overall accuracy:   {overall_accuracy*100:.1f}%")
    print(f"  ROC-AUC score:      {auc:.3f}  "
          f"(1.0 = perfect, 0.5 = random guessing)")
    print(f"  Dropout recall:     {dropout_recall*100:.1f}%  "
          f"← % of real dropouts we caught")
    print(f"  Dropout precision:  {dropout_precision*100:.1f}%  "
          f"← % of our alerts that were correct")
    print(f"  Dropout F1:         {dropout_f1:.3f}")
    print(f"  Confusion matrix:")
    print(f"    Predicted →      Not Dropout  Dropout")
    print(f"    Actually Not Dropout  {cm[0][0]:>5}       {cm[0][1]:>5}")
    print(f"    Actually Dropout      {cm[1][0]:>5}       {cm[1][1]:>5}")

    return auc, dropout_recall


# ----------------------------------------------------------------
# STEP 4: Train Model 1 — Logistic Regression
#
# The simplest model. Finds a straight boundary between
# dropouts and non-dropouts. Fast to train, easy to explain.
# We use it as a baseline — if a complex model can't beat
# this, something is wrong.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 3-5: Training and evaluating all 3 models")
print("=" * 55)

print("\nTraining Model 1: Logistic Regression...")
lr = LogisticRegression(
    class_weight=class_weight_dict,
    max_iter=1000,       # give it enough iterations to converge
    random_state=42
)
lr.fit(X_train, y_train)
lr_auc, lr_recall = evaluate(
    "Logistic Regression", lr, X_test, y_test)


# ----------------------------------------------------------------
# STEP 5: Train Model 2 — Random Forest
#
# Builds 100 decision trees, each trained on a random
# subset of the data and features.
# Final prediction = majority vote across all 100 trees.
# Much harder to overfit than a single tree.
# ----------------------------------------------------------------
print("\nTraining Model 2: Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,    # 100 trees
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1            # use all CPU cores to train faster
)
rf.fit(X_train, y_train)
rf_auc, rf_recall = evaluate(
    "Random Forest", rf, X_test, y_test)


# ----------------------------------------------------------------
# STEP 6: Train Model 3 — XGBoost
#
# Gradient boosting: builds trees one at a time, where
# each new tree focuses on correcting the mistakes of
# the previous trees.
#
# scale_pos_weight handles imbalance for XGBoost specifically:
# it's the ratio of non-dropouts to dropouts.
# ----------------------------------------------------------------
print("\nTraining Model 3: XGBoost...")
scale = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale,  # handles imbalance
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss",
    verbosity=0              # suppress training output
)
xgb_model.fit(X_train, y_train)
xgb_auc, xgb_recall = evaluate(
    "XGBoost", xgb_model, X_test, y_test)


# ----------------------------------------------------------------
# STEP 7: Compare all 3 and pick a winner
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 7: Model comparison summary")
print("=" * 55)

results = {
    "Logistic Regression": (lr_auc, lr_recall),
    "Random Forest":       (rf_auc, rf_recall),
    "XGBoost":             (xgb_auc, xgb_recall),
}

print(f"\n  {'Model':<25} {'ROC-AUC':>8}  {'Dropout Recall':>15}")
print("  " + "-" * 50)
for name, (auc, recall) in results.items():
    print(f"  {name:<25} {auc:>8.3f}  {recall*100:>14.1f}%")

best = max(results, key=lambda k: results[k][0])
print(f"\n  ✓ Best model by ROC-AUC: {best}")


# ----------------------------------------------------------------
# STEP 8: Save the best model
#
# joblib saves the trained model to a file so we can
# load it in Stage 5 (SHAP) without retraining.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 8: Saving best model")
print("=" * 55)

os.makedirs("data/processed", exist_ok=True)

# We save XGBoost regardless — it's what SHAP works best with
xgb_model.save_model("data/processed/xgb_model.json")
joblib.dump(lr, "data/processed/lr_model.pkl")
joblib.dump(rf, "data/processed/rf_model.pkl")

print("  ✓ Saved xgb_model.json")
print("  ✓ Saved lr_model.pkl")
print("  ✓ Saved rf_model.pkl")
print("\n✓ Stage 4 complete — models trained and saved!")
print("  Next: Stage 5 — SHAP explainability")
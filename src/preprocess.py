# ================================================================
# Stage 3: Data Preprocessing
#
# We're preparing the raw data so a machine learning model
# can actually learn from it. We do this in 4 clear steps.
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ----------------------------------------------------------------
# STEP 1: Load the data
# ----------------------------------------------------------------
print("=" * 55)
print("STEP 1: Loading data")
print("=" * 55)

df = pd.read_csv("data/raw/students.csv", sep=";")
print(f"  Loaded {df.shape[0]} students, {df.shape[1]} columns")


# ----------------------------------------------------------------
# STEP 2: Simplify the target column
#
# Original: "Graduate", "Dropout", "Enrolled"  (3 classes)
# New:       1 = Dropout,  0 = Not Dropout      (2 classes)
#
# Why? Because the question we want to answer is simply:
# "Is this student at risk of dropping out — yes or no?"
# That's what a university advisor can actually act on.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 2: Simplifying target to binary (Dropout vs Not)")
print("=" * 55)

df["Target"] = df["Target"].map({
    "Dropout":  1,
    "Graduate": 0,
    "Enrolled": 0
})

counts = df["Target"].value_counts()
print(f"  Not Dropout (0): {counts[0]} students")
print(f"  Dropout     (1): {counts[1]} students")
print(f"  Ratio: 1 dropout for every "
      f"{counts[0]/counts[1]:.1f} non-dropouts")


# ----------------------------------------------------------------
# STEP 3: Separate features (X) from target (y)
#
# X = everything the model is ALLOWED to look at to make
#     its prediction (all columns except Target)
# y = the answer we're trying to predict (just Target)
#
# Think of X as the student's file, y as the outcome.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 3: Separating features (X) from target (y)")
print("=" * 55)

X = df.drop(columns=["Target"])
y = df["Target"]

print(f"  X shape: {X.shape}  ← 4424 students × 36 features")
print(f"  y shape: {y.shape}  ← 4424 answers (0 or 1)")


# ----------------------------------------------------------------
# STEP 4: Split into train and test sets
#
# We keep 20% of students hidden from the model during training.
# We only use them at the very end to check real performance.
#
# stratify=y means: keep the same dropout ratio in both sets.
# random_state=42 means: same split every time we run this.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 4: Train/test split (80% train, 20% test)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps dropout ratio balanced in both sets
)

print(f"  Training set:  {X_train.shape[0]} students")
print(f"  Test set:      {X_test.shape[0]} students")
print(f"  Dropout % in train: "
      f"{y_train.mean()*100:.1f}%")
print(f"  Dropout % in test:  "
      f"{y_test.mean()*100:.1f}%")
print("  ✓ Stratify worked — both sets have same dropout ratio")


# ----------------------------------------------------------------
# STEP 5: Scale the features
#
# Problem: "Age at enrollment" ranges 17–70.
#          "Curricular units approved" ranges 0–26.
#          Some models get confused when one feature has
#          much bigger numbers than another.
#
# Solution: StandardScaler shifts every column so that:
#   - The average value becomes 0
#   - Values are measured in "how far from average" units
#
# IMPORTANT RULE: we fit the scaler ONLY on training data,
# then apply it to both. Why? Because in real life, you
# won't have test data when you build the scaler. Fitting
# on test data would be "cheating" — leaking future info.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 5: Scaling features")
print("=" * 55)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn + apply
X_test_scaled  = scaler.transform(X_test)        # apply only

print("  Before scaling — 'Age at enrollment' range:")
print(f"    min={X_train['Age at enrollment'].min()},  "
      f"max={X_train['Age at enrollment'].max()}")

# Convert back to DataFrame to check
import numpy as np
age_col_index = list(X.columns).index("Age at enrollment")
print("  After scaling — same column:")
print(f"    min={X_train_scaled[:, age_col_index].min():.2f},  "
      f"max={X_train_scaled[:, age_col_index].max():.2f}")
print("  ✓ Now centred around 0")


# ----------------------------------------------------------------
# STEP 6: Save the processed data
#
# We save to files so Stage 4 (modelling) can just load them
# directly without repeating all this work.
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 6: Saving processed data")
print("=" * 55)

os.makedirs("data/processed", exist_ok=True)

# Save as CSV — keeps column names intact
pd.DataFrame(X_train_scaled,
             columns=X.columns).to_csv(
    "data/processed/X_train.csv", index=False)

pd.DataFrame(X_test_scaled,
             columns=X.columns).to_csv(
    "data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv( "data/processed/y_test.csv",  index=False)

# Also save the column names for later use in dashboard
pd.Series(X.columns).to_csv(
    "data/processed/feature_names.csv", index=False)

print("  ✓ Saved X_train.csv")
print("  ✓ Saved X_test.csv")
print("  ✓ Saved y_train.csv")
print("  ✓ Saved y_test.csv")
print("  ✓ Saved feature_names.csv")
print("\n  All files in data/processed/")
print("\n✓ Stage 3 complete — data is ready for modelling!")
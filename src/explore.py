# ================================================================
# Stage 2: Exploratory Data Analysis (EDA)
# We're just LOOKING at the data here — no modelling yet.
# The goal is to understand what columns exist, spot problems,
# and find patterns that will guide our model later.
# ================================================================

import pandas as pd

# ----------------------------------------------------------------
# STEP 1: Load the data
# The file uses semicolons (;) instead of commas to separate
# columns — that's why we pass sep=";"
# ----------------------------------------------------------------
df = pd.read_csv("data/raw/students.csv", sep=";")

print("=" * 50)
print("STEP 1: Basic shape")
print("=" * 50)
print(f"Rows (students): {df.shape[0]}")
print(f"Columns (features): {df.shape[1]}")

# ----------------------------------------------------------------
# STEP 2: See the column names
# These are all the things we know about each student
# ----------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: Column names")
print("=" * 50)
for i, col in enumerate(df.columns):
    print(f"  {i+1:>2}. {col}")

# ----------------------------------------------------------------
# STEP 3: Look at the target column
# "Target" is what we're trying to predict.
# We need to know what values it contains and how many of each.
# ----------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: Target column — what we're predicting")
print("=" * 50)
counts = df["Target"].value_counts()
total = len(df)
for label, count in counts.items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<12} {count:>5} students  ({pct:.1f}%)  {bar}")

# ----------------------------------------------------------------
# STEP 4: Check for missing values
# Missing values cause models to crash or give wrong answers.
# We need to know if any exist before going further.
# ----------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4: Missing values check")
print("=" * 50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found — clean dataset!")
else:
    print(missing[missing > 0])

# ----------------------------------------------------------------
# STEP 5: Peek at the first 3 rows
# Just to see what actual data looks like
# ----------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: First 3 rows (first 6 columns)")
print("=" * 50)
print(df.iloc[:3, :6].to_string())

print("\n✓ EDA complete — tell me what you see!")
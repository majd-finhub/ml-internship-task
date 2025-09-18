# 0) Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packaging import version
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# 1) Load data
CSV_PATH = "Stock Prices Data Set.csv"  # <- update if needed
df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df.columns = [c.strip().lower() for c in df.columns]
df = (df
      .sort_values(["symbol", "date"])
      .drop_duplicates(["symbol", "date"])
      .reset_index(drop=True))
assert df.duplicated(["symbol","date"]).sum()==0, "Duplicates remain. Check input."

# 2) Target engineering (next-day return + direction)
df["return"] = df.groupby("symbol")["close"].pct_change()
df["targeted_return"] = df.groupby("symbol")["return"].shift(-1)
df["targeted_direction"] = (df["targeted_return"] > 0).astype(int)
df = df.dropna(subset=["targeted_return", "targeted_direction"]).reset_index(drop=True)

# 3) Feature sets
numerical_columns   = ["open","high","low","close","volume"]
categorical_columns = ["symbol"]
meta_columns        = ["date"]

feature_columns = numerical_columns + categorical_columns
X = df[feature_columns].copy()
y_reg = df["targeted_return"].copy()
y_clf = df["targeted_direction"].copy()

# 4) Time-based split (80% quantile)
cutoff = df["date"].quantile(0.8)
train_mask = df["date"] <= cutoff
test_mask  = df["date"] >  cutoff

X_train, X_test = X[train_mask], X[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]
y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]

n_train = train_mask.sum()
assert len(X_train)==n_train and len(y_reg_train)==n_train and len(y_clf_train)==n_train

# 5) Build preprocessing (version-safe OHE)
ohe_kwargs = {"handle_unknown": "ignore"}
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(**ohe_kwargs))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numerical_columns),
        ("cat", categorical_pipe, categorical_columns),
    ],
    remainder="drop"
)

# 6) Fit & transform
preprocessor.fit(X_train)
Xt_train = preprocessor.transform(X_train)
Xt_test  = preprocessor.transform(X_test)

# 7) Linear Regression (regress next-day return)
lin = LinearRegression()
lin.fit(Xt_train, y_reg_train)
y_reg_pred = lin.predict(Xt_test)

rmse = mean_squared_error(y_reg_test, y_reg_pred)
r2   = r2_score(y_reg_test, y_reg_pred)

# Parity plot
plt.figure()
mn = float(min(y_reg_test.min(), y_reg_pred.min()))
mx = float(max(y_reg_test.max(), y_reg_pred.max()))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True next-day return")
plt.ylabel("Predicted next-day return")
plt.title("Linear Regression — Parity (Test)")
plt.tight_layout()
plt.show()

# Time plot (first N points)
N = min(300, len(y_reg_test))
plt.figure()
plt.plot(y_reg_test.iloc[:N].index, y_reg_test.iloc[:N].values, label="True")
plt.plot(y_reg_test.iloc[:N].index, y_reg_pred[:N], label="Pred")
plt.xlabel("Index (test subset)")
plt.ylabel("Next-day return")
plt.title("Linear Regression — True vs Pred (first N)")
plt.legend()
plt.tight_layout()
plt.show()

print("Linear Regression Results")
print("  RMSE:", rmse)
print("  R²  :", r2)

# 8) KNN Classifier on direction
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xt_train, y_clf_train)
y_clf_pred = knn.predict(Xt_test)

acc = accuracy_score(y_clf_test, y_clf_pred)
prec = precision_score(y_clf_test, y_clf_pred, zero_division=0)
rec  = recall_score(y_clf_test, y_clf_pred, zero_division=0)
f1   = f1_score(y_clf_test, y_clf_pred, zero_division=0)

print("\nKNN Classifier Results")
print("  Accuracy :", acc)
print("  Precision:", prec)
print("  Recall   :", rec)
print("  F1 Score :", f1)
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_clf_pred))

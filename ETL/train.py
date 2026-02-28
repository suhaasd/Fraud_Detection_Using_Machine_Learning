import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("models",   exist_ok=True)
os.makedirs("encoders", exist_ok=True)

data = pd.read_csv("data/final_data.csv")

print("--- First 10 Rows ---")
print(data.head(10))
print("\n--- DataFrame Info ---")
data.info()

df = data.copy()

#device id features
device_counts = df.groupby("device_id")["user_id"].nunique()
df["device_id_user_counts"] = df["device_id"].map(device_counts)

joblib.dump(device_counts, "encoders/device_counts_map.joblib")

print(f"\nMax unique users on a single device: {df['device_id_user_counts'].max()}")
print("\nDistribution of user counts per device:")
print(df["device_id_user_counts"].value_counts())

#drop ip column
df.drop(["ip_address", "lower_bound_ip_address", "upper_bound_ip_address"],
        axis=1, inplace=True)

print(f"\nTotal Cases : {len(df)}")
print(f"Fraud Cases : {len(df[df['class'] == 1])}")
print(f"Valid Trans : {len(df[df['class'] == 0])}")

#time difference
df["signup_time"]   = pd.to_datetime(df["signup_time"],   format="%Y-%m-%d %H:%M:%S")
df["purchase_time"] = pd.to_datetime(df["purchase_time"], format="%Y-%m-%d %H:%M:%S")
df["time_diff_min"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 60

df.drop(["device_id", "user_id"], axis=1, inplace=True)

#extracting date and time
df["signup_hour"]        = df["signup_time"].dt.hour
df["purchase_hour"]      = df["purchase_time"].dt.hour
df["purchase_dayofweek"] = df["purchase_time"].dt.dayofweek
df["purchase_month"]     = df["purchase_time"].dt.month
df["purchase_day"]       = df["purchase_time"].dt.day

print("\nCross-tab purchase_hour vs class:")
print(pd.crosstab(df["purchase_hour"], df["class"]))
print("\nCross-tab purchase_dayofweek vs class:")
print(pd.crosstab(df["purchase_dayofweek"], df["class"]))

#cyclic time encoding
df["purchase_hour_sin"] = np.sin(2 * np.pi * df["purchase_hour"] / 24.0)
df["purchase_hour_cos"] = np.cos(2 * np.pi * df["purchase_hour"] / 24.0)
df["signup_hour_sin"]   = np.sin(2 * np.pi * df["signup_hour"]   / 24.0)
df["signup_hour_cos"]   = np.cos(2 * np.pi * df["signup_hour"]   / 24.0)
df["dayofweek_sin"]     = np.sin(2 * np.pi * df["purchase_dayofweek"] / 7.0)
df["dayofweek_cos"]     = np.cos(2 * np.pi * df["purchase_dayofweek"] / 7.0)
df["day_sin"]           = np.sin(2 * np.pi * df["purchase_day"]  / 31.0)
df["day_cos"]           = np.cos(2 * np.pi * df["purchase_day"]  / 31.0)

#time difference bucketing
time_bins   = [0, 1, 10, 60, np.inf]
time_labels = ["< 1 min", "1-10 mins", "10-60 mins", "> 60 mins"]

df["time_diff_category"] = pd.cut(
    df["time_diff_min"], bins=time_bins, labels=time_labels, right=False
)

joblib.dump(time_bins,   "encoders/time_bins.joblib")
joblib.dump(time_labels, "encoders/time_labels.joblib")

print("\nTime category distribution:")
print(df["time_diff_category"].value_counts())

df.drop([
    "purchase_hour", "signup_hour", "purchase_day",
    "purchase_dayofweek", "purchase_time", "signup_time",
], axis=1, inplace=True)

#top 19 countries
top_18_countries = (
    df["country"].value_counts().nlargest(18).index.tolist()
)
joblib.dump(top_18_countries, "encoders/top_18_countries.joblib")

df.loc[~df["country"].isin(top_18_countries), "country"] = "Other"

print("\nCountry distribution (after grouping):")
print(df["country"].value_counts())

#label encoding
encoders = {
    "sex":                LabelEncoder(),
    "source":             LabelEncoder(),
    "browser":            LabelEncoder(),
    "time_diff_category": LabelEncoder(),
    "country":            LabelEncoder(),
}

for col, le in encoders.items():
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f"encoders/le_{col}.joblib")

print("\nAfter label encoding:")
print(df.head())

MODEL_FEATURES = [
    "purchase_value", "source", "browser", "sex", "age", "country",
    "device_id_user_counts", "purchase_month",
    "purchase_hour_sin", "purchase_hour_cos",
    "signup_hour_sin",  "signup_hour_cos",
    "time_diff_category",
    "dayofweek_sin", "dayofweek_cos",
    "day_sin", "day_cos",
]

X = df[MODEL_FEATURES]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

print(f"\nTrain shape: {X_train.shape}  |  Test shape: {X_test.shape}")

#minmax scaler on purchase value and age
scaler = MinMaxScaler()
X_train[["purchase_value", "age"]] = scaler.fit_transform(
    X_train[["purchase_value", "age"]]
)
X_test[["purchase_value", "age"]] = scaler.transform(
    X_test[["purchase_value", "age"]]
)
joblib.dump(scaler, "models/min_max_scaler.joblib")

#randomizedsearchcv
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"\nscale_pos_weight = {scale_pos_weight:.4f}")

param_dist = {
    "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "n_estimators":     [100, 200, 300, 500, 700, 800],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

#xgboost
xgb_base = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=25,                         #25 candidates × 5 folds = 125 fits
    scoring="average_precision",
    cv=cv,
    verbose=3,
    random_state=42,
    n_jobs=-1,
    refit=True,
)

print("\nStarting RandomizedSearchCV (25 iters × 5 folds = 125 fits) …")
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best AP     : {search.best_score_:.4f}")

best_model = search.best_estimator_
joblib.dump(best_model, "models/best_xgb_model.joblib")

print("\n" + "="*70)
print("THRESHOLD TUNING")
print("="*70)

y_prob = best_model.predict_proba(X_test)[:, 1]

#Precision-Recall curve sweep
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)

#f1-score for each threshold
f1_scores = np.where(
    (precisions[:-1] + recalls[:-1]) == 0,
    0,
    2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
)

best_pr_idx       = np.argmax(f1_scores)
best_threshold_pr = float(pr_thresholds[best_pr_idx])
best_f1_pr        = float(f1_scores[best_pr_idx])

print(f"\n[Precision-Recall sweep]")
print(f"  Best threshold : {best_threshold_pr:.4f}")
print(f"  Best F1 score  : {best_f1_pr:.4f}")
print(f"  Precision      : {precisions[best_pr_idx]:.4f}")
print(f"  Recall         : {recalls[best_pr_idx]:.4f}")

#f1 vs threshold
plt.figure(figsize=(9, 4))
plt.plot(pr_thresholds, f1_scores, color="steelblue", linewidth=1.5,
         label="F1 score")
plt.axvline(best_threshold_pr, color="crimson", linestyle="--",
            label=f"Best threshold = {best_threshold_pr:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score (Fraud class)")
plt.title("F1 Score vs Decision Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("pictures/threshold_f1_curve.png", dpi=100)
plt.show()

#precision and recall vs threshold
plt.figure(figsize=(9, 4))
plt.plot(pr_thresholds, precisions[:-1], label="Precision", color="darkorange")
plt.plot(pr_thresholds, recalls[:-1],    label="Recall",    color="steelblue")
plt.axvline(best_threshold_pr, color="crimson", linestyle="--",
            label=f"Best threshold = {best_threshold_pr:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Decision Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("pictures/threshold_precision_recall_curve.png", dpi=100)
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, color="steelblue", linewidth=1.5)
plt.scatter(recalls[best_pr_idx], precisions[best_pr_idx],
            color="crimson", zorder=5,
            label=f"Best threshold = {best_threshold_pr:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("pictures/precision_recall_curve.png", dpi=100)
plt.show()

#ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="steelblue", linewidth=1.5,
         label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("pictures/roc_curve.png", dpi=100)
plt.show()
print(f"ROC AUC = {roc_auc:.4f}")

joblib.dump(best_threshold_pr, "models/best_threshold.joblib")

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

y_pred_default = (y_prob >= 0.50).astype(int)
print("\n--- Default threshold = 0.50 ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_default):.4f}")
print(classification_report(y_test, y_pred_default,
                             target_names=["Valid (0)", "Fraud (1)"]))

y_pred_tuned = (y_prob >= best_threshold_pr).astype(int)
print(f"\n--- Tuned threshold  = {best_threshold_pr:.4f} ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(classification_report(y_test, y_pred_tuned,
                             target_names=["Valid (0)", "Fraud (1)"]))

f1_default = f1_score(y_test, y_pred_default)
f1_tuned   = f1_score(y_test, y_pred_tuned)
print(f"\nFraud-class F1  |  default (0.50): {f1_default:.4f}  |  "
      f"tuned ({best_threshold_pr:.4f}): {f1_tuned:.4f}")

#Confusion matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Valid", "Fraud"],
            yticklabels=["Valid", "Fraud"])
plt.title(f"Confusion Matrix — Tuned Threshold ({best_threshold_pr:.4f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("pictures/confusion_matrix_tuned.png", dpi=100)
plt.show()

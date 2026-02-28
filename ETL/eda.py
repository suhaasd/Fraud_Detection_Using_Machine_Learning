import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

df = pd.read_csv("data/Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])

print("Shape (rows, cols):", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\ndtypes & non-null counts")
df.info()

print("\n=== Missing values per column")
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])

#time features
df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
df["purchase_hour"]    = df["purchase_time"].dt.hour
df["purchase_day"]     = df["purchase_time"].dt.day_name()
print("\nAdded columns: ['time_to_purchase', 'purchase_hour', 'purchase_day']")

#class balance
counts = df["class"].value_counts().sort_index()
print("\nClass balance:\n", counts)

plt.figure()
plt.bar(["Not fraud (0)", "Fraud (1)"], counts.values)
plt.title("Class Balance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("pictures/eda_class_balance.png", dpi=100)
plt.show()

#histograms
for col, title in [
    ("purchase_value", "Distribution: purchase_value"),
    ("age",            "Distribution: age"),
]:
    plt.figure()
    df[col].dropna().hist(bins=30)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"pictures/eda_hist_{col}.png", dpi=100)
    plt.show()

# log1p of time_to_purchase
plt.figure()
np.log1p(df["time_to_purchase"].clip(lower=0)).dropna().hist(bins=30)
plt.title("Distribution: log1p(time_to_purchase)")
plt.xlabel("log1p(seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("pictures/eda_hist_time_to_purchase.png", dpi=100)
plt.show()

#categorical columns
cat_cols = [c for c in ["source", "browser", "sex"] if c in df.columns]

for col in cat_cols:
    print(f"\nValue counts for '{col}':\n", df[col].value_counts())

    top_counts = df[col].value_counts().head(10)
    plt.figure()
    top_counts.plot(kind="bar")
    plt.title(f"Top {len(top_counts)} {col} (counts)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"pictures/eda_bar_{col}_counts.png", dpi=100)
    plt.show()

    fraud_rate = df.groupby(col)["class"].mean().sort_values(ascending=False).head(10)
    print(f"\nFraud rate by {col} (top 10):\n", fraud_rate)

    plt.figure()
    fraud_rate.plot(kind="bar")
    plt.title(f"Top {len(fraud_rate)} {col} by Fraud Rate (mean of class)")
    plt.ylabel("Fraud rate")
    plt.tight_layout()
    plt.savefig(f"pictures/eda_bar_{col}_fraud_rate.png", dpi=100)
    plt.show()

#purchase hour and day of the week
hour_counts = df["purchase_hour"].value_counts().sort_index()
plt.figure()
hour_counts.plot(kind="bar")
plt.title("Purchases by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("pictures/eda_bar_hour_counts.png", dpi=100)
plt.show()

hour_fraud_rate = df.groupby("purchase_hour")["class"].mean()
plt.figure()
hour_fraud_rate.plot(kind="bar")
plt.title("Fraud Rate by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Fraud rate")
plt.tight_layout()
plt.savefig("pictures/eda_bar_hour_fraud_rate.png", dpi=100)
plt.show()

day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_counts = df["purchase_day"].value_counts().reindex(day_order)
plt.figure()
day_counts.plot(kind="bar")
plt.title("Purchases by Day of Week")
plt.xlabel("Day")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("pictures/eda_bar_day_counts.png", dpi=100)
plt.show()

day_fraud_rate = df.groupby("purchase_day")["class"].mean().reindex(day_order)
plt.figure()
day_fraud_rate.plot(kind="bar")
plt.title("Fraud Rate by Day of Week")
plt.xlabel("Day")
plt.ylabel("Fraud rate")
plt.tight_layout()
plt.savefig("pictures/eda_bar_day_fraud_rate.png", dpi=100)
plt.show()

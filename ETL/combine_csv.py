import pandas as pd
import numpy as np

fraud = pd.read_csv("data/Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
ip_map = pd.read_csv(
    "data/IpAddress_to_Country.csv",
    dtype={"lower_bound_ip_address": float, "upper_bound_ip_address": float},
)

print(f"Fraud_Data shape     : {fraud.shape}")
print(f"IpAddress_to_Country : {ip_map.shape}")

#sort ip for merge_asof
ip_map = ip_map.sort_values("lower_bound_ip_address").reset_index(drop=True)
fraud_sorted = fraud.sort_values("ip_address").reset_index(drop=True)

#merge by nearest lower bound ip
merged = pd.merge_asof(
    fraud_sorted,
    ip_map,
    left_on="ip_address",
    right_on="lower_bound_ip_address",
    direction="backward",
)

# Drop rows where the IP falls outside every range
merged = merged[merged["ip_address"] <= merged["upper_bound_ip_address"]]

final_data = merged.sort_values("user_id").reset_index(drop=True)

print(f"\nfinal_data shape     : {final_data.shape}")
print(final_data.head(5))

final_data.to_csv("data/final_data.csv", index=False)
import gradio as gr
import pandas as pd
import numpy as np
import joblib

model            = joblib.load("models/best_xgb_model.joblib")
scaler           = joblib.load("models/min_max_scaler.joblib")
best_threshold   = joblib.load("models/best_threshold.joblib")

device_map       = joblib.load("encoders/device_counts_map.joblib")
top_18_countries = joblib.load("encoders/top_18_countries.joblib")
time_bins        = joblib.load("encoders/time_bins.joblib")
time_labels      = joblib.load("encoders/time_labels.joblib")

le_sex           = joblib.load("encoders/le_sex.joblib")
le_source        = joblib.load("encoders/le_source.joblib")
le_browser       = joblib.load("encoders/le_browser.joblib")
le_time_category = joblib.load("encoders/le_time_diff_category.joblib")
le_country       = joblib.load("encoders/le_country.joblib")

MODEL_FEATURES = [
    "purchase_value", "source", "browser", "sex", "age", "country",
    "device_id_user_counts", "purchase_month",
    "purchase_hour_sin", "purchase_hour_cos",
    "signup_hour_sin",   "signup_hour_cos",
    "time_diff_category",
    "dayofweek_sin", "dayofweek_cos",
    "day_sin",       "day_cos",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["signup_time"]   = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["device_id_user_counts"] = df["device_id"].map(device_map).fillna(1).astype(int)
    df["time_diff_min"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 60

    df["signup_hour"]        = df["signup_time"].dt.hour
    df["purchase_hour"]      = df["purchase_time"].dt.hour
    df["purchase_dayofweek"] = df["purchase_time"].dt.dayofweek
    df["purchase_month"]     = df["purchase_time"].dt.month
    df["purchase_day"]       = df["purchase_time"].dt.day

    df["purchase_hour_sin"] = np.sin(2 * np.pi * df["purchase_hour"] / 24.0)
    df["purchase_hour_cos"] = np.cos(2 * np.pi * df["purchase_hour"] / 24.0)

    df["signup_hour_sin"] = np.sin(2 * np.pi * df["signup_hour"] / 24.0)
    df["signup_hour_cos"] = np.cos(2 * np.pi * df["signup_hour"] / 24.0)

    df["dayofweek_sin"] = np.sin(2 * np.pi * df["purchase_dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["purchase_dayofweek"] / 7.0)

    df["day_sin"] = np.sin(2 * np.pi * df["purchase_day"] / 31.0)
    df["day_cos"] = np.cos(2 * np.pi * df["purchase_day"] / 31.0)

    df["time_diff_category"] = pd.cut(
        df["time_diff_min"], bins=time_bins, labels=time_labels, right=False
    )

    df.loc[~df["country"].isin(top_18_countries), "country"] = "Other"

    df["sex"]                = le_sex.transform(df["sex"])
    df["source"]             = le_source.transform(df["source"])
    df["browser"]            = le_browser.transform(df["browser"])
    df["time_diff_category"] = le_time_category.transform(df["time_diff_category"].astype(str))  # ← fixed
    df["country"]            = le_country.transform(df["country"])

    df[["purchase_value", "age"]] = scaler.transform(df[["purchase_value", "age"]])

    return df[MODEL_FEATURES]

def predict_from_csv(uploaded_file):
    try:
        df_raw = pd.read_csv(uploaded_file.name)
    except Exception as e:
        return f"Error reading CSV: {e}", None

    try:
        X = engineer_features(df_raw)
    except Exception as e:
        return f"Feature engineering failed: {e}", None

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    df_raw["fraud_probability"] = y_prob.round(4)
    df_raw["prediction"]        = y_pred
    df_raw["prediction_label"]  = df_raw["prediction"].map({0: "✅ Valid", 1: "🚨 FRAUD"})

    n_fraud = y_pred.sum()
    summary = (
        f"**Total transactions:** {len(df_raw)}\n\n"
        f"**Flagged as FRAUD:** {n_fraud}  ({100 * n_fraud / len(df_raw):.1f}%)\n\n"
        f"**Valid:** {len(df_raw) - n_fraud}\n\n"
        f"*(Decision threshold: {best_threshold:.4f})*"
    )
    result_cols = ["prediction_label", "fraud_probability"] + [
        c for c in df_raw.columns if c not in ("prediction_label", "fraud_probability", "prediction")
    ]
    return summary, df_raw[result_cols].to_html(index=False)

def predict_single(
    user_id, device_id, signup_time, purchase_time,
    purchase_value, source, browser, sex, age, country,
):
    row = pd.DataFrame([{
        "user_id":        user_id,
        "device_id":      device_id,
        "signup_time":    signup_time,
        "purchase_time":  purchase_time,
        "purchase_value": float(purchase_value),
        "source":         source,
        "browser":        browser,
        "sex":            sex,
        "age":            int(age),
        "country":        country,
    }])

    try:
        X = engineer_features(row)
    except Exception as e:
        return f"Feature engineering error: {e}"

    prob  = model.predict_proba(X)[0, 1]
    label = "🚨 FRAUD" if prob >= best_threshold else "✅ Valid"
    return (
        f"{label}\n\n"
        f"Fraud probability : {prob:.4f}\n"
        f"Decision threshold: {best_threshold:.4f}"
    )


with gr.Blocks(title="Fraud Detection — XGBoost") as demo:
    gr.Markdown(
        "# 🔍 Fraud Detection — XGBoost\n"
    )

    with gr.Tab("📂 Batch CSV Prediction"):
        gr.Markdown(
            "Upload a CSV with columns: `user_id, device_id, signup_time, purchase_time, "
            "purchase_value, source, browser, sex, age, country`"
        )
        csv_input   = gr.File(label="Upload CSV", file_types=[".csv"])
        csv_button  = gr.Button("Predict")
        csv_summary = gr.Markdown()
        csv_table   = gr.HTML()
        csv_button.click(
            predict_from_csv,
            inputs=csv_input,
            outputs=[csv_summary, csv_table],
        )

    with gr.Tab("✏️ Single Transaction"):
        gr.Markdown("Fill in the transaction details below:")
        with gr.Row():
            uid    = gr.Textbox(label="User ID",   value="123456")
            did    = gr.Textbox(label="Device ID", value="ABCDEFGHIJKLM")
        with gr.Row():
            st     = gr.Textbox(label="Signup Time",   value="2015-01-01 10:00:00", placeholder="YYYY-MM-DD HH:MM:SS")
            pt     = gr.Textbox(label="Purchase Time", value="2015-01-01 10:00:01", placeholder="YYYY-MM-DD HH:MM:SS")
        with gr.Row():
            pv     = gr.Number(label="Purchase Value ($)", value=50)
            age_in = gr.Number(label="Age",               value=30)
        with gr.Row():
            src    = gr.Dropdown(label="Source",  choices=["SEO", "Ads", "Direct"],                         value="SEO")
            brw    = gr.Dropdown(label="Browser", choices=["Chrome", "Safari", "IE", "FireFox", "Opera"],   value="Chrome")
            sx     = gr.Dropdown(label="Sex",     choices=["M", "F"],                                       value="M")
        country_in = gr.Textbox(label="Country", value="United States")
        single_btn = gr.Button("Predict")
        single_out = gr.Textbox(label="Result")
        single_btn.click(
            predict_single,
            inputs=[uid, did, st, pt, pv, src, brw, sx, age_in, country_in],
            outputs=single_out,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
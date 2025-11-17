import gradio as gr
import pandas as pd
import numpy as np
import joblib

model = joblib.load('best_xgb_model.joblib')
scaler = joblib.load('min_max_scaler.joblib')
device_map = joblib.load('device_counts_map.joblib')
top_18_countries = joblib.load('top_18_countries.joblib')
time_bins = joblib.load('time_bins.joblib')
time_labels = joblib.load('time_labels.joblib')

le_sex = joblib.load('le_sex.joblib')
le_source = joblib.load('le_source.joblib')
le_browser = joblib.load('le_browser.joblib')
le_time_category = joblib.load('le_time_category.joblib')
le_country = joblib.load('le_country.joblib')


MODEL_FEATURES = [
    'purchase_value', 'source', 'browser', 'sex', 'age', 'country',
    'device_id_user_counts', 'purchase_month', 'purchase_hour_sin',
    'purchase_hour_cos', 'signup_hour_sin', 'signup_hour_cos',
    'time_diff_category', 'dayofweek_sin', 'dayofweek_cos', 'day_sin', 'day_cos'
]

def predict_fraud_csv(uploaded_file):
    
    try:
        df = pd.read_csv(uploaded_file.name)
    except Exception as e:
        return f"Error reading CSV: {e}"
        
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    
    df['device_id_user_counts'] = df['device_id'].map(device_map).fillna(1)
    
    df["time_diff_min"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 60
    
    df["signup_hour"] = df["signup_time"].dt.hour
    df["purchase_hour"] = df["purchase_time"].dt.hour
    df["purchase_dayofweek"] = df["purchase_time"].dt.dayofweek
    df["purchase_month"] = df["purchase_time"].dt.month
    df["purchase_day"] = df["purchase_time"].dt.day
    
    df['purchase_hour_sin'] = np.sin(2 * np.pi * df['purchase_hour']/24.0)
    df['purchase_hour_cos'] = np.cos(2 * np.pi * df['purchase_hour']/24.0)
    df['signup_hour_sin'] = np.sin(2 * np.pi * df['signup_hour']/24.0)
    df['signup_hour_cos'] = np.cos(2 * np.pi * df['signup_hour']/24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['purchase_dayofweek']/7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['purchase_dayofweek']/7.0)
    df['day_sin'] = np.sin(2 * np.pi * df['purchase_day']/31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['purchase_day']/31.0)

    df['time_diff_category'] = pd.cut(df['time_diff_min'], bins=time_bins, labels=time_labels, right=False)
    
    df.loc[~df['country'].isin(top_18_countries), 'country'] = 'Other'

    df['sex'] = le_sex.transform(df['sex'])
    df['source'] = le_source.transform(df['source'])
    df['browser'] = le_browser.transform(df['browser'])
    df['time_diff_category'] = le_time_category.transform(df['time_diff_category'])
    df['country'] = le_country.transform(df['country'])
    
    df[["purchase_value", "age"]] = scaler.transform(df[["purchase_value", "age"]])
    
    final_df = df[MODEL_FEATURES]

    probability = model.predict_proba(final_df)[:, 1][0]
    
    if probability >= 0.45:
        return {'Fraud': probability, 'Valid': 1 - probability}
    else:
        return {'Valid': 1 - probability, 'Fraud': probability}

frontend = gr.Interface(
    fn=predict_fraud_csv,
    inputs=gr.File(label="Upload Input CSV"),
    outputs=gr.Label(label="Prediction"),
    title="Fraud Detection Application",
    description="Upload a CSV file to get a fraud prediction"
)

frontend.launch()
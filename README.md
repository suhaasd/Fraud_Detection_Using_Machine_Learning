### 🛡️ Fraud Detection Using Machine Learning
End-to-end project for detecting fraudulent online transactions — from data cleaning and exploratory analysis, through feature engineering and model training, to a deployable Gradio app for inference.
Clean, reproducible, and structured for both academic evaluation and portfolio presentation.

## 📋 Overview

Domain: Transaction fraud detection

Approach: Feature engineering + XGBoost (with careful imbalance handling)

Deliverables: Trained model artifacts, Gradio app (Dockerized), live Hugging Face Space, and presentation slides

Why this project: demonstrates handling of extreme class imbalance, preventing data leakage (SMOTE timing), model tuning (scale_pos_weight), and production-ready inference.

### 🎯 Key Features

- **High-Performance Model**: XGBoost-based classifier optimized for fraud detection
- **Advanced Feature Engineering**: Temporal, cyclical, and behavioral pattern analysis
- **Interactive Web Interface**: User-friendly Gradio application for real-time predictions
- **Comprehensive Analysis**: Detailed exploratory data analysis and feature importance insights
- **Production-Ready**: Dockerized app deployed live on Hugging Face Spaces

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Input Data (CSV)                │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│      Feature Engineering Pipeline       │
│  • Temporal Features                    │
│  • Cyclical Encoding                    │
│  • Device Frequency Mapping             │
│  • Geographic Categorization            │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│       Data Preprocessing                │
│  • Label Encoding                       │
│  • Min-Max Scaling                      │
│  • Feature Selection                    │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│      XGBoost Classifier                 │
│  • Optimized Hyperparameters            │
│  • Probability Threshold: 0.45          │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│    Prediction & Risk Assessment         │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option A — Live Demo (no setup required)

The app is deployed and publicly accessible on Hugging Face Spaces:
**[https://huggingface.co/spaces/suhaasd/fraud_detection](https://huggingface.co/spaces/suhaasd/fraud_detection)**

### Option B — Run with Docker

```bash
cd ETL
docker build -t fraud-detection .
docker run -p 7860:7860 fraud-detection
```
Open `http://localhost:7860`

### Option C — Run locally with Python

#### Prerequisites
```bash
Python 3.10.0 or higher
pip package manager
```

1. **Install dependencies**
```bash
cd ETL
pip install -r requirements.txt
```

2. **Run the application**
```bash
python app.py
```

3. **Access the web interface**
Open your browser and navigate to `http://localhost:7860`

## 📊 Features & Methodology

### Feature Engineering

| Feature Category | Description | Techniques Used |
|-----------------|-------------|-----------------|
| **Temporal Features** | Time-based patterns between signup and purchase | Time difference categorization, Hour/Day extraction |
| **Cyclical Encoding** | Circular representation of time variables | Sine/Cosine transformations for hours, days, months |
| **Device Analytics** | Device usage patterns and frequency | User count mapping per device ID |
| **Geographic Features** | Country-based risk assessment | Top 18 countries + "Other" categorization |
| **Behavioral Patterns** | User interaction characteristics | Source, Browser, Demographics analysis |

### Model Performance

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Threshold Optimization**: Custom threshold of 0.45 for optimal precision-recall balance
- **Cross-validation**: Stratified k-fold validation ensuring robust performance

## 📁 Project Structure

```
Fraud_Detection_Using_Machine_Learning/
└─ ETL/
   ├─ app.py                      # Gradio app — inference pipeline (batch + single)
   ├─ Dockerfile                  # Docker build for containerized deployment
   ├─ requirements.txt            # Python dependencies
   ├─ train.py                    # Model training script
   ├─ eda.py                      # Exploratory data analysis script
   ├─ combine_csv.py              # Merges raw CSVs (IP → country mapping)
   ├─ data/
   │   ├─ Fraud_Data.csv          # Raw transaction data
   │   ├─ IpAddress_to_Country.csv
   │   └─ final_data.csv          # Cleaned & merged dataset used for training
   ├─ encoders/                   # Label encoders & mapping artifacts (joblib)
   │   ├─ le_country.joblib
   │   ├─ le_browser.joblib
   │   ├─ le_sex.joblib
   │   ├─ le_source.joblib
   │   ├─ le_time_diff_category.joblib
   │   ├─ device_counts_map.joblib
   │   ├─ top_18_countries.joblib
   │   ├─ time_bins.joblib
   │   └─ time_labels.joblib
   ├─ models/                     # Saved model + scaler + threshold (joblib)
   │   ├─ best_xgb_model.joblib
   │   ├─ min_max_scaler.joblib
   │   └─ best_threshold.joblib
   └─ pictures/                   # EDA & evaluation plots
```

## 🔧 Usage

### Web Interface

**Batch CSV Prediction tab:**
1. Upload a CSV file with transaction data
2. Click "Predict" to score all rows at once
3. View per-row fraud probability and prediction label

**Single Transaction tab:**
1. Fill in the transaction form fields
2. Click "Predict" to get an instant result

### CSV Input Format

Your input CSV should contain the following columns:

|     Column     |   Type   |            Description          |
|----------------|----------|---------------------------------|
| user_id        | int      | Unique user identifier          |
| signup_time    | datetime | User registration timestamp     |
| purchase_time  | datetime | Transaction timestamp           |
| purchase_value | float    | Transaction amount              |
| device_id      | string   | Device identifier               |
| source         | string   | Traffic source (SEO/Ads/Direct) |
| browser        | string   | Browser type                    |
| sex            | string   | User gender (M/F)               |
| age            | int      | User age                        |
| country        | string   | Transaction country             |

## 📈 Model Details

### Training Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Parse datetime columns
   - Encode categorical variables

2. **Feature Engineering**
   - Create time-based features
   - Apply cyclical transformations
   - Generate device frequency maps

3. **Model Training**
   - XGBoost with hyperparameter tuning
   - Class weight balancing
   - Cross-validation for stability

4. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - ROC-AUC analysis
   - Confusion matrix visualization


## 📌 Notes on evaluation & best practices

PR-AUC over ROC-AUC: for highly imbalanced classification PR-AUC reflects precision/recall tradeoffs better than ROC.

SMOTE timing matters: creating synthetic minority samples before splitting causes target leakage — always apply synthetic oversampling after the split only on training data.

Model explainability (future): add SHAP plots or feature contribution breakdown to help stakeholders understand high-risk flags.


## 👨‍💻 Author

**Suhaas D**  
Masters in Computer Science Student  
Manipal School of Information Sciences

- GitHub: [@suhaasd](https://github.com/suhaasd)
- LinkedIn: [suhaasd](https://www.linkedin.com/in/suhaasd/)
- Email: suhaasdmurthy@gmail.com

## 🙏 Acknowledgments

- Manipal School of Information Sciences for academic support
- XGBoost and Gradio communities for excellent documentation
- Dataset providers for enabling research in fraud detection

## 📊 Results & Insights

The fraud detection system demonstrates:
- **High Precision**: Minimizes false positives in fraud alerts
- **Scalable Architecture**: Handles large-scale transaction volumes
- **Real-time Processing**: Sub-second prediction times
- **Interpretable Results**: Clear probability scores for decision-making

## 🔮 Future Enhancements

- Add SHAP-based explanations for flagged transactions to improve stakeholder trust.
- Compare LightGBM / CatBoost to XGBoost with the same pipeline.
- Set up CI to run pipeline tests and ensure artifacts stay in sync with training code.

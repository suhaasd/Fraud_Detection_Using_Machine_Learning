### 🛡️ Fraud Detection Using Machine Learning
End-to-end project for detecting fraudulent online transactions — from data cleaning and exploratory analysis, through feature engineering and model training, to a deployable Gradio app for inference.
Clean, reproducible, and structured for both academic evaluation and portfolio presentation.

## 📋 Overview

Domain: Transaction fraud detection

Approach: Feature engineering + XGBoost (with careful imbalance handling)

Deliverables: Jupyter notebook, trained model artifacts, Gradio app, sample inputs, and presentation slides

Why this project: demonstrates handling of extreme class imbalance, preventing data leakage (SMOTE timing), model tuning (scale_pos_weight), and production-ready inference.

### 🎯 Key Features

- **High-Performance Model**: XGBoost-based classifier optimized for fraud detection
- **Advanced Feature Engineering**: Temporal, cyclical, and behavioral pattern analysis
- **Interactive Web Interface**: User-friendly Gradio application for real-time predictions
- **Comprehensive Analysis**: Detailed exploratory data analysis and feature importance insights
- **Production-Ready**: Scalable architecture with saved model artifacts for deployment

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

### Prerequisites

```bash
Python 3.10.0 or higher
pip package manager
```

### Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Download model artifacts**
Ensure the following pre-trained model files are in your project directory:
- `best_xgb_model.joblib`
- `min_max_scaler.joblib`
- `device_counts_map.joblib`
- `top_18_countries.joblib`
- `time_bins.joblib`
- `time_labels.joblib`
- Label encoder files (`le_*.joblib`)

3. **Run the application**
```bash
python app.py
```

4. **Access the web interface**
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
Fraud_Detection/
├─ app.py                         # Gradio app — model + preprocessing pipeline for inference
├─ requirements.txt               # Python dependencies
├─ fraud_detection.ipynb          # Full notebook: EDA → FE → training → evaluation
├─ fraud detection.pptx           # Project presentation (slides)
├─ final_data.csv                 # Cleaned dataset used for experiments
├─ encoders/                      # Label encoders & mapping artifacts (joblib)
│   ├─ le_country.joblib
│   ├─ le_browser.joblib
│   └─ ...
├─ models/                        # Saved model + scalers + feature maps (joblib)
│   ├─ best_xgb_model.joblib
│   ├─ min_max_scaler.joblib
│   ├─ device_counts_map.joblib
│   └─ ...
└─ user_input/                    # Example CSVs for demo / bulk prediction
  ├─ input1.csv
  ├─ input2.csv
  ├─ input3.csv
  └─ input4.csv
```

## 🔧 Usage

### Web Interface

1. Launch the Gradio application
2. Upload a CSV file with transaction data
3. Click "Submit" to get fraud predictions
4. View probability scores for fraud/valid classification

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
| ip_address     | float    | IP address (numeric)            |
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

Add SHAP-based explanations for flagged transactions.
Compare LightGBM / CatBoost to XGBoost with same pipeline.
Set up CI to run notebook tests and ensure artifacts are up to date.
Deploy to Hugging Face Spaces, Streamlit Cloud, or a small cloud VM for public demo.

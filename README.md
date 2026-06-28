## Disease Prediction and Medical Report Analysis System

## Project Overview

This project was built as a summer internship project. It walks through the full ML lifecycle — from raw medical data to a deployed, user-facing prediction app:

1. Explore and visualize a synthetic medical dataset
2. Preprocess and encode the data
3. Train and evaluate a classification model
4. Save the trained artifacts
5. Serve predictions through a Streamlit interface

---

##  Dataset

- **File:** `medical_dataset_1500_records_10_diseases.csv`
- **Records:** 1,500 synthetic patient records
- **Classes:** 10 disease categories (including a "Healthy" class)
- **Features:** 14 attributes — a mix of binary symptom flags (e.g. `Cough`, `Fever`) and continuous vitals (`Age`, `BP`, `Sugar`, `Oxygen_Level`)

### Exploratory Data Analysis
- **Class distribution** — count plot of records per disease
- **Correlation heatmap** — relationships between all numeric features
- **Symptom averages by disease** — bar plots comparing symptom prevalence across disease classes

---

##  ML Pipeline

| Step | Details |
|------|---------|
| **Label Encoding** | `Disease` (text) encoded into `Disease_encoded` (numeric) via `LabelEncoder` |
| **Train/Test Split** | 80/20 split, stratified to preserve class balance (`random_state=42`) |
| **Feature Scaling** | `StandardScaler` applied only to continuous columns: `Age`, `BP`, `Sugar`, `Oxygen_Level` |
| **Model** | `RandomForestClassifier` (`n_estimators=100`, `random_state=42`) |
| **Evaluation** | Confusion matrix (actual vs. predicted) and feature importance ranking |

### Why Random Forest?
Random Forest was selected for its strong performance on tabular, mixed binary/continuous medical data, its resistance to overfitting, and its built-in feature importance scores — useful for understanding which symptoms drive each prediction.

---

##  Saved Artifacts

The training script saves everything needed to reproduce predictions without retraining:

| File | Purpose |
|------|---------|
| `disease_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted `StandardScaler` for continuous features |
| `label_encoder.pkl` | `LabelEncoder` mapping disease names ↔ numeric labels |
| `continuous_cols.pkl` | List of continuous column names |
| `feature_columns.pkl` | Full ordered list of input feature columns |

---

##  Streamlit App (`app.py`)

The deployed app provides:

- **Vitals input** — sliders for Age, Blood Pressure, Sugar Level, Oxygen Level
- **Symptom checklist** — checkboxes for all binary symptom features
- **Live vitals reference** — age-adjusted healthy ranges for BP, Sugar, and Oxygen with color-coded status (🟢 Normal / 🟡 Elevated / 🔴 High / 🔵 Low) — shown *before* prediction
- **Disease prediction** — top predicted disease with confidence score
- **Probability breakdown** — horizontal bar chart of prediction probability across all 10 disease classes (Plotly)
- **Care tips** — curated, disease-specific lifestyle and care guidance for each of the 10 outcomes
- **Safety disclaimer** — persistent reminder that this is an educational tool, not a diagnosis

### Age-Adjusted Reference Ranges
Healthy BP thresholds shift with age (≤40, 41–60, 60+), while Sugar (70–140 mg/dL) and Oxygen (95–100%) ranges follow standard general guidelines.

---

##  Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML:** Scikit-learn (`RandomForestClassifier`, `LabelEncoder`, `StandardScaler`, `train_test_split`, `confusion_matrix`)
- **Persistence:** Joblib
- **App framework:** Streamlit
- **Tunneling (for Colab demo):** Pyngrok
- **Development environment:** Google Colab

---

##  Running the Project

### 1. In Google Colab (as built)
1. Upload `medical_dataset_1500_records_10_diseases.csv` to the Colab session.
2. Run the notebook cells in order — EDA → preprocessing → training → artifact saving.
3. The `app.py` file is written out via the `%%writefile` magic cell.
4. Install Streamlit and Pyngrok, then launch the app via `subprocess` + ngrok tunnel for a public preview link.

### 2. Locally
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib seaborn

# Ensure these files are in the same folder as app.py:
# disease_model.pkl, scaler.pkl, label_encoder.pkl,
# continuous_cols.pkl, feature_columns.pkl

streamlit run app.py
```

> **Security note:** Replace any hardcoded ngrok auth token with an environment variable before sharing or committing this code (e.g. `ngrok.set_auth_token(os.environ["NGROK_TOKEN"])`).

---

##  Results

- Confusion matrix visualizes per-class prediction accuracy across all 10 diseases.
- Feature importance ranking highlights which symptoms and vitals most strongly influence the model's decisions — useful both for model interpretability and for sanity-checking against medical intuition.

---

##  Possible Extensions

- Cross-validate across multiple model types (Logistic Regression, SVM, Gradient Boosting) and compare via GridSearchCV
- Add SHAP-based explainability for individual predictions
- Expand the dataset beyond synthetic records with proper anonymized clinical data
- Add a downloadable PDF medical report summarizing the prediction, vitals status, and care tips
- Deploy permanently (Streamlit Community Cloud / Render) instead of relying on an ngrok tunnel

---

## 👤 Author

Built by Dinesh as part of a summer internship project — B.Tech, Artificial Intelligence and Machine Learning, Aditya University.

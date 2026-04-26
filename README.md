# ❤️ Heart Failure Prediction System

An AI-powered **Clinical Decision Support System** that predicts heart failure mortality risk using XGBoost and provides transparent, interpretable predictions using SHAP (SHapley Additive exPlanations).

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange)

## 🔬 Features

- **3-Tier Risk Classification** — Classifies patients as Low, Moderate, or High risk based on 11 clinical biomarkers
- **SHAP Explainability** — Per-patient feature attribution plots showing *why* the model made its prediction
- **Heart Health Scoring** — Automated scoring against cardiology reference ranges (0–100 scale)
- **Clinical Recommendations** — Evidence-based, tier-specific action plans for clinicians
- **PDF Report Generation** — Downloadable clinical decision support reports
- **Interactive Dashboard** — Real-time inference with dynamic visualizations

## 🧬 Clinical Biomarkers Used

| Biomarker | Type | Unit |
|-----------|------|------|
| Age | Demographic | years |
| Ejection Fraction | Cardiac | % |
| Serum Creatinine | Renal | mg/dL |
| Serum Sodium | Electrolyte | mEq/L |
| Creatinine Phosphokinase | Enzyme | mcg/L |
| Platelets | Hematologic | kiloplatelets/mL |
| Anaemia | Binary | Yes/No |
| Diabetes | Binary | Yes/No |
| High Blood Pressure | Binary | Yes/No |
| Sex | Binary | M/F |
| Smoking | Binary | Yes/No |

## 🛠️ Tech Stack

- **ML Model**: XGBoost (Gradient Boosted Trees)
- **Explainability**: SHAP (TreeExplainer)
- **Frontend/Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, SHAP plots
- **Report Generation**: FPDF2

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/Rajkamal08/heart-failure-prediction.git
cd heart-failure-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 How It Works

1. **Input** — Enter 11 clinical parameters via the interactive form
2. **Predict** — XGBoost model outputs mortality probability
3. **Classify** — 3-tier risk stratification (Low < 10%, Moderate 10–35%, High > 35%)
4. **Explain** — SHAP values show per-feature contributions (positive = increases risk, negative = decreases risk)
5. **Recommend** — Tier-specific clinical action plans generated automatically
6. **Report** — Download PDF clinical decision support report

## ⚠️ Disclaimer

This system is designed as a **clinical decision support tool** and should not replace professional medical judgment. Always consult qualified healthcare providers before making clinical decisions.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

# app.py
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
from datetime import datetime
import io

st.set_page_config(
    page_title="Heart Failure Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Utility: Robust model loader
# -------------------------
def load_model_and_scaler():
    """
    Try to load model from several common files:
      1) heart_model.json (preferred for XGBoost)
      2) heart_model.pkl  (pickled sklearn wrapper)
      3) heart_model.* other names
    Also load scaler.pkl
    Returns (model, scaler)
    """
    model = None
    scaler = None

    # try JSON XGBoost model first
    json_path = "heart_model.json"
    pkl_path = "heart_model.pkl"
    alt_pkl = "xgb_model.pkl"
    alt_pkl2 = "heart_model.pkl"
    scaler_paths = ["scaler.pkl", "scaler.sav"]

    try:
        if os.path.exists(json_path):
            # load as XGBClassifier then call load_model
            model = xgb.XGBClassifier()
            model.load_model(json_path)
            st.sidebar.success(f"✅ Loaded model: {json_path}")
        elif os.path.exists(pkl_path):
            model = pickle.load(open(pkl_path, "rb"))
            st.sidebar.success(f"✅ Loaded model: {pkl_path}")
        elif os.path.exists(alt_pkl):
            model = pickle.load(open(alt_pkl, "rb"))
            st.sidebar.success(f"✅ Loaded model: {alt_pkl}")
        else:
            # try any xgb*.pkl in folder
            for f in os.listdir("."):
                if f.lower().endswith(".pkl") and "model" in f.lower():
                    try:
                        model = pickle.load(open(f, "rb"))
                        st.sidebar.success(f"✅ Loaded model: {f}")
                        break
                    except Exception:
                        model = None
            if model is None:
                st.warning("No model file found in JSON or pickles. Please put heart_model.json or heart_model.pkl in this folder.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

    # scaler
    for sp in scaler_paths:
        if os.path.exists(sp):
            try:
                scaler = pickle.load(open(sp, "rb"))
                st.sidebar.success(f"✅ Loaded scaler: {sp}")
                break
            except Exception:
                scaler = None

    # fallback: if no scaler, create a StandardScaler (NOT recommended, used only to avoid crash)
    if scaler is None:
        st.error("⚠️ CRITICAL: No scaler file found (scaler.pkl). Predictions will be INCORRECT without proper scaling. Please add scaler.pkl to this folder.")
        scaler = StandardScaler()

    return model, scaler

model, scaler = load_model_and_scaler()

# If model still None, show helpful message and stop
if model is None:
    st.markdown("**ERROR:** No model could be loaded. Place `heart_model.json` or `heart_model.pkl` in this folder and restart the app.")
    st.stop()

# -------------------------
# Feature names (from your CSV)
# -------------------------
# ensure these match training order/columns
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
    "sex", "smoking"
]

# Load some sample rows (if dataset exists) to create presets
DATA_CSV = "heart_failure_no_time.csv"
sample_df = None
if os.path.exists(DATA_CSV):
    try:
        df_all = pd.read_csv(DATA_CSV)
        sample_df = df_all[FEATURES].copy()
    except Exception:
        sample_df = None

# -------------------------
# SHAP explainer (TreeExplainer)
# -------------------------
# Precompute explainer on a small background dataset if available
try:
    if sample_df is not None and len(sample_df) >= 20:
        background = sample_df.sample(min(100, len(sample_df)), random_state=42)
        bg_scaled = scaler.transform(background)
        explainer = shap.TreeExplainer(model)
        # compute once (may show warning about additivity; we'll pass check_additivity=False when needed)
    else:
        explainer = shap.TreeExplainer(model)
except Exception:
    explainer = shap.TreeExplainer(model)

# -------------------------
# 3-tier classification
# -------------------------
def classify_risk_tier(prob):
    if prob < 0.10:
        return "Low risk"
    elif prob < 0.35:
        return "Moderate risk"
    else:
        return "High risk"

def get_clinical_recommendations(tier):
    if tier == "Low risk":
        return [
            "✓ Routine follow-up: Schedule visit in 6–12 months or as clinically indicated.",
            "✓ Lifestyle modifications: Sodium restriction (<2g/day), fluid management, regular light-moderate exercise.",
            "✓ Medication adherence: Ensure compliance with prescribed HF medications (ACE-I/ARB, beta-blockers, diuretics).",
            "✓ Monitor vitals: Track daily weight, blood pressure, heart rate at home.",
            "✓ Laboratory monitoring: Check renal function (creatinine, eGFR), electrolytes (sodium, potassium) every 6 months.",
            "✓ Patient education: Recognize early warning signs (shortness of breath, swelling, fatigue)."
        ]
    elif tier == "Moderate risk":
        return [
            "⚠ Follow-up timeline: Schedule appointment within 2–3 months, sooner if symptoms emerge.",
            "⚠ Medication optimization: Review and titrate guideline-directed medical therapy (GDMT):",
            "   - ACE inhibitors/ARBs or ARNI (sacubitril-valsartan)",
            "   - Beta-blockers (carvedilol, metoprolol, bisoprolol)",
            "   - Mineralocorticoid receptor antagonists (spironolactone, eplerenone)",
            "   - SGLT2 inhibitors (dapagliflozin, empagliflozin) if applicable",
            "⚠ Enhanced monitoring: Check labs (BNP/NT-proBNP, creatinine, potassium, sodium) every 2-3 months.",
            "⚠ Symptom assessment: Evaluate NYHA functional class, 6-minute walk test if available.",
            "⚠ Dietary counseling: Strict sodium restriction, daily weight monitoring, fluid restriction if needed.",
            "⚠ Warning signs education: Seek immediate care if weight gain >2-3 lbs in 1 day or >5 lbs in 1 week."
        ]
    else:  # High risk
        return [
            "🚨 URGENT ACTION: Schedule follow-up within 1–2 weeks or consider immediate cardiology referral.",
            "🚨 Hospitalization consideration: Admit if acute decompensation signs present:",
            "   - Severe dyspnea at rest, orthopnea, paroxysmal nocturnal dyspnea",
            "   - Hypotension (SBP <90 mmHg), tachycardia (HR >100), oxygen saturation <90%",
            "   - Significant volume overload (pulmonary edema, severe peripheral edema)",
            "🚨 Medication review (STAT): Optimize GDMT, adjust diuretics (IV if necessary), assess need for inotropes.",
            "🚨 Advanced therapies: Consider referral for advanced HF evaluation (LVAD, transplant candidacy).",
            "🚨 Intensive monitoring: Daily to weekly assessment until stable:",
            "   - Daily weights, I/O monitoring if hospitalized",
            "   - Serial BNP/NT-proBNP, electrolytes, renal function (Cr, BUN, eGFR)",
            "   - Echocardiography to assess ejection fraction and cardiac function",
            "🚨 RED FLAG symptoms - Instruct patient to call 911 or go to ER immediately if:",
            "   - Severe shortness of breath, cannot lie flat, gasping for air",
            "   - Chest pain or pressure (rule out acute coronary syndrome)",
            "   - Sudden weight gain (>3 lbs overnight or >5 lbs in 2-3 days)",
            "   - Severe swelling in legs/abdomen, inability to wear shoes",
            "   - Extreme fatigue, confusion, dizziness, fainting",
            "   - Rapid or irregular heartbeat (palpitations, >120 bpm at rest)",
            "🚨 Care coordination: Involve multidisciplinary team (cardiologist, HF nurse, dietitian, pharmacist)."
        ]

# -------------------------
# PDF Report Generation
# -------------------------
def generate_pdf_report(input_df, prob, tier, recommendations):
    """
    Generate a professional PDF report with patient data, risk assessment, and clinical recommendations
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(220, 53, 69)  # Red color
    pdf.cell(0, 15, 'HEART FAILURE PREDICTION SYSTEM', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Clinical Decision Support Report', 0, 1, 'C')
    
    # Timestamp
    pdf.set_font('Arial', 'I', 10)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, f'Generated: {current_time}', 0, 1, 'C')
    
    pdf.ln(10)
    
    # Risk Assessment Section
    pdf.set_font('Arial', 'B', 16)
    if tier == "Low risk":
        pdf.set_text_color(40, 167, 69)  # Green
    elif tier == "Moderate risk":
        pdf.set_text_color(255, 193, 7)  # Orange
    else:
        pdf.set_text_color(220, 53, 69)  # Red
    
    pdf.cell(0, 10, f'RISK TIER: {tier.upper()}', 0, 1, 'C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Predicted Mortality Probability: {prob:.3f} ({prob*100:.1f}%)', 0, 1, 'C')
    
    pdf.ln(10)
    
    # Patient Data Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Patient Clinical Data', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Create two-column layout for patient data
    data_items = [
        ('Age', f"{int(input_df['age'].values[0])} years"),
        ('Anaemia', 'Yes' if input_df['anaemia'].values[0] == 1 else 'No'),
        ('Creatinine Phosphokinase', f"{int(input_df['creatinine_phosphokinase'].values[0])} mcg/L"),
        ('Diabetes', 'Yes' if input_df['diabetes'].values[0] == 1 else 'No'),
        ('Ejection Fraction', f"{int(input_df['ejection_fraction'].values[0])}%"),
        ('High Blood Pressure', 'Yes' if input_df['high_blood_pressure'].values[0] == 1 else 'No'),
        ('Platelets', f"{int(input_df['platelets'].values[0])} kiloplatelets/mL"),
        ('Serum Creatinine', f"{input_df['serum_creatinine'].values[0]:.2f} mg/dL"),
        ('Serum Sodium', f"{int(input_df['serum_sodium'].values[0])} mEq/L"),
        ('Sex', 'Male' if input_df['sex'].values[0] == 1 else 'Female'),
        ('Smoking', 'Yes' if input_df['smoking'].values[0] == 1 else 'No'),
    ]
    
    for i in range(0, len(data_items), 2):
        if i + 1 < len(data_items):
            pdf.cell(95, 6, f"{data_items[i][0]}: {data_items[i][1]}", 0, 0)
            pdf.cell(95, 6, f"{data_items[i+1][0]}: {data_items[i+1][1]}", 0, 1)
        else:
            pdf.cell(0, 6, f"{data_items[i][0]}: {data_items[i][1]}", 0, 1)
    
    pdf.ln(10)
    
    # Clinical Recommendations Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommended Clinical Actions', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for rec in recommendations:
        # Handle sub-items (indented with spaces)
        if rec.startswith('   -'):
            pdf.cell(10, 6, '', 0, 0)  # Indent
            pdf.multi_cell(0, 6, rec.strip())
        else:
            pdf.multi_cell(0, 6, f'• {rec}')
    
    pdf.ln(5)
    
    # Disclaimer
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, 'DISCLAIMER: This report is generated by an AI-based clinical decision support system. It should not replace professional medical judgment. Always consult with qualified healthcare providers before making clinical decisions. This system is designed to assist, not replace, clinical expertise.')
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')


# -------------------------
# Prediction + SHAP helper
# -------------------------
def predict_and_explain(input_df):
    """
    input_df: raw values (pd.DataFrame with one row, columns=FEATURES)
    returns: prob, tier, shap_vals (1D), expected_value
    """
    # scale
    Xs = scaler.transform(input_df[FEATURES])
    prob = float(model.predict_proba(Xs)[0][1])
    tier = classify_risk_tier(prob)

    # SHAP values (disable additivity check to avoid errors)
    try:
        shap_vals_all = explainer.shap_values(Xs, check_additivity=False)
        # For XGBoost/TreeExplainer this may return shape (n_outputs, n_samples, n_features)
        # or (n_samples, n_features) depending on version; normalize below:
        if isinstance(shap_vals_all, list) or (hasattr(shap_vals_all, "shape") and shap_vals_all.ndim == 3):
            # shap_vals_all[class_index][sample_index][feature_index]
            if len(shap_vals_all) >= 2:
                shap_vals = np.array(shap_vals_all[1])[0]  # class 1 contributions for sample
            else:
                shap_vals = np.array(shap_vals_all)[0]
        else:
            # newer SHAP returns object with .values or plain array
            if hasattr(shap_vals_all, "values"):
                shap_vals = shap_vals_all.values[0]
            else:
                shap_vals = np.array(shap_vals_all)[0]
    except Exception:
        # fallback: use explainer(X).values if available
        try:
            ev = explainer(Xs)
            shap_vals = np.array(ev.values[0])
            expected = float(ev.base_values[0]) if hasattr(ev, "base_values") else float(explainer.expected_value)
        except Exception:
            shap_vals = np.zeros(len(FEATURES))
            expected = 0.0
        return prob, tier, shap_vals, expected

    # expected value
    try:
        if hasattr(explainer, "expected_value"):
            expected = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1])
        else:
            expected = 0.0
    except Exception:
        expected = 0.0

    return prob, tier, shap_vals, expected

# -------------------------
# SHAP plotting helpers (matplotlib static plots)
# -------------------------
def plot_shap_bar_global(X_df, max_display=11):
    """
    Plot mean(|SHAP|) bar plot for training/background data (global importance)
    """
    try:
        # compute shap values on background quickly
        Xs = scaler.transform(X_df[FEATURES])
        sv = explainer.shap_values(Xs, check_additivity=False)
        # handle list/array differences
        if isinstance(sv, list) or (hasattr(sv, "shape") and np.ndim(sv) == 3):
            if len(sv) >= 2:
                sv_for_class = np.array(sv[1])
            else:
                sv_for_class = np.array(sv)
        else:
            if hasattr(sv, "values"):
                sv_for_class = sv.values
            else:
                sv_for_class = np.array(sv)
        mean_abs = np.mean(np.abs(sv_for_class), axis=0)
        order = np.argsort(mean_abs)[::-1][:max_display]
        feat_names = np.array(FEATURES)[order]
        mean_abs_sorted = mean_abs[order]

        fig, ax = plt.subplots(figsize=(7, max(3, 0.4*len(feat_names))))
        bars = ax.barh(np.arange(len(feat_names))[::-1], mean_abs_sorted[::-1])
        ax.set_yticks(np.arange(len(feat_names))[::-1])
        ax.set_yticklabels(feat_names[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP value| (impact on prediction)")
        ax.set_title("Global feature importance (mean absolute SHAP)")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Could not compute global SHAP plot: {e}")
        return None

def plot_shap_patient_contrib(shap_vals, feature_names=FEATURES, top_n=10):
    """
    Plot per-patient SHAP contributions (signed) as horizontal bar chart.
    Positive -> pushes toward death (higher risk), Negative -> pushes away from death.
    """
    feat = np.array(feature_names)
    vals = np.array(shap_vals)
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    labels = feat[order]
    values = vals[order]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]  # red positive, blue negative

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4*len(labels))))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values[::-1], color=[colors[i] for i in range(len(colors))][::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_xlabel("SHAP value (effect on model output)")
    ax.set_title("Top feature contributions for this patient")
    plt.tight_layout()
    return fig

# -------------------------
# Heart Health Analysis Functions
# -------------------------

# Clinical reference ranges based on cardiology guidelines
PARAMETER_RANGES = {
    "age": {"normal": (18, 65), "borderline": (66, 75), "abnormal": (76, 85), "critical": (86, 120), "unit": "years", "lower_is_better": True},
    "ejection_fraction": {"normal": (50, 70), "borderline": (40, 49), "abnormal": (30, 39), "critical": (10, 29), "unit": "%", "lower_is_better": False},
    "serum_creatinine": {"normal": (0.6, 1.2), "borderline": (1.3, 1.5), "abnormal": (1.6, 3.0), "critical": (3.1, 15.0), "unit": "mg/dL", "lower_is_better": True},
    "serum_sodium": {"normal": (135, 145), "borderline": [(130, 134), (146, 150)], "abnormal": [(125, 129), (151, 155)], "critical": [(100, 124), (156, 160)], "unit": "mEq/L", "lower_is_better": False},
    "platelets": {"normal": (150000, 400000), "borderline": [(100000, 149999), (400001, 500000)], "abnormal": [(50000, 99999), (500001, 700000)], "critical": [(20000, 49999), (700001, 1500000)], "unit": "kiloplatelets/mL", "lower_is_better": False},
    "creatinine_phosphokinase": {"normal": (0, 200), "borderline": (201, 400), "abnormal": (401, 800), "critical": (801, 20000), "unit": "mcg/L", "lower_is_better": True},
}

def evaluate_parameter(param_name, value):
    """
    Evaluate a clinical parameter and return status, color, icon
    Returns: (status_text, color, icon, severity_score)
    severity_score: 0=normal, 1=borderline, 2=abnormal, 3=critical
    """
    if param_name not in PARAMETER_RANGES:
        return "Unknown", "gray", "❓", 0
    
    ranges = PARAMETER_RANGES[param_name]
    
    # Helper function to check if value is in range(s)
    def in_range(range_spec):
        if isinstance(range_spec, tuple):
            return range_spec[0] <= value <= range_spec[1]
        elif isinstance(range_spec, list):
            return any(r[0] <= value <= r[1] for r in range_spec)
        return False
    
    if in_range(ranges["normal"]):
        return "Normal", "#4caf50", "✅", 0
    elif in_range(ranges["borderline"]):
        return "Borderline", "#ff9800", "⚠️", 1
    elif in_range(ranges["abnormal"]):
        return "Abnormal", "#ff5722", "🟠", 2
    else:
        return "Critical", "#d32f2f", "🔴", 3

def calculate_health_score(input_df):
    """
    Calculate overall heart health score (0-100) based on parameters in normal range
    Weighted by clinical importance
    """
    weights = {
        "ejection_fraction": 3.0,  # Most important
        "serum_creatinine": 2.0,
        "serum_sodium": 1.5,
        "creatinine_phosphokinase": 1.0,
        "platelets": 1.0,
        "age": 0.5,  # Less controllable
    }
    
    total_weight = 0
    weighted_score = 0
    
    for param, weight in weights.items():
        if param in input_df.columns:
            value = float(input_df[param].values[0])
            _, _, _, severity = evaluate_parameter(param, value)
            
            # Score: 100 for normal, 70 for borderline, 40 for abnormal, 0 for critical
            param_score = {0: 100, 1: 70, 2: 40, 3: 0}[severity]
            
            weighted_score += param_score * weight
            total_weight += weight
    
    if total_weight == 0:
        return 50  # Default
    
    return int(weighted_score / total_weight)

def get_parameter_explanation(param_name, status, value):
    """
    Get patient-friendly explanation for a parameter
    """
    explanations = {
        "ejection_fraction": {
            "what": "Ejection Fraction measures how much blood your heart pumps out with each beat",
            "normal": "Your heart is pumping normally - this is excellent!",
            "borderline": "Your heart's pumping ability is slightly reduced. Close monitoring recommended.",
            "abnormal": "Your heart is pumping less blood than normal. This indicates weakened heart muscle.",
            "critical": "Your heart's pumping ability is severely reduced. This requires immediate medical attention.",
            "action": "Follow medication plan, reduce salt intake, regular cardiology follow-up"
        },
        "serum_creatinine": {
            "what": "Serum Creatinine measures how well your kidneys are filtering waste",
            "normal": "Your kidney function is normal - great!",
            "borderline": "Your kidney function is slightly reduced. May need monitoring.",
            "abnormal": "Your kidneys are not filtering waste efficiently. This can affect heart health.",
            "critical": "Severe kidney dysfunction detected. Requires urgent nephrology consultation.",
            "action": "Monitor kidney function, stay hydrated, review medications with doctor"
        },
        "serum_sodium": {
            "what": "Serum Sodium measures electrolyte balance in your blood",
            "normal": "Your sodium levels are balanced - perfect!",
            "borderline": "Your sodium is slightly outside normal range. Monitor fluid intake.",
            "abnormal": "Abnormal sodium levels can affect heart rhythm and fluid balance.",
            "critical": "Severe sodium imbalance. Can cause serious complications if not corrected.",
            "action": "Adjust fluid intake, review diuretic medications, monitor closely"
        },
        "creatinine_phosphokinase": {
            "what": "CPK is an enzyme that indicates heart or muscle damage",
            "normal": "No signs of recent heart or muscle damage - good!",
            "borderline": "Mildly elevated CPK. Could indicate minor muscle strain or heart stress.",
            "abnormal": "Elevated CPK may indicate heart muscle damage or stress.",
            "critical": "Very high CPK - possible acute heart injury. Requires immediate evaluation.",
            "action": "Avoid strenuous activity, follow up with cardiac enzyme tests, ECG monitoring"
        },
        "platelets": {
            "what": "Platelets help your blood clot properly",
            "normal": "Your platelet count is in the healthy range!",
            "borderline": "Platelet count is outside normal range. May affect clotting.",
            "abnormal": "Abnormal platelet count can increase bleeding or clotting risk.",
            "critical": "Severe platelet abnormality. Risk of bleeding or dangerous clots.",
            "action": "Monitor complete blood count, review medications affecting platelets"
        },
        "age": {
            "what": "Age is a non-modifiable risk factor for heart disease",
            "normal": "Younger age - lower baseline cardiovascular risk.",
            "borderline": "Middle age - moderate cardiovascular risk. Prevention is key.",
            "abnormal": "Older age - higher cardiovascular risk. Regular screening important.",
            "critical": "Advanced age - highest cardiovascular risk. Close monitoring needed.",
            "action": "Focus on modifiable risk factors: diet, exercise, blood pressure, cholesterol"
        }
    }
    
    if param_name not in explanations:
        return {}
    
    exp = explanations[param_name]
    status_lower = status.lower()
    
    return {
        "what": exp["what"],
        "interpretation": exp.get(status_lower, "Status unclear"),
        "action": exp["action"]
    }

def get_priority_concerns(input_df):
    """
    Identify top 3 most concerning parameters
    Returns list of (param_name, value, status, severity_score)
    """
    concerns = []
    
    for param in PARAMETER_RANGES.keys():
        if param in input_df.columns:
            value = float(input_df[param].values[0])
            status, color, icon, severity = evaluate_parameter(param, value)
            if severity >= 2:  # Abnormal or Critical only
                concerns.append((param, value, status, severity, icon))
    
    # Sort by severity (highest first)
    concerns.sort(key=lambda x: x[3], reverse=True)
    return concerns[:3]  # Top 3

def get_modifiable_factors(input_df):
    """
    Identify which risk factors the patient can potentially modify
    """
    modifiable = []
    
    # Check binary factors
    if input_df["smoking"].values[0] == 1:
        modifiable.append(("Smoking", "🚬", "Quit smoking to significantly reduce cardiovascular risk"))
    
    if input_df["high_blood_pressure"].values[0] == 1:
        modifiable.append(("High Blood Pressure", "💊", "Control blood pressure through medication and lifestyle changes"))
    
    if input_df["diabetes"].values[0] == 1:
        modifiable.append(("Diabetes", "🩺", "Manage blood sugar levels to protect heart health"))
    
    if input_df["anaemia"].values[0] == 1:
        modifiable.append(("Anaemia", "🔴", "Treat anaemia to improve oxygen delivery to heart"))
    
    # Check if any lab values are abnormal and modifiable
    ef_val = float(input_df["ejection_fraction"].values[0])
    if ef_val < 50:
        modifiable.append(("Low Ejection Fraction", "❤️", "Improve with cardiac medications, exercise, and lifestyle modifications"))
    
    return modifiable

# -------------------------
# Streamlit UI layout
# -------------------------
st.title("❤️ HEART FAILURE PREDICTION SYSTEM")
st.write("**Clinical Decision Support System** — Predicts mortality risk (Low/Moderate/High), explains predictions using AI explainability (SHAP), and provides evidence-based clinical recommendations to support real-world medical decision-making.")

# left: inputs, right: results
col1, col2 = st.columns((1, 1.3))

with col1:
    st.header("✍️ Enter Patient Data")
    st.markdown("")
    
    # Demographics Section
    with st.expander("👤 **Demographics**", expanded=True):
        st.markdown('<div style="padding: 8px;">', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=60, help="Patient's age in years")
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1, horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Clinical Laboratory Values Section
    with st.expander("🔬 **Clinical Laboratory Values**", expanded=True):
        st.markdown('<div style="padding: 8px;">', unsafe_allow_html=True)
        cpk = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=20000, value=250, 
                             help="Enzyme indicating heart/muscle damage")
        creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.1, format="%.2f",
                                    help="Kidney function marker")
        sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=160, value=137,
                                help="Electrolyte balance indicator")
        platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=20000, max_value=1500000, value=262000,
                                   help="Blood clotting cells")
        ef = st.number_input("Ejection Fraction (%)", min_value=10, max_value=80, value=38,
                            help="Percentage of blood pumped out of heart per beat")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Medical History Section
    with st.expander("📋 **Medical History**", expanded=True):
        st.markdown('<div style="padding: 8px;">', unsafe_allow_html=True)
        anaemia = st.radio("Anaemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0, horizontal=True,
                          help="Decrease in red blood cells")
        diabetes = st.radio("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0, horizontal=True,
                           help="High blood sugar condition")
        hbp = st.radio("High Blood Pressure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0, horizontal=True,
                      help="Hypertension")
        smoking = st.radio("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0, horizontal=True,
                          help="Current or past smoking history")
        st.markdown('</div>', unsafe_allow_html=True)

    # Preset buttons using example rows from dataset if available
    st.markdown("---")
    st.write("Or load a preset example:")
    if sample_df is not None:
        # pick three presets: one low, one moderate, one high by sampling probabilities if available
        if st.button("Load sample #1 (example)"):
            row = sample_df.iloc[0]
            age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking = [
                float(row["age"]), int(row["anaemia"]), float(row["creatinine_phosphokinase"]),
                int(row["diabetes"]), float(row["ejection_fraction"]), int(row["high_blood_pressure"]),
                float(row["platelets"]), float(row["serum_creatinine"]), float(row["serum_sodium"]),
                int(row["sex"]), int(row["smoking"])
            ]
            st.rerun()
        if st.button("Load sample #2 (example)"):
            row = sample_df.iloc[min(5, len(sample_df)-1)]
            age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking = [
                float(row["age"]), int(row["anaemia"]), float(row["creatinine_phosphokinase"]),
                int(row["diabetes"]), float(row["ejection_fraction"]), int(row["high_blood_pressure"]),
                float(row["platelets"]), float(row["serum_creatinine"]), float(row["serum_sodium"]),
                int(row["sex"]), int(row["smoking"])
            ]
            st.rerun()
    else:
        st.info("Dataset not found on server — sample presets disabled.")

    st.markdown("")
    st.markdown("---")
    
    # Enhanced Predict Button
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 16px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Predict Risk", key="predict_btn"):
        input_df = pd.DataFrame([{
            "age": age, "anaemia": anaemia, "creatinine_phosphokinase": cpk,
            "diabetes": diabetes, "ejection_fraction": ef, "high_blood_pressure": hbp,
            "platelets": platelets, "serum_creatinine": creatinine, "serum_sodium": sodium,
            "sex": sex, "smoking": smoking
        }], columns=FEATURES)

        prob, tier, shap_vals, expected = predict_and_explain(input_df)

        # store to state for display on right column
        st.session_state["last_input_df"] = input_df
        st.session_state["last_prob"] = prob
        st.session_state["last_tier"] = tier
        st.session_state["last_shap_vals"] = shap_vals
        st.session_state["last_expected"] = expected

        st.rerun()

with col2:
    st.header("🩺 Prediction & Explanation")
    if "last_prob" not in st.session_state:
        st.write("No prediction yet. Enter patient data and press **Predict Risk**.")
    else:
        prob = st.session_state["last_prob"]
        tier = st.session_state["last_tier"]
        shap_vals = st.session_state["last_shap_vals"]
        expected = st.session_state["last_expected"]
        input_df = st.session_state["last_input_df"]

        # Enhanced Risk Tier Card with Visual Impact
        st.markdown("")
        
        # Define colors and gradients for each tier
        if tier == "Low risk":
            gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            bg_color = "#e8f5e9"
            icon = "✅"
            bar_color = "#4caf50"
            text_color = "#2e7d32"
        elif tier == "Moderate risk":
            gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            bg_color = "#fff8e1"
            icon = "⚠️"
            bar_color = "#ff9800"
            text_color = "#e65100"
        else:  # High risk
            gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
            bg_color = "#ffebee"
            icon = "🚨"
            bar_color = "#f44336"
            text_color = "#c62828"
        
        # Risk Card HTML
        st.markdown(f"""
        <div style="
            background: {gradient};
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 72px; margin-bottom: 10px;">{icon}</div>
            <h1 style="color: white; margin: 10px 0; font-size: 36px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                {tier.upper()}
            </h1>
            <p style="font-size: 20px; margin: 5px 0; opacity: 0.95;">
                Mortality Probability: <strong>{prob:.1%}</strong>
            </p>
            <p style="font-size: 14px; margin: 5px 0; opacity: 0.8;">
                Risk Score: {prob:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Visualization Bar
        st.markdown("**Risk Probability Gauge**")
        prob_percent = int(prob * 100)
        st.markdown(f"""
        <div style="
            background-color: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin-bottom: 25px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                background: {gradient};
                width: {prob_percent}%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
                font-size: 14px;
                transition: width 1s ease;
            ">
                {prob_percent}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")

        # ═══════════════════════════════════════════════════════════
        # 🫀 HEART HEALTH PARAMETER ANALYSIS
        # ═══════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("")
        st.markdown("## 🫀 Heart Health Parameter Analysis")
        st.markdown("Comprehensive evaluation of all clinical parameters against established reference ranges")
        st.markdown("")
        
        # Calculate health score and get analysis
        health_score = calculate_health_score(input_df)
        priority_concerns = get_priority_concerns(input_df)
        modifiable_factors = get_modifiable_factors(input_df)
        
        # Determine health score category and color
        if health_score >= 80:
            score_category = "Excellent"
            score_color = "#4caf50"
            score_gradient = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
        elif health_score >= 60:
            score_category = "Good"
            score_color = "#8bc34a"
            score_gradient = "linear-gradient(135deg, #96c93d 0%, #00b09b 100%)"
        elif health_score >= 40:
            score_category = "Fair"
            score_color = "#ff9800"
            score_gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        else:
            score_category = "Needs Attention"
            score_color = "#f44336"
            score_gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
        
        # Overall Health Score Card
        st.markdown(f"""
        <div style="
            background: {score_gradient};
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            text-align: center;
            color: white;
            margin-bottom: 20px;
        ">
            <h3 style="margin: 0; color: white; font-size: 18px; opacity: 0.9;">Overall Heart Health Score</h3>
            <div style="font-size: 56px; font-weight: bold; margin: 10px 0;">{health_score}/100</div>
            <div style="font-size: 20px; font-weight: 600; opacity: 0.95;">{score_category}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Concerns Section (if any)
        if priority_concerns:
            st.markdown("")
            st.markdown("### ⚠️ Priority Concerns")
            st.markdown("""
            <div style="
                background-color: #fff3e0;
                border-left: 5px solid #ff9800;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            ">
            """, unsafe_allow_html=True)
            
            for concern in priority_concerns:
                param_name, value, status, severity, icon = concern
                param_display = param_name.replace("_", " ").title()
                unit = PARAMETER_RANGES[param_name]["unit"]
                
                explanation = get_parameter_explanation(param_name, status, value)
                
                # Format value for display
                if isinstance(value, float) and value < 1000:
                    value_display = f"{value:.2f}"
                else:
                    value_display = str(int(value))
                
                st.markdown(f"""
                **{icon} {param_display}:** {value_display} {unit} - **{status}**
                """)
                if explanation:
                    st.markdown(f"*{explanation.get('interpretation', '')}*")
                st.markdown("")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("")
        
        # Parameter Details in Expandable Section
        with st.expander("📊 **Detailed Parameter Analysis**", expanded=True):
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            
            # Create parameter comparison table
            params_to_check = [
                ("ejection_fraction", "Ejection Fraction"),
                ("serum_creatinine", "Serum Creatinine"),
                ("serum_sodium", "Serum Sodium"),
                ("platelets", "Platelets"),
                ("creatinine_phosphokinase", "Creatinine Phosphokinase"),
                ("age", "Age"),
            ]
            
            for param_key, param_display in params_to_check:
                if param_key in input_df.columns:
                    value = float(input_df[param_key].values[0])
                    status, color, icon, severity = evaluate_parameter(param_key, value)
                    unit = PARAMETER_RANGES[param_key]["unit"]
                    
                    # Get normal range for display
                    normal_range = PARAMETER_RANGES[param_key]["normal"]
                    if isinstance(normal_range, tuple):
                        range_str = f"{normal_range[0]}-{normal_range[1]} {unit}"
                    else:
                        range_str = "See guidelines"
                    
                    # Format value for display
                    if value >= 1000:
                        value_str = f"{int(value):,}"
                    elif value < 10:
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:.1f}"
                    
                    # Create parameter card
                    st.markdown(f"""
                    <div style="
                        background-color: {'#e8f5e9' if severity == 0 else '#fff3e0' if severity == 1 else '#ffe0e0'};
                        border-left: 4px solid {color};
                        padding: 12px 16px;
                        border-radius: 8px;
                        margin-bottom: 12px;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <strong style="font-size: 15px;">{icon} {param_display}</strong><br>
                                <span style="font-size: 13px; color: #666;">Normal Range: {range_str}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 22px; font-weight: bold; color: {color};">{value_str} {unit}</div>
                                <span style="
                                    background-color: {color};
                                    color: white;
                                    padding: 3px 10px;
                                    border-radius: 12px;
                                    font-size: 11px;
                                    font-weight: 600;
                                ">{status.upper()}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add explanation for abnormal values
                    if severity >= 2:
                        explanation = get_parameter_explanation(param_key, status, value)
                        if explanation:
                            with st.expander(f"ℹ️ What does this mean?", expanded=False):
                                st.markdown(f"**{explanation.get('what', '')}**")
                                st.markdown(f"*{explanation.get('interpretation', '')}*")
                                st.markdown(f"**💡 What you can do:** {explanation.get('action', '')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Modifiable Risk Factors Section
        if modifiable_factors:
            with st.expander("🎯 **Modifiable Risk Factors - You Can Change These!**", expanded=True):
                st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
                st.markdown("These are risk factors that you have control over. Making changes here can significantly improve your heart health:")
                st.markdown("")
                
                for factor_name, icon, recommendation in modifiable_factors:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
                        border-left: 4px solid #667eea;
                        padding: 12px 16px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                    ">
                        <strong style="font-size: 15px;">{icon} {factor_name}</strong><br>
                        <span style="font-size: 13px; color: #555;">{recommendation}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("---")
        st.markdown("")

        # Recommendations - Improved UI with functional collapsible sections
        st.subheader("📋 Recommended Clinical Actions")
        
        # Add CSS to style expanders with individual colors
        st.markdown("""
        <style>
        /* Base expander styling */
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stExpander"] details {
            border: none !important;
        }
        
        div[data-testid="stExpander"] summary {
            background-color: rgba(33, 150, 243, 0.1) !important;
            border-left: 4px solid #2196F3 !important;
            border-radius: 6px !important;
            padding: 12px 16px !important;
            font-weight: 600 !important;
            margin: 8px 0 !important;
        }
        
        div[data-testid="stExpander"] summary:hover {
            background-color: rgba(33, 150, 243, 0.2) !important;
        }
        
        /* Color coding based on risk tier - using nth-child */
        /* Low Risk - Blue and Teal */
        div[data-testid="stExpander"]:nth-child(1) summary {
            background-color: rgba(33, 150, 243, 0.1) !important;
            border-left: 4px solid #2196F3 !important;
        }
        div[data-testid="stExpander"]:nth-child(2) summary {
            background-color: rgba(0, 150, 136, 0.1) !important;
            border-left: 4px solid #009688 !important;
        }
        
        /* Moderate Risk - Orange, Amber, Red-Orange */
        div[data-testid="stExpander"]:nth-child(1) summary {
            background-color: rgba(255, 152, 0, 0.1) !important;
            border-left: 4px solid #ff9800 !important;
        }
        div[data-testid="stExpander"]:nth-child(2) summary {
            background-color: rgba(255, 193, 7, 0.1) !important;
            border-left: 4px solid #ffc107 !important;
        }
        div[data-testid="stExpander"]:nth-child(3) summary {
            background-color: rgba(255, 87, 34, 0.1) !important;
            border-left: 4px solid #ff5722 !important;
        }
        
        /* High Risk - Red, Pink, Purple */
        div[data-testid="stExpander"]:nth-child(1) summary {
            background-color: rgba(244, 67, 54, 0.1) !important;
            border-left: 4px solid #f44336 !important;
        }
        div[data-testid="stExpander"]:nth-child(2) summary {
            background-color: rgba(233, 30, 99, 0.1) !important;
            border-left: 4px solid #e91e63 !important;
        }
        div[data-testid="stExpander"]:nth-child(3) summary {
            background-color: rgba(156, 39, 176, 0.1) !important;
            border-left: 4px solid #9c27b0 !important;
        }
        div[data-testid="stExpander"]:nth-child(4) summary {
            background-color: rgba(211, 47, 47, 0.15) !important;
            border-left: 5px solid #d32f2f !important;
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        recommendations = get_clinical_recommendations(tier)
        
        # Display first recommendation prominently
        st.markdown(f"**{recommendations[0]}**")
        st.markdown("")  # spacing
        
        # Group remaining recommendations into categories - ALL COLLAPSED by default
        if tier == "Low risk":
            # Lifestyle & Monitoring - Light Blue
            with st.expander("🏥 **Lifestyle & Monitoring**", expanded=False):
                st.markdown("""
                <div style='background-color: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                for r in recommendations[1:4]:
                    st.markdown(f"- {r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Laboratory & Education - Light Teal
            with st.expander("🔬 **Laboratory & Education**", expanded=False):
                st.markdown("""
                <div style='background-color: #e0f2f1; border-left: 4px solid #009688; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                for r in recommendations[4:]:
                    st.markdown(f"- {r}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        elif tier == "Moderate risk":
            # Medication Optimization - Light Orange
            with st.expander("💊 **Medication Optimization (GDMT)**", expanded=False):
                st.markdown("""
                <div style='background-color: #fff3e0; border-left: 4px solid #ff9800; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                st.markdown(recommendations[1])  # Medication optimization header
                for r in recommendations[2:6]:  # Drug list
                    st.markdown(f"{r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Monitoring & Assessment - Light Amber
            with st.expander("📊 **Monitoring & Assessment**", expanded=False):
                st.markdown("""
                <div style='background-color: #fff8e1; border-left: 4px solid #ffc107; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                for r in recommendations[6:9]:
                    st.markdown(f"- {r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Warning Signs - Light Red/Orange
            with st.expander("⚠️ **Warning Signs**", expanded=False):
                st.markdown("""
                <div style='background-color: #ffebee; border-left: 4px solid #ff5722; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                st.markdown(f"- {recommendations[9]}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        else:  # High risk
            # URGENT: Hospitalization Criteria - Light Red
            with st.expander("🚨 **URGENT: Hospitalization Criteria**", expanded=False):
                st.markdown("""
                <div style='background-color: #ffebee; border-left: 4px solid #f44336; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                st.markdown(recommendations[1])  # Hospitalization consideration
                for r in recommendations[2:5]:
                    st.markdown(f"{r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Medication & Advanced Therapies - Light Pink
            with st.expander("💉 **Medication & Advanced Therapies**", expanded=False):
                st.markdown("""
                <div style='background-color: #fce4ec; border-left: 4px solid #e91e63; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                for r in recommendations[5:7]:
                    st.markdown(f"- {r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Intensive Monitoring Protocol - Light Purple
            with st.expander("🔬 **Intensive Monitoring Protocol**", expanded=False):
                st.markdown("""
                <div style='background-color: #f3e5f5; border-left: 4px solid #9c27b0; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                st.markdown(recommendations[7])  # Intensive monitoring header
                for r in recommendations[8:11]:
                    st.markdown(f"{r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # RED FLAG SYMPTOMS - Deep Red with alert
            with st.expander("🆘 **RED FLAG SYMPTOMS — EMERGENCY**", expanded=False):
                st.markdown("""
                <div style='background-color: #ffcdd2; border-left: 5px solid #d32f2f; border-radius: 6px; padding: 12px; margin: 8px 0;'>
                """, unsafe_allow_html=True)
                st.error("**Instruct patient to call 911 or go to ER immediately if:**")
                for r in recommendations[12:18]:
                    st.markdown(f"{r}")
                st.markdown(f"- {recommendations[18]}")  # Care coordination
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("")

        # SHAP Plots in Tabs for Better Organization
        st.subheader("🔍 AI Explainability - Why This Prediction?")
        st.markdown("Understanding the factors driving this risk assessment using SHAP (SHapley Additive exPlanations)")
        st.markdown("")
        
        tab1, tab2, tab3 = st.tabs(["📊 Patient Contribution", "🌐 Global Importance", "ℹ️ Interpretation Guide"])
        
        with tab1:
            st.markdown("")
            st.markdown("**Individual Feature Contributions for This Patient**")
            st.markdown("This chart shows how each clinical feature pushed the prediction higher (red) or lower (blue) for this specific patient.")
            st.markdown("")
            fig_patient = plot_shap_patient_contrib(shap_vals, feature_names=FEATURES, top_n=len(FEATURES))
            st.pyplot(fig_patient)
            st.markdown("""
            - **Red bars** → Features pushing **toward higher risk** (mortality)
            - **Blue bars** → Features pushing **toward lower risk** (survival)
            - **Longer bars** → Stronger impact on the prediction
            """)
        
        with tab2:
            st.markdown("")
            if sample_df is not None:
                st.markdown("**Overall Feature Importance Across All Patients**")
                st.markdown("This chart shows which features are most important on average across the entire dataset.")
                st.markdown("")
                fig_global = plot_shap_bar_global(sample_df, max_display=len(FEATURES))
                if fig_global:
                    st.pyplot(fig_global)
                st.markdown("""
                - Features are ranked by their **average absolute impact** on predictions
                - Higher values indicate features that more strongly influence the model overall
                - This helps identify the most critical clinical markers across all patients
                """)
            else:
                st.info("Global importance plot requires the training dataset to be available.")
        
        with tab3:
            st.markdown("")
            st.markdown("### 📖 How to Interpret SHAP Values")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** provides a unified measure of feature importance based on game theory.
            
            #### Patient Contribution Plot
            - Shows **why** the model made its prediction for this specific patient
            - Each feature's contribution is calculated relative to an "average" patient
            - Positive values push the prediction toward **higher mortality risk**
            - Negative values push the prediction toward **lower mortality risk**
            
            #### Global Importance Plot
            - Shows which features are **most important overall** across all patients
            - Calculated as the mean absolute SHAP value for each feature
            - Helps identify which clinical markers matter most for the model's decisions
            
            #### Clinical Application
            1. **Review top contributors** - Focus on features with largest absolute SHAP values
            2. **Validate clinical relevance** - Ensure important features align with medical knowledge
            3. **Identify intervention targets** - Modifiable features pushing toward high risk may be targets for treatment
            4. **Build trust** - Transparent explanations help clinicians trust and validate AI predictions
            
            #### Example Interpretation
            If `ejection_fraction` has a large negative SHAP value (blue bar):
            - This patient's ejection fraction is **protective** (reducing mortality risk)
            - It's pulling the prediction **away from** the high-risk category
            
            If `serum_creatinine` has a large positive SHAP value (red bar):
            - This patient's elevated creatinine is **concerning** (increasing mortality risk)
            - It's pushing the prediction **toward** the high-risk category
            
            ---
            
            💡 **Remember**: SHAP values explain the model's reasoning, but clinical judgment should always guide final decisions.
            """)

        # Offer download of result as JSON or CSV
        st.markdown("---")
        if st.button("Download result (CSV)"):
            out = input_df.copy()
            out["predicted_prob"] = prob
            out["risk_tier"] = tier
            st.download_button("Click to download CSV", out.to_csv(index=False), file_name="patient_risk.csv", mime="text/csv")

# Footer / notes
st.markdown("---")
st.caption("Notes: This app predicts *risk* (probability) not a deterministic outcome. Use clinical judgement. Always validate models locally before clinical use.")

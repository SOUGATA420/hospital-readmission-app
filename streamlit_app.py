import streamlit as st
import pickle
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64


# ---------------------
# Page Configuration
# ---------------------
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------------
# Load model, selector, metrics
# ---------------------
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

with open('model_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# ---------------------
# Page Header
# ---------------------
st.markdown("<h1 style='text-align: center; color: navy;'>🏥 Hospital Readmission Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict the likelihood of patient readmission using clinical data</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------
# Input Section (Responsive)
# ---------------------
st.markdown("## 📝 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age Group", [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ], help="Select the patient's age range")

    num_lab_procedures = st.slider("🧪 Lab Procedures", 1, 150, 40,
                                   help="Number of lab tests performed during hospitalization")

    num_medications = st.slider("💊 Medications", 1, 100, 18,
                                 help="Total number of different medications administered")

    number_emergency = st.slider("🚑 Emergency Visits", 0, 10, 0,
                                  help="Number of emergency department visits made by the patient")

    number_diagnoses = st.slider("📋 Diagnoses", 1, 16, 9,
                                 help="Number of distinct diagnoses assigned during the visit")

with col2:
    time_in_hospital = st.slider("⏳ Time in Hospital (days)", 1, 20, 3,
                                 help="Days the patient stayed in the hospital")

    num_procedures = st.slider("🔧 Procedures", 0, 6, 1,
                                help="Number of procedures done during admission")

    number_inpatient = st.slider("🏥 Inpatient Visits", 0, 20, 0,
                                 help="Times previously admitted as inpatient")

    number_outpatient = st.slider("🏨 Outpatient Visits", 0, 20, 0,
                                  help="Number of outpatient visits (no admission)")

st.markdown("<br>", unsafe_allow_html=True)
submitted = st.button("✅ Submit")

# ---------------------
# Prediction
# ---------------------

age_mapping = {
    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
    '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
}

if submitted:
    st.markdown("## 📄 Patient Summary")
    with st.container():
        st.markdown(f"""
        - **Age Group:** {age}  
        - **Time in Hospital:** {time_in_hospital} days  
        - **Lab Procedures:** {num_lab_procedures}  
        - **Procedures:** {num_procedures}  
        - **Medications:** {num_medications}  
        - **Diagnoses:** {number_diagnoses}  
        - **Inpatient Visits:** {number_inpatient}  
        - **Emergency Visits:** {number_emergency}  
        - **Outpatient Visits:** {number_outpatient}  
        """)
        st.markdown("---")

    # Prepare input
    input_dict = {
        'age': age_mapping[age],
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_diagnoses': number_diagnoses,
        'number_inpatient': number_inpatient,
        'number_emergency': number_emergency,
        'number_outpatient': number_outpatient
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure all expected features are present
    expected_features = selector.feature_names_in_
    for col in set(expected_features) - set(input_df.columns):
        input_df[col] = 0
    input_df = input_df[list(expected_features)]

    # Transform and predict
    transformed = selector.transform(input_df)
    prediction = model.predict(transformed)[0]
    proba = model.predict_proba(transformed)[0][1]

    # Prediction result layout
    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.subheader("🩺 Prediction")
        if prediction == 1:
            st.error(f"⚠️ **High Risk** of Readmission\nProbability: `{proba:.2f}`")
        else:
            st.success(f"✅ **Low Risk** of Readmission\nProbability: `{proba:.2f}`")

    with result_col2:
        st.subheader("🧭 Risk Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of Readmission", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0.0, 0.4], 'color': "#90ee90"},   # Green
                    {'range': [0.4, 0.7], 'color': "#ffeb3b"},   # Yellow
                    {'range': [0.7, 1.0], 'color': "#f44336"}    # Red
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'value': proba
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    #function for download the pdf report
    def create_pdf_report(input_dict, prediction, proba):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Add watermark
        pdf.set_text_color(200, 200, 200)
        pdf.set_font("Arial", size=40)
        pdf.rotate(45, x=None, y=None)
        pdf.text(20, 100, "Maity-enp")
        pdf.rotate(0)

        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt=" Hospital Readmission Prediction Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        # Patient Summary
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt=" Patient Summary:", ln=True)
        pdf.set_font("Arial", size=12)
        for key, value in input_dict.items():
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)

        pdf.ln(5)
        # Prediction Result
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt=" Prediction Result:", ln=True)
        pdf.set_font("Arial", size=12)
        result_text = "High Risk of Readmission" if prediction == 1 else "Low Risk of Readmission"
        pdf.cell(200, 8, txt=f"Prediction: {result_text}", ln=True)
        pdf.cell(200, 8, txt=f"Probability: {proba:.2f}", ln=True)

        # Evaluation Metrics (Optional Summary)
        pdf.ln(5)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt=" Model Evaluation Metrics:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Accuracy: {metrics['accuracy']*100:.2f}%", ln=True)
        pdf.cell(200, 8, txt=f"Precision: {metrics['precision']:.2f}", ln=True)
        pdf.cell(200, 8, txt=f"Recall: {metrics['recall']:.2f}", ln=True)
        pdf.cell(200, 8, txt=f"F1 Score: {metrics['f1']:.2f}", ln=True)
        # Signature
        pdf.ln(15)
        pdf.set_font("Arial", style='I', size=12)
        pdf.cell(200, 10, txt="Authorized by: Sougata Maity", ln=True, align='R')
        # Save to bytes
        return pdf.output(dest='S')
    #download button
    pdf_bytes = create_pdf_report(input_dict, prediction, proba)
    b64 = base64.b64encode(pdf_bytes).decode()

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="readmission_report.pdf">📄 Download Prediction Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------------------
# Model Information & Evaluation
# ---------------------
st.markdown("## 📊 Model Info & Evaluation Metrics")

info_col1, info_col2 = st.columns([1, 2])

with info_col1:
    st.markdown("### 🧠 Model Used")
    st.write("**StackingClassifier** with base learners:")
    st.markdown("- Random Forest\n- XGBoost\n- Logistic Regression (meta)")

    st.markdown("### 📈 Evaluation Metrics")
    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("Precision", f"{metrics['precision']:.2f}")
    st.metric("Recall", f"{metrics['recall']:.2f}")
    st.metric("F1 Score", f"{metrics['f1']:.2f}")

with info_col2:
    st.markdown("### 🔲 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### 📋 Classification Report")
    st.code(metrics['report'], language='text')

# ---------------------
# Footer
# ---------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    f"© {datetime.datetime.now().year} | Built  using Streamlit by <b>Sougata Maity</b>"
    f"</div>",
    unsafe_allow_html=True
)

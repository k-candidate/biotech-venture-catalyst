# =====================================================================
# app.py - Streamlit Biotech Clinical Trial Risk Prediction Dashboard
# =====================================================================
import streamlit as st
import pickle
import re
import pandas as pd

st.set_page_config(
    page_title="Biotech Venture Catalyst",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_production_pipeline():
    with open("model.bin", "rb") as f_in:
        return pickle.load(f_in)

try:
    model_pipeline = load_production_pipeline()
except Exception as e:
    st.error(f"❌ Failed to load model.bin file artifact: {e}")

def local_feature_transformer(start_date, complete_date, icdcodes, eligibility_criteria):
    record = {}
    start_dt = pd.to_datetime(start_date, errors='coerce', format='mixed')
    complete_dt = pd.to_datetime(complete_date, errors='coerce', format='mixed')
    
    record['start_year'] = int(start_dt.year) if pd.notnull(start_dt) else -1
    record['start_month'] = int(start_dt.month) if pd.notnull(start_dt) else -1
    record['duration_days'] = int((complete_dt - start_dt).days) if pd.notnull(start_dt) and pd.notnull(complete_dt) else -1
    
    record['num_conditions'] = len(icdcodes.split('__')) if icdcodes else 0
    record['primary_disease_chapter'] = str(icdcodes)[0] if icdcodes and len(str(icdcodes)) > 0 else 'Unknown'
    
    criteria_text = eligibility_criteria or ""
    record['criteria_word_count'] = len(criteria_text.split())
    record['exclusion_keyword_count'] = sum(1 for _ in re.finditer(r'(?i)\b(exclude|exclusion|no history|prior concurrent)\b', criteria_text))
    return record

# --- UI Layout ---
st.title("🧬 Biotech Venture Catalyst")
st.subheader("Predicting Phase 1 Clinical Trial Probability of Success (PoS)")
st.markdown("---")

col_input, col_results = st.columns([1.2, 1])

with col_input:
    st.header("📋 Clinical Trial Operational Metadata")
    
    start_date = st.text_input("Trial Start Date", value="March 1996", help="Format: 'Month Year' or 'YYYY-MM-DD'")
    complete_date = st.text_input("Expected Completion Date", value="July 2000", help="Leave blank if trial was halted")
    
    icdcodes = st.text_input(
        "Target Indication ICD-10 Codes", 
        value="C96.A__C46.9", 
        help="Separate multiple ICD-10 codes with '__'."
    )
    
    eligibility_criteria = st.text_area(
        "Eligibility Criteria",
        value="Patient Characteristics: Age 13 and over. Performance status: stable. No uncontrolled systemic infection. Effective contraception required.",
        height=220
    )

with col_results:
    st.header("📊 Operational Risk Assessment")
    
    if st.button("Calculate Probability of Success", type="primary"):
        # Step 1: Process raw inputs through the transformer
        engineered_record = local_feature_transformer(start_date, complete_date, icdcodes, eligibility_criteria)
        
        # Step 2: Extract scalar Probability of Success (Class 1)
        pos_probability = float(model_pipeline.predict_proba([engineered_record])[0, 1])
        
        # Step 3: Display Core Prediction Metrics
        st.markdown("### 🎯 Prediction")
        st.metric(
            label="Probability of Success (PoS)", 
            value=f"{pos_probability * 100:.2f}%"
        )
        st.progress(pos_probability)
        st.caption("Continuous probability score derived from Phase 1 operational metadata.")
                
        st.markdown("---")
        with st.expander("🔬 View Derived Features", expanded=True):
            st.json(engineered_record)

import streamlit as st
import pandas as pd
import joblib

# Load trained model
dt_model = joblib.load("mental_health_model.pkl")

# Page config
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    layout="centered"
)

# Background CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/1345881892/vector/people-human-concept-abstract-color-background-vector-ilustration.jpg?s=612x612&w=0&k=20&c=amHctoUOt52HD7WSg5cNs5ZolTLcz-UoP0GvgFNAgEI=");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        padding: 2rem;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<div class='block-container'>", unsafe_allow_html=True)
st.title("🧠 Mental Health Risk Prediction")

# Sidebar Inputs
st.subheader("Please provide the following information:")

gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed", "Student"])
work_env = st.selectbox("Work Environment", ["Remote", "On-site", "Hybrid"])
mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
seeks_treatment = st.selectbox("Seeking Treatment", ["Yes", "No"])

stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
physical_days = st.slider("Physical Activity Days/Week", 0, 7, 3)
age = st.slider("Age", 10, 100, 30)
depression_score = st.slider("Depression Score (0-30)", 0, 30, 10)
anxiety_score = st.slider("Anxiety Score (0-30)", 0, 30, 10)
support_score = st.slider("Social Support Score (0-100)", 0, 100, 50)
productivity_score = st.slider("Productivity Score (0-100)", 0, 100, 70)

# Predict Button
if st.button("Predict Mental Health Risk"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'gender': [gender],
        'employment_status': [employment_status],
        'work_environment': [work_env],
        'mental_health_history': [mental_health_history],
        'seeks_treatment': [seeks_treatment],
        'stress_level': [stress_level],
        'sleep_hours': [sleep_hours],
        'physical_activity_days': [physical_days],
        'age': [age],
        'depression_score': [depression_score],
        'anxiety_score': [anxiety_score],
        'social_support_score': [support_score],
        'productivity_score': [productivity_score]
    })

    # One-hot encode and reindex to match model input
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=dt_model.feature_names_in_, fill_value=0)

    # Predict
    prediction = dt_model.predict(input_df)[0]
    st.success(f"🩺 Predicted Mental Health Risk: **{prediction}**")

st.markdown("</div>", unsafe_allow_html=True)

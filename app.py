import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="KMeans Predictor", layout="centered")
st.title("🤖 Customer Cluster Prediction (KMeans Model)")

# Load model and scaler
@st.cache_resource
def load_model():
    kmeans = joblib.load("km.pkl")
    scaler = joblib.load("sc.pkl")
    return kmeans, scaler

kmeans, scaler = load_model()

# Feature Inputs
st.subheader("Enter Customer Features")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Ever Married", ["Yes", "No"])
age = st.number_input("Age", value=30.0, step=1.0)
graduated = st.selectbox("Graduated", ["Yes", "No"])
profession = st.selectbox(
    "Profession",
    ["Artist", "Doctor", "Engineer", "Entertainment", "Executive",
     "Healthcare", "Homemaker", "Lawyer", "Marketing", "Other"]
)
work_exp = st.number_input("Work Experience (Years)", value=3, step=1)
spending_score = st.number_input("Spending Score", value=60.0, step=1.0)
family_size = st.number_input("Family Size", value=2, step=1)

# Encode categorical inputs
gender_enc = 1 if gender.lower() == "male" else 0
married_enc = 1 if married.lower() == "yes" else 0
graduated_enc = 1 if graduated.lower() == "yes" else 0

# Label encoding for profession
profession_map = {
    "Artist": 0, "Doctor": 1, "Engineer": 2, "Entertainment": 3,
    "Executive": 4, "Healthcare": 5, "Homemaker": 6, "Lawyer": 7,
    "Marketing": 8, "Other": 9
}
profession_enc = profession_map.get(profession, 9)

# Combine into feature array (8 features in this exact order)
features = np.array([[gender_enc, married_enc, age, graduated_enc,
                      profession_enc, int(work_exp), spending_score, int(family_size)]])

# DEBUG INFO
st.write("🔍 Input Features (before scaling):", features)
st.write("📏 Shape:", features.shape)

# Predict
if st.button("Predict Cluster"):
    try:
        scaled_input = scaler.transform(features)
        st.write("🧪 Scaled Input Shape:", scaled_input.shape)

        cluster = kmeans.predict(scaled_input)
        st.success(f"✅ Predicted Cluster: {cluster[0]}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

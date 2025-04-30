import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

st.set_page_config(page_title="ML Classification Model Prediction", layout="centered")
st.title("ML Classification Model Prediction")          

# --- Clear cache ---
if st.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared! Please re-upload or proceed.")

st.markdown("""
Welcome to the **Recommendation Model Predictor**! This app helps predict the best recommendation model for users based on their preferences and behavior.

### Key Features:
- **Interactive User Input**: Users can input personal details (e.g., age, cuisine preference, taste) to get a model recommendation.
- **Data Upload**: Option to upload a custom dataset or use the default dataset.
- **Model Prediction**: A trained Random Forest Classifier predicts the best recommendation model based on the user's input.
- **Feature Importance**: Visual display of the top 10 most important features influencing the model's recommendations.
- **Simulated User Predictions**: Predictions for sample users are displayed to demonstrate the model's functionality.
- **Download Results**: Users can download simulated predictions in CSV format for further analysis.
""")

# --- File Upload Section ---
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with user data", type=["csv"])

@st.cache_data
def load_default_data():
    return pd.read_csv("recommendation_model_updated_v5.csv")

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()
else:
    st.info("ℹ️ Using default dataset (recommendation_model_updated_v5.csv)")
    df = load_default_data()

# --- Train Model ---
@st.cache_resource
def train_model(df):
    features = [
        "Donations ($)", "Recommendation_Accuracy (%)", "Engagement_(min/session)", "user_age",
        "user_cuisine", "gender", "taste", "likes", "rating", "Time_Spent (min)",
        "Conversion_Rate (%)", "occasion", "place", "dietary_preferences", "budget"
    ]
    target = "Model_Used"

    X = df[features]
    y = df[target]

    categorical_features = ["user_cuisine", "gender", "taste", "occasion", "place", "dietary_preferences", "budget"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder='passthrough')

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, "model_recommender.pkl")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, acc, report

# Show dataset
st.write("### Preview of Data:")
st.write(df.head())
st.write("### Summary Statistics:")
st.write(df.describe())

# Grouped summary
st.write("### Group by Model_Used and calculate the mean of Conversion_Rate (%)")
st.write(df.groupby("Model_Used").agg({
    'gender': 'count', 'user_age': 'mean', 'occasion': 'count', 'user_cuisine': 'count',
    'taste': 'count', 'Conversion_Rate (%)': 'mean', 'likes': 'count', 'rating': 'count',
    'place': 'count', 'dietary_preferences': 'count', 'budget': 'count'
}))

# Conversion rate summary
conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()
max_conversion_model = conversion_rate_summary.idxmax()
max_conversion_value = conversion_rate_summary.max()
st.write(f"✅ The model with the highest Conversion Rate is **Model {max_conversion_model}**, with a Conversion Rate of **{max_conversion_value:.2f}%**")

# Train model
clf, accuracy, report = train_model(df)

# --- User Input Section ---
st.header("🧑 ML Model Prediction for a New User")

with st.form("user_form"):
    user_age = st.slider("User Age", 1, 100, 25)
    user_cuisine = st.selectbox("Preferred Cuisine", df["user_cuisine"].unique())
    gender = st.radio("Gender",

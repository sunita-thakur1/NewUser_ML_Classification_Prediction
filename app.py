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
- **Interactive User Input**: Users can input personal details (e.g., cuisine preference, taste, etc.) to get a model recommendation.
- **Data Upload**: Upload your own dataset (CSV required).
- **Model Prediction**: A trained Random Forest Classifier predicts the best recommendation model based on the user's input.
- **Feature Importance**: Visual display of the top 10 most important features influencing the model's recommendations.
- **Simulated User Predictions**: Predictions for sample users are displayed to demonstrate the model's functionality.
- **Download Results**: Users can download simulated predictions in CSV format for further analysis.
""")

# --- File Upload Section ---
st.subheader("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with user data", type=["csv"])

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload a CSV file to continue.")
    st.stop()

# --- Train Model ---
@st.cache_resource
def train_model(df):
    features = [
        "user_cuisine", "taste", "Time_Spent (min)", 
        "occasion", "place", "dietary_preferences", "budget"
    ]
    target = "Model_Used"

    X = df[features]
    y = df[target]

    categorical_features = [
        "user_cuisine", "taste", "occasion", 
        "place", "dietary_preferences", "budget"
    ]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder='passthrough')

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "model_recommender.pkl")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, acc, report

# --- Display Dataset Info ---
st.write("### Preview of Data:")
st.write(df.head())
st.write("### Summary Statistics:")
st.write(df.describe())

# --- Grouped Summary ---
st.write("### Group by Model_Used and calculate the mean of Conversion_Rate (%)")
st.write(df.groupby("Model_Used").agg({
    'gender': 'count', 'user_age': 'mean', 'occasion': 'count', 'user_cuisine': 'count',
    'taste': 'count', 'Conversion_Rate (%)': 'mean', 'likes': 'count', 'rating': 'count',
    'place': 'count', 'dietary_preferences': 'count', 'budget': 'count'
}))

# --- Best Performing Model ---
conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()
max_conversion_model = conversion_rate_summary.idxmax()
max_conversion_value = conversion_rate_summary.max()
st.write(f"✅ The model with the highest Conversion Rate is **Model {max_conversion_model}**, with a Conversion Rate of **{max_conversion_value:.2f}%**")

# --- Train the model ---
clf, accuracy, report = train_model(df)

# --- User Input Section ---
st.header("🧑 ML Model Prediction for a New User")

with st.form("user_form"):
    user_cuisine = st.selectbox("Preferred Cuisine", df["user_cuisine"].unique())
    taste = st.selectbox("Taste Preference", df["taste"].unique())
    time_spent = st.slider("Time Spent (min)", 0, 120, 30)
    occasion = st.selectbox("Occasion", df["occasion"].unique())
    place = st.selectbox("Place", df["place"].unique())
    dietary_preferences = st.selectbox("Preferred Diet", df["dietary_preferences"].unique())
    budget = st.selectbox("Budget", df["budget"].unique())

    submitted = st.form_submit_button("Predict Model")

if submitted:
    new_user = pd.DataFrame([{
        "user_cuisine": user_cuisine,
        "taste": taste,
        "Time_Spent (min)": time_spent,
        "occasion": occasion,
        "place": place,
        "dietary_preferences": dietary_preferences,
        "budget": budget
    }])

    prediction = clf.predict(new_user)[0]
    st.success(f"✅ Recommended Model: **Model {prediction}**")

# --- Feature Importance ---
with st.expander("📊 Show Feature Importances"):
    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["classifier"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx][:10][::-1], 
            [importances[i] for i in sorted_idx][:10][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

# --- Simulated Users ---
st.header("🧪 Simulated Users")
simulated_users = pd.DataFrame([
    {
        "user_cuisine": "Mexican", "taste": "Spicy",
        "Time_Spent (min)": 10, "occasion": "Party", "place": "Home",
        "dietary_preferences": "None", "budget": "Low"
    },
    {
        "user_cuisine": "Japanese", "taste": "Umami",
        "Time_Spent (min)": 60, "occasion": "Date", "place": "Restaurant",
        "dietary_preferences": "Vegetarian", "budget": "High"
    }
])

predicted_models = clf.predict(simulated_users)
simulated_users["Recommended_Model"] = predicted_models
st.dataframe(simulated_users)

# --- Download results ---
csv = simulated_users.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Simulated Results", csv, "simulated_predictions.csv", "text/csv")


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

st.set_page_config(page_title="Recommendation Model Predictor", layout="centered")
st.title("🔮 Recommendation Model Predictor")
st.markdown("Upload your dataset or use the default. Enter user info below to predict the best recommendation model.")

# --- Load or train model ---
@st.cache_data
def load_data():
    return pd.read_csv("recommendation_model_updated_v4.csv")  
    

@st.cache_resource
def train_model(df):
    features = [
        "user_age", "user_cuisine", "sex", "taste",
        "Likes", "Dislikes", "Time_Spent (min)", "Conversion_Rate (%)"
    ]
    target = "Model_Used"

    X = df[features]
    y = df[target]

    categorical_features = ["user_cuisine", "sex", "taste"]

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

# Load and train
df = load_data()
clf, accuracy, report = train_model(df)
st.write("### Preview of Data:")
st.write(df.head())
st.write(df.groupby("Model_Used").agg({'sex': 'count',  'user_age': 'mean', 'user_cuisine': 'count', 'user_cuisine':'count', 'taste': 'count', 'Conversion_Rate (%)': 'mean', 'Likes': 'count' }))
#st.markdown(f"### 🎯 Model Accuracy: `{accuracy:.2f}` on test set")

# --- User Input Section ---
st.header("🧑 Prediction for a New User")

with st.form("user_form"):
    user_age = st.slider("User Age", 1, 100, 25)
    user_cuisine = st.selectbox("Preferred Cuisine", df["user_cuisine"].unique())
    sex = st.radio("Sex", df["sex"].unique())
    taste = st.selectbox("Taste Preference", df["taste"].unique())
    likes = st.number_input("Likes", min_value=0, value=10)
    dislikes = st.number_input("Dislikes", min_value=0, value=2)
    time_spent = st.slider("Time Spent (min)", 0, 120, 30)
    conversion_rate = st.slider("Conversion Rate (%)", 0, 100, 5)

    submitted = st.form_submit_button("Predict Model")

    if submitted:
        new_user = pd.DataFrame([{
            "user_age": user_age,
            "user_cuisine": user_cuisine,
            "sex": sex,
            "taste": taste,
            "Likes": likes,
            "Dislikes": dislikes,
            "Time_Spent (min)": time_spent,
            "Conversion_Rate (%)": conversion_rate
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
    {"user_age": 5, "user_cuisine": "Mexican", "sex": "Male", "taste": "Spicy",
     "Likes": 15, "Dislikes": 3, "Time_Spent (min)": 10, "Conversion_Rate (%)": 2},

    {"user_age": 95, "user_cuisine": "Japanese", "sex": "Female", "taste": "Umami",
     "Likes": 8, "Dislikes": 2, "Time_Spent (min)": 60, "Conversion_Rate (%)": 7}
])

predicted_models = clf.predict(simulated_users)
simulated_users["Recommended_Model"] = predicted_models

st.dataframe(simulated_users)

# Optional: Download predictions
csv = simulated_users.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Simulated Results", csv, "simulated_predictions.csv", "text/csv")

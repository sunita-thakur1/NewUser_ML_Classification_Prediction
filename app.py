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
- **Interactive User Input**: Users can input personal details to get a model recommendation.
- **Data Upload**: Upload your own dataset (CSV required).
- **Model Prediction**: A trained Random Forest Classifier predicts the best recommendation model.
- **Feature Importance**: Visual display of important features.
- **Simulated Predictions**: See example predictions.
- **Download Results**: Export prediction results in CSV format.
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

# --- Display Dataset ---
st.write("### Preview of Data:")
st.write(df.head())
st.write("### Summary Statistics:")
st.write(df.describe())

# --- Feature Selection ---
st.subheader("🔧 Select Model Configuration")

columns = df.columns.tolist()

features = st.multiselect("Select Feature Columns", options=columns, default=[
    "user_cuisine", "taste", "Time_Spent (min)", "occasion", "place", "dietary_preferences", "budget"
])

target = st.selectbox("Select Target Column", options=columns, index=columns.index("Model_Used") if "Model_Used" in columns else 0)

categorical_features = st.multiselect(
    "Select Categorical Features (subset of features)", 
    options=features,
    default=[col for col in features if df[col].dtype == 'object']
)

# --- Train Model ---
@st.cache_resource
def train_model(df, features, target, categorical_features):
    X = df[features]
    y = df[target]

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

# --- Grouped Summary ---
if "Conversion_Rate (%)" in df.columns and "Model_Used" in df.columns:
    st.write("### Group by Model_Used and calculate the mean of Conversion_Rate (%)")
    st.write(df.groupby("Model_Used")['Conversion_Rate (%)'].mean())

    conversion_rate_summary = df.groupby("Model_Used")['Conversion_Rate (%)'].mean()
    max_model = conversion_rate_summary.idxmax()
    max_val = conversion_rate_summary.max()
    st.write(f"✅ Highest Conversion Rate: **Model {max_model}** at **{max_val:.2f}%**")

# --- Train Model ---
clf, accuracy, report = train_model(df, features, target, categorical_features)

# --- User Input Section ---
st.header("🧑 Prediction for a New User")

with st.form("user_form"):
    user_input = {}
    for feature in features:
        if df[feature].dtype == 'object':
            user_input[feature] = st.selectbox(f"{feature}", df[feature].unique())
        elif np.issubdtype(df[feature].dtype, np.number):
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            user_input[feature] = st.slider(f"{feature}", min_val, max_val, int(df[feature].mean()))
        else:
            user_input[feature] = st.text_input(f"{feature}")

    submitted = st.form_submit_button("Predict Model")

if submitted:
    new_user = pd.DataFrame([user_input])
    prediction = clf.predict(new_user)[0]
    st.success(f"✅ Predicted {target}: **{prediction}**")

# --- Feature Importance ---
with st.expander("📊 Show Feature Importances"):
    try:
        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    except:
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
# Use only selected features
simulated_users = pd.DataFrame([
    {f: df[f].iloc[0] for f in features},
    {f: df[f].iloc[1] for f in features}
])

predicted_models = clf.predict(simulated_users)
simulated_users[f"Predicted_{target}"] = predicted_models
st.dataframe(simulated_users)

# --- Download ---
csv = simulated_users.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Simulated Results", csv, "simulated_predictions.csv", "text/csv")

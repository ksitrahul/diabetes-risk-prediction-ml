import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "diabetes_model.joblib"
DATA_PATH = "Diabetes_Prediction_Dataset.csv"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH).drop(columns=["Patient_ID"])
    df["Blood_Pressure"] = df["Blood_Pressure"].replace(999, np.nan)
    X = df.drop(columns=["Diabetes"])
    y = df["Diabetes"].map({"Yes": 1, "No": 0})

    num_cols = ["Age", "BMI", "Glucose", "Blood_Pressure", "Cholesterol", "Insulin", "HbA1c"]
    cat_cols = ["Gender", "Smoking_Status", "Physical_Activity", "Family_History"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    artifact = {
        "pipeline": pipe,
        "model_name": "Gradient Boosting Classifier",
        "features": num_cols + cat_cols,
        "metrics": {
            "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, probs)), 4)
        }
    }
    joblib.dump(artifact, MODEL_PATH)
    return artifact

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Blood_Pressure"] = df["Blood_Pressure"].replace(999, np.nan)
        return df
    return None

artifact = load_or_train_model()
pipeline = artifact["pipeline"]
metrics = artifact.get("metrics", {})

# Sidebar
st.sidebar.title("🩺 Clinical Navigation")
page = st.sidebar.radio("Select View", ["Single Patient Prediction", "Batch CSV Prediction", "Dataset & Model Insights"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Metadata")
st.sidebar.write(f"**Algorithm:** {artifact.get('model_name', 'Gradient Boosting')}")
if "Accuracy" in metrics:
    st.sidebar.metric("Test Accuracy", f"{metrics['Accuracy']*100:.2f}%")
if "ROC-AUC" in metrics:
    st.sidebar.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

st.sidebar.info("Note: For educational and screening support only. Always consult a qualified physician for medical diagnosis.")

# Main Header
st.title("Diabetes Risk Assessment & ML Analytics")
st.markdown("Enter patient clinical biomarkers and lifestyle parameters below to estimate diabetes probability.")

if page == "Single Patient Prediction":
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics & Lifestyle")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            activity = st.selectbox("Physical Activity", ["Moderate", "High", "Low"])
            fam_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

        with col2:
            st.subheader("Glycemic & Metabolic Markers")
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=350.0, value=120.0, step=1.0)
            hba1c = st.number_input("HbA1c (%)", min_value=3.5, max_value=15.0, value=6.0, step=0.1)
            insulin = st.number_input("Insulin (mu U/ml)", min_value=10, max_value=400, value=110, step=1)

        with col3:
            st.subheader("Vitals & Lipid Profile")
            bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=60.0, value=26.5, step=0.1)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=60, max_value=240, value=125, step=1)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)

        submitted = st.form_submit_button("Predict Diabetes Risk", use_container_width=True, type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age,
            "BMI": bmi,
            "Glucose": glucose,
            "Blood_Pressure": bp,
            "Cholesterol": cholesterol,
            "Insulin": insulin,
            "HbA1c": hba1c,
            "Gender": gender,
            "Smoking_Status": smoking,
            "Physical_Activity": activity,
            "Family_History": fam_history
        }])

        prob = float(pipeline.predict_proba(input_df)[0][1])
        pred = int(pipeline.predict(input_df)[0])

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Predicted Diabetes Probability", f"{prob*100:.1f}%")
            if prob >= 0.70:
                st.error("High Risk — Positive Prediction (Yes)")
            elif prob >= 0.40:
                st.warning("Moderate / Borderline Risk")
            else:
                st.success("Low Risk — Negative Prediction (No)")

        with res_col2:
            st.subheader("Clinical Biomarker Reference Check")
            flags = []
            if glucose >= 126:
                flags.append(f"• **Elevated Glucose ({glucose} mg/dL):** Fasting levels ≥ 126 mg/dL are in the diabetic range.")
            elif glucose >= 100:
                flags.append(f"• **Borderline Glucose ({glucose} mg/dL):** Fasting levels between 100–125 mg/dL indicate prediabetes.")
            if hba1c >= 6.5:
                flags.append(f"• **Elevated HbA1c ({hba1c}%):** Levels ≥ 6.5% are consistent with diabetes.")
            elif hba1c >= 5.7:
                flags.append(f"• **Borderline HbA1c ({hba1c}%):** Levels between 5.7%–6.4% indicate prediabetes.")
            if bmi >= 30.0:
                flags.append(f"• **High BMI ({bmi} kg/m²):** BMI ≥ 30 falls in the obesity range, increasing insulin resistance risk.")
            if fam_history == "Yes":
                flags.append("• **Family History:** Positive family history is a major non-modifiable risk contributor.")

            if flags:
                for f_text in flags:
                    st.markdown(f_text)
            else:
                st.markdown("• Key glycemic and BMI markers are within standard reference ranges.")

elif page == "Batch CSV Prediction":
    st.subheader("Upload Patient CSV for Batch Screening")
    st.markdown("Upload a CSV file containing the columns: `Age`, `Gender`, `BMI`, `Glucose`, `Blood_Pressure`, `Cholesterol`, `Insulin`, `HbA1c`, `Smoking_Status`, `Physical_Activity`, `Family_History`.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "Blood_Pressure" in batch_df.columns:
            batch_df["Blood_Pressure"] = batch_df["Blood_Pressure"].replace(999, np.nan)

        required_cols = artifact["features"]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            probs = pipeline.predict_proba(batch_df[required_cols])[:, 1]
            preds = np.where(probs >= 0.5, "Yes", "No")

            out_df = batch_df.copy()
            out_df["Predicted_Diabetes"] = preds
            out_df["Diabetes_Probability_%"] = (probs * 100).round(2)

            st.success(f"Processed {len(out_df)} records successfully.")
            st.dataframe(out_df.head(50), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_bytes,
                file_name="diabetes_batch_predictions.csv",
                mime="text/csv"
            )

elif page == "Dataset & Model Insights":
    st.subheader("Training Dataset Overview & Feature Importance")
    df_data = load_dataset()
    if df_data is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients", len(df_data))
        c2.metric("Diabetes Positive (%)", f"{(df_data['Diabetes'] == 'Yes').mean()*100:.1f}%")
        c3.metric("Mean Glucose", f"{df_data['Glucose'].mean():.1f} mg/dL")
        c4.metric("Mean HbA1c", f"{df_data['HbA1c'].mean():.2f}%")

        st.markdown("### Sample Records")
        st.dataframe(df_data.head(10), use_container_width=True)

        cat_cols = ["Gender", "Smoking_Status", "Physical_Activity", "Family_History"]
        num_cols = ["Age", "BMI", "Glucose", "Blood_Pressure", "Cholesterol", "Insulin", "HbA1c"]
        ohe_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
        feat_names = num_cols + list(ohe_names)
        importances = pipeline.named_steps["classifier"].feature_importances_
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(10)

        st.markdown("### Top 10 Predictive Features")
        st.bar_chart(imp_df.set_index("Feature"))

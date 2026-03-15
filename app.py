import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr    = pickle.load(open('LogisticR.pkl',    'rb'))
    svm   = pickle.load(open('SVM.pkl',          'rb'))
    dtree = pickle.load(open('DecisionTree.pkl', 'rb'))
    rf    = pickle.load(open('RandomForest.pkl', 'rb'))
    return lr, svm, dtree, rf

lr, svm, dtree, rf = load_models()

model_map = {
    "Logistic Regression": lr,
    "SVM":                 svm,
    "Decision Tree":       dtree,
    "Random Forest":       rf
}

# ── Encoding helper ────────────────────────────────────────────────────────────
def encode(Age, Sex, ChestPainType, RestingBP, Cholesterol,
           FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    sex_enc    = 0 if Sex == "Male" else 1
    cp_enc     = {"ATA (Atypical Angina)": 0, "NAP (Non-Anginal Pain)": 1,
                  "ASY (Asymptomatic)": 2, "TA (Typical Angina)": 3}[ChestPainType]
    fbs_enc    = int(FastingBS)
    ecg_enc    = {"Normal": 0, "ST (ST-T wave abnormality)": 1,
                  "LVH (Left Ventricular Hypertrophy)": 2}[RestingECG]
    angina_enc = 1 if ExerciseAngina == "Yes" else 0
    slope_enc  = {"Up (Upsloping)": 0, "Flat": 1, "Down (Downsloping)": 2}[ST_Slope]
    return np.array([[Age, sex_enc, cp_enc, RestingBP, Cholesterol,
                      fbs_enc, ecg_enc, MaxHR, angina_enc, Oldpeak, slope_enc]])

# ── Page layout ────────────────────────────────────────────────────────────────
st.title("Heart Disease Predictor")

tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Patient Prediction")
    col1, col2 = st.columns(2)

    with col1:
        Age           = st.number_input("Age (years)", min_value=0, max_value=150, value=40)
        Sex           = st.selectbox("Sex", ["Male", "Female"],
                                     help="0: Male, 1: Female")
        ChestPainType = st.selectbox("Chest Pain Type",
                                     ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)",
                                      "ASY (Asymptomatic)", "TA (Typical Angina)"],
                                     help="TA=3, ATA=0, NAP=1, ASY=2")
        RestingBP     = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        Cholesterol   = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=700, value=200)
        FastingBS     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                     help="1: FastingBS > 120 mg/dl, 0: otherwise")

    with col2:
        RestingECG     = st.selectbox("Resting ECG",
                                      ["Normal", "ST (ST-T wave abnormality)",
                                       "LVH (Left Ventricular Hypertrophy)"],
                                      help="0: Normal, 1: ST, 2: LVH")
        MaxHR          = st.number_input("Max Heart Rate", min_value=60, max_value=202, value=150,
                                         help="Numeric value between 60 and 202")
        ExerciseAngina = st.selectbox("Exercise Angina", ["Yes", "No"],
                                      help="1: Yes, 0: No")
        Oldpeak        = st.number_input("Oldpeak (ST depression)", min_value=-5.0,
                                         max_value=10.0, value=0.0, step=0.1,
                                         help="Numeric value measured in depression")
        ST_Slope       = st.selectbox("ST Slope",
                                      ["Up (Upsloping)", "Flat", "Down (Downsloping)"],
                                      help="0: Upsloping, 1: Flat, 2: Downsloping")
        model_choice   = st.selectbox("Select Model", list(model_map.keys()))

    if st.button("Predict", use_container_width=True, type="primary"):
        data       = encode(Age, Sex, ChestPainType, RestingBP, Cholesterol,
                            FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)
        prediction = model_map[model_choice].predict(data)[0]
        st.divider()
        if prediction == 1:
            st.error("⚠️ High Risk: This patient is likely to have Heart Disease.")
        else:
            st.success("✅ Low Risk: This patient is unlikely to have Heart Disease.")
        st.caption(f"Model used: **{model_choice}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - Bulk Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bulk Prediction from CSV")
    st.markdown("Upload a CSV file with the same columns as the training data (without `HeartDisease`).")

    model_choice_bulk = st.selectbox("Select Model ", list(model_map.keys()))
    uploaded_file     = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview (original):**", df.head())

        if df['Sex'].dtype == object:
            df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
        if df['ChestPainType'].dtype == object:
            df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
        if df['RestingECG'].dtype == object:
            df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
        if df['ExerciseAngina'].dtype == object:
            df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
        if df['ST_Slope'].dtype == object:
            df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

        if 'HeartDisease' in df.columns:
            df = df.drop('HeartDisease', axis=1)

        if df.isnull().any().any():
            st.error("❌ Your CSV contains missing or unrecognized values in these columns:")
            st.write(df.isnull().sum()[df.isnull().sum() > 0])
            st.info("💡 Make sure all values match the expected encoding shown in the Model Information tab.")
        else:
            predictions      = model_map[model_choice_bulk].predict(df)
            result_df        = df.copy()
            result_df['Prediction'] = predictions
            result_df['Prediction'] = result_df['Prediction'].map(
                {1: '⚠️ Heart Disease', 0: '✅ No Heart Disease'})

            st.divider()
            st.subheader("Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Model Information
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Information")

    # ── Accuracy Bar Chart ─────────────────────────────────────────────────────
    models     = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    accuracies = [81.0, 82.6, 87.0, 82.6]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    bars = ax.bar(models, accuracies, color='#636EFA')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracies', color='white')
    ax.set_xlabel('Models', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc}%', ha='center', va='bottom', color='white', fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Feature Conventions ────────────────────────────────────────────────────
    st.markdown("**Instructions for using this app:**")
    st.markdown("""
1. No NaN values allowed.
2. Total 11 features in this order: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
3. Check the spellings of the feature names.
4. Feature values conventions:
    - **Age**: age of the patient [years]
    - **Sex**: sex of the patient [0: Male, 1: Female]
    - **ChestPainType**: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]
    - **RestingBP**: resting blood pressure [mm Hg]
    - **Cholesterol**: serum cholesterol [mm/dl]
    - **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
    - **RestingECG**: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]
    - **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
    - **ExerciseAngina**: exercise-induced angina [1: Yes, 0: No]
    - **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
    - **ST_Slope**: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
    """)

    st.divider()
    st.markdown("**Models used:**")
    st.markdown("""
| Model | Accuracy |
|---|---|
| **Logistic Regression** | 82.6% |
| **SVM** | 82.6% |
| **Decision Tree** | 81.0% |
| **Random Forest** | 87.0% |
    """)

    st.divider()
    st.caption("⚠️ This app is for educational purposes only. Always consult a medical professional.")

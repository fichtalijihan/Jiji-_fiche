import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Diabetes Prediction & Visualization")

tab1, tab2 = st.tabs(["Prediction", "Visualisation"])

with tab1:
    st.header("Enter Patient Data for Prediction")

    # Numeric inputs
    age = st.number_input("Age", 0, 120, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    waist = st.number_input("Waist Circumference (cm)", 0.0, 200.0, 80.0)
    fasting_glucose = st.number_input("Fasting Blood Glucose (mg/dL)", 0.0, 300.0, 90.0)
    hba1c = st.number_input("HbA1c (%)", 0.0, 15.0, 5.5)
    systolic = st.number_input("Blood Pressure Systolic (mmHg)", 0, 250, 120)
    diastolic = st.number_input("Blood Pressure Diastolic (mmHg)", 0, 150, 80)
    chol_total = st.number_input("Total Cholesterol (mg/dL)", 0.0, 400.0, 180.0)
    chol_hdl = st.number_input("HDL Cholesterol (mg/dL)", 0.0, 100.0, 50.0)
    chol_ldl = st.number_input("LDL Cholesterol (mg/dL)", 0.0, 300.0, 100.0)
    ggt = st.number_input("GGT (U/L)", 0.0, 200.0, 30.0)
    urate = st.number_input("Serum Urate (mg/dL)", 0.0, 12.0, 5.0)
    calories = st.number_input("Dietary Intake Calories", 500, 5000, 2000)
    family_history = st.selectbox("Family History of Diabetes", [0, 1])
    gestational = st.selectbox("Previous Gestational Diabetes", [0, 1])

    # Categorical inputs with dummy variables
    ethnicity = st.selectbox("Ethnicity", ["Black", "Hispanic", "White", "Other"])
    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
    smoking = st.selectbox("Smoking Status", ["Current", "Former", "Never"])

    # Convert categorical to dummy variables
    sex = 1 if sex == "Male" else 0

    # Ethnicity dummies
    ethnicity_black = 1 if ethnicity == "Black" else 0
    ethnicity_hispanic = 1 if ethnicity == "Hispanic" else 0
    ethnicity_white = 1 if ethnicity == "White" else 0
    # "Other" is reference => all zero

    # Physical Activity dummies
    activity_low = 1 if activity == "Low" else 0
    activity_moderate = 1 if activity == "Moderate" else 0
    # High is reference

    # Alcohol Consumption dummies
    alcohol_moderate = 1 if alcohol == "Moderate" else 0
    alcohol_heavy = 1 if alcohol == "Heavy" else 0
    # None is reference

    # Smoking Status dummies
    smoking_former = 1 if smoking == "Former" else 0
    smoking_never = 1 if smoking == "Never" else 0
    # Current is reference

    # Arrange input data in correct order expected by model
    input_data = np.array([[
        age,
        sex,
        bmi,
        waist,
        fasting_glucose,
        hba1c,
        systolic,
        diastolic,
        chol_total,
        chol_hdl,
        chol_ldl,
        ggt,
        urate,
        calories,
        family_history,
        gestational,
        ethnicity_black,
        ethnicity_hispanic,
        ethnicity_white,
        activity_low,
        activity_moderate,
        alcohol_moderate,
        alcohol_heavy,
        smoking_former,
        smoking_never
    ]])

    if st.button("Predict"):
        pred = model.predict(input_data)
        if pred[0] == 1:
            st.error("High risk of diabetes.")
        else:
            st.success("Low risk of diabetes.")

with tab2:
    st.header("Sample Data Visualization")

    dt = pd.read_csv('diabetes_dataset.csv')

    st.subheader("Histogram of Numerical Features")
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))  # 4x4 = 16 axes
    dt.hist(ax=axes)
    st.pyplot(fig)
    plt.clf()

    st.subheader("Distribution of Age")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(dt['Age'], kde=True, bins=20, color='skyblue', ax=ax2)
    ax2.set_title('Distribution de l\'âge des patients')
    ax2.set_xlabel('Âge')
    ax2.set_ylabel('Fréquence')
    st.pyplot(fig2)
    plt.clf()

    st.subheader("Patient Sex Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Sex', data=dt, palette='Set2', ax=ax3)
    ax3.set_title('Répartition par sexe')
    ax3.set_xlabel('Sexe')
    ax3.set_ylabel('Nombre de patients')
    st.pyplot(fig3)
    plt.clf()

    st.subheader("Alcohol Consumption Distribution")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Alcohol_Consumption', data=dt, palette='viridis', ax=ax4)
    ax4.set_title('Répartition de la consommation d\'alcool')
    ax4.set_xlabel('Consommation d\'alcool')
    ax4.set_ylabel('Nombre de patients')
    st.pyplot(fig4)
    plt.clf()

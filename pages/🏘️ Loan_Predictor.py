import streamlit as st
import pandas as pd
import joblib

st.markdown("""
<style>
/* Main background */
.stApp{
    background-color:#F5F7FA;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    border:none;
    border-radius:18px;
    background:#F8FBFF;
    background-color:#FAFCFF;
    padding:20px;
    margin-bottom:25px;
    box-shadow:0 4px 12px rgba(0,0,0,0.06)
}

/* Buttons */
.stButton>button{

    width:100%;
    background:2563EB;
    color:White;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold'
    padding:12px;
}

.stButton>button:hover{
    background:#1D4ED8;
}

/* Number Input */
.stNumberInput input{
    border-radius:10px;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"]{
    border-radius:10px;
}

/* Metric-Cards */
div[data-testid="metric-container"]{
    background:White;
    border-radius:15px;
    padding:18px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.10);
}

/* DataFrame */
div[data-testid="stDataFrame"]{
    border-radius:15px;
}

/* Progress bar */
.stProgress . div .div{
    background:#16A34A;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Loan Predictor",
    page_icon="🏘️",
    layout="wide"
)

st.title("🏘️ Creditwise Loan Approval Predictor")
st.caption(" AI-Powered Loan Eligibility Prediction System")

st.markdown("""
Enter the applicant's details below to predict wether the loan application is likly to be approved.
""")

pipeline = joblib.load(r"C:\Users\hp\projects\Creditwise_Dashboard\model\pipeline.pkl")
target_encoder = joblib.load(r"C:\Users\hp\projects\Creditwise_Dashboard\model\target_encoder.pkl")


with st.container(border=True):

    st.subheader("🤵🏻 Personal Information")
    col1 ,col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender",["Male","Female"])

        marital_status = st.selectbox("Married",["Yes","No"])

        education_level = st.selectbox("Education",["Graduate","Not Graduate"])

    with col2:

        age = st.number_input("Age",min_value=18,max_value=80,value=30)

        dependents = st.number_input("Dependents",min_value=0,max_value=10,value=0)

with st.container(border=True):

    st.subheader("💼 Employment Details")
    col1,col2 = st.columns(2)

    with col1:
        employment_status= st.selectbox("Employment Status",["Employed","Self-Employed","Unemployed"])

    with col2:
        employer_category=st.selectbox("Employer Category",["Private","Government","Business","Other"])

with st.container(border=True):

    st.subheader("💰 financial Details")
    col1,col2=st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0,value=5000.0)

        savings = st.number_input("Savings",min_value=0.0,value=10000.0)

        credit_score = st.number_input("Credit Score",min_value=300,max_value=900,value=700)

    with col2:
        coapplicant_income = st.number_input("Coapplicant Income",min_value=0.0,value=0.0)

        existing_loans = st.number_input("Existing Loans",min_value=0,value=0)

        dti_ratio = st.number_input("DTI Ratio",min_value=0.0,max_value=1.0,value=0.30)

with st.container(border=True):

    st.subheader("🏡 Loan Details")
    col1,col2 = st.columns(2)

    with col1:

        loan_amount = st.number_input("Loan Amount",min_value=0.0,value=150000.0)

        loan_term = st.number_input("Loan Term (Months)",min_value=1,value=360)

        collateral_value = st.number_input("Collateral Input",min_value=0.0,value=200000.0)

    with col2:

        loan_purpose = st.selectbox("Loan Purpose",["Home","Education","Business","Vehicle","Personal"])

        property_area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])

predict = st.button("🔍 Predict Loan Approval")

if predict:
    input_df = pd.DataFrame({
        "Applicant_Income" : [applicant_income],
        "Coapplicant_Income" : [coapplicant_income],
        "Employment_Status" : [employment_status],
        "Marital_Status" : [marital_status],
        "Age" : [age],
        "Dependents" : [dependents],
        "Credit_Score" : [credit_score],
        "Existing_Loans" : [existing_loans],
        "DTI_Ratio" : [dti_ratio],
        "Savings" : [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount" : [loan_amount],
        "Loan_Term" : [loan_term],
        "Loan_Purpose" : [loan_purpose],
        "Property_Area" : [property_area],
        "Education_Level" : [education_level],
        "Gender" : [gender],
        "Employer_Category" : [employer_category]
    })

    st.subheader("Input Data")
    st.dataframe(input_df)

    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"]**2
    input_df["Credit_Score_sq"] = input_df["Credit_Score"]**2

    input_df.drop(["Credit_Score","DTI_Ratio"],axis=1,inplace=True)

    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)

    result = target_encoder.inverse_transform(prediction)


    approval_Prob= probability[0][1]*100

    rejection_Prob= probability[0][0]*100

    with st.container(border=True):

        st.markdown("## Prediction Result")

        col1,col2 = st.columns(2)

        with col1:
            if prediction == 1 :
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

        with col2:
            st.metric("Approval Probability", f"{approval_Prob : .2f}%")

        col3 , col4 = st.columns(2)

        if approval_Prob >= 80:
            risk = "🟢 Low Risk"
        elif approval_Prob>=50:
            risk = "🟡 Medium Risk"
        else:
            risk = "🔴 High Risk"

        with col3:
            st.metric("Risk Level",risk)
        with col4:
            st.metric("Model","Logistic Regression")

    st.markdown("### 📊 Confidence Analysis")

    st.progress(int(approval_Prob))
    st.write(f"Approval Chance : **{approval_Prob:.2f}%**")

    
    st.markdown("### 💡 Recomendations")

    if prediction[0] == 1:
        st.success("🎉 congratulations! The applicant appears eligible for the requested Loan.")

    else:
        st.warning("Here are some suggestions to improve the chance of approval:")

        if credit_score < 650:
            st.write("✔️ Improve your Credit Score")
        if dti_ratio > 40:
            st.write("✔️ Reduce your DTI Ratio")
        if savings<50000:
            st.write("✔️ Increase your Savings")
        if existing_loans > 2:
            st.write("Reduce Existing Loans")
        if collateral_value< loan_amount:
            st.write("✔️ Increase Collateral Value")
        if applicant_income < 30000:
            st.write("✔️ Increase monthly income")

st.markdown("---")

st.markdown("""
<div style="
background:#1E293B;
padding:25px;
border-radius:15px;
color:White;
text-align:center;
margin-top:40px;
">

<h3> CreditWise Loan Approval Predictor</h3>

<p>
AI-Powered Loan Eligibility Predicton System
</p>

<p style="font-size:15px;color:#CBD5E1">
This prediction is generated using a Machine Learning Model.
Actual Loan approval depends on the bank's policies document Verification.
</p>

<hr style="border:1px solid #475569;">

<p style="font-size:14px;color:#94A3B8;">
Developed by <b>Roshni Jaiswal ❤️</b> using python, Scikit-learn and Streamlit
</p>
</div>
""",unsafe_allow_html=True)

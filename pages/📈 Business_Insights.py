import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Business Insights",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Business Insights Dashboard")

st.markdown("""
This page provides key bussiness insights from the loan approval dataset,
helping stackholders understand applicant trends and loan approval patterns.
""")

df = pd.read_csv("data/loan_approval_data.csv")

st.subheader("📌Key Metrics")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Total Applications",df.shape[0])

with col2:
    st.metric("Average Loan Amount",f"{df["Loan_Amount"].mean():.0f}")

with col3:
    st.metric("Average Applicant Income",f"{df["Applicant_Income"].mean():.0f}")

with col4:
    st.metric("Average Loan Term",f"{df["Loan_Term"].mean():.0f}")

st.subheader("📌 Loan Purpose Distribution")
fig = px.pie(
    df,
    names="Loan_Purpose",
    title="Loan Purpose"
)
st.plotly_chart(fig,use_container_width=True)

st.subheader("🏡 Property Area")
fig = px.bar(
    df,
    x="Property_Area",
    color="Loan_Approved"
)
st.plotly_chart(fig,use_container_width=True)

st.subheader("💼 Employer type")
fig=px.histogram(
    df,
    x="Employer_Category",
    color="Loan_Approved"
)
st.plotly_chart(fig,use_container_width=True)

st.subheader("💡 Key Business Observations")
st.info("""
- Identify the most common loan purpose.
- Compare approval rates across different property areas.
- Understand how employment category influence loan applications.
- Observe income and loan amount trends.
- Highlight any significant bussiness patterns during analysis.
""")

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

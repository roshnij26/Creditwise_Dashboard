import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA",page_icon="📊",layout="wide")

st.title("📊 Exploratory Data Analysis")

st.write("This page will contain the exploratory data analysis for the loan dataset.")

df = pd.read_csv("data/loan_approval_data.csv")

st.subheader("📌 Dashboard Overview")
col1,col2,col3,col4 =st.columns(4)

with col1:
    st.metric("📃 Total Applications",df.shape[0])

with col2:
    st.metric("📋 Total Features",df.shape[1])

with col3:
    st.metric("❓ Missing Values",df.isnull().sum().sum())

with col4:
    approval_rate = (df["Loan_Approved"]=="Yes").mean()*100
    st.metric("✅ Approval Rate",f"{approval_rate:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🤵🏻 Applicant Analysis",
    "💰 Loan Analysis",
    "📈 Business Insights"
])


with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows",df.shape[0])

    with col2:
        st.metric("Total Columns",df.shape[1])

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Loan Approval Distribution")

    fig = px.histogram(
        df,
        x="Loan_Approved",
        color = "Loan_Approved",
        text_auto=True,
        title= "Loan Approval Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Loan Approval Percentage")

    fig = px.pie(
        df,
        names="Loan_Approved",
        title="Loan Approval Percentage"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Applicant Income")

    fig = px.histogram(
        df,
        x="Applicant_Income",
        color="Loan_Approved"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Coapplicant Income")
    fig = px.histogram(
        df,
        x= "Coapplicant_Income",
        color="Loan_Approved"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Education Level")
    fig = px.histogram(
        df,
        x="Education_Level",
        color = "Loan_Approved",
        barmode="group"
        )
    st.plotly_chart(fig,use_container_width=True)


    st.subheader("Marital Status")
    fig = px.histogram(
        df,
        x="Marital_Status",
        color = "Loan_Approved",
        barmode="group"
        )
    st.plotly_chart(fig,use_container_width=True)


    st.subheader("Employment Status")
    fig = px.histogram(
        df,
        x="Employment_Status",
        color = "Loan_Approved"
        )
    st.plotly_chart(fig,use_container_width=True)


    st.subheader("Gender")
    fig = px.histogram(
        df,
        x="Gender",
        color = "Loan_Approved"
        )
    st.plotly_chart(fig,use_container_width=True)

with tab3:
    st.subheader("Loan Amount")

    fig = px.histogram(
        df,
        x="Loan_Amount",
        color="Loan_Approved",
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Loan Term")
    fig = px.histogram(
        df,
        x="Loan_Term",
        color = "Loan_Approved"
        )

    st.plotly_chart(fig,use_container_width=True)

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



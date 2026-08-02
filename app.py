import streamlit as st
st.set_page_config(
    page_title="CreditWise Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 CreditWise Loan Approval Dashboard")

st.markdown(""" 
### Welcome
This Dashboard helps banks analyze loan application and predict wether a loan should be approved.
### Features
- 📊 Data Analysis
- 📈 Business Insights
- 🏘️ Loan Prediction
- ⚡ Interactive Dashboard            
 """)

st.info("Use the sidebar to navigate between pages.")

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

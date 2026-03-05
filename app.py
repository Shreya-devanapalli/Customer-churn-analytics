import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import time
from streamlit_lottie import st_lottie
import requests



# PAGE CONFIG

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# CUSTOM CSS (Dashboard Styling)

st.markdown("""
<style>

.main {
    background-color:#f4f6f7;
}

h1 {
    color:#2C3E50;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.sidebar .sidebar-content {
    background: linear-gradient(#2C3E50,#4CA1AF);
}

</style>
""", unsafe_allow_html=True)


# LOAD DATA

model = joblib.load("models/churn_model.pkl")
df = pd.read_csv("data/telco_customer_churn.csv")


# HEADER
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

dashboard_animation = load_lottie(
    "https://assets5.lottiefiles.com/packages/lf20_49rdyysj.json"
)

ai_animation = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

# ---------------------------------
# HEADER WITH ANIMATION
# ---------------------------------
col1, col2 = st.columns([3,1])

with col1:
    st.title("📊 Customer Churn Analytics Dashboard")
    st.write(
        "A machine learning powered dashboard to predict and analyze telecom customer churn."
    )

with col2:
    st_lottie(dashboard_animation, height=150)


# SIDEBAR NAVIGATION

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Model Insights"]
)


# DASHBOARD

if page == "Dashboard":

    st.subheader("Business Overview")

    total_customers = len(df)
    churn_rate = (df["Churn"] == "Yes").mean()
    avg_charges = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Churn Rate", f"{churn_rate:.2%}")
    col3.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
    col4.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

    st.markdown("---")

    # INTERACTIVE CHARTS
   
    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            title="Monthly Charges Distribution",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.box(
            df,
            x="Churn",
            y="tenure",
            title="Tenure vs Churn",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    fig = px.pie(
        df,
        names="Churn",
        title="Customer Churn Breakdown",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


# PREDICTION

elif page == "Prediction":

    st.subheader("🔮 Customer Churn Prediction")
    st_lottie(ai_animation, height=200)

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider("Tenure (months)", 0, 72, 12)

        monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)

        total_charges = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            1000.0
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    
    # CREATE INPUT DATA
    
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_data["InternetService_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
    input_data["Contract_Two year"] = 1 if contract == "Two year" else 0
    input_data["Contract_One year"] = 1 if contract == "One year" else 0
    input_data["OnlineSecurity_Yes"] = 1 if online_security == "Yes" else 0
    input_data["TechSupport_Yes"] = 1 if tech_support == "Yes" else 0
    input_data["PaymentMethod_Electronic check"] = 1 if payment_method == "Electronic check" else 0

    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model.feature_names_in_]

    # PREDICTION
  
    if st.button("Predict Churn"):

        with st.spinner("Analyzing customer behavior..."):
            time.sleep(1.5)

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High churn risk ({probability:.2%})")
        else:
            st.success(f"✅ Customer likely to stay ({1-probability:.2%})")

        st.progress(float(probability))



# MODEL INSIGHTS

elif page == "Model Insights":

    st.subheader("🧠 Model Insights")

    st.markdown("### Feature Importance")

    importances = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    )

    fig, ax = plt.subplots()

    importances.sort_values().tail(10).plot(
        kind="barh",
        ax=ax,
        color="#4CA1AF"
    )

    ax.set_title("Top Churn Drivers")

    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### SHAP Explainability")

    X_sample = df.sample(200)

    X_sample = pd.get_dummies(X_sample, drop_first=True)

    for col in model.feature_names_in_:
        if col not in X_sample.columns:
            X_sample[col] = 0

    X_sample = X_sample[model.feature_names_in_]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    shap.plots.bar(shap_values[:,:,1], show=False)

    st.pyplot(plt.gcf())


# FOOTER

st.markdown("---")
st.caption("Customer Churn Analytics Dashboard")
st.caption("Built with Streamlit | Author: Shreya Devanapalli")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle

from ai_explainer import explain_with_ai  

st.set_page_config(page_title="AI Financial Assistant", layout="wide")

st.title("💼 AI Customer Churn Analyst")

model = pickle.load(open("model.pkl", "rb"))

tab1, tab2 = st.tabs(["🎯 Prediction & AI", "📊 Dashboard"])

with tab1:

    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    if "customer_data" not in st.session_state:
        st.session_state.customer_data = None

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80)
        tenure = st.slider("Tenure", 0, 10)
        credit_score = st.number_input("Credit Score", value=500)

    with col2:
        balance = st.number_input("Balance", value=10000)
        products = st.selectbox("Products", [1, 2, 3, 4])
        active = st.selectbox("Active Member", [0, 1])

    if st.button("🔍 Predict Churn"):

        data = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1,
            "IsActiveMember": active,
            "EstimatedSalary": 50000,
            "Geography_Germany": 1,
            "Geography_Spain": 0,
            "Gender_Male": 1
        }

        st.session_state.customer_data = data

        feature_order = list(data.keys())
        df = pd.DataFrame([data])[feature_order]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        explanation = explain_with_ai(data, prediction)

        st.session_state.prediction_done = True
        st.session_state.result = {
            "prediction": prediction,
            "probability": probability,
            "explanation": explanation
        }

    if st.session_state.prediction_done:

        result = st.session_state.result

        st.subheader("📊 Prediction Result")

        if result["prediction"] == 1:
            st.error("⚠ HIGH CHURN RISK")
        else:
            st.success("✅ LOW CHURN RISK")

        prob = result["probability"] * 100
        st.progress(int(prob))
        st.write(f"**Churn Probability:** {prob:.2f}%")

        st.subheader("🧠 AI Explanation")
        st.info(result["explanation"])

        st.subheader("🔍 SHAP Explanation")

        df = pd.DataFrame([st.session_state.customer_data])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df)

        shap.plots.waterfall(shap_values[0, :, 1], show=False)

        fig = plt.gcf()
        fig.set_size_inches(6, 3)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("💬 Ask AI about this customer")

        user_question = st.text_input("Ask a question:")

        if user_question:
            chat_response = explain_with_ai(
                st.session_state.customer_data,
                result["prediction"],
                user_question
            )
            st.write("🤖", chat_response)

with tab2:

    st.subheader("📊 Customer Churn Dashboard")

    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    total = len(df)
    churned = df["Exited"].sum()
    churn_rate = (churned / total) * 100
    avg_balance = df["Balance"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Customers", total)
    col2.metric("❌ Churned", churned)
    col3.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("💰 Avg Balance", f"{avg_balance:.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Churn Distribution")
        fig, ax = plt.subplots(figsize=(4,3))
        df["Exited"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"], rotation=0)
        st.pyplot(fig)

    with col2:
        st.markdown("### 👤 Age Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.hist(df["Age"], bins=25)
        st.pyplot(fig2)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Balance vs Churn")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.boxplot(
            [df[df["Exited"]==0]["Balance"],
             df[df["Exited"]==1]["Balance"]],
            labels=["Stayed", "Churned"]
        )
        st.pyplot(fig3)

    with col2:
        st.markdown("### 🛍 Products vs Churn")
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        df.groupby("NumOfProducts")["Exited"].mean().plot(kind="bar", ax=ax4)
        st.pyplot(fig4)

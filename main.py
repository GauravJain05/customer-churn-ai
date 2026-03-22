import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import matplotlib

from ai_explainer import explain_with_ai  

matplotlib.rcParams['text.usetex'] = False

st.set_page_config(page_title="AI Customer Churn Analyst", layout="wide")

st.title("💼 AI Customer Churn Analyst")

model = pickle.load(open("model.pkl", "rb"))

tab1, tab2 = st.tabs(["🎯 Prediction & AI", "📊 Dashboard"])

with tab1:

    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    if "customer_data" not in st.session_state:
        st.session_state.customer_data = None

    if "shap_values" not in st.session_state:
        st.session_state.shap_values = None

    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80)
        tenure = st.slider("Tenure", 0, 10)
        credit_score = st.number_input("Credit Score", value=500)
        salary = st.number_input("Estimated Salary", value=50000)
        cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])

    with col2:
        balance = st.number_input("Balance", value=10000)
        products = st.selectbox("Products", [1, 2, 3, 4])
        active = st.selectbox("Active Member", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"])

    if st.button("🔍 Predict Churn"):

        data = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if cr_card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0,
            "EstimatedSalary": salary,
            "Geography_Germany": 1 if geo == "Germany" else 0,
            "Geography_Spain": 1 if geo == "Spain" else 0,
            "Gender_Male": 1 if gender == "Male" else 0
        }

        st.session_state.customer_data = data

        st.session_state.chat_response = ""
        st.session_state.last_question = ""

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df)

        st.session_state.shap_values = shap_values  

        import numpy as np

        values = shap_values.values[0][:, 1]
        features = df.columns

        importance = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
        top_features = importance[:3]

        shap_text = "\n".join([
            f"{f}: impact={'positive' if v > 0 else 'negative'}, strength={abs(v):.3f}"
            for f, v in top_features
        ])

        explanation = explain_with_ai(data, prediction, shap_text)

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

        # shap_values = st.session_state.shap_values

        # import matplotlib.pyplot as plt

        # plt.clf()  

        # shap.plots.bar(shap_values[0, :, 1], show=False)

        # fig = plt.gcf()
        # fig.set_size_inches(8.8, 3)   

        # plt.tight_layout()

        # st.pyplot(fig)

        # plt.close(fig)
        shap_container = st.container()

        with shap_container:
            st.subheader("🔍 SHAP Explanation")

            if st.session_state.shap_values is not None:
                shap_values = st.session_state.shap_values

                import matplotlib.pyplot as plt

                plt.clf()  

                shap.plots.bar(shap_values[0, :, 1], show=False)

                fig = plt.gcf()
                fig.set_size_inches(8.8, 3)   

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

        st.divider()
        st.subheader("💬 Ask AI about this customer")

        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., Why is this customer likely to churn?"
        )

        if st.button("Ask AI"):

            if user_question:

                st.session_state.last_question = user_question

                response = explain_with_ai(
                    st.session_state.customer_data,
                    result["prediction"],
                    question=user_question
                )

                st.session_state.chat_response = response

        if st.session_state.last_question:
            st.markdown(f"**🧑 {st.session_state.last_question}**")
            st.markdown(f"**🤖 {st.session_state.chat_response}**")


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
        fig, ax = plt.subplots(figsize=(4, 3))
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
            [df[df["Exited"] == 0]["Balance"],
            df[df["Exited"] == 1]["Balance"]],
            labels=["Stayed", "Churned"],
            widths=0.5   
        )

        ax3.set_ylim(0, df["Balance"].max()) 
        plt.tight_layout()                   

        st.pyplot(fig3)
        plt.close(fig3)


    with col2:
        st.markdown("### 🛍 Products vs Churn")
        fig4, ax4 = plt.subplots(figsize=(4, 3))

        df.groupby("NumOfProducts")["Exited"].mean().plot(
            kind="bar",
            ax=ax4,
            width=0.5   
        )

        ax4.set_ylim(0, 1)                  
        plt.tight_layout()                 
        st.pyplot(fig4)
        plt.close(fig4)

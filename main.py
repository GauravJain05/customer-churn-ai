import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap
import pickle
import matplotlib

from ai_explainer import explain_with_ai

matplotlib.rcParams['text.usetex'] = False

st.set_page_config(page_title="AI Customer Churn Analyst", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f5f6fa !important;
    color: #111827 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1300px;
}
[data-testid="stAppViewContainer"] > div {
    background-color: #f5f6fa !important;
}
.card {
    background: #ffffff !important;
}
.heading-card {
    background: #ffffff !important;
}
.main-title-card {
    background: #ffffff !important;
}
.page-heading {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    line-height: 1.2;
}
.page-subheading {
    font-size: 15px;
    color: #6b7280;
    font-weight: 400;
    margin-bottom: 28px;
    display:inline-block;
    text-align: center;
    text-align: center;
}
.page-heading1 {
    font-size: 22px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    line-height: 1.2;
    
}
.heading-card {
    background: #f5f6fa;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px 28px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
    font-size: 17px;
    font-weight: 700;
    color: #111827;
    display: inline-block;
    width: 100%;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500;
    font-size: 20px;
    border-radius: 7px;
    padding: 9px 22px;
    color: #374151 !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.card {
    background: #f5f6fa;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.sec-label {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: none;
    color: #111827;
    margin-top: 0;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e7eb;
    width: 100%;
    text-align: left;
    display: block;
}

.field-group-label {
    font-size: 13px;
    font-weight: 600;
    color: #111827;
    margin: 18px 0 10px;
    letter-spacing: 0;
}

div[data-testid="stSlider"] > div > div > div {
    background: #2563eb !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: #2563eb !important;
    border-color: #2563eb !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #111827 !important;
    letter-spacing: 0 !important;
}

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    border: none !important;
    background: #2563eb !important;
    color: #ffffff !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.3) !important;
}

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    color: #111827;
}
.risk-badge.high   { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.risk-badge.medium { background:#fffbeb; color:#d97706; border:2px solid #d97706; }
.risk-badge.low    { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }

.ai-box {
    background: #eef0f5;
    border: 1px solid #dbeafe;
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: 20px 22px;
    font-size: 15px;
    line-height: 1.8;
    color: #111827;
    white-space: pre-wrap;
    text-align: left;
}

.chat-q {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px 10px 10px 3px;
    padding: 12px 18px;
    font-size: 15px;
    color: #1e40af;
    margin-bottom: 12px;
    display: block;
    font-weight: 500;
    width: 100%;
    text-align: left;
}
.chat-a {
    background: #eef0f5;
    border: 1px solid #e5e7eb;
    border-radius: 10px 10px 3px 10px;
    padding: 14px 18px;
    font-size: 15px;
    color: #111827;
    line-height: 1.8;
    white-space: pre-wrap;
    width: 100%;
    display: block;
    text-align: left;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 480px;
    background: #f5f6fa;
    border: 1.5px dashed #d1d5db;
    border-radius: 12px;
    color: #9ca3af;
    text-align: center;
    gap: 10px;
}

.dash-kpi {
    background: #f5f6fa;
    border: 1.5px solid #111827;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-val {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.kpi-lbl {
    font-size: 14px;
    color: #374151;
    margin-top: 5px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
}
.kpi-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.main-title-card {
    background: #f5f6fa;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #111827;
    width: 100%;
}

hr { border-color: #f3f4f6 !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title-card page-heading">💼 AI Customer Churn Analyst</div>', unsafe_allow_html=True)
st.markdown('<center>Enter customer details to predict churn risk and get an AI-powered explanation.</center>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

def gauge_chart(prob):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#f5f6fa')
    ax.set_facecolor('#f5f6fa')

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#e5e7eb', linewidth=18, solid_capstyle='round')

    fill_theta = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    color = '#dc2626' if prob >= 0.7 else '#d97706' if prob >= 0.4 else '#16a34a'
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=18, solid_capstyle='round')

    ax.text(0, -0.15, f"{prob*100:.1f}%",
            ha='center', va='center', fontsize=32, fontweight='800',
            color=color, fontfamily='DejaVu Sans')
    ax.text(0, -0.50, "CHURN PROBABILITY",
            ha='center', va='center', fontsize=8.5, color='#374151',
            fontfamily='DejaVu Sans', fontweight='600')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.68, 1.15)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def risk_tier(prob):
    if prob >= 0.7:   return "High Risk",   "high"
    elif prob >= 0.4: return "Medium Risk", "medium"
    else:             return "Low Risk",    "low"

def risk_dot(cls):
    return {"high": "🔴", "medium": "🟡", "low": "🟢"}[cls]

for key, val in {
    "prediction_done": False,
    "customer_data":   None,
    "shap_values":     None,
    "chat_response":   "",
    "last_question":   "",
    "result":          None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


tab1, tab2 = st.tabs(["🎯  Prediction & AI", "📊  Dashboard"])


with tab1:

    inp_col, gauge_col = st.columns([3, 2], gap="large")

    with inp_col:
        st.markdown('<div class="heading-card">Customer Details</div>', unsafe_allow_html=True)
        

        st.markdown('<div class="field-group-label">Customer Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age          = st.slider("Age", 18, 80, 40)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=500, step=10)
        with c2:
            tenure  = st.slider("Tenure (years)", 0, 10, 3)
            balance = st.number_input("Balance ($)", min_value=0, max_value=250000, value=10000, step=1000)
        with c3:
            salary   = st.number_input("Estimated Salary ($)", min_value=10000, max_value=200000, value=50000, step=1000)
            products = st.selectbox("Number of Products", [1, 2, 3, 4])

        st.markdown('<div class="field-group-label">Account Flags</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1: active  = st.selectbox("Active Member",   ["Yes", "No"])
        with f2: cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        with f3: gender  = st.selectbox("Gender",           ["Male", "Female"])
        with f4: geo     = st.selectbox("Geography",        ["France", "Germany", "Spain"])

        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 1, 1])
        with btn_col:
            predict_btn = st.button("Run Churn Analysis →")
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col:
        if st.session_state.prediction_done:
            result = st.session_state.result
            prob   = result["probability"]
            tier_label, tier_cls = risk_tier(prob)

           
            st.markdown('<div class="heading-card">Risk Assessment</div>', unsafe_allow_html=True)
            gauge_fig = gauge_chart(prob)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)
            st.markdown(
                f'<div style="text-align:center;margin-top:4px;margin-bottom:6px">'
                f'<span class="risk-badge {tier_cls}">{risk_dot(tier_cls)}&nbsp; {tier_label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:36px">📊</div>
                <div style="font-size:16px;font-weight:600;color:#374151">No prediction yet</div>
                <div style="font-size:14px;color:#6b7280;max-width:220px;line-height:1.6">
                    Fill in the customer details and click<br><strong>Run Churn Analysis</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if predict_btn:
        data = {
            "CreditScore":       credit_score,
            "Age":               age,
            "Tenure":            tenure,
            "Balance":           balance,
            "NumOfProducts":     products,
            "HasCrCard":         1 if cr_card == "Yes" else 0,
            "IsActiveMember":    1 if active  == "Yes" else 0,
            "EstimatedSalary":   salary,
            "Geography_Germany": 1 if geo == "Germany" else 0,
            "Geography_Spain":   1 if geo == "Spain"   else 0,
            "Gender_Male":       1 if gender == "Male"  else 0,
        }
        st.session_state.customer_data = data
        st.session_state.chat_response = ""
        st.session_state.last_question = ""

        df_pred     = pd.DataFrame([data])
        prediction  = model.predict(df_pred)[0]
        probability = model.predict_proba(df_pred)[0][1]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(df_pred)
        st.session_state.shap_values = shap_values

        vals       = shap_values.values[0][:, 1]
        importance = sorted(zip(df_pred.columns, vals), key=lambda x: abs(x[1]), reverse=True)
        shap_text  = "\n".join([
            f"{f}: {'INCREASES churn risk' if v > 0 else 'DECREASES churn risk'} (magnitude={abs(v):.3f})"
            for f, v in importance[:3]
        ])

        explanation = explain_with_ai(data, prediction, shap_text)
        st.session_state.prediction_done = True
        st.session_state.result = {
            "prediction":  prediction,
            "probability": probability,
            "explanation": explanation,
        }
        st.rerun()

    if st.session_state.prediction_done:
        result = st.session_state.result
        prob   = result["probability"]

        _, center, _ = st.columns([1, 4, 1])
        with center:

            
            st.markdown('<div class="heading-card">AI Analyst Explanation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{result["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="heading-card">Feature Impact · SHAP Values</div>', unsafe_allow_html=True)
            

            if st.session_state.shap_values is not None:
                sv         = st.session_state.shap_values
                vals_plot  = sv.values[0][:, 1]
                feat_names = list(pd.DataFrame([st.session_state.customer_data]).columns)
                sorted_idx   = np.argsort(np.abs(vals_plot))[-8:]
                vals_sorted  = vals_plot[sorted_idx]
                names_sorted = [feat_names[i] for i in sorted_idx]
                bar_colors   = ['#dc2626' if v > 0 else '#2563eb' for v in vals_sorted]

                fig_s, ax_s = plt.subplots(figsize=(8, 3.2))
                fig_s.patch.set_facecolor('#f5f6fa')
                ax_s.set_facecolor('#f0f1f5')
                ax_s.barh(names_sorted, vals_sorted, color=bar_colors, height=0.52)
                ax_s.axvline(0, color='#d1d5db', linewidth=1)
                ax_s.tick_params(colors='#111827', labelsize=11)
                for spine in ax_s.spines.values():
                    spine.set_edgecolor('#e5e7eb')
                ax_s.set_xlabel("SHAP value", color='#111827', fontsize=10)
                ax_s.grid(axis='x', color='#e5e7eb', linewidth=0.7, linestyle='--')
                ax_s.set_axisbelow(True)

                red_p  = mpatches.Patch(color='#dc2626', label='Increases churn risk')
                blue_p = mpatches.Patch(color='#2563eb', label='Decreases churn risk')
                ax_s.legend(handles=[red_p, blue_p], fontsize=10,
                            facecolor='#f5f6fa', edgecolor='#e5e7eb',
                            labelcolor='#111827', loc='lower right')
                plt.tight_layout(pad=0.6)
                st.pyplot(fig_s, use_container_width=True)
                plt.close(fig_s)

            st.markdown('</div>', unsafe_allow_html=True)

           
            st.markdown('<div class="heading-card">Ask a Follow-up Question</div>', unsafe_allow_html=True)
            
            user_question = st.text_input(
                "question",
                placeholder="e.g. What is the biggest risk factor for this customer?",
                label_visibility="collapsed"
            )
            if st.button("Ask AI"):
                if user_question.strip():
                    st.session_state.last_question = user_question
                    st.session_state.chat_response = explain_with_ai(
                        st.session_state.customer_data,
                        result["prediction"],
                        question=user_question
                    )
                    st.rerun()

            if st.session_state.last_question:
                st.markdown(f'<div class="chat-q">🧑&nbsp; {st.session_state.last_question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-a">🤖&nbsp; {st.session_state.chat_response}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)



with tab2:

    st.markdown('<div class="page-heading1">📊 Customer Churn Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Overview of churn trends across the customer base.</div>', unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("Churn_Modelling.csv")
        return df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    df = load_data()

    total       = len(df)
    churned     = int(df["Exited"].sum())
    churn_rate  = churned / total * 100
    avg_balance = df["Balance"].mean()

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, f"{total:,}",           "Total Customers", "#2563eb"),
        (k2, f"{churned:,}",         "Churned",         "#dc2626"),
        (k3, f"{churn_rate:.1f}%",   "Churn Rate",      "#d97706"),
        (k4, f"${avg_balance:,.0f}", "Avg Balance",     "#16a34a"),
    ]
    for col, val, lbl, dot_color in kpis:
        col.markdown(f"""
        <div class="dash-kpi">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">
                <span class="kpi-dot" style="background:{dot_color}"></span>{lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def style_chart(fig, ax):
        fig.patch.set_facecolor('#f5f6fa')
        ax.set_facecolor('#f0f1f5')
        ax.tick_params(colors='#111827', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#e5e7eb')
        ax.grid(axis='y', color='#e5e7eb', linewidth=0.8, linestyle='--')
        ax.set_axisbelow(True)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), color='#111827', fontsize=11)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), color='#111827', fontsize=11)

    col1, col2 = st.columns(2)

    with col1:
       
        st.markdown('<div class="heading-card">Churn Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["Exited"].value_counts()
        ax.bar(["Stayed", "Churned"], [counts[0], counts[1]],
               color=['#2563eb', '#dc2626'], width=0.45)
        for i, v in enumerate([counts[0], counts[1]]):
            ax.text(i, v + 30, f"{v:,}", ha='center', color='#111827', fontsize=11, fontweight='600')
        ax.set_ylabel("Customers")
        style_chart(fig, ax)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
       
        st.markdown('<div class="heading-card">Age Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(df["Age"], bins=25, color='#2563eb', alpha=0.85)
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Count")
        style_chart(fig2, ax2)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
       
        st.markdown('<div class="heading-card">Balance vs Churn</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.boxplot(
            [df[df["Exited"]==0]["Balance"], df[df["Exited"]==1]["Balance"]],
            labels=["Stayed", "Churned"],
            patch_artist=True,
            widths=0.45,
            boxprops=dict(facecolor='#eff6ff', color='#2563eb'),
            medianprops=dict(color='#2563eb', linewidth=2),
            whiskerprops=dict(color='#93c5fd'),
            capprops=dict(color='#93c5fd'),
            flierprops=dict(marker='o', color='#93c5fd', alpha=0.4, markersize=3)
        )
        ax3.set_ylim(0, df["Balance"].max())
        ax3.set_ylabel("Balance ($)")
        style_chart(fig3, ax3)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        
        st.markdown('<div class="heading-card">Products vs Churn Rate</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        prod_data = df.groupby("NumOfProducts")["Exited"].mean()
        ax4.bar(prod_data.index.astype(str), prod_data.values,
                color='#2563eb', width=0.45, alpha=0.85)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel("Number of Products")
        ax4.set_ylabel("Churn Rate")
        style_chart(fig4, ax4)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_CONTEXT = """
Feature ranges and CORRECT churn insights from this specific banking dataset:

- CreditScore: 300 to 850. Lower credit scores slightly increase churn risk.
- Age: 18 to 80. Age is one of the STRONGEST churn predictors in this dataset.
  Customers aged 40+ churn significantly MORE. Customers aged 60+ are at the HIGHEST risk.
  NEVER describe old age as a protective or positive factor — it ALWAYS increases churn risk.
  Young customers (under 40) have lower churn risk.
- Balance: 0 to 250000. Customers with ZERO balance churn MORE.
  Very high balance (above 100,000) also increases churn risk slightly.
  Mid-range balance is the safest zone.
- NumOfProducts: 1 to 4. This is a critical factor.
  1 or 2 products = LOWER churn risk.
  3 or 4 products = VERY HIGH churn rate (~83%). Always flag 3-4 products as a strong risk signal.
- EstimatedSalary: 10000 to 200000. Salary has very low impact on churn. Do not over-emphasise it.
- Tenure: 0 to 10 years. Very low tenure (0-1 years) = higher churn risk.
  Longer tenure = more loyalty, lower churn risk.
- IsActiveMember: 1 = Active, 0 = Inactive.
  Inactive members churn significantly MORE than active members.
  Active membership is a strong protective factor.
- Geography: Germany has the highest churn rate (~32%).
  France (~16%) and Spain (~16%) are much lower.
  Being from Germany increases churn risk.
- Gender: Female customers churn slightly more than male customers. Minor factor.
- HasCrCard: Very low impact on churn. Do not use this as a key reason.
"""


def explain_with_ai(data, prediction, shap_info="", question=""):

    risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

    if question:
        prompt = f"""
You are a financial analyst. A customer churn prediction has already been explained to the user.
The user is now asking a follow-up question. Answer it directly and concisely.

Customer data:
{data}

Prediction: {risk}

Key factors from model:
{shap_info}

Dataset-specific rules you MUST follow when answering:
- Age above 40 ALWAYS increases churn risk. Never say older age is a good or protective sign.
- NumOfProducts 3 or 4 = VERY HIGH churn risk (~83%). 1 or 2 = lower risk.
- Inactive members churn more. Active membership reduces risk.
- Zero balance increases churn risk.
- Germany geography increases churn risk.
- Salary has very low impact — do not over-emphasise it.

ANSWER RULES:
- Answer ONLY the question asked — do NOT repeat the full prediction explanation
- Do NOT restate the risk level unless directly asked
- Keep the answer to 2-4 sentences maximum
- Be direct, clear, and human-friendly
- Base your answer strictly on the data and dataset rules above

User question: {question}
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI explanation error: {str(e)}"

    prompt = f"""
You are a financial analyst explaining customer churn for a bank.

{DATA_CONTEXT}

Customer data:
{data}

Prediction: {risk}

Key factors identified by the model (SHAP values):
{shap_info}

STRICT RULES — you MUST follow all of these:
- Use ONLY the customer data and SHAP factors provided
- Do NOT guess or assume anything not in the data
- Do NOT use technical terms like "impact=positive", "strength", or "SHAP"
- Do NOT repeat similar phrases across bullet points
- Age above 40 ALWAYS increases churn risk — NEVER describe old age as a protective or positive factor
- NumOfProducts 3 or 4 is ALWAYS a high risk signal — never say more products means more loyalty
- If a factor reduces risk, explain clearly why it is protective
- If a factor increases risk, explain clearly why it is a warning sign
- Do NOT contradict the prediction — if prediction is LOW RISK, the conclusion must be low risk overall
- Each bullet point must correctly reflect whether the factor is helping or hurting churn risk

TASK:
1. Start with a one-line summary that matches the prediction (HIGH or LOW risk)
2. Give 2–3 key reasons using bullet points based on the SHAP factors
3. For each reason:
   - Name the factor clearly
   - State whether it increases or reduces churn risk for this customer
   - Explain in 1-2 simple sentences why it matters
4. End with a short one-line overall conclusion

STYLE:
- Simple, clear, human-friendly language
- Feels like a real bank analyst wrote it
- Not too short, not too long

FORMAT:
This customer is at [high/low/medium] risk of churn mainly due to:

• Factor 1: explanation

• Factor 2: explanation

• Factor 3: explanation

Overall conclusion in one line.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI explanation error: {str(e)}"

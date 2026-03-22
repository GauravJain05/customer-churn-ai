import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_CONTEXT = """
Feature ranges in dataset:

- CreditScore: 300 to 850
- Age: 18 to 80
- Balance: 0 to 250000
- NumOfProducts: 1 to 4
- EstimatedSalary: 10000 to 200000
- Tenure: 0 to 10
"""


def explain_with_ai(data, prediction, shap_info="", question=""):

    risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

    prompt = f"""
You are a financial analyst explaining customer churn.

{DATA_CONTEXT}

Customer data:
{data}

Prediction: {risk}

Key factors from model:
{shap_info}

IMPORTANT RULES:
- Use ONLY given data and factors
- Do NOT guess or assume anything
- Do NOT use technical terms like "impact=positive" or "strength"
- Do NOT repeat similar phrases
- Do NOT over-explain dataset ranges
- Each bullet point should have 1–2 lines explaining "why it matters"

TASK:
1. Start with a one-line summary of risk
2. Give 2–3 key reasons using bullet points
3. For each reason:
   - Explain the factor
   - Explain why it affects churn
   - Take a new line 
4. Add a short final conclusion

STYLE:
- Keep it simple and human-friendly
- Add a little detail (not too short, not too long)
- Make it feel like a real analyst explanation

FORMAT EXAMPLE:
This customer is at high risk of churn mainly due to:

• Reason 1  
• Reason 2  
• Reason 3  

Overall conclusion in one line.
"""

    if question:
        prompt += f"\n\nUser question: {question}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI explanation error: {str(e)}"

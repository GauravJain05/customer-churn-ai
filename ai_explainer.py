import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_with_ai(data, prediction, question=""):
    print("🔥 USING GROQ AI")

    risk = "HIGH RISK (Churn)" if prediction == 1 else "LOW RISK (No Churn)"

    if question:
        prompt = f"""
        You are a financial analyst chatbot.

        Customer data:
        {data}

        Prediction: {risk}

        User question: {question}

        Answer the question clearly and professionally.
        Keep it short (3-4 lines max).
        """
    else:
        prompt = f"""
        You are a financial analyst.

        Customer data:
        {data}

        Prediction: {risk}

        Explain in simple and professional language:
        - Why this customer may churn or not churn
        - Give 2-3 clear reasons
        """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
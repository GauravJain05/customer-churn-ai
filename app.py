from flask import Flask, request, jsonify
import pickle
import pandas as pd

from ai_explainer import explain_with_ai
from utils import explain_churn

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return "AI Financial Analyst Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        question = data.get("question", "")

        model_data = {k: v for k, v in data.items() if k != "question"}

        df = pd.DataFrame([model_data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        try:
            explanation = explain_with_ai(model_data, prediction, question)
        except Exception as e:
            print("❌ GROQ ERROR:", e)
            explanation = explain_churn(model_data, prediction)

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "explanation": explanation
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

def explain_churn(data, prediction):
    reasons = []

    if data['IsActiveMember'] == 0:
        reasons.append("Customer is not active")

    if data['Tenure'] < 3:
        reasons.append("Customer has low tenure")

    if data['NumOfProducts'] == 1:
        reasons.append("Customer uses only one product")

    if data['Balance'] < 50000:
        reasons.append("Customer has low balance")

    if prediction == 1:
        return "High churn risk because: " + ", ".join(reasons)
    else:
        return "Low churn risk. Customer is stable."
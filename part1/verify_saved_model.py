import joblib

model = joblib.load("models/return_risk_model.pkl")

print("Model type:", type(model))
print("Pipeline steps:", model.named_steps)
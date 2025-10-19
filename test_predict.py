import joblib
import pandas as pd

# Load the pipeline and label encoder
clf = joblib.load("fitness_pipeline.pkl")
label_enc = joblib.load("label_encoder.pkl")

# Example user input
goal = "lose face fat"
level = "beginner"

# Put into DataFrame
user_df = pd.DataFrame([[goal, level]], columns=["Goal", "Fitness Level"])

# Predict
pred = clf.predict(user_df)[0]
recommendation = label_enc.inverse_transform([pred])[0]

print(f"Recommendation: {recommendation}")

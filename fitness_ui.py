#inputs needed for project
import joblib
import pandas as pd

clf = joblib.load("fitness_pipeline.pkl")
label_enc = joblib.load("label_encoder.pkl")

print("Welcome to the Fitness Recommendation System!\n")

while True:
    goal = input("Enter your fitness goal (or 'quit' to exit): ").strip().lower()
    if goal == "quit":
        break
    level = input("Enter your fitness level (beginner/intermediate/advanced): ").strip().lower()

    user_df = pd.DataFrame([[goal, level]], columns=["Goal", "Fitness Level"])

    #predictions
    pred_enc = clf.predict(user_df)[0]
    recommendation = label_enc.inverse_transform([pred_enc])[0]

    print(f"\nRecommendation: {recommendation}\n")

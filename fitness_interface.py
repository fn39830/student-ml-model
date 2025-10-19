import pandas as pd
import joblib


# 1. Load the trained model and label encoder
clf = joblib.load('train_model.py')   # update with your actual model filename
label_enc = joblib.load("label_encoder.pkl")  # update if you saved separately

# 2. Take user inputs
goal = input("Enter your goal (e.g., lose face fat, lose weight, build muscle, maintain): ").strip().lower()
fitness_level = input("Enter your fitness level (beginner, intermediate, advanced): ").strip().lower()

# 3. Put input into DataFrame
user_df = pd.DataFrame([[goal, fitness_level]], columns=["Goal", "Fitness Level"])

# 4. Make prediction
predicted = clf.predict(user_df)[0]

# 5. Convert prediction back to text
recommendation = label_enc.inverse_transform([predicted])[0]

print(f"\nRecommended Plan: {recommendation}")

import pandas as pd
import joblib


#load the trained model and label encoder to assign a unique numerical value to each different category in the dataset
clf = joblib.load('train_model.py')   
label_enc = joblib.load("label_encoder.pkl")  

#takes user inputs
goal = input("Enter your goal (e.g., lose face fat, lose weight, build muscle, maintain): ").strip().lower()
fitness_level = input("Enter your fitness level (beginner, intermediate, advanced): ").strip().lower()

#puts data into dataframe
user_df = pd.DataFrame([[goal, fitness_level]], columns=["Goal", "Fitness Level"])

#makes prediction
predicted = clf.predict(user_df)[0]

#converst prediction back to text
recommendation = label_enc.inverse_transform([predicted])[0]

print(f"\nRecommended Plan: {recommendation}")

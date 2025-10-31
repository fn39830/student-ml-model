#import necesary libraries
import joblib
import pandas as pd

#load pipeline and label encoder
clf = joblib.load("fitness_pipeline.pkl")
label_enc = joblib.load("label_encoder.pkl")

#example of user input
goal = "lose face fat"
level = "beginner"

#puts data into dataframe
user_df = pd.DataFrame([[goal, level]], columns=["Goal", "Fitness Level"])

#prediction
pred = clf.predict(user_df)[0]
recommendation = label_enc.inverse_transform([pred])[0]

#prints recommendation
print(f"Recommendation: {recommendation}")

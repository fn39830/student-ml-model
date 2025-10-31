#imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


CSV_PATH = "fitness_data_balanced.csv"
RANDOM_STATE = 42
MODEL_OUT = "fitness_pipeline.pkl"
LABEL_ENCODER_OUT = "label_encoder.pkl"


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded rows:", len(df))
    print(df.head())


    for col in ["Goal", "Fitness Level", "Recommendation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()


    df = df.dropna(subset=["Goal", "Fitness Level", "Recommendation"])
    return df

def train(df):

    X = df[["Goal", "Fitness Level"]]
    y = df["Recommendation"]


    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Goal", "Fitness Level"])
        ],
        remainder='drop'
    )

    #create pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    if len(X) < 10: 
        clf.fit(X, y_enc)
        print("Model trained on all data (tiny dataset mode).")
    else:
        #train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=RANDOM_STATE)

        #fit
        clf.fit(X_train, y_train)

        #predictions on the test set
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\nTest accuracy: {test_acc:.3f}\n")
        print("Classification report (labels are encoded integers):")
        print(classification_report(y_test, y_pred, zero_division=0))

        #cross-validation for more robust metric
        cv_folds = min(3, len(X))  # safer for small datasets
        cv_scores = cross_val_score(clf, X, y_enc, cv=cv_folds)
        print(f"Cross-val scores: {np.round(cv_scores, 3)}  mean={cv_scores.mean():.3f}")

    #save the pipeline and label encoder
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(label_encoder, LABEL_ENCODER_OUT)
    print(f"\nSaved pipeline -> {MODEL_OUT}")
    print(f"Saved label encoder -> {LABEL_ENCODER_OUT}")

    return clf, label_encoder

def predict_single(clf, label_encoder, goal, level):
    # normalise inputs the same way as training
    goal = str(goal).strip().lower()
    level = str(level).strip().lower()

    input_df = pd.DataFrame([{"Goal": goal, "Fitness Level": level}])
    pred_enc = clf.predict(input_df)[0]
    pred_text = label_encoder.inverse_transform([pred_enc])[0]
    return pred_text

if __name__ == "__main__":
    df = load_and_clean(CSV_PATH)
    clf, label_enc = train(df)

    #interactive test (change the strings to try different inputs)
    sample_goal = "lose face fat"
    sample_level = "beginner"
    recommendation = predict_single(clf, label_enc, sample_goal, sample_level)
    print(f"\nSample input -> Goal: '{sample_goal}', Level: '{sample_level}'")
    print("Model recommends:", recommendation)

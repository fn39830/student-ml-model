import tkinter as tk
from tkinter import messagebox
import joblib

# Load trained model and label encoder
clf = joblib.load("fitness_pipeline.pkl")
label_enc = joblib.load("label_encoder.pkl")

def get_recommendation():
    goal = goal_entry.get()
    level = level_entry.get()
    if not goal or not level:
        messagebox.showwarning("Input Error", "Please enter both Goal and Fitness Level.")
        return
    # Predict
    input_df = {"Goal": goal.strip().lower(), "Fitness Level": level.strip().lower()}
    import pandas as pd
    pred_enc = clf.predict(pd.DataFrame([input_df]))[0]
    pred_text = label_enc.inverse_transform([pred_enc])[0]
    result_label.config(text=f"Recommendation: {pred_text}")

# Create main window
root = tk.Tk()
root.title("Fitness Recommendation")

# Goal input
tk.Label(root, text="Enter your Goal:").grid(row=0, column=0, padx=5, pady=5)
goal_entry = tk.Entry(root, width=30)
goal_entry.grid(row=0, column=1, padx=5, pady=5)

# Fitness Level input
tk.Label(root, text="Enter your Fitness Level:").grid(row=1, column=0, padx=5, pady=5)
level_entry = tk.Entry(root, width=30)
level_entry.grid(row=1, column=1, padx=5, pady=5)

# Button to get recommendation
tk.Button(root, text="Enter Goal & Level", command=get_recommendation).grid(row=2, column=0, columnspan=2, pady=10)

# Label to show result
result_label = tk.Label(root, text="", fg="blue")
result_label.grid(row=3, column=0, columnspan=2, pady=5)

root.mainloop()

from tkinter import ttk, messagebox
import tkinter as tk
import numpy as np
from BTL_NB import NaiveBayes, preprocess_data, model
import pandas as pd
import joblib
root = tk.Tk()

myLabel = tk.Label(root, text="Dự đoán mắc bệnh trầm cảm")
myLabel.pack()

# Create input fields
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Gender input
gender_label = ttk.Label(frame, text="Gender:")
gender_label.grid(row=0, column=0, sticky=tk.W, pady=5)
gender_var = tk.StringVar()
gender_cb = ttk.Combobox(frame, textvariable=gender_var)
gender_cb['values'] = ('Male', 'Female')
gender_cb.grid(row=0, column=1, pady=5)

# Age input
age_label = ttk.Label(frame, text="Age:")
age_label.grid(row=1, column=0, sticky=tk.W, pady=5)
age_var = tk.StringVar()
age_cb = ttk.Combobox(frame, textvariable=age_var)
age_cb['values'] = (1,2,3)
age_cb.grid(row=1, column=1, pady=5)

# Academic_Pressure Input
Academic_Pressure_label = ttk.Label(frame, text="Academic_Pressure:")
Academic_Pressure_label.grid(row=2, column=0, sticky=tk.W, pady=5)
Academic_Pressure_var = tk.StringVar()
Academic_Pressure_cb = ttk.Combobox(frame, textvariable=Academic_Pressure_var)  
Academic_Pressure_cb['values'] = (0,1,2,3,4,5)
Academic_Pressure_cb.grid(row=2, column=1, pady=5)  

# CGPA Input
CGPA_label = ttk.Label(frame, text="CGPA:")
CGPA_label.grid(row=3, column=0, sticky=tk.W, pady=5)
CGPA_var = tk.StringVar()
CGPA_cb = ttk.Combobox(frame, textvariable=CGPA_var)  
CGPA_cb['values'] = (1,2,3)
CGPA_cb.grid(row=3, column=1, pady=5)  

# Study_Satisfaction input
Study_Satisfaction_label = ttk.Label(frame, text="Study Satisfaction:")  
Study_Satisfaction_label.grid(row=4, column=0, sticky=tk.W, pady=5)
Study_Satisfaction_var = tk.StringVar()
Study_Satisfaction_cb = ttk.Combobox(frame, textvariable=Study_Satisfaction_var)  
Study_Satisfaction_cb['values'] = (0,1,2,3,4,5)
Study_Satisfaction_cb.grid(row=4, column=1, pady=5)  

# Sleep_Duration input
Sleep_Duration_label = ttk.Label(frame, text="Sleep Duration:")
Sleep_Duration_label.grid(row=5, column=0, sticky=tk.W, pady=5)
Sleep_Duration_var = tk.StringVar()
Sleep_Duration_cb = ttk.Combobox(frame, textvariable=Sleep_Duration_var)  
Sleep_Duration_cb['values'] = ('5-6 hours','Less than 5 hours','7-8 hours','More than 8 hours','Others')
Sleep_Duration_cb.grid(row=5, column=1, pady=5) 

# Dietary_habits input
Dietary_habits_label = ttk.Label(frame, text="Dietary Habits:")
Dietary_habits_label.grid(row=6, column=0, sticky=tk.W, pady=5)
Dietary_habits_var = tk.StringVar()
Dietary_habits_cb = ttk.Combobox(frame, textvariable=Dietary_habits_var)  
Dietary_habits_cb['values'] = ('Healthy','Moderate','Unhealthy','Others')
Dietary_habits_cb.grid(row=6, column=1, pady=5)  

# Have_you_ever_had_suicidal_thoughts input
suicidal_thoughts_label = ttk.Label(frame, text="Have_you_ever_had_suicidal_thoughts:")
suicidal_thoughts_label.grid(row=7, column=0, sticky=tk.W, pady=5)
suicidal_thoughts_var = tk.StringVar()
suicidal_thoughts_cb = ttk.Combobox(frame, textvariable=suicidal_thoughts_var)
suicidal_thoughts_cb['values'] = ('Yes','No')
suicidal_thoughts_cb.grid(row=7, column=1, pady=5)  

#Work/Study_Hours
Work_Study_Hours_label = ttk.Label(frame, text="Work/Study Hours:")
Work_Study_Hours_label.grid(row=8, column=0, sticky=tk.W, pady=5)
Work_Study_Hours_var = tk.StringVar()
Work_Study_Hours_cb = ttk.Combobox(frame, textvariable=Work_Study_Hours_var)  
Work_Study_Hours_cb['values'] = (0,1,2,3,4,5,6,7,8,9,10,11,12)
Work_Study_Hours_cb.grid(row=8, column=1, pady=5)  

#Financial_Stress
Financial_Stress_label = ttk.Label(frame, text="Financial Stress:")
Financial_Stress_label.grid(row=9, column=0, sticky=tk.W, pady=5)
Financial_Stress_var = tk.StringVar()
Financial_Stress_cb = ttk.Combobox(frame, textvariable=Financial_Stress_var)  
Financial_Stress_cb['values'] = ('1', '2', '3', '4', '5')
Financial_Stress_cb.grid(row=9, column=1, pady=5)  

#Family_History_of_Mental_Illness
Family_History_of_Mental_Illness_label = ttk.Label(frame, text="Family History of Mental Illness:")
Family_History_of_Mental_Illness_label.grid(row=10, column=0, sticky=tk.W, pady=5)
Family_History_of_Mental_Illness_var = tk.StringVar()
Family_History_of_Mental_Illness_cb = ttk.Combobox(frame, textvariable=Family_History_of_Mental_Illness_var)  # Fix var
Family_History_of_Mental_Illness_cb['values'] = ('Yes','No')
Family_History_of_Mental_Illness_cb.grid(row=10, column=1, pady=5)  

# Submit button
def submit():
    data = [
        gender_cb.get(),  
        int(age_cb.get()),  
        Academic_Pressure_cb.get(), 
        float(CGPA_cb.get()),  
        int(Study_Satisfaction_cb.get()),  
        Sleep_Duration_cb.get(), 
        Dietary_habits_cb.get(),  
        suicidal_thoughts_cb.get(),  
        int(Work_Study_Hours_cb.get()),  
        float( Financial_Stress_cb.get()),  
        Family_History_of_Mental_Illness_cb.get() 
    ]

    data_encoded = [
        0 if data[0] == 'Male' else 1,  # Gender
        data[1],  # Age
        0 if data[2] == 'Undergraduate' else 1,  # Academic
        data[3],  # CGPA
        data[4],  # Study Satisfaction
        {'5-6 hours': 0, 'Less than 5 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3, 'Others': 4}[data[5]],  # Sleep Duration
        {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2, 'Others': 3}[data[6]],  # Dietary Habits
        1 if data[7] == 'Yes' else 0,  # Suicidal Thoughts
        data[8],  # Work/Study Hours
        data[9],  # Financial
        1 if data[10] == 'Yes' else 0  # Family
    ]
    
    prediction = model.predict([data_encoded])
    result = 'Depressed' if prediction[0] == 1 else 'Not Depressed'

    name = ttk.Label(frame, text='Result: ' + result)
    name.grid(row=12, column=0, columnspan=2, pady=20)

submit_button = ttk.Button(frame, text="Dự đoán",command=submit)
submit_button.grid(row=11, column=0, columnspan=2, pady=20)
root.mainloop()

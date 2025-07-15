# Predicting Depression
This project aim to build a machine learning model to predict wether an individual is suffering from depression based on survey and behavioral data. It is part of a university project on mental health analytics.
# Dataset: 
Source: Collected via Google Form from voluntary and anonymous participants (students and young adults). 
Features collected: 19
- Gender
- Age
- Levels of stress, anxiety, and loneliness
- History of suicidal thoughts
- Family history of mental illness
- Daily screen time (phone/computer usage per day)
- Financial pressure
- ...
# Labels: The dataset is labeled based on the self-reported responses and scored using a predefined scale. It includes three levels of depression
- No depression  
- Mild depression   
- Severe depression
<img width="945" height="1012" alt="image" src="https://github.com/user-attachments/assets/42902456-1f0c-40bc-8a59-c5a5caf35e36" />
# Model Pipeline
1. Data cleaning: Handling missing values, noise filtering
2. Feature Engineering: Normalization, encoding categorical variables
3. Model Trainning:
- Logistic Regression
- Random Forest
- XGBoost
# Results
| Model               | Accuracy 
|------------------   |----------
| Logistic Regression | 0.84 
| Random Forest       | 0.9502% 
| XGBoost             | 0.9905% 


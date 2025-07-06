#  Hospital Readmission Prediction using Machine Learning

This project aims to predict whether a patient will be readmitted to the hospital based on their medical and demographic data. Built as a **Streamlit web app**, it leverages **machine learning** to support hospital staff and healthcare providers in making data-driven decisions for reducing readmission rates.

---

##  Data Science Overview

###  Problem Statement

Hospital readmissions are costly and often avoidable. The goal is to build a **classification model** that accurately predicts the likelihood of readmission based on patient records, enabling proactive interventions.

###  Approach

- **Data Preprocessing**: Missing value imputation, outlier handling, and encoding categorical variables.
- **Feature Engineering**: Derived features from diagnosis codes, inpatient visits, and length of stay.
- **Modeling**: Trained various classifiers like Random Forest, XGBoost, and Logistic Regression.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC AUC


---

##  Web App Features

-    Predict hospital readmission based on patient input
-    Visualizations for input distribution and prediction results
-    Downloadable prediction report 
-    Built with a clean and responsive UI using Streamlit
-    Risk Probability Gauge meter for better UI

---

## 📁 Project Structure
hospital_readmission_app/
│
├── MAIN.py # Streamlit frontend code
├── model_matrics.pkl # Trained ML model evaluation matrices   (pickle format)
├── requirements.txt # Project dependencies
├── README.md # Project documentation
|-- stacking_model.pkl #trained ML model(pickle format)
|-- feature_selector.pkl
├── diabetic_data.csv # Sample input CSV file


## Model Info
- Model Used

StackingClassifier with base learners:

- Random Forest
- XGBoost
- Logistic Regression (meta)
- 


## Model Performance
    Metric	Value (%)
    Accuracy	81.59
    Precision	91
    Recall	    74
    F1-Score	82
    ROC AUC	    89.8






## Tech Stack
Python 3.10

Streamlit – frontend web UI

Pandas, NumPy – data manipulation

Scikit-learn – ML modeling

Matplotlib, Seaborn – visualization

Pickle – model serialization

# processes
- select the dataset named "diabetic_data.csv " from kaggle 
- perform EDA
- preprocess and clean the data 
- feature Eng.
- Feature Selection 
- perform advance Feature Eng.
- Build a stacking based Ensemble model 
- compare multiple models evaluation matrices 
- select the best 2 model rendomforest and XGBoost for base learning model in stacking 
- Threshold tuning using ROC curve
- save Model , selector, and metrics in pkl formate

## Live App

## Sample Input Format
    Upload a .csv file with columns like:
  - 1	number_inpatient	->Count of previous inpatient visits (strongest predictor)

  - 2	total_visits	->Sum of inpatient, emergency, outpatient visits

  - 3	had_inpatient	_>Indicator whether patient had an inpatient visit

  - 4	number_emergency	=>Number of emergency visits

  - 5	number_diagnoses	->Number of diagnosis codes present

  - 6	discharge_disposition_id	

## Author
  Sougata Maity
  https://www.linkedin.com/in/sougatamaity501/
  maitysougata420@gmail.com


## Future Enhancements
- Add SHAP/LIME model interpretation

- Database integration for tracking predictions

- Deploy using  pipelines

- Integrate patient feedback analysis



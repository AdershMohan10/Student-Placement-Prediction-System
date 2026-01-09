🎓 Student Placement Prediction System
📌 Project Overview

The Student Placement Prediction System is a machine learning project that predicts whether a student will get placed based on their academic performance, work experience, and demographic attributes.
This project demonstrates an end-to-end data science workflow including data preprocessing, visualization, machine learning modeling, and evaluation.

🎯 Objectives

Analyze factors affecting student placements

Build predictive machine learning models

Compare multiple models using standard evaluation metrics

Create insights suitable for dashboards and business decisions

📊 Dataset

Source: Publicly available student placement dataset (MBA students)

Records: 215 students

Features include:

Academic scores (SSC, HSC, Degree, MBA)

Work experience

Gender, specialization, board types

Placement status (Target variable)

🔧 Technologies Used

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Scikit-learn

Tools: VS Code, Power BI

🔄 Project Workflow

Data loading and cleaning

Exploratory Data Analysis (EDA)

Feature encoding and preprocessing

Machine Learning model training

Model evaluation and comparison

Dataset export for Power BI dashboard

🤖 Machine Learning Models

Logistic Regression

Random Forest Classifier

📈 Model Evaluation Metrics
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	86%	87.9%	93.5%	90.6%
Random Forest	76.7%	80%	90.3%	84.8%
🔍 Key Insights

Students with work experience have a significantly higher chance of placement

Academic performance (SSC, Degree percentage) strongly impacts outcomes

Logistic Regression performed better than Random Forest for this dataset

📊 Visualizations

Placement distribution (Placed vs Not Placed)

MBA percentage vs placement status

Feature importance analysis

Clean dataset prepared for Power BI dashboarding

📁 Project Structure
Student-Placement-Prediction/
│
├── placement_analysis.py        # Main Python script
├── placement_data.csv           # Original dataset
├── placement_cleaned.csv        # Cleaned dataset for Power BI
├── visuals/                     # Generated plots
└── README.md

🚀 How to Run the Project
pip install pandas numpy matplotlib scikit-learn
python placement_analysis.py

📌 Future Improvements

Add more ML models (XGBoost, SVM)

Hyperparameter tuning

Deploy as a web app using Streamlit

Integrate live Power BI dashboard

👨‍💻 Author

Adersh Mohan Puthiyedath
B.Tech Computer Science & Engineering

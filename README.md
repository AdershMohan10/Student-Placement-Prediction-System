🎓 Student Placement Prediction System

📌 Project Overview

The Student Placement Prediction System is a data science and machine learning project designed to predict whether a student will get placed based on academic performance, work experience, and other profile attributes. The project combines Python-based data analysis and machine learning with Power BI dashboarding to deliver both predictive insights and visual analytics.

This project was developed as a course-level main Data Science project and follows an end-to-end pipeline: data preprocessing, visualization, model building, evaluation, and dashboard creation.

📂 Dataset

Source: Kaggle (Campus Placement Dataset)

Size: 215 records, 15 features

Target Variable: status (Placed / Not Placed)

Key Features:

Academic scores (SSC, HSC, Degree, MBA)

Work experience

Gender, specialization, board types

Placement status (target)

🛠️ Technologies Used

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Scikit-learn

Visualization: Python (Matplotlib), Power BI

IDE: VS Code

🔄 Project Workflow

Dataset loading and inspection

Data cleaning and preprocessing

Encoding categorical variables

Exploratory Data Analysis (EDA) using Python

Machine learning model training

Model evaluation using standard metrics

Feature importance analysis

Clean dataset export for Power BI

Interactive Power BI dashboard creation

📊 Visualizations & Dashboarding
🔹 Python-Based Visualizations

Placement status distribution (Placed vs Not Placed)

Academic performance comparison

MBA percentage vs placement status

Feature importance visualization

🔹 Power BI Dashboard

Interactive dashboard built using the cleaned dataset

Visualizes placement trends and key influencing factors

Includes KPIs, bar charts, and filters for deeper insights

🤖 Machine Learning Models

Two models were trained and evaluated:

1️⃣ Logistic Regression

Accuracy: 86%

Precision: 87.9%

Recall: 93.5%

F1 Score: 90.6%

2️⃣ Random Forest Classifier

Accuracy: 76.7%

Precision: 80%

Recall: 90.3%

F1 Score: 84.8%

📌 Logistic Regression performed better overall and was selected as the final model.

🔑 Key Insights

Work experience is the strongest factor influencing placement

Academic consistency plays a major role

MBA percentage alone does not guarantee placement

Logistic Regression provides better generalization on this dataset

📁 Project Structure
Student-Placement-Prediction/
│
├── placement_analysis.py        # Main Python script
├── clean_dataset_for_powerbi.csv
├── README.md
└── PowerBI_Dashboard.pbix       # Power BI dashboard file

🚀 How to Run the Project

Clone the repository

Install dependencies:

pip install pandas numpy matplotlib scikit-learn


Run the Python script:

python placement_analysis.py


Open the Power BI file using the cleaned CSV dataset

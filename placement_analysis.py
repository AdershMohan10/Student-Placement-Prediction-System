import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Placement_Data_Full_Class.csv")

# Show first 5 rows
print(df.head())

# Show dataset shape
print("\nDataset Shape:", df.shape)

# Show column names
print("\nColumns:")
print(df.columns)


print("\nMissing Values in Each Column:")
print(df.isnull().sum())


# Drop salary column
df = df.drop(columns=['salary'])
print("\nSalary column dropped")


print("\nPlacement Status Count:")
print(df['status'].value_counts())


# Encode target variable
df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})

print("\nEncoded Target Values:")
print(df['status'].value_counts())


# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nEncoded Data Shape:", df_encoded.shape)


# Split features and target
X = df_encoded.drop('status', axis=1)
y = df_encoded['status']

print("\nX shape:", X.shape)
print("y shape:", y.shape)


from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


from sklearn.linear_model import LogisticRegression

# Initialize model
log_model = LogisticRegression(max_iter=1000)

# Train model
log_model.fit(X_train, y_train)

print("Logistic Regression model trained")


# Make predictions
y_pred = log_model.predict(X_test)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLogistic Regression Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

print("Random Forest model trained")


# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nRandom Forest Performance:")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy, rf_accuracy],
    "Precision": [precision, rf_precision],
    "Recall": [recall, rf_recall],
    "F1 Score": [f1, rf_f1]
})

print("\nModel Comparison:")
print(results)


# Save cleaned dataset for Power BI
df_encoded.to_csv("powerbi/placement_cleaned.csv", index=False)

print("Clean dataset saved for Power BI")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(x=y)
plt.title("Placement Distribution (0 = Not Placed, 1 = Placed)")
plt.xlabel("Placement Status")
plt.ylabel("Count")
plt.show()


plt.figure()
sns.boxplot(x=y, y=df['mba_p'])
plt.title("MBA Percentage vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("MBA Percentage")
plt.show()


feature_importance = pd.Series(
    log_model.coef_[0],
    index=X.columns
).sort_values(ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())

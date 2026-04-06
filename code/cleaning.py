import pandas as pd

# Load the raw dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges column - convert text to numbers
# Blank values become NaN and then we fill them with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert Churn column from Yes/No text to 1/0 numbers
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID because it is just an identifier
# and has no predictive value for the model
df = df.drop('customerID', axis=1)

# Apply one-hot encoding to all remaining text columns
# This converts categories like month-to-month into
# separate yes or no columns the model can read
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset to the data folder
df.to_csv('../data/telco_cleaned.csv', index=False)

# Confirm everything looks right
print("Cleaning complete")
print("Shape after cleaning:", df.shape)
print(df.head())
import pandas as pd

# Load the dataset
df = pd.read_csv('../Sdata/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Look at the first 5 rows
print(df.head())

# Check how many rows and columns there are
print(df.shape)

# Check what data types each column is
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())
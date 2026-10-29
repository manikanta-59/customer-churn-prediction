import pandas as pd

# Load the customer dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print("Dataset shape:", df.shape)

# Display column names
print("Columns:", df.columns.tolist())

# Display data types and missing values information
df.info()
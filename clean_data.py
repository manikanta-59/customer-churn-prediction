import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Check missing values
print("Missing values:")
print(df.isnull().sum())

# 2. Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Convert TotalCharges from text to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 4. Check missing values again
print("\nMissing values after conversion:")
print(df.isnull().sum())

# 5. Display data types
print("\nUpdated data types:")
print(df.dtypes)

# Check the rows with missing TotalCharges
missing_rows = df[df["TotalCharges"].isnull()]

print("\nCustomers with missing TotalCharges:")
print(missing_rows[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# Check whether these customers have zero tenure
print("\nTenure values in missing rows:")
print(missing_rows["tenure"].value_counts())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Amita\OneDrive\Desktop\Bank-Churn-ML\data\bank_churn_500.csv")
# 2. Display first 5 rows
print("First 5 rows:")
print(df.head())

# 3. Dataset information
print("\nDataset Information:")
print(df.info())

# 4. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 6. Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# 7. Churn count
print("\nCustomer Churn:")
print(df["Exited"].value_counts())

# 8. Churn percentage
churn_percentage = df["Exited"].value_counts(normalize=True) * 100
print("\nChurn Percentage:")
print(churn_percentage)

# 9. Churn visualization
plt.figure(figsize=(6, 4))
sns.countplot(x="Exited", data=df)
plt.title("Bank Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.show()

# 10. Churn by gender
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", hue="Exited", data=df)
plt.title("Churn by Gender")
plt.show()

# 11. Churn by geography
plt.figure(figsize=(7, 4))
sns.countplot(x="Geography", hue="Exited", data=df)
plt.title("Churn by Geography")
plt.show()

# 12. Churn by active membership
plt.figure(figsize=(7, 4))
sns.countplot(x="IsActiveMember", hue="Exited", data=df)
plt.title("Churn by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.show()

# 13. Churn by number of products
plt.figure(figsize=(7, 4))
sns.countplot(x="NumOfProducts", hue="Exited", data=df)
plt.title("Churn by Number of Products")
plt.show()

print("\nData analysis completed successfully!")
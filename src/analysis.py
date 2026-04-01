# ---------------------------------------------------
# Import Libraries
# ---------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
print("\nLoading Dataset...")
data = pd.read_csv("data/cleaned_housing_data.csv")
print("Dataset Loaded Successfully")


# ---------------------------------------------------
# Basic Checks
# ---------------------------------------------------
print("\nFirst 5 Records")
print(data.head())

print("\nDataset Info")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())


# ---------------------------------------------------
# Data Cleaning (IMPROVED)
# ---------------------------------------------------

# 1. Remove duplicates
data = data.drop_duplicates()

# 2. Fill missing values (better than drop)
data.fillna(data.median(numeric_only=True), inplace=True)

print("\nMissing values handled + duplicates removed")


# ---------------------------------------------------
# Convert Categorical to Numeric
# ---------------------------------------------------
cols = ["mainroad","guestroom","basement",
        "hotwaterheating","airconditioning","prefarea"]

for c in cols:
    data[c] = data[c].str.lower().map({"yes":1,"no":0})

data["furnishingstatus"] = data["furnishingstatus"].map({
    "furnished":2,
    "semi-furnished":1,
    "unfurnished":0
})

print("Categorical conversion done")


# ---------------------------------------------------
# Remove Outliers (MULTIPLE COLUMNS)
# ---------------------------------------------------
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ["price", "area", "bedrooms", "bathrooms"]:
    data = remove_outliers(data, col)

print("Outliers removed from multiple columns")


# ---------------------------------------------------
# Feature Engineering (NEW 🔥)
# ---------------------------------------------------
data["price_per_sqft"] = data["price"] / data["area"]
data["total_rooms"] = data["bedrooms"] + data["bathrooms"]

print("New features added")


# ---------------------------------------------------
# EDA
# ---------------------------------------------------
print("\nStatistical Summary")
print(data.describe())


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure()
plt.scatter(data["area"], data["price"])
plt.title("Area vs Price")
plt.show()

sns.histplot(data["price"], kde=True)
plt.title("Price Distribution")
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# ---------------------------------------------------
# Machine Learning
# ---------------------------------------------------
X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
print("\nModel Performance")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))


# ---------------------------------------------------
# Export Clean Dataset
# ---------------------------------------------------
data.to_csv("cleaned_housing_data.csv", index=False)

print("\nProject Completed Successfully")

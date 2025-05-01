# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("sp.csv")  # Replace with your dataset file

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Handling missing values
imputer = SimpleImputer(strategy="mean")
df.fillna(df.mean(), inplace=True)

# Encode categorical features
encoder = LabelEncoder()
if 'Category' in df.columns:
    df['Category'] = encoder.fit_transform(df['Category'])

# Feature selection
features = [col for col in df.columns if col not in ['Sales', 'Date']]  # Excluding target & date
target = 'Sales'

# Splitting dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    print(f"\nModel: {name}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# --- VISUALIZATIONS ---

# 1. Sales Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Sales'], bins=30, kde=True, color="blue")
plt.title("Sales Distribution")
plt.show()

# 2. Category-wise Sales
if 'Category' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df["Category"], y=df["Sales"], palette="coolwarm", estimator=np.sum)
    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")
    plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Feature Importance (Random Forest)
rf_model = models["Random Forest"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="magma")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.show()

# 5. Actual vs Predicted Sales (Linear Regression)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=predictions["Linear Regression"], alpha=0.6, label="Linear Regression", color="red")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Linear Regression)")
plt.legend()
plt.show()

# 6. Actual vs Predicted Sales (Random Forest)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=predictions["Random Forest"], alpha=0.6, label="Random Forest", color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Random Forest)")
plt.legend()
plt.show()

# 7. Residual Plot (Linear Regression)
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=predictions["Linear Regression"], lowess=True, color="red")
plt.title("Residual Plot (Linear Regression)")
plt.xlabel("Actual Sales")
plt.ylabel("Residuals")
plt.show()

# 8. Residual Plot (Random Forest)
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=predictions["Random Forest"], lowess=True, color="blue")
plt.title("Residual Plot (Random Forest)")
plt.xlabel("Actual Sales")
plt.ylabel("Residuals")
plt.show()

# 9. Boxplot of Sales by Category
if 'Category' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df["Category"], y=df["Sales"], palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title("Boxplot of Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")
    plt.show()

# 10. Scatter Plot of Features vs. Sales
for col in features[:3]:  # Plot only first 3 features
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[col], y=df["Sales"], alpha=0.6)
    plt.title(f"{col} vs Sales")
    plt.xlabel(col)
    plt.ylabel("Sales")
    plt.show()

# 11. Sales Trend Over Time
if 'Date' in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime if not already
    df = df.sort_values(by="Date")
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df["Date"], y=df["Sales"], color="green")
    plt.title("Sales Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.show()

# 12. Sales vs. Discount (or another numeric feature)
if 'Discount' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["Discount"], y=df["Sales"], alpha=0.6, color="purple")
    plt.title("Discount vs. Sales")
    plt.xlabel("Discount")
    plt.ylabel("Sales")
    plt.show()

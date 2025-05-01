# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Mapping target values to actual species names
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
df["species"] = df["species"].map(species_map)

# Display dataset info
print(df.head())
print(df.info())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 6))
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()

# Check class distribution
sns.countplot(x="species", data=df)
plt.title("Class Distribution of Iris Species")
plt.show()

# Feature selection
X = df.iloc[:, :-1]  # Selecting all columns except the target
y = df.iloc[:, -1]   # Target column

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"\n🔹 Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing decision boundaries (using only first two features)
X_subset = X.iloc[:, :2].values  # Use only first two features for visualization
X_subset = scaler.fit_transform(X_subset)

plt.figure(figsize=(12, 4))
for i, (name, model) in enumerate(models.items()):
    plt.subplot(1, 3, i+1)
    model.fit(X_subset, y)
    plot_decision_regions(X_subset, y, clf=model, legend=2)
    plt.title(name)

plt.show()

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # To split data into training and validation sets
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.ensemble import RandomForestRegressor  # Random forest regressor for non-linear relationships
from sklearn.ensemble import GradientBoostingRegressor  # Gradient boosting regressor
from sklearn.metrics import mean_squared_error  # To evaluate model performance (RMSE)
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For scaling numeric data and encoding categorical data
from sklearn.compose import ColumnTransformer  # To preprocess different feature types in a single pipeline
from sklearn.pipeline import Pipeline  # To streamline preprocessing and modeling
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For enhanced visualizations

# Load the training and test datasets
train_path = "train.csv"
test_path = "test.csv"

try:
    train_data = pd.read_csv(train_path)  # Training data
    test_data = pd.read_csv(test_path)  # Test data
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Visualization: Missing Values
missing = train_data.isnull().sum()
if missing[missing > 0].empty:
    print("No missing values in the dataset.")
else:
    plt.figure(figsize=(10, 6))
    missing[missing > 0].plot(kind='bar', color='skyblue')
    plt.title("Missing Values Count per Column")
    plt.ylabel("Count")
    plt.xlabel("Columns")
    plt.show()

# Visualization: Distribution of Numeric Features
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
train_data[numeric_features].hist(figsize=(12, 10), bins=30, edgecolor='k')
plt.suptitle("Distribution of Numeric Features")
plt.show()

# Visualization: Categorical Features
categorical_features = train_data.select_dtypes(include=['object']).columns
for col in categorical_features:
    if train_data[col].nunique() > 20:
        print(f"Skipping bar chart for {col} due to high cardinality.")
        continue
    plt.figure(figsize=(8, 4))
    train_data[col].value_counts().plot(kind='bar', color='lightcoral')
    plt.title(f"Frequency of Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Visualization: Boxplots for Numeric Features
for col in numeric_features:
    plt.figure(figsize=(8, 4))
    train_data.boxplot(column=col)
    plt.title(f"Boxplot for {col}")
    plt.show()

# Visualization: Correlation Matrix
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Data Preprocessing
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()
numeric_features.remove("price")  # Remove target variable from numeric features

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data into features and target variable
X = train_data.drop(columns=["price"])
y = train_data["price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_val_pred = lr_pipeline.predict(X_val)
lr_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Validation Set: Actual vs Predicted Prices (Linear Regression)")
plt.show()

# Train and Evaluate Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_val_pred_rf = rf_pipeline.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))

# Train and Evaluate Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_val_pred_gb = gb_pipeline.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_gb))

print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Gradient Boosting RMSE: {gb_rmse}")

# Apply Best Model to Test Data
if rf_rmse < lr_rmse and rf_rmse < gb_rmse:
    final_model = rf_pipeline
elif gb_rmse < lr_rmse:
    final_model = gb_pipeline
else:
    final_model = lr_pipeline

try:
    test_predictions = final_model.predict(test_data)
    print("Test predictions generated successfully.")
except Exception as e:
    print(f"Error during test predictions: {e}")
    exit()

# Save Test Predictions
pd.DataFrame({"Test Predictions": test_predictions}).to_csv("test_predictions.csv", index=False)
print("Sample Test Predictions:")
print(pd.DataFrame({"Test Predictions": test_predictions}).head(10))

# Summarize Results
summary = {
    "Linear Regression RMSE": lr_rmse,
    "Random Forest RMSE": rf_rmse,
    "Gradient Boosting RMSE": gb_rmse,
    "Selected Model": "Gradient Boosting" if gb_rmse < lr_rmse and gb_rmse < rf_rmse else
                     "Random Forest" if rf_rmse < lr_rmse else
                     "Linear Regression",
    "Min Prediction": test_predictions.min(),
    "Max Prediction": test_predictions.max(),
    "Mean Prediction": test_predictions.mean()
}
print(pd.DataFrame([summary]))

# Visualization: Distribution of Predicted Prices
plt.hist(test_predictions, bins=50, edgecolor='k')
plt.xlabel("Predicted Prices")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Prices")
plt.show()
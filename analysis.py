# Import necessary libraries
import pandas as pd # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split # To split data into training and validation sets
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.ensemble import RandomForestRegressor # Random forest regressor for non-linear relationships
from sklearn.metrics import mean_squared_error # To evaluate model performance (RMSE)
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For scaling numeric data and encoding categorical data
from sklearn.compose import ColumnTransformer # To preprocess different feature types in a single pipeline
from sklearn.pipeline import Pipeline # To streamline preprocessing and modeling

import matplotlib.pyplot as pyplot
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Validation Set: Actual vs Predicted Prices")
plt.show()


# Load the training and test datasets
# train.csv contains both features and target variable ('price')
# test.csv contains only features, and we need to predict 'price' for this dataset
train_path = "train.csv"
test_path = "test.csv"

try:
    train_data = pd.read_csv(train_path) # Training data
    test_data = pd.read_csv(test_path) # Test data
except: FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


# Step 1: Data Exploration
# Explore the training dataset
train_summary = train_data.describe(include='all').transpose()  # Summary statistics for all columns
missing_values = train_data.isnull().sum() # Count of missing values in each column


# Step 2: Data Preprocessing
# Identify numeric and categorical columns
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist() # List of numeric features
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()  # List of categorical features


# Remove the target variable ('price') from numeric features, as it shouldn't be preprocessed
numeric_features.remove("price")


# Define transformations:
# Scale numeric features to standardize their range (mean=0, std=1)
numeric_transformer = StandardScaler()

# Encode categorical features into binary/one-hot representation
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# Combine numeric and categorical transformations into a single preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features), # Apply scaling to numeric features
        ('cat', categorical_transformer, categorical_features) # Apply encoding to categorical features
    ])


# Step 3: Split data
# Separate features (X) and target variable (y) from the training dataset
X = train_data.drop(columns=["price"]) # Features
y = train_data["price"] # Target variable (housing prices)

# Split the data into training and validation sets
# 80% of the data is used for training, and 20% is reserved for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 4: Model Training (Linear Regression as a baseline) and Evaluation
# Train a Linear Regression model as a baseline
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), # Apply preprocessing
                              ('regressor', LinearRegression())])  # Use Linear Regression as the model


# Fit the pipeline to the training data
lr_pipeline.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = lr_pipeline.predict(X_val)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model's performance
lr_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))


# Step 5: Model Training (Random Forest Regressor for improvement)

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), # Apply preprocessing
                              ('regressor', RandomForestRegressor(random_state=42))]) # Use Random Forest Regressor as the model


# Fit the pipeline to the training data
rf_pipeline.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_rf = rf_pipeline.predict(X_val)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model's performance
rf_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))


# Step 6: Apply best model to test data
# Select the best-performing model based on RMSE
# Use Random Forest if it performs better; otherwise, use Linear Regression
final_model = rf_pipeline if rf_rmse < lr_rmse else lr_pipeline
test_predictions = final_model.predict(test_data)

# Summarize results
summary = {
    "Data Overview": train_summary,
    "Missing Values": missing_values,
    "Linear Regression RMSE": lr_rmse,
    "Random Forest RMSE": rf_rmse,
    "Selected Model": "Random Forest" if rf_rmse < lr_rmse else "Linear Regression"
}

#import ace_tools as tools; tools.display_dataframe_to_user(name="Data Summary and Model Results", dataframe=pd.DataFrame([summary]))
print(pd.DataFrame([summary]))

# Save the summary of results to a CSV file for documentation
pd.DataFrame([summary]).to_csv("output_summary.csv", index=False)
print("Summary saved to output_summary.csv")
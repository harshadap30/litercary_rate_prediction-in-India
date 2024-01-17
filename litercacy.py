import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the data
file_path = 'literacy_rate.csv'
df = pd.read_csv(file_path)

# Replace 'na' values with NaN
df.replace('na', np.nan, inplace=True)

# Identify non-numeric columns
non_numeric_columns = ['States/Union_Territories']

# Convert numeric columns to numeric (excluding non-numeric columns)
numeric_columns = df.columns.difference(non_numeric_columns)
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Display the first few rows of the DataFrame
print(df.head())

# Data Analysis and Visualization
# Let's visualize the literacy rates over the years for each state
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=df.melt(id_vars=['No.', 'States/Union_Territories'], var_name='Year', value_name='Literacy Rate'),
#              x='Year', y='Literacy Rate', hue='No.')
# plt.title('State-wise Literacy Rates Over the Years')
# plt.show()

# Machine Learning: Linear Regression Example
# Predicting literacy rates for the year 2011 based on previous years
X = df[['1951', '1961', '1971', '1981', '1991', '2001']].fillna(0)  # Replace NaN with 0 for simplicity
y = df['2011'].fillna(0)  # Replace NaN with 0 for simplicity

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Root Mean Squared Error: {rmse}')




# Feature Importance
# feature_importance = pd.Series(model.coef_, index=X.columns)
# feature_importance.plot(kind='barh')
# plt.title('Feature Importance')
# plt.xlabel('Coefficient Magnitude')
# plt.show()

# Residual Analysis
# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.title('Residual Analysis')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.show()


# Predictive Visualizations
# plt.scatter(X_test['2001'], y_test, color='black', label='Actual')
# plt.scatter(X_test['2001'], y_pred, color='blue', label='Predicted')
# plt.xlabel('Literacy Rate in 2001')
# plt.ylabel('Literacy Rate in 2011')
# plt.title('Linear Regression: Predicted vs Actual Literacy Rates (2011)')
# plt.legend()
# plt.show()

from sklearn.model_selection import cross_val_score

# Cross-Validation
cv_mse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
cv_rmse = np.sqrt(-cv_mse)
print(f'Cross-Validation RMSE: {cv_rmse.mean()}')


# # Example of creating a feature representing the change in literacy rate from 1951 to 2001
# df['change_rate_1951_to_2001'] = df['2001'] - df['1951']

# # Include the new feature in the analysis
# X_updated = df[['1951', '1961', '1971', '1981', '1991', '2001', 'change_rate_1951_to_2001']].fillna(0)

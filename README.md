# financial-stock-market-prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_excel('/content/yahoo_data.xlsx')  

# Check the first few rows of the dataset to verify the data format and column names
print(data.head())

# Extracting relevant columns
selected_columns = ['Open', 'High', 'Low', 'Volume']  # Adjust column names as per your dataset
data = data[selected_columns].dropna()

# Splitting the data into features and target variable
X = data[['Open', 'High', 'Low']]  # Features
y = data['Volume']  # Target variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting linear regression model to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Printing the coefficients
print("Coefficients:", regressor.coef_)

# Printing the intercept
print("Intercept:", regressor.intercept_)

# Calculate the number of observations
n = len(y_test)
print("Number of Observations:", n)

# Calculate the Multiple R (which is the square root of R-squared)
multiple_r = r2 ** 0.5
print("Multiple R:", multiple_r)

# Calculate the Adjusted R-squared
p = X_test.shape[1]  # Number of independent variables
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R-squared:", adjusted_r2)

# Calculate standard deviation of residuals
std_dev = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("Standard Deviation of Residuals:", std_dev)

import matplotlib.pyplot as plt

# Plotting actual vs predicted volume
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Volume (Test Set)')
plt.xlabel('Actual Volume')
plt.ylabel('Predicted Volume')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the relationship between each independent variable and the target variable (Volume)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, column in enumerate(X.columns):
    axs[i].scatter(X_test[column], y_test, color='blue', label='Actual')
    axs[i].scatter(X_test[column], y_pred, color='red', label='Predicted')
    axs[i].set_title(f'{column} vs. Volume')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Volume')
    axs[i].legend()

plt.tight_layout()
plt.show()

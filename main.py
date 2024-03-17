import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the data into a Pandas DataFrame
df = pd.read_csv('C:\project\Financial Analytics data.csv')

# Display basic statistics and information about the dataset
print(df.info())
print(df.describe())

# Pair plot for key metrics
sns.pairplot(df[['Mar Cap - Crore', 'Sales Qtr - Crore']])
plt.suptitle('Pair Plot of Market Capitalization and Quarterly Sales', y=1.02)
plt.show()

# Correlation heatmap
correlation_matrix = df[['Mar Cap - Crore', 'Sales Qtr - Crore']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap between Market Capitalization and Quarterly Sales')
plt.show()

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df[['Mar Cap - Crore', 'Sales Qtr - Crore']] = imputer.fit_transform(df[['Mar Cap - Crore', 'Sales Qtr - Crore']])

# Splitting the data into features (X) and target variable (y)
X = df[['Mar Cap - Crore']]  # Feature: Market Capitalization
y = df['Sales Qtr - Crore']  # Target: Quarterly Sales

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Coefficients:", model.coef_)

# Plotting the regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Market Capitalization (in crores)')
plt.ylabel('Quarterly Sales (in crores)')
plt.title('Linear Regression Model: Market Capitalization vs Quarterly Sales')
plt.show()

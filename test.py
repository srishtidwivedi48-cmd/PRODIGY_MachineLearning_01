import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Python is running")

data = pd.read_csv("train.csv")
print("File opened successfully")
print(data.head())
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
print(data.isnull().sum())
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_house = np.array([[2000, 3, 2]])
predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()


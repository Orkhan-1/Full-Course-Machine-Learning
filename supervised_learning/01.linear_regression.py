from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample Data: House size vs. Price
x = np.array([[500], [750], [1000], [1250], [1500]])
y = np.array([100, 150, 200, 250, 300])  # in $1000s

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Predict the price of a 1100 sq ft house
predicted_price = model.predict([[1100]])
print(f"Predicted price: ${predicted_price[0]*1000:.2f}")

# Plotting
plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title("Linear Regression - House Price Prediction")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.show()

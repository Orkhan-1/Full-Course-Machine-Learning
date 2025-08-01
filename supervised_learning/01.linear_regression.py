from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([50, 60, 65, 70, 75])

model = LinearRegression()
model.fit(x, y)

print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

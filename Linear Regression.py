import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv("weight-height2.csv")
x = data['Height']
y = data['Weight']

slope, intercept, r_value, p_value, std_err = linregress(x, y)

mean_x = np.mean(x)
mean_y = np.mean(y)
covar = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)
var = np.sum((x - mean_x) ** 2) / (len(x) - 1)

print("Covariance:", covar)
print("Variance:", var)
print("Slope:", slope)
print("Intercept:", intercept)

height_input = int(input("Enter Height: "))
predicted_weight = slope * height_input + intercept
print("The predicted weight is", predicted_weight)

model = slope * x + intercept

plt.figure(figsize=(10, 10))
plt.scatter(x, y, color="violet", label="Data")
plt.plot(x, model, color="cyan", label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()

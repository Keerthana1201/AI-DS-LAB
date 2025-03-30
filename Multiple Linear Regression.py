from numpy import *
from matplotlib.pyplot import *
from sklearn.linear_model import *
from pandas import *
from seaborn import *

data = read_csv("Multiple.csv")
x = data[['size', 'bed', 'age']]
y = data['price']

model = LinearRegression()
model.fit(x, y)

coef = model.coef_
intercept = model.intercept_

size = int(input("Enter size: "))
bed = int(input("Enter no. of bedrooms: "))
age = int(input("Enter age: "))

price = intercept + coef[0] * size + coef[1] * bed + coef[2] * age
print("Predicted price:", price)

pairplot(data, x_vars=['size', 'bed', 'age'], y_vars='price', hue='price')
show()

from models.linear_regression import LinearRegressionModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_PATH = "./data/altura_peso.csv"

data = pd.read_csv(DATA_PATH)

x = data["Altura"]
normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))

min_x = data["Altura"].min()
max_x = data["Altura"].max()

y = data["Peso"]
normalized_y = (y - np.min(y)) / (np.max(y) - np.min(y))

plt.scatter(x, y)

model = LinearRegressionModel()
model.summary()

model.train_with(x=normalized_x, y=normalized_y, verbose=True)

w, b = model.get_slope_and_y_intercept()

print("w, b")

x_axis = np.linspace(min_x, max_x, 100)
normalized_axis = (x_axis - np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
y_axis = np.min(y) + model.predict(normalized_axis) * (np.max(y) - np.min(y))

plt.plot(x_axis, y_axis)

plt.show()

while True:
    str = input("Enter something to predict on: ")
    if str == "q":
        break
    value = float(str)
    predicted = model.predict(tf.constant([value]))

    print("Predicted: {}", predicted)



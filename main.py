from models.linear_regression import LinearRegressionModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_PATH = "./data/altura_peso.csv"

data = pd.read_csv(DATA_PATH)

x = data["Altura"]
y = data["Peso"]

plt.scatter(x, y)

model = LinearRegressionModel()
model.summary()

model.train_with(x=x, y=y, verbose=True)

w, b = model.get_slope_and_y_intercept()

print("w, b")

x_axis = np.linspace(np.min(x), np.max(x), 100)
y_axis = model.predict(x_axis) 

plt.plot(x_axis, y_axis)

plt.show()



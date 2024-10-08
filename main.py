from models.linear_regression import LinearRegressionModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


DATA_PATH = "./data/altura_peso.csv"

def plot_height_vs_weight(height: tf.Tensor, weight: tf.Tensor):
    """
        Grafica un diagrama de dispersión de la relación
        entre las variables "Altura" y "Peso"
    """
    plt.xlabel("Altura (cm)")
    plt.ylabel("Peso (kg)")
    plt.title("Altura vs Peso")
    return plt.scatter(height, weight)

def plot_predicted(model: LinearRegressionModel, *, min_x: float, max_x: float, n: int):
    """
        Grafica las predicciones del modelo para un rango dado
    """
    x_axis = np.linspace(min_x, max_x, 100)
    y_axis = model.predict(x_axis) 
    plt.plot(x_axis, y_axis, color="red")

def plot_train_history(loss_history: tf.Tensor):
    """
        Grafica la pérdida a lo largo de las épocas 
    """
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (Error Cuadrático medio)")
    plt.plot(loss_history.history["loss"])

def plot_loss_and_predictions(x: tf.Tensor, y: tf.Tensor, model: LinearRegressionModel):
    """
        Grafica información sobre el entrenamiento de `model` con los 
        tensores `x` e `y`, así como de las predicciones del mismo.
    """
    plt.subplot(1, 2, 1)
    plt.title("Épocas vs Pérdida")

    # Realizamos el entrenamiento
    history = model.train_with(x=x, y=y, verbose=True)
    plot_train_history(history)

    # Imprimimos un resumen de la topología del modelo
    model.summary()

    plt.subplot(1, 2, 2)
    plt.title("Altura (cm) vs Peso (kg)")

    plot_height_vs_weight(x, y)

    # Graficamos los valores predichos por el modelo
    plot_predicted(model, min_x=np.min(x), max_x=np.max(x), n=100)

    # Imprimimos en pantalla los parámetros del modelo
    w, b = model.get_weights()
    print(f"Parámetros: w = {w}, b = {b}")

    plt.show()


def main():
    model = LinearRegressionModel()

    data = pd.read_csv(DATA_PATH)
    x = data["Altura"]
    y = data["Peso"]

    plot_loss_and_predictions(x, y, model)

    # Predicción específica
    predicted = model.predict(170)
    print(f"Para una altura de 1,70m (170cm), el modelo predice un peso {predicted[0][0]:.2f}kg")


if __name__ == "__main__":
    main()

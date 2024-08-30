import keras.api
from keras.api.models import Sequential
from keras.api.layers import Dense, InputLayer
from keras.api.losses import mean_squared_error
from keras.api.optimizers import SGD
import keras
import tensorflow as tf
import numpy as np

LEARNING_RATE = 3.2 * (10 ** (-5))
EPOCHS = 10_000

class LinearRegressionModel:
    """
        Clase que implementa un modelo de regresión lineal.
        Permite entrenarlo con entradas y salidas definidas y ofrecer
        predicciones en base a una entrada concreta.
    """

    model: keras.Sequential

    def __init__(self):
        self.model = Sequential()

        # Establecer una semilla para la generación de números aleatorios
        # para que el modelo se comporte de manera homógenea entre ejecuciones
        # del script 
        np.random.seed(230490)

        # Como tratamos con regresión lineal, tanto la entrada como la salida
        # son números reales inviduales.
        self.model.add(Dense(1, activation="linear", input_dim=1))

        # Usamos el descenso estocástico del gradiente como función de optimización...
        sgd = SGD(learning_rate=LEARNING_RATE, momentum=0.0001)
        # ...y el error cuadrático medio como functión de coste
        self.model.compile(optimizer=sgd, loss=mean_squared_error)

    def summary(self):
        """
            Muestra un resumen de las características de la red neuronal
        """
        self.model.summary()

    def train_with(self, x: tf.Tensor, y: tf.Tensor, *, verbose: bool = False, epochs: int = EPOCHS):
        """
            Entrena el modelo con una entrada lineal (no por lotes). 
            La forma de los tensores debe ser (n,), donde n es el tamaño de la entrada
            y su correspondiente salida.

            :returns La historia del entrenamiento
        """
        if len(x.shape) > 1:
            raise ValueError("Forma incorrecta para el tensor x")
        if len(y.shape) > 1:
            raise ValueError("Forma incorrecta para el tensor y")
        # El tamaño de cada batch de entrenamiento lo obtenemos
        # de la forma del input
        batch_size = x.shape[0]

        story = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=int(verbose))
        return story

    def get_slope_and_y_intercept(self) -> tuple[float, float]:
        """
            Obtiene la pendiente y la ordenada al origen de la recta de regresión. 
            El modelo debería ser entrenado antes de que esta función pueda
            retornar algún valor útil

            :returns Una tupla con (pendiente, ordenada al origen)
            :rtype tuple[float, float]
        """
        layer = self.model.layers[0]
        w, b =  layer.get_weights()
        return w[0][0], b[0]

    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """
            Predice la salida de cada uno de los elementos de `x` siguiendo
            la recta de regresión
        """
        if len(x.shape) > 1:
            raise ValueError("Forma de tensor a predecir incorrecta")
        
        return self.model.predict(x)



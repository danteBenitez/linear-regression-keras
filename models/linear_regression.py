import keras.api
from keras.api.models import Sequential
from keras.api.layers import Dense, InputLayer
from keras.api.losses import mean_squared_error
from keras.api.optimizers import SGD
from keras.api.utils import normalize as keras_normalize
import keras
import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.0004
EPOCHS = 10_000

def normalize(x: tf.Tensor) -> tf.Tensor:
    """
        Normaliza un tensor usando escalado min-max

        :returns Una tupla con (max, min, normalized)
        :rtype (float, float, tf.Tensor)
    """
    min_x = np.min(x)
    max_x = np.max(x)
    normalized = (x - min_x) / (max_x - min_x)
    return (max_x, min_x, normalized)

def normalize_value(x: float, min_x: float, max_x: float) -> float:
    """
        Normaliza un valor individual dado un máximo y un mínimo

        :returns El valor normalizado entre 0 y 1
        :rtype float
    """
    normalized = (x - min_x) / (max_x - min_x)
    return normalized

def denormalize(x: tf.Tensor, min_value: float, max_value: float) -> tf.Tensor:
    """
        Denormaliza un tensor que fue reescalado con el método min-max
    """
    return min_value + x * (max_value - min_value)

class LinearRegressionModel:
    """
        Clase que implementa un modelo de regresión lineal.
        Permite entrenarlo con entradas y salidas definidas y ofrecer
        predicciones en base a una entrada concreta.
    """

    model: keras.Sequential

    # Valores mínimos y máximos usados para entrenamientos
    min_y: float
    max_y: float
    min_x: float
    max_x: float

    def __init__(self):
        self.model = Sequential()

        # Establecer una semilla para la generación de números aleatorios
        # para que el modelo se comporte de manera homógenea entre ejecuciones
        # del script 
        keras.utils.set_random_seed(2)

        # Como tratamos con regresión lineal, tanto la entrada como la salida
        # son números reales inviduales.
        self.model.add(Dense(1, activation="linear", input_dim=1))

        # Usamos el descenso estocástico del gradiente como función de optimización...
        sgd = SGD(learning_rate=LEARNING_RATE)
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

        max_x, min_x, normalized_x = normalize(x)
        max_y, min_y, normalized_y = normalize(y)

        self.max_y = max_y
        self.min_y = min_y
        self.max_x = max_x
        self.min_x = min_x

        story = self.model.fit(x=normalized_x, y=normalized_y, epochs=epochs, batch_size=batch_size, verbose=int(verbose))
        return story

    def get_weights(self) -> tuple[float, float]:
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

    def predict(self, x: tf.Tensor | float) -> tf.Tensor:
        """
            Predice la salida de cada uno de los elementos de `x` siguiendo
            la recta de regresión
        """


        if isinstance(x, float) or isinstance(x, int):
            print(f"Normalizing value with {self.min_x} and {self.max_x}")
            normalized = tf.constant([normalize_value(x, self.min_x, self.max_x)])
        else:
            if len(x.shape) > 1:
                raise ValueError("Forma de tensor a predecir incorrecta")
            _, _, normalized = normalize(x) 

        print("normalized: ", normalized)
        return denormalize(self.model.predict(normalized), min_value=self.min_y, max_value=self.max_y)



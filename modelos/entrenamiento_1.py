import os

import cv2
import numpy as np
from keras.layers import InputLayer, Conv2D, MaxPool2D, Reshape, Dense, Flatten
from keras.models import Sequential


class NeuralNetwork:
    def __init__(self):
        self.ancho = 128
        self.alto = 128
        self.pixeles = self.ancho * self.alto
        self.numeroCanales = 1
        self.formaImagen = (self.ancho, self.alto, self.numeroCanales)
        self.numeroCategorias = 7
        self.model = None

    def cargarDatos(self, rutaOrigen):
        imagenesCargadas = []
        valorEsperado = []
        for categoria in range(7,self.numeroCategorias+7):
            dir = f"{rutaOrigen}{categoria}"
            files = os.listdir(dir)
            for file_name in files:
                ruta= f"{dir}/{file_name}"
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (self.ancho, self.alto), interpolation=cv2.INTER_AREA)
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(self.numeroCategorias)
                probabilidades[categoria-7] = 1
                valorEsperado.append(probabilidades)
        imagenesEntrenamiento = np.array(imagenesCargadas)
        valoresEsperados = np.array(valorEsperado)
        return imagenesEntrenamiento, valoresEsperados

    def create_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.pixeles,)))
        model.add(Reshape(self.formaImagen))
        model.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.numeroCategorias, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def train(self, imagenes, probabilidades, epochs, batch_size):
        self.model.fit(x=imagenes, y=probabilidades, epochs=epochs, batch_size=batch_size)

    def evaluate(self, imagenesPrueba, probabilidadesPrueba):
        resultados = self.model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
        print("Accuracy =", resultados[1])

    def save_model(self, ruta):
        self.model.save(ruta)

    def summary(self):
        self.model.summary()


    def crearModel(self,nombre):
        imagenes, probabilidades = self.cargarDatos("dataset/train/")

        # Create the model
        self.create_model()

        # Train the model
        self.train(imagenes, probabilidades, epochs=30, batch_size=60)

        # Load test images
        imagenesPrueba, probabilidadesPrueba = self.cargarDatos("dataset/test/")

        # Evaluate the model
        self.evaluate(imagenesPrueba, probabilidadesPrueba)

        # Save the model
        ruta = f"modelos/models/{nombre}.h5"
        self.save_model(ruta)

        # Print the model summary
        self.summary()




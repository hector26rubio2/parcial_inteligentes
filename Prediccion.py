import os
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import numpy as np
import cv2

class Prediccion():
    def __init__(self,ruta,ancho,alto):
        self.ruta = ruta
        self.modelo=load_model(self.ruta, custom_objects={"Addons>F1Score": tfa.metrics.F1Score(num_classes=2, average="micro")})
        self.alto=alto
        self.ancho=ancho

    # Función para realizar una predicción utilizando el modelo de clasificación.

    # Parámetros:
    # - imagen: la imagen de entrada a predecir (numpy array)

    # Retorna:
    # - La clase predicha para la imagen (entero)
    
    def predecir(self,imagen):
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas=[]
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA=np.array(imagenesCargadas)
        predicciones=self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=",predicciones)
        clasesMayores=np.argmax(predicciones,axis=1)
        return clasesMayores[0]

# Función para cargar y preparar los datos para el entrenamiento del modelo de clasificación.

#     Parámetros:
#     - rutaOrigen: la ruta de origen de las imágenes (string)
#     - numeroCategorias: el número total de categorías/clases (entero)
#     - ancho: el ancho deseado para las imágenes redimensionadas (entero)
#     - alto: el alto deseado para las imágenes redimensionadas (entero)

#     Retorna:
#     - imagenesEntrenamiento: un numpy array de imágenes preparadas para el entrenamiento (numpy array)
#     - valoresEsperados: un numpy array de los valores esperados para cada imagen en formato one-hot encoding (numpy array)

    def cargarDatos(self, rutaOrigen, numeroCategorias, ancho, alto):
        imagenesCargadas = []
        valorEsperado = []
        for categoria in range(1, numeroCategorias):
            dir = rutaOrigen + str(categoria)
            files = os.listdir(dir)
            for file_name in files:
                ruta = dir + "/" + file_name
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria] = 1
                valorEsperado.append(probabilidades)
        imagenesEntrenamiento = np.array(imagenesCargadas)
        valoresEsperados = np.array(valorEsperado)
        return imagenesEntrenamiento, valoresEsperados
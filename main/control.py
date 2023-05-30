import cv2
import numpy as np
from tensorflow.python.keras.models import load_model


class Prediccion:
    def __init__(self, ruta, ancho, alto):
        print(ruta)
        self.modelo = load_model(ruta)
        self.alto = alto
        self.ancho = ancho

    def predecir(self, imagen):
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas = []
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA = np.array(imagenesCargadas)
        predicciones = self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=", predicciones)
        clasesMayores = np.argmax(predicciones, axis=1)
        return clasesMayores[0]


clases=["numero 0","numero 1","numero 2","numero 3","numero 4","numero 5","numero 6","numero 7","numero 8","numero 9"]

ancho=128
alto=128

miModeloCNN=Prediccion("C:/Users/hecto/PycharmProjects/parcial_2/modelos/models/modeloA.h5",ancho,alto)
imagen=cv2.imread("dataset/test/8/8_15.jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()
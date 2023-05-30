import cv2
import numpy as np

from cargarDatos.cut import Cut


class CargarImagen:
    def __init__(self,carta):
        self.nameWindow = "Calculadora Canny"
        self.camara = cv2.VideoCapture(1)
        self.recorte = Cut(carta=carta)
        self.idImg = 1

        self.constructorVentana()  # Create trackbars

    def nothing(self, x):
        pass

    def constructorVentana(self):
        cv2.namedWindow(self.nameWindow)
        cv2.createTrackbar("min", self.nameWindow, 59, 255, self.nothing)
        cv2.createTrackbar("max", self.nameWindow, 86, 255, self.nothing)
        cv2.createTrackbar("kernel", self.nameWindow, 2, 100, self.nothing)
        cv2.createTrackbar("areaMin", self.nameWindow, 350, 2000, self.nothing)
        cv2.createTrackbar("areaMax", self.nameWindow, 1300, 5000, self.nothing)

    def calcularAreas(self, figuras):
        areas = []
        for figuraActual in figuras:
            areas.append(cv2.contourArea(figuraActual))
        return areas

    def detectarFormas(self):
        _, imagen = self.camara.read()
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        min = cv2.getTrackbarPos("min", self.nameWindow)
        max = cv2.getTrackbarPos("max", self.nameWindow)
        bordes = cv2.Canny(imagen_gris, min, max)
        tamaño_kernel = cv2.getTrackbarPos("kernel", self.nameWindow)
        kernel = np.ones((tamaño_kernel, tamaño_kernel), np.uint8)
        bordes = cv2.dilate(bordes, kernel)
        cv2.imshow("Bordes", bordes)
        # Deteccion de la figura
        figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = self.calcularAreas(figuras)
        area_min = cv2.getTrackbarPos("areaMin", self.nameWindow)
        i = 0
        for figura_actual in figuras:
            if areas[i] >= area_min:
                vertices = cv2.approxPolyDP(figura_actual, 0.05 * cv2.arcLength(figura_actual, True), True)
                if len(vertices) == 4:

                    self.idImg, imagen = self.recorte.crop(imagen, figuras, self.idImg, bordes, imagen_gris)
                    i += 1
        return imagen

    def run(self):
        while True:
            k = cv2.waitKey(1)
            imagen = self.detectarFormas()
            cv2.imshow('Imagen', imagen)
            if k == ord('e'):
                break

        cv2.destroyAllWindows()


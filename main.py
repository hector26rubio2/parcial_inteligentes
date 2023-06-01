import cv2
import numpy as np
import base64
import os
from Cut import Cut

def nothing(x):
    pass

def constructorVentana(nameWindow):
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 59, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 86, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 2, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 350, 2000, nothing)
    cv2.createTrackbar("areaMax", nameWindow, 1300, 5000, nothing)

# Función para calcular el área de cada figura en una lista de contornos.

# Parámetros:
# - figuras: una lista de contornos (lista de numpy arrays)

# Retorna:
# - areas: una lista de áreas correspondientes a cada figura (lista de floats)
def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas


# Función para detectar formas en una imagen y realizar recortes.

# Parámetros:
# - imagen: la imagen de entrada en formato BGR (numpy array)
# - idImg: el identificador de la imagen (entero)
# - recorte: objeto de la clase Recorte utilizado para realizar los recortes (objeto)
# - nameWindow: el nombre de la ventana utilizada para los controles trackbar (string)

# Retorna:
# - imagen: la imagen de entrada con las formas detectadas y recortadas (numpy array)
# - idImg: el identificador de la imagen actualizado después de los recortes (entero)
def detectarFormas(imagen, idImg, recorte, nameWindow):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagen_gris, min, max)
    tamaño_kernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamaño_kernel, tamaño_kernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    area_min = cv2.getTrackbarPos("areaMin", nameWindow)
    i = 0
    for figura_actual in figuras:
        if areas[i] >= area_min:
            vertices = cv2.approxPolyDP(figura_actual, 0.05 * cv2.arcLength(figura_actual, True), True)
            if len(vertices) == 4:
                idImg, imagen = recorte.crop2(imagen, figuras, idImg, bordes,imagen_gris)
                i += 1
    return imagen, idImg



# Función para iniciar la interfaz gráfica.

# Parámetros:
# - nameWindow: el nombre de la ventana principal (string)
def gui(nameWindow):
    camara = cv2.VideoCapture(1)
    recorte = Cut
    k = 0
    constructorVentana(nameWindow)
    idImg = 27
    while True:
        k = cv2.waitKey(1)
        _, imagen = camara.read()
        imagen, idImg = detectarFormas(imagen, idImg, recorte,nameWindow)
        cv2.imshow('Imagen', imagen)
        if k == ord('e'):
            break
    
def main():
    nameWindow = "Calculadora Canny"
    gui(nameWindow)
    
if __name__ == "__main__":
    main()
import cv2

class Recorta:

    # Función para recortar y guardar las regiones de interés de una imagen.

    # Parámetros:
    # - ruta: la ruta donde se guardarán las imágenes recortadas (string)
    # - image: la imagen original de la cual se extraerán las regiones de interés (numpy array)
    # - contours: los contornos de las regiones de interés (lista de numpy arrays)
    # - category: el número de categoría o identificador de las imágenes recortadas (entero)
    # - num: el número de imagen dentro de una categoría (entero)

    # Retorna:
    # - num: el número actualizado de imagen dentro de una categoría (entero)
    def recortar(self, ruta, image, contours, category, num):
        idNum = category
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if w > 50 and h > 50:
                # Redimenciona
                new_img = image[y:y + h, x:x + w]
                cv2.imwrite(f"{ruta}{idNum}_{num}.jpg", new_img)
                idNum = idNum + 1
        return num
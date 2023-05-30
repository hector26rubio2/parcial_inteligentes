import os

import cv2


class Cut:

    def __init__(self, carta):
        self.carta = carta

    def crop(self, image, contours, num, bordes, gris):
        pru = image
        new_img = bordes
        idNum = num
        for c in contours:
            area = cv2.contourArea(c)
            if area == 0:
                break
            x, y, w, h = cv2.boundingRect(c)
            if (w > 370 and h > 180) or (h > 370 and w > 180):
                new_img = bordes[y:y + h, x:x + w]
                gris = gris[y:y + h, x:x + w]
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    idNum += 1
                    dir = f"copy_datos/{self.carta}"
                    print(f"{dir}/{self.carta}_{idNum}.jpg")
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    cv2.imwrite(f"{dir}/{self.carta}_{idNum}.jpg", new_img)
        return idNum, new_img

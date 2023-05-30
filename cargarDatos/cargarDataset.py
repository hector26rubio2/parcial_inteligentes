import os
import shutil

from sklearn.model_selection import train_test_split


class CargarDataSet:


    def cargarData(self):
        carpetas = [7, 8, 9, 10,11,12,13 ]
        dir = "datos/"
        newd = "dataset/"
        for i in carpetas:
            # The list of items
            files = os.listdir(f"{dir}{i}")

            xTrain, xTest = train_test_split(files, test_size=0.20, random_state=42)

            # Loop to print each filename separately
            for filename in xTrain:
                carp = f"{newd}train/{i}"
                if not os.path.exists(carp):
                    os.makedirs(carp)
                shutil.move(f"{dir}{i}/{filename}", f"{carp}/{filename}")
                # print(filename)

            for filename in xTest:
                carp = f"{newd}test/{i}"
                if not os.path.exists(carp):
                    os.makedirs(carp)
                shutil.move(f"{dir}{i}/{filename}", f"{carp}/{filename}")
            # print(filename)
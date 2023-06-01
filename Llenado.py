import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil

dir ="C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/datos/"
newd = "C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/dataset/"

for i in range(1,8):
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


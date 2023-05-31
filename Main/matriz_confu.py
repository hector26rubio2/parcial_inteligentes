from tensorflow.python.keras.models import load_model
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
sys.path.append("..") 
from Prediccion import Prediccion
## Matrices de confusión
width = 128
height = 128
num_class = 8
dirc = "C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/Crops/dataset/"
dir_root = "C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/Modelos/models/"
###  CAMBIAR RUTAS PARA MOSTRAR LOS OTROS MODELOS    ###
miModeloCNN = Prediccion(f"{dir_root}/modelo_2.h5", width, height)
imagenesPrueba,probabilidadesPrueba = miModeloCNN.cargarDatos( dirc+"test/", num_class, width, height)

model= load_model(f"{dir_root}/modelo_3.h5",custom_objects={"Addons>F1Score":tfa.metrics.F1Score(num_classes=2, average="micro")})
YPred= model.predict(imagenesPrueba)
yPred= np.argmax(YPred, axis=1)
MatrixConf= confusion_matrix( np.argmax(probabilidadesPrueba,axis=1),yPred)
print(MatrixConf)

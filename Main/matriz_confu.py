from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
sys.path.append("..") 
from Prediccion import Prediccion
import tensorflow.keras.optimizers as optimizers

## Matrices de confusión
width = 128
height = 128
num_class = 8
dirc = "C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/dataset/"
dir_root = "C:/Users/hecto/OneDrive/Documentos/GitHub/Parcial/Modelos/models/"
###  CAMBIAR RUTAS PARA MOSTRAR LOS OTROS MODELOS    ###
# Define and register the custom optimizer
def matriz_confusion(modelo):
  custom_objects = {"CustomAdam": optimizers.Adam}

  # Load the model with custom objects
  modelo_path = f"{dir_root}{modelo}"
  modeloCNN = Prediccion(modelo_path, 128, 128)  # Ca
  imagenesPrueba,probabilidadesPrueba = modeloCNN.cargarDatos( dirc+"test/", num_class, width, height)

  model= load_model(f"{dir_root}{modelo}", custom_objects={"Addons>F1Score": tfa.metrics.F1Score(num_classes=2, average="micro")})
  YPred= model.predict(imagenesPrueba)
  yPred= np.argmax(YPred, axis=1)
  MatrixConf= confusion_matrix( np.argmax(probabilidadesPrueba,axis=1),yPred)
  print(MatrixConf)


print("matriz confusion modelo 1")
matriz_confusion("modelo_1.h5")
print("\n--------------------\n")
print("matriz confusion modelo 2")
matriz_confusion("modelo_2.h5")
print("\n--------------------\n")
print("matriz confusion modelo 3")
matriz_confusion("modelo_3.h5")

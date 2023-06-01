# parcial_inteligentes

## link 
[![]()

## Dataset
- Entrenamiento: 'dataset/train'
  - categoria 7: 'dataset/train/1' 
  - categoria 8: 'dataset/train/2' 
  - categoria 9: 'dataset/train/3' 
  - categoria 10: 'dataset/train/4' 
  - categoria 11: 'dataset/train/5' 
  - categoria 12: 'dataset/train/6' 
  - categoria 13: 'dataset/train/7' 
- Prueba --> 'dataset/test'
  - categoria 7: 'dataset/test/1' 
  - categoria 8: 'dataset/test/2' 
  - categoria 9: 'dataset/test/3' 
  - categoria 10: 'dataset/test/4' 
  - categoria 11: 'dataset/test/5' 
  - categoria 12: 'dataset/test/6' 
  - categoria 13: 'dataset/test/7' 

## Matriz de confusion
### modelo1
![modelo 1](/img/modelo1.jpg)

### modelo2

![modelo 2](/img/model2.jpg)

### modelo3
![modelo 3](/img/modelo3.jpg)



## Metricas

![texto_alternativo](/img/tabla.jpg)
## Analisis comparativo
Basado en los resultados obtenidos, se recomienda seleccionar el Modelo2 como la elección final para su implementación en la solución. Este modelo ha demostrado un rendimiento sólido en todas las métricas evaluadas, con altos valores de exactitud, precisión, recall y F1 score, así como una pérdida muy baja. También se destaca que tiene un tiempo de respuesta más rápido en comparación con el Modelo1.

Según su rendimiento destacado en las métricas evaluadas, el Modelo2 se considera el candidato más sólido para la implementación. Sin embargo, debemos tener en cuenta los posibles escenarios en los que el modelo podría fallar y considerar estrategias para abordarlos, como la monitorización continua del rendimiento, la actualización del modelo con nuevos datos y la realización de pruebas rigurosas en diferentes condiciones.



### posibles escenarios de fallo
Debemos considerar diversos escenarios en los que el Modelo2 podría fallar:

Datos de prueba no representativos: Si el conjunto de datos de prueba difiere significativamente del conjunto de entrenamiento, el rendimiento del modelo puede verse afectado y dar lugar a predicciones incorrectas.

Cambios en la distribución de los datos: Si la distribución de los datos cambia en el entorno de producción en comparación con el conjunto de entrenamiento, el modelo puede tener dificultades para generalizar y exhibir un rendimiento inferior.

Datos atípicos o sesgados: Si el conjunto de datos contiene datos atípicos o está sesgado hacia una clase en particular, el modelo podría tener dificultades para realizar predicciones precisas en casos no previamente observados.

Errores en los datos de entrada: Si los datos de entrada contienen errores o ruido, el modelo puede verse afectado y generar predicciones incorrectas.

Cambios en el contexto o en las condiciones del problema: Si las condiciones del problema cambian en el entorno de producción, el modelo podría no ser adecuado para enfrentar los cambios y, como consecuencia, producir resultados inexactos.

#### Reconocimiento de máscaras


¡Este repositorio contiene una secuencia de comandos Python3 que reconoce si una cara lleva una máscara o no!

El código usa la biblioteca OpenCV para el procesamiento de imágenes y scikit-learn para entrenar el modelo que clasifica una cara que usa una máscara o no.

El conjunto de imágenes utilizadas para el entrenamiento de modelos se puede encontrar en la carpeta [imagenes] (./ imagenes)


#### Más sobre el código

En [abrir_cam.py] (./ abrir_cam.py) (con detección de rostros) y en [abrir_cam_sin_deteccion_rostro.py] (./abrir_cam_sin_deteccion_rostro.py) (sin detección de rostros) tenemos scripts que permiten que se inicie la cámara web de su computadora.

En estos scripts cargamos un marco de datos del conjunto de imágenes que tenemos y entrenamos un modelo [K-Nearest Neighbor]  para clasificar las caras.

Para detectar rostros, use el [CascadeClassifier], ya incluido en la biblioteca OpenCV. En general, este método de entrenamiento usa un archivo .xml, que también se incluye en el paquete, para entrenar un modelo que reconoce rostros de manera genérica, usando el método [Viola-Jones]  y [AdaBoost] 
El algoritmo de aprendizaje automático elegido para la clasificación fue el vecino más cercano K, ya que presentó el mejor rendimiento en comparación con el conjunto de prueba y validación. 


** Modo de detección de rostros y modo sin detección de rostros **
Debido a un problema delsobre la predilección a las máscaras de luz y también una dificultad con OpenCV para detectar caras cuando las sombras de las máscaras son más oscuras, se propuso esta división de modos. El modo [DETECCIÓN DE CARA] (abrir_cam.py) utiliza una solución OpenCV para identificar la cara en la imagen, mientras que el modo [SIN DETECCIÓN DE CARA] (abrir_cam_sin_deteccion_rostro.py) le pide al usuario que centre su cara en la imagen para que la clasificación se cumpla.

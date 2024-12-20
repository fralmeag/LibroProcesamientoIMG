"""
Este script realiza la segmentación de una imagen en el espacio 
de color CIELab para identificar y resaltar áreas de color amarillo. 
Primero, se carga una imagen y se reduce su tamaño para facilitar 
el procesamiento. 
Luego, se convierte la imagen de RGB a CIELab y se separan los 
canales L, a y b. 
Se definen umbrales específicos para cada canal para crear máscaras 
que identifican las áreas de interés. 
Finalmente, se combinan las máscaras y se visualizan los resultados, 
mostrando tanto las máscaras individuales como la imagen segmentada.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
from ip_functions import *

# Abrir imagen
RGB=plt.imread("cartagena.jpg")

#Realizar Slicing para reducir procesamiento
S=10
RGB=np.array(RGB[1::S,1::S,:])

"""#Extraer Capas Cielab"""

# RGB es la  imagen de entrada
LAB=rgb2lab(RGB)
# Separar los canales LAB
L, a, b= imsplit(LAB)

"""#Segmenta LAB"""

# Segmentar el color Amarillo en CIELab

# Definir umbrales para l (luminosidad)
UmbralInferiorL, UmbralSuperiorL = 0, 65
# Definir umbrales para a (componente a)
UmbralInferiorA, UmbralSuperiorA = -10, 25
# Definir umbrales para b (componente b)
UmbralInferiorB, UmbralSuperiorB = 40, 64


# Crear máscaras
MascaraL = (L >= UmbralInferiorL) & (L <= UmbralSuperiorL)
MascaraA = (a >= UmbralInferiorA) & (a <= UmbralSuperiorA)
MascaraB = (b >= UmbralInferiorB) & (b <= UmbralSuperiorB)

# Graficar resultados
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(L, cmap='gray')
plt.title('L (Luminosidad)')

plt.subplot(2, 3, 2)
plt.imshow(a, cmap='gray')
plt.title('a (Verde-Rojo)')

plt.subplot(2, 3, 3)
plt.imshow(b, cmap='gray')
plt.title('b (Azul-Amarillo)')

plt.subplot(2, 3, 4)
plt.imshow(MascaraL, cmap='gray')
plt.title('Máscara L')

plt.subplot(2, 3, 5)
plt.imshow(MascaraA, cmap='gray')
plt.title('Máscara a')

plt.subplot(2, 3, 6)
plt.imshow(MascaraB, cmap='gray')
plt.title('Máscara b')

plt.tight_layout()

# Combinar máscaras
MascaraFinal = MascaraL & MascaraA & MascaraB

"""#Imagen Segmentada en HSV"""

# Graficar resultados
plt.figure(2, figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.imshow(RGB)
plt.title('Imagen RGB Original')
plt.subplot(1, 3, 2)
plt.imshow(MascaraFinal, cmap='gray')
plt.title('Máscara Final')
plt.subplot(1, 3, 3)
plt.imshow(MascaraFinal[:,:,np.newaxis] * RGB)
plt.title('Imagen CieLAB Enmascarada')

plt.show()
"""
Este script realiza la segmentación de una imagen en el espacio de 
color HSV para identificar y resaltar áreas de color amarillo. 
Primero, se carga una imagen RGB y se reduce su tamaño para facilitar 
el procesamiento. Luego, se convierte la imagen al espacio de color 
HSV y se extraen las capas de tono (Hue), saturación (Saturation) y 
valor (Value). 
Se definen umbrales específicos para cada una de estas capas con el fin 
de crear máscaras que identifiquen las áreas de color amarillo. 
Finalmente, se combinan estas máscaras y se visualizan los resultados, 
mostrando tanto las capas individuales como la imagen segmentada.
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

"""#Extraer Capas HSV"""

# RGB es la  imagen de entrada
HSV=rgb2hsv(RGB)
h,s,v= imsplit(HSV)

"""#Segmenta HSV"""

#Segmentar el color Amarillo
# Definir umbrales para H (tono)
UmbralInferiorH, UmbralSuperiorH = 30/360, 55/360
# Definir umbrales para S (saturación)
UmbralInferiorS, UmbralSuperiorS = 0.8, 1
# Definir umbrales para V (valor)
UmbralInferiorV, UmbralSuperiorV = 0.7, 1

# Crear máscaras
MascaraH = (h >= UmbralInferiorH) & (h <= UmbralSuperiorH)
MascaraS = (s >= UmbralInferiorS) & (s <= UmbralSuperiorS)
MascaraV = (v >= UmbralInferiorV) & (v <= UmbralSuperiorV)

# Graficar resultados
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(h, cmap='gray')
plt.title('Hue (Tono)')

plt.subplot(2, 3, 2)
plt.imshow(s, cmap='gray')
plt.title('Saturation (Saturación)')

plt.subplot(2, 3, 3)
plt.imshow(v, cmap='gray')
plt.title('Value (Valor)')

plt.subplot(2, 3, 4)
plt.imshow(MascaraH, cmap='gray')
plt.title('Máscara Hue')

plt.subplot(2, 3, 5)
plt.imshow(MascaraS, cmap='gray')
plt.title('Máscara Saturation')

plt.subplot(2, 3, 6)
plt.imshow(MascaraV, cmap='gray')
plt.title('Máscara Value')

plt.tight_layout()

# Combinar máscaras
MascaraFinal = MascaraH & MascaraS & MascaraV

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
plt.title('Imagen HSV Enmascarada')

plt.show()
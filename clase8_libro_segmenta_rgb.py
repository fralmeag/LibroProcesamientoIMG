"""
Este script segmenta una imagen RGB para identificar y enmascarar 
áreas de color amarillo. 
El proceso incluye cargar la imagen, reducir su resolución, separar 
sus componentes de color, definir umbrales para el color amarillo, 
crear máscaras binarias para cada componente, visualizar las componentes 
y sus máscaras, combinar las máscaras para obtener una máscara final, 
y visualizar la imagen original, la máscara final y la imagen enmascarada.
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

"""#Extraer Capas RGB"""

# RGB es la  imagen de entrada
r, g, b = imsplit(RGB)

"""#Segmenta RGB"""

#Segmentar ewl color Amarillo

# Rojo
# UmbralInferiorR, UmbralSuperiorR = 75, 255
# UmbralInferiorG, UmbralSuperiorG = 0, 25
# UmbralInferiorB, UmbralSuperiorB = 0, 75

# Azul
# UmbralInferiorR, UmbralSuperiorR = 0, 40
# UmbralInferiorG, UmbralSuperiorG = 0, 80
# UmbralInferiorB, UmbralSuperiorB = 40, 225

# Amarillo
UmbralInferiorR, UmbralSuperiorR = 75, 185
UmbralInferiorG, UmbralSuperiorG = 75, 185
UmbralInferiorB, UmbralSuperiorB = 0, 40

# Crear máscaras
MascaraR = (r >= UmbralInferiorR) & (r <= UmbralSuperiorR)
MascaraG = (g >= UmbralInferiorG) & (g <= UmbralSuperiorG)
MascaraB = (b >= UmbralInferiorB) & (b <= UmbralSuperiorB)

# Graficar resultados
plt.figure(2, figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(r, cmap='gray')
plt.title('Rojo')

plt.subplot(2, 3, 2)
plt.imshow(g, cmap='gray')
plt.title('Verde')

plt.subplot(2, 3, 3)
plt.imshow(b, cmap='gray')
plt.title('Azul')

plt.subplot(2, 3, 4)
plt.imshow(MascaraR, cmap='gray')
plt.title('Máscara Roja')

plt.subplot(2, 3, 5)
plt.imshow(MascaraG, cmap='gray')
plt.title('Máscara Verde')

plt.subplot(2, 3, 6)
plt.imshow(MascaraB, cmap='gray')
plt.title('Máscara Azul')

plt.tight_layout()

# Combinar máscaras
MascaraFinal = MascaraR & MascaraG & MascaraB

"""#Imagen Segmentada en RGB"""

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
plt.title('Imagen RGB Enmascarada')

plt.show()
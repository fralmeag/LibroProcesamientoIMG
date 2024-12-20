"""
Este script realiza la ecualización del histograma de una imagen de 
entrada utilizando una imagen de referencia.
Se importan las bibliotecas necesarias, se cargan las imágenes y se 
seleccionan las capas rojas de cada una.
Luego, se realiza un slicing en la imagen de entrada para reducir 
el procesamiento.
Se muestran las imágenes y sus histogramas antes y después de la 
ecualización.
"""
#Importación de Liberia Básicas
import matplotlib.pyplot as plt
import numpy as np
import sys
from ip_functions import *

"""#Cargar las Imágenes"""

# Abrir Imagen
RGBr=plt.imread("negativo_chica.jpg")
RGBr=255-np.array(RGBr);
#Seleccionar la capa Roja Positiva
ref=RGBr[:,:,0]
RGBe=plt.imread("cartagena.jpg")
#Seleccionar la capa Roja Positiva
rent=RGBe[:,:,0]


#Realizar Slicing para reducir procesamiento
S=15
rent=np.array(rent[1::S,1::S])

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
plt.imshow(ref, cmap='gray')
plt.axis('off')
plt.title('Referencia')

plt.subplot(2,2,2)
plt.imshow(rent, cmap='gray')
plt.axis('off')
plt.title('Entrada')

ax3 = plt.subplot(2,2,3)
imhist(ref, ax=ax3)


ax4 = plt.subplot(2,2,4)
imhist(rent, ax=ax4)


plt.tight_layout()
plt.show()

href=imhist(ref, None, False)
Is=histeq(rent, href)

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
plt.imshow(ref, cmap='gray')
plt.axis('off')
plt.title('Referencia')

plt.subplot(2,2,2)
plt.imshow(Is, cmap='gray')
plt.axis('off')
plt.title('Salida')

ax3 = plt.subplot(2,2,3)
imhist(ref, ax=ax3)


ax4 = plt.subplot(2,2,4)
imhist(Is, ax=ax4)


plt.tight_layout()
plt.show()
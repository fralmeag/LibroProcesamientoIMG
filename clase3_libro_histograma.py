"""
Este código carga una imagen, reduce su tamaño para facilitar el 
procesamiento, separa la imagen en sus componentes de color rojo, 
verde y azul, y luego muestra tanto las imágenes de cada componente 
de color como sus histogramas correspondientes.
"""
#Importación de Liberia Básicas
import matplotlib.pyplot as plt
import numpy as np
import sys
from ip_functions import *

# Abrir  imagen
RGB=plt.imread("cartagena.jpg")
RGB=np.array(RGB)
#Realizar Slicing para reducir procesamiento
S=5
RGB=np.array(RGB[1::S,1::S,:])
#Seleccionar la capas
r=RGB[:,:,0]
g=RGB[:,:,1]
b=RGB[:,:,2]

plt.figure(figsize=(15,15))

plt.subplot(2,3,1)
plt.imshow(r, cmap='gray')
plt.axis('off')
plt.title("Rojo")

plt.subplot(2,3,2)
plt.imshow(g, cmap='gray')
plt.axis('off')
plt.title("Verde")

plt.subplot(2,3,3)
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.title("Azul")

ax4 = plt.subplot(2,3,4)
imhist(r, ax=ax4)

ax5 = plt.subplot(2,3,5)
imhist(g, ax=ax5)

ax6 = plt.subplot(2,3,6)
imhist(b, ax=ax6)

plt.tight_layout()
plt.show()


"""
Este script crea una imagen RGB de 3x3 píxeles utilizando matrices NumPy 
y la visualiza con Matplotlib.
Cada canal de color (rojo, verde y azul) se define por separado y se 
combinan para formar la imagen final.
"""

import matplotlib.pyplot as plt
import numpy as np

#tamaño de las matrices a visualizar
size=(3,3,3)
#Una matriz de ceros 3x3
RGB=np.zeros(size)
RGB=np.uint8(RGB)

#Asignar Valores a cada pixel
RGB[:,:,0]=([[0,255,255],[255,128,0],[255,0,0]])
RGB[:,:,1]=([[255,255,0],[0,128,255],[255,0,0]])
RGB[:,:,2]=([[255,255,0],[255,128,0],[0,0,255]])
plt.figure(figsize=(5,5))
plt.imshow(RGB)
plt.show()
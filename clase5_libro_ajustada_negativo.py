"""
Este script realiza el procesamiento de una imagen para ajustar 
su histograma. 
El proceso incluye la carga de una imagen, la reducción de su 
tamaño mediante slicing, la inversión de colores para obtener 
un negativo, y la separación de las capas de color (rojo, verde 
y azul). Luego, se ajustan los histogramas de cada capa de color 
utilizando la función `imadjust` y se muestran las imágenes y sus 
histogramas antes y después del ajuste. 
Finalmente, se compone una nueva imagen con las capas ajustadas 
y se visualiza junto con 
la imagen original y su negativo.
"""
#Importación de Liberia Básicas
import matplotlib.pyplot as plt
import numpy as np
import sys
from ip_functions import *

# Abrir Imagen
rgb_negativo=plt.imread("fam2.jpg")

#Realizar Slicing para reducir procesamiento
S=10
rgb_negativo=np.array(rgb_negativo[1::S,1::S,:])


rgb=255-np.array(rgb_negativo);
#Seleccionar Capas
r=np.array(rgb[:,:,0])
g=np.array(rgb[:,:,1])
b=np.array(rgb[:,:,2])

# r=non_overflowing_sum(r,100)

plt.figure(figsize=(15,5))
fig, axs = plt.subplots(2, 3, figsize=(15, 15))

axs[0, 0].imshow(r, cmap='gray')
axs[0, 0].axis('off')
axs[0, 0].set_title("Rojo")

axs[0, 1].imshow(g, cmap='gray')
axs[0, 1].axis('off')
axs[0, 1].set_title("Verde")

axs[0, 2].imshow(b, cmap='gray')
axs[0, 2].axis('off')
axs[0, 2].set_title("Azul")

imhist(r, ax=axs[1, 0])
axs[1, 0].set_title("Histograma Rojo")

imhist(g, ax=axs[1, 1])
axs[1, 1].set_title("Histograma Verde")

imhist(b, ax=axs[1, 2])
axs[1, 2].set_title("Histograma Azul")

plt.tight_layout()
#plt.show()

"""Ajuste del Histograma"""

#Capa Roja
#Emr=0/255
#EMr=205/
Emr,EMr=stretchlim(r);
Smr=0/255
SMr=255/255

n=1 #El mismo para todas las capas

#Capa Verde
#Emg=140/255
#EMg=250/255
Emg,EMg=stretchlim(g);
Smg=0/255
SMg=255/255

#Capa VerAzulde
#Emb=190/255
#EMb=255/255
Emb,EMb=stretchlim(b);
Smb=0/255
SMb=255/255

Isr=imadjust(r,[Emr,EMr],[Smr,SMr],n)
Isg=imadjust(g,[Emg,EMg],[Smg,SMg],n)
Isb=imadjust(b,[Emb,EMb],[Smb,SMb],n)

"""Gráficas Ajustadas de Salida"""

plt.figure(figsize=(15,15))
plt.subplot(2,3,1)
plt.imshow(Isr,cmap='gray')
plt.axis('off')
plt.title("Rojo")
plt.subplot(2,3,2)
plt.imshow(Isg,cmap='gray')
plt.axis('off')
plt.title("Verde")
plt.subplot(2,3,3)
plt.imshow(Isb,cmap='gray')
plt.axis('off')
plt.title("Azul")

ax4=plt.subplot(2,3,4)
imhist(Isr,ax=ax4)
plt.title("Rojo")
ax5=plt.subplot(2,3,5)
imhist(Isg,ax=ax5)
plt.title("Verde")
ax6=plt.subplot(2,3,6)
imhist(Isb,ax=ax6)
plt.title("Azul")

"""Histogramas Ajustados de Salida

Imagen Compuesta
"""

[F,C,L]=np.shape(rgb_negativo)
rgbn=np.uint8(np.zeros([F,C,L]))
rgbn[:,:,0]=Isr
rgbn[:,:,1]=Isg
rgbn[:,:,2]=Isb

plt.figure(figsize=(18,15))
plt.subplot(1,3,1)
plt.imshow(rgb_negativo)
plt.title('Negativo')
plt.subplot(1,3,2)
plt.imshow(rgb)
plt.title('Positivo')
plt.subplot(1,3,3)
plt.imshow(rgbn)
plt.title('Ajustada')
plt.axis('off')

plt.show()
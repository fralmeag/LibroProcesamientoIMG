"""
Este script realiza el procesamiento de imágenes para ajustar los 
histogramas de las capas de color rojo, verde y azul de una imagen. 
Utiliza bibliotecas como matplotlib y numpy para la manipulación y 
visualización de imágenes. 
El proceso incluye la lectura de imágenes, reducción de tamaño para 
optimizar el procesamiento, selección de capas de color, visualización 
de histogramas y ajuste de histogramas para mejorar la calidad de la imagen.
"""
#Importación de Liberia Básicas
import matplotlib.pyplot as plt
import numpy as np
import sys
from ip_functions import *

# Abrir Imagen
Color=plt.imread("ColorA.jpg")
R=plt.imread("RojoA.jpg")
G=plt.imread("VerdeA.jpg")
B=plt.imread("AzulA.jpg")

#Realizar Slicing para reducir procesamiento
S=10
Color=np.array(Color[1::S,1::S,:])
R=np.array(R[1::S,1::S,:])
G=np.array(G[1::S,1::S,:])
B=np.array(B[1::S,1::S,:])

#Seleccionar la capa Roja Positiva
r=np.array(R[:,:,0])
g=np.array(G[:,:,1])
b=np.array(B[:,:,2])

#Mostar las Capas
plt.figure(figsize=(15,15))
plt.subplot(2,3,1)
plt.imshow(r,cmap='gray')
plt.axis('off')
plt.title("Rojo")
plt.subplot(2,3,2)
plt.imshow(g,cmap='gray')
plt.axis('off')
plt.title("Verde")
plt.subplot(2,3,3)
plt.imshow(b,cmap='gray')
plt.axis('off')
plt.title("Azul")

ax4=plt.subplot(2,3,4)
imhist(r,ax=ax4)
plt.title("Rojo")
ax5=plt.subplot(2,3,5)
imhist(g,ax=ax5)
plt.title("Verde")
ax6=plt.subplot(2,3,6)
imhist(r,ax=ax6)
plt.title("Azul")

#Ajuste del Histograma

#Capa Roja
#Emr=25/255
#EMr=80/255
Emr,EMr=stretchlim(r);
Smr=0/255
SMr=255/255

n=1 #El mismo para todas las capas

#Capa Verde
#Emg=25/255
#EMg=140/255
Emg,EMg=stretchlim(g);
Smg=0/255
SMg=255/255

#Capa VerAzulde
#Emb=25/255
#EMb=80/255
Emb,EMb=stretchlim(b);
Smb=0/255
SMb=255/255

Isr=imadjust(r,[Emr,EMr],[Smr,SMr],n)
Isg=imadjust(g,[Emg,EMg],[Smg,SMg],n)
Isb=imadjust(b,[Emb,EMb],[Smb,SMb],n)

#Gráficas Ajustadas de Salida
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

#Histogramas Ajustados de Salida

#Imágenes de Referencia, Compuesta y Ajusta
In=np.uint8(np.stack((r,g,b),axis=-1))
Ia=np.uint8(np.stack((Isr,Isg,Isb),axis=-1))
plt.figure(figsize=(18,15))
plt.subplot(1,3,1)
plt.imshow(Color)
plt.title('Referencia')
plt.subplot(1,3,2)
plt.imshow(In)
plt.title('Compuesta')
plt.subplot(1,3,3)
plt.imshow(Ia)
plt.title('Ajustada')
plt.axis('off')

plt.show()
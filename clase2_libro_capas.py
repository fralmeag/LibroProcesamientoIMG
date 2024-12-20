"""
Este script carga una imagen y extrae sus capas de color rojo, verde y azul.
Luego, muestra las capas individuales en escala de grises y en color utilizando 
subplots.
"""
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    # Abrir imagen
    RGB=plt.imread("cartagena.jpg")

    """Extraer Capas"""

    r=RGB[:,:,0]
    g=RGB[:,:,1]
    b=RGB[:,:,2]

    """Imágenes con Subplot Gris"""

    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(RGB)
    plt.title("Color")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(r,cmap='gray')
    plt.title("Rojo")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(g,cmap='gray')
    plt.title("Verde")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(b,cmap='gray')
    plt.title("Azul")
    plt.axis('off')

    """Imágenes con Subplot Color"""

    R=np.array(RGB)
    R[:,:,1]=0
    R[:,:,2]=0

    G=np.array(RGB)
    G[:,:,0]=0
    G[:,:,2]=0

    B=np.array(RGB)
    B[:,:,0]=0
    B[:,:,1]=0

    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(RGB)
    plt.title("Color")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(R)
    plt.title("Rojo")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(G)
    plt.title("Verde")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(B)
    plt.title("Azul")
    plt.axis('off')

    plt.show()
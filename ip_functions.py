import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import random

#----------------------------------------------------------------------
def imhist(r, ax=None, ver=True):
    """
    Calcula y muestra el histograma de una imagen en escala de grises.

    Parámetros:
    r (numpy.ndarray): Imagen en escala de grises representada como 
        un arreglo 2D de numpy.
    ax (matplotlib.axes.Axes, opcional): Eje de Matplotlib donde se 
        dibujará el histograma. Si no se proporciona, se usará el eje actual.
    ver (bool, opcional): Si es True, se muestra el histograma usando Matplotlib. 
        Si es False, se devuelve el histograma como un arreglo.

    Retorna:
    numpy.ndarray: Si `ver` es False, retorna un arreglo 1D de numpy con el 
        histograma de la imagen. Si `ver` es True, no retorna nada.

    Ejemplo de uso:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ip_functions import imhist
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> imhist(img)
    """
    h = np.zeros([256, 1])
    i, j = r.shape
    for x in range(i):
        for y in range(j):
            h[r[x, y]] += 1
    
    if ver:
        x = range(256)
        hbar = np.reshape(h, 256)
        if ax is None:
            ax = plt.gca()
        ax.bar(x, hbar)
        norm = mpl.colors.Normalize(vmin=0, vmax=255)
        escala = plt.cm.ScalarMappable(cmap='gray', norm=norm)
        escala.set_array([])
        plt.colorbar(escala, ax=ax, orientation="horizontal", ticks=[0, 50, 100, 150, 200, 255])
        ax.set_xlabel("Intensidades")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Histograma")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, np.amax(h) * 0.3)
        ax.grid(True)
    else:
        return h

#---------------------------------------------------------
def stretchlim(I,Tol=0.01):
    """
    Calcula los límites de estiramiento de contraste para una imagen en escala de grises.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    Tol (float, opcional): Tolerancia para el cálculo de los límites. Debe estar en el rango [0, 1]. El valor por defecto es 0.01.

    Retorna:
    tuple: Una tupla (Em, EM) donde Em es el límite inferior y EM es el límite superior, ambos normalizados en el rango [0, 1].

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import stretchlim
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> Em, EM = stretchlim(img)
    >>> print(Em, EM)
    """        
    Em=0
    EM=255
    h=imhist(I,None,False)
    i,j=np.shape(I)
    ha=np.zeros([256,1])
    hp=np.zeros([256,1])
    L=256
    for k in range(L):
        ha[k]=np.sum(h[0:k+1])
        hp[k]=ha[k]/(i*j)
        if hp[k]<=Tol:
             Em=k
        if hp[k]<=1-Tol:
             EM=k              
    return Em/255,EM/255    
#---------------------------------------------------------    
def imadjust(I,E,S=(0,1),n=1):
    """
    Ajusta la intensidad de una imagen en escala de grises según los límites especificados.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    E (tuple): Tupla (Em, EM) que representa los límites de entrada normalizados en el rango [0, 1].
    S (tuple, opcional): Tupla (Sm, SM) que representa los límites de salida normalizados en el rango [0, 1]. El valor por defecto es (0, 1).
    n (int, opcional): Exponente para el ajuste de intensidad. El valor por defecto es 1.

    Retorna:
    numpy.ndarray: Imagen ajustada con las nuevas intensidades.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imadjust
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> Em, EM = 0.2, 0.8
    >>> adjusted_img = imadjust(img, (Em, EM))
    >>> print(adjusted_img)
    """
    Em=E[0]*255
    EM=E[1]*255
    Sm=S[0]*255
    SM=S[1]*255
    I=np.float16(I)
    Is=((SM-Sm)/(EM-Em)**n)*(np.absolute(I-Em))**n+Sm
    #Ajusta el overflow de los valores
    Is[np.where(Is>255)] = 255
    return np.uint8(Is)
        
#---------------------------------------------------------        
def histeq(I, hR=None):
    """
    Realiza la ecualización del histograma de una imagen en escala de grises.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    hR (numpy.ndarray, opcional): Histograma de referencia. Si se proporciona, la ecualización se realizará en función de este histograma. Si no se proporciona, se realizará una ecualización normal.

    Retorna:
    numpy.ndarray: Imagen con el histograma ecualizado.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import histeq
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> eq_img = histeq(img)
    >>> print(eq_img)
    """  

    # Determinar el histograma de la imagen de entrada I
    hI= imhist(I,None,False)
        
    # Calcula la CDF de la imagen de entrada I
    cdfI = np.cumsum(hI) / np.sum(hI)
        
    if hR is None:
        # Se realiza la ecualización normal
        LUT = np.uint8(255 * cdfI)
    else:
        # Se hace la especificación del histograma
        # Calcula la CDF de la imagen de referencia
        cdfR = np.cumsum(hR) / np.sum(hR)
            
        # Tabla de búsqueda (LUT) por proximidad
        LUT = np.zeros(256, dtype=np.uint8)
        for idx in range(256):
            minIndex = np.argmin(np.abs(cdfR - cdfI[idx]))
            LUT[idx] = minIndex  # Se indexa desde 0
        
    # Aplica la LUT a toda la imagen de entrada usando indexación directa
    S = LUT[I]
        
    return S

#---------------------------------------------------------
def imsplit(I):
    """
    Divide una imagen en sus canales RGB.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, 
    donde la tercera dimensión corresponde a los canales de color (RGB).

    Retorna:
    tuple: Una tupla (r, g, b) donde r, g y b son arreglos 2D de numpy 
            que representan los canales rojo, verde y azul de la imagen, 
            respectivamente.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imsplit
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> r, g, b = imsplit(img)
    >>> print(r.shape, g.shape, b.shape)
            (100, 100) (100, 100) (100, 100)
    """
    r=np.array(I[:,:,0])
    g=np.array(I[:,:,1])
    b=np.array(I[:,:,2])
    return r,g,b

#---------------------------------------------------------
def rgb2gray(RGB):
    """
    Convierte una imagen RGB a escala de grises utilizando la fórmula de luminancia.

    Parámetros      :
    RGB (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (RGB).

    Retorna:
    numpy.ndarray: Imagen en escala de grises representada como un arreglo 2D de numpy.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import rgb2gray
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> gray_img = rgb2gray(img)
    >>> print(gray_img.shape)
    (100, 100)
    """
    r,g,b=imsplit(RGB)
    gris=np.uint8(0.299*np.double(r)+0.587*np.double(g)+0.114*np.double(b))
    return gris

#---------------------------------------------------------
def non_overflowing_sum(a,b):
    """
    Realiza la suma de dos arreglos de numpy sin desbordamiento, limitando los valores resultantes al rango [0, 255].

    Parámetros:
    a (numpy.ndarray): Primer arreglo de entrada.
    b (numpy.ndarray): Segundo arreglo de entrada.

    Retorna:
    numpy.ndarray: Arreglo resultante de la suma de `a` y `b`, con valores limitados al rango [0, 255].

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import non_overflowing_sum
    >>> a = np.array([100, 150, 200], dtype=np.uint8)
    >>> b = np.array([100, 150, 200], dtype=np.uint8)
    >>> result = non_overflowing_sum(a, b)
    >>> print(result)
    [200 255 255]
    """
    c = np.uint16(a)+b
    c[np.where(c>255)] = 255
    c[np.where(c<0)] = 0
    return np.uint8(c)

#---------------------------------------------------------
def graythresh(I):
    """
    Calcula un umbral global para convertir una imagen en escala de grises a una imagen binaria utilizando el método de Otsu.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.

    Retorna:
    float: Umbral calculado en el rango [0, 1].

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import graythresh
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> threshold = graythresh(img)
    >>> print(threshold)
    """
    h=imhist(I,None,False)
    return otsuthresh(h)

#---------------------------------------------------------
def otsuthresh(h):
    """
    Calcula el umbral óptimo de Otsu para una distribución de intensidades dada.
    
    Parámetros:
    h (numpy.ndarray): Histograma de intensidades de la imagen.
    
    Retorna:
    float: Umbral de Otsu normalizado en el rango [0, 1].

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import otsuthresh
    >>> hist = np.random.randint(0, 100, 256)
    >>> threshold = otsuthresh(hist)
    >>> print(threshold)
    """
    lh=len(h)
    tam=np.sum(h)
    maxV=0
    
    for T in np.arange(0,lh,1):
         Wb=np.sum(h[0:T])/tam
         Acub=np.sum(h[0:T])
         Acuf=np.sum(h[T+1:lh])
         if  Acub==0:
             Ub=0
         else:
             Ub=np.dot(np.arange(0,T,1),h[0:T])/Acub
         if Acuf==0:
             Uf=0
         else:
             Uf=np.dot(np.arange(T+1,lh,1),h[T+1:lh])/Acuf
         Wf=1-Wb
         BCV=Wb*Wf*(Ub-Uf)**2;
         
         if BCV>=maxV:
                maxV=BCV
                umbral=(T+1)/255
    return umbral

#---------------------------------------------------------
def im2bw(I, threshold):
    """
    Convierte una imagen en escala de grises a una imagen binaria utilizando un umbral especificado o calculado automáticamente.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    threshold (float, opcional): Umbral para la binarización. Si no se proporciona, se calculará automáticamente utilizando el método de Otsu.

    Retorna:
    numpy.ndarray: Imagen binaria representada como un arreglo 2D de numpy con valores 0 y 1.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import im2bw
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> binary_img = im2bw(img)
    >>> print(binary_img)
    """
    # Convierte la imagen I a binaria usando el umbral especificado
    return (I >= threshold * 255).astype(np.uint8)

#---------------------------------------------------------
def adaptthresh(I, P=None, V=None):
    """
    Calcula un umbral adaptativo para cada píxel de una imagen en escala de grises utilizando el método de la media ponderada de los vecinos.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    P (float, opcional): Sensibilidad del umbral adaptativo. Debe estar en el rango [0, 1]. El valor por defecto es 0.5.
    V (tuple, opcional): Tamaño de la ventana de vecindario para el cálculo del umbral adaptativo. El valor por defecto es 2 veces el tamaño de la imagen dividido por 16.

    Retorna:
    numpy.ndarray: Imagen binaria representada como un arreglo 2D de numpy con valores 0 y 1.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import adaptthresh
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> binary_img = adaptthresh(img)
    >>> print(binary_img)
    """
    # Valores por defecto
    S = 0.5  # Sensibilidad por defecto
    W = 2 * np.floor(np.array(I.shape) / 16).astype(int) + 1  # Tamaño de ventana por defecto

    # Si no se proporciona V, usar la fórmula por defecto
    if V is None:
        V = W

    # Si no se proporciona P, usar el valor por defecto
    if P is None:
        P = S

    # Parámetros de la ventana
    Tx, Ty = V
    Finix = (Tx + 1) // 2
    Ciniy = (Ty + 1) // 2
    Ffinx = Finix - 1
    Cfiny = Ciniy - 1

    # Preparar la imagen con padarray replicado
    Io = I.copy()
    I_padded = np.pad(Io, ((Ffinx, Ffinx), (Cfiny, Cfiny)), mode='edge')
    F, C = Io.shape
    T = np.zeros((F, C))

    # Calcular el umbral local
    for i in range(Finix, F - Ffinx):
        for j in range(Ciniy, C - Cfiny):
            # Definir límites del vecindario
            start_i = i - Ffinx
            end_i = i + Ffinx + 1
            start_j = j - Cfiny
            end_j = j + Cfiny + 1

            # Extraer el vecindario y calcular el umbral local
            W = I_padded[start_i:end_i, start_j:end_j]
            T[i, j] = np.mean(W) * (1 - P)  # Aplicar la sensibilidad P al cálculo del umbral


    return T/255

#---------------------------------------------------------
def imbinarize(I, *args):
    """
    Convierte una imagen en escala de grises a una imagen binaria utilizando un umbral especificado o calculado automáticamente.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    *args: Argumentos adicionales para especificar el umbral o el método de binarización. Puede ser un umbral específico, el modo de binarización ('global' o 'adaptive') y argumentos adicionales para el modo adaptativo.

    Retorna:
    numpy.ndarray: Imagen binaria representada como un arreglo 2D de numpy con valores 0 y 1.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imbinarize
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> binary_img = imbinarize(img, 128)  # Usar un umbral específico
    >>> print(binary_img)
    >>> binary_img_auto = imbinarize(img)  # Usar umbral automático
    >>> print(binary_img_auto)
    """
    # Comprobar si el primer argumento es un escalar (umbral)
    if len(args) == 1 and isinstance(args[0], (int, float, np.ndarray)):
        T = args[0]
        return im2bw(I, T)
    
    mode = args[0].lower() if len(args) > 0 and isinstance(args[0], str) else 'global'
    kwargs = dict(zip(args[1::2], args[2::2]))  # Extraer argumentos adicionales
    
    # Convertir todas las claves de kwargs a minúsculas
    kwargs = {k.lower(): v for k, v in kwargs.items()}

    if mode == 'global':
        # Umbral global utilizando el método de Otsu
        Thres = graythresh(I)
        bw = im2bw(I, Thres)
    elif mode == 'adaptive':
        # Umbral adaptativo utilizando el método de Bradley
        sensitivity = kwargs.get('sensitivity', 0.5)  # Usar 0.5 si no se proporciona
        window_size = kwargs.get('windowsize', None)
        Tadap = adaptthresh(I, P=sensitivity, V=window_size)
        bw = im2bw(I, Tadap)
    else:
        raise ValueError("Modo no válido. Debe ser 'global', 'adaptive' o proporcionar el umbral T.")
    
    return bw

#---------------------------------------------------------
def immse(Iref, I):
    """
    Calcula el error cuadrático medio (MSE) entre dos imágenes.

    Parámetros:
    I1 (numpy.ndarray): Primera imagen representada como un arreglo 2D de numpy.
    I2 (numpy.ndarray): Segunda imagen representada como un arreglo 2D de numpy.

    Retorna:
    float: El error cuadrático medio entre las dos imágenes.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import immse
    >>> img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> mse = immse(img1, img2)
    >>> print(mse)
    """
    fil,col=np.shape(Iref)
    I1=np.array(Iref,dtype=float)
    I2=np.array(I,dtype=float)
    MSE=np.sum((I1-I2)**2)/(fil*col)
    return MSE 
#---------------------------------------------------------
def psnr(Iref, I):
    """
    Calcula el error cuadrático medio (MSE) entre dos imágenes.

    Parámetros:
    Iref (numpy.ndarray): Primera imagen representada como un arreglo 2D de numpy.
    I (numpy.ndarray): Segunda imagen representada como un arreglo 2D de numpy.

    Retorna:
    float: El error cuadrático medio entre las dos imágenes.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import immse
    >>> img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> mse = immse(img1, img2)
    >>> print(mse)
    """    
    I2=np.array(I,dtype=float)
    fil,col=np.shape(Iref)
    A=immse(Iref, I)
    psnr=10*np.log10(255**2/A)
    Mean_noise=np.mean(I2**2)
    snr=10*np.log10(Mean_noise/A)
    return psnr,snr

#---------------------------------------------------------
def imnoise(I,tipo='gaussian',P=0.5,sigma=0.1):
    """
    Añade ruido a una imagen en escala de grises.

    Parámetros:
    I (numpy.ndarray): Imagen en escala de grises representada como un arreglo 2D de numpy.
    tipo (str, opcional): Tipo de ruido a añadir. Puede ser 'salt & pepper', 'gaussian' o 'speckle'. El valor por defecto es 'gaussian'.
    P (float, opcional): Probabilidad de ruido para el ruido de sal y pimienta. El valor por defecto es 0.5.
    sigma (float, opcional): Desviación estándar para el ruido gaussiano o speckle. El valor por defecto es 0.1.

    Retorna:
    numpy.ndarray: Imagen con ruido añadido.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imnoise
    >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    >>> noisy_img = imnoise(img, mode='gaussian', mean=0, var=0.01)
    >>> print(noisy_img)
    """      
    mu=0
    fil,col=np.shape(I)
    pnt=int(fil*col*P)
    tipo=tipo.lower()
    if tipo=='salt & pepper':
        Ir=np.copy(I)
        for i in range(pnt):      
            x=random.randint(0, fil-1)
            y=random.randint(0, col-1)
            Ir[x,y]=255*random.randint(0,1)
            Ir=np.uint8(Ir)
              
    elif tipo=='gaussian':
        mu=P
        s = np.random.normal(mu, sigma*255, (fil,col))
        Ir=non_overflowing_sum(I,s)
        
    elif tipo=='speckle':
        sigma=P
        s = np.random.normal(mu, sigma, (fil,col))
        Ir=non_overflowing_sum(I,s*I)
        Ir=np.uint8(Ir)
    else:
        Ir=np.copy(I)
    
    return Ir


#---------------------------------------------------------

def rgb2hsv(I):
    """
    Convierte una imagen de color RGB a una imagen en el espacio de color HSV.
    
    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (RGB).
    
    Retorna:
    numpy.ndarray: Imagen en el espacio de color HSV representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (HSV).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import rgb2hsv
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> hsv_img = rgb2hsv(img)
    >>> print(hsv_img.shape)
    (100, 100, 3)
    """    
    I=I/255
    r,g,b=imsplit(I)
    [fil,col,pro]=np.shape(I)
    Cmax=np.zeros((fil,col))
    Cmin=np.zeros((fil,col))
    d=np.zeros((fil,col))
    H=np.zeros((fil,col))
    S=np.zeros((fil,col))
    V=np.zeros((fil,col))
    for i in range(fil):
        for j in range(col):
            maximo=max([r[i,j],g[i,j],b[i,j]])
            minimo=min([r[i,j],g[i,j],b[i,j]])
            Cmax[i,j]=maximo
            Cmin[i,j]=minimo
            d[i,j]=maximo-minimo
            if d[i,j]==0:
                H[i,j]=0
            elif maximo==r[i,j]:
                H[i,j]=60*(((g[i,j]-b[i,j])/d[i,j])%6)
            elif maximo==g[i,j]:
                H[i,j]=60*(((g[i,j]-b[i,j])/d[i,j])+2)
            elif maximo==b[i,j]:
                H[i,j]=60*(((g[i,j]-b[i,j])/d[i,j])+4)
            if maximo==0:
                S[i,j]=0
            else:
                S[i,j]=d[i,j]/maximo
            V[i,j]=maximo
    H=H/360.0
    hsv=np.dstack((H,S,V))
    return hsv

#---------------------------------------------------------

def hsv2rgb(H):
    """
    Convierte una imagen en el espacio de color HSV a una imagen de color RGB.

    Parámetros:
    H (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (HSV).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color RGB representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (RGB).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import hsv2rgb
    >>> hsv_img = np.random.rand(100, 100, 3)  # Valores en el rango [0, 1] para H, S, V
    >>> rgb_img = hsv2rgb(hsv_img)
    >>> print(rgb_img.shape)
    (100, 100, 3)
    """    
    h,s,v=imsplit(H)
    N,M,L=np.shape(H)
    X=np.zeros((N,M))
    m=np.zeros((N,M))
    r=np.zeros((N,M))
    g=np.zeros((N,M))
    b=np.zeros((N,M))
    h=h*360
    C=np.zeros((N,M)) #s*v
    for i in range(N):
        for j in range(M):
            C[i,j]=v[i,j]*s[i,j]
            X[i,j]=C[i,j]*(1-abs((h[i,j]/60)%2-1))
            m[i,j]=v[i,j]-C[i,j]
            if 0<=h[i,j] and h[i,j]<60:
                r[i,j],g[i,j],b[i,j]=C[i,j],X[i,j],0
            elif 60<=h[i,j] and h[i,j]<120:
                r[i,j],g[i,j],b[i,j]=X[i,j],C[i,j],0
            elif 120<=h[i,j] and h[i,j]<180:
                r[i,j],g[i,j],b[i,j]=0,C[i,j],X[i,j]
            elif 180<=h[i,j] and h[i,j]<240:
                r[i,j],g[i,j],b[i,j]=0,X[i,j],C[i,j]
            elif 240<=h[i,j] and h[i,j]<300:
                r[i,j],g[i,j],b[i,j]=X[i,j],0,C[i,j]
            elif 300<=h[i,j] and h[i,j]<360:
                r[i,j],g[i,j],b[i,j]=C[i,j],0,X[i,j]
    R,G,B=255*(r+m),255*(g+m),255*(b+m)
    RGB=np.dstack((R,G,B))
    RGB=np.uint8(RGB)
    return RGB
  
#---------------------------------------------------------
def xyz2lab(myXYZ):
    """
    Convierte una imagen en el espacio de color XYZ a una imagen en el espacio de color CIELAB.

    Parámetros:
    myXYZ (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (XYZ).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color CIELAB representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (L, a, b).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import xyz2lab
    >>> xyz_img = np.random.rand(100, 100, 3)  # Valores en el rango [0, 1] para X, Y, Z
    >>> lab_img = xyz2lab(xyz_img)
    >>> print(lab_img.shape)
    (100, 100, 3)
    """
    F, C, L = myXYZ.shape
    if L != 3:
        raise ValueError('xyz2lab: La imagen debe ser MxNx3')
    
    X = myXYZ[:,:,0]
    Y = myXYZ[:,:,1]
    Z = myXYZ[:,:,2]
    
    CIELAB = np.zeros((F, C, L))
    CIEL = np.zeros((F, C))
    CIEa = np.zeros((F, C))
    CIEb = np.zeros((F, C))
    
    # D65
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    Reference = M @ np.array([100, 100, 100])  # Respecto a 100 en D65
    ReferenceX, ReferenceY, ReferenceZ = Reference / 100
    
    var_X = X / ReferenceX
    var_Y = Y / ReferenceY
    var_Z = Z / ReferenceZ
    
    def evaluar(x):
        return np.where(x > 0.008856, x**(1/3), 7.787 * x + 16/116)
    
    var_X = evaluar(var_X)
    var_Y = evaluar(var_Y)
    var_Z = evaluar(var_Z)
    
    CIEL = 116 * var_Y - 16
    CIEa = 500 * (var_X - var_Y)
    CIEb = 200 * (var_Y - var_Z)
    
    CIELAB[:,:,0] = CIEL
    CIELAB[:,:,1] = CIEa
    CIELAB[:,:,2] = CIEb
    
    return CIELAB.astype(np.float64)
    

#---------------------------------------------------------
def lab2xyz(lab):
    """
    Convierte una imagen en el espacio de color CIELAB a una imagen en el espacio de color XYZ.

    Parámetros:
    lab (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (L*, a*, b*).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color XYZ representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (X, Y, Z).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import lab2xyz
    >>> lab_img = np.random.rand(100, 100, 3) * [100, 255, 255]  # Valores en el rango [0, 100] para L y [-128, 127] para a, b
    >>> xyz_img = lab2xyz(lab_img)
    >>> print(xyz_img.shape)
    (100, 100, 3)
    """    
    # Asegurarse de que el último eje tenga tamaño 3
    if lab.shape[-1] != 3:
        raise ValueError("El último eje debe tener tamaño 3 (L*, a*, b*)")

    # Referencia D65 a 2 grados
    Reference = np.array([0.950456, 1.000000, 1.088754])

    def evaluar(x):
        return np.where(x > 0.008856, x ** 3, (x - 16 / 116) / 7.787)

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    nL = (L + 16) / 116
    na = a / 500
    nb = b / 200

    var_X = evaluar(nL + na)
    var_Y = evaluar(nL)
    var_Z = evaluar(nL - nb)

    XYZ = np.zeros_like(lab)
    XYZ[..., 0] = Reference[0] * var_X
    XYZ[..., 1] = Reference[1] * var_Y
    XYZ[..., 2] = Reference[2] * var_Z

    return XYZ

#---------------------------------------------------------
def xyz2rgb(XYZ):
    """
    Convierte una imagen en el espacio de color XYZ a una imagen en el espacio de color RGB.

    Parámetros:
    XYZ (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (X, Y, Z).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color RGB representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (R, G, B).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import xyz2rgb
    >>> xyz_img = np.random.rand(100, 100, 3)  # Valores en el rango [0, 1] para X, Y, Z
    >>> rgb_img = xyz2rgb(xyz_img)
    >>> print(rgb_img.shape)
    (100, 100, 3)
    """
    # Asegurarse de que el último eje tenga tamaño 3
    if XYZ.shape[-1] != 3:
        raise ValueError("El último eje debe tener tamaño 3 (X, Y, Z)")

    def evaluar(x):
        return np.where(x > 0.0031308,
                        1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055,
                        12.92 * x)

    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])

    var_RGB = np.dot(XYZ, M.T)
    var_RGB = np.clip(var_RGB, 0, None)  # Asegura que no haya valores negativos

    RGB = evaluar(var_RGB)
    RGB = (RGB * 255).astype(np.uint8)

    return RGB

#---------------------------------------------------------
def rgb2xyz(RGB):
    """
    Convierte una imagen en el espacio de color RGB a una imagen en el espacio de color XYZ.

    Parámetros:
    RGB (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (RGB).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color XYZ representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (X, Y, Z).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import rgb2xyz
    >>> rgb_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> xyz_img = rgb2xyz(rgb_img)
    >>> print(xyz_img.shape)
    (100, 100, 3)
    """
    F, C, L = RGB.shape
    if L != 3:
        raise ValueError('rgb2xyz: La imagen debe ser MxNx3')
    
    if RGB.dtype == np.uint8:
        RGB = RGB.astype(np.float64)
        div = 255
    else:
        div = 1
    
    sR = RGB[:,:,0]
    sG = RGB[:,:,1]
    sB = RGB[:,:,2]
    
    var_R = sR / div
    var_G = sG / div
    var_B = sB / div
    
    def evaluar(x):
        return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    
    var_R = evaluar(var_R)
    var_G = evaluar(var_G)
    var_B = evaluar(var_B)
    
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    XYZ = np.zeros((F, C, L))
    
    for i in range(F):
        for j in range(C):
            var_RGB = M @ np.array([var_R[i,j], var_G[i,j], var_B[i,j]])
            XYZ[i,j,:] = var_RGB
    
    return XYZ
    
#---------------------------------------------------------
def rgb2lab(RGB):
    """
    Convierte una imagen en el espacio de color RGB a una imagen en el espacio de color CIELAB.

    Parámetros:
    RGB (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (RGB).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color CIELAB representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (L, a, b).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import rgb2lab
    >>> rgb_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> lab_img = rgb2lab(rgb_img)
    >>> print(lab_img.shape)
    (100, 100, 3)
    """
    F, C, L = np.shape(RGB)
    if L != 3:
        raise ValueError('rgb2lab: La imagen debe ser MxNx3')
    
    xyz = rgb2xyz(RGB)
    CIEL = xyz2lab(xyz)
    
    return CIEL

#---------------------------------------------------------    
def lab2rgb(LAB):
    """
    Convierte una imagen en el espacio de color CIELAB a una imagen en el espacio de color RGB.

    Parámetros:
    LAB (numpy.ndarray): Imagen representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (L, a, b).

    Retorna:
    numpy.ndarray: Imagen en el espacio de color RGB representada como un arreglo 3D de numpy, donde la tercera dimensión corresponde a los canales de color (R, G, B).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import lab2rgb
    >>> lab_img = np.random.rand(100, 100, 3) * [100, 255, 255]  # Valores en el rango [0, 100] para L y [-128, 127] para a, b
    >>> rgb_img = lab2rgb(lab_img)
    >>> print(rgb_img.shape)
    (100, 100, 3)
    """
    # Verificar si la entrada es una matriz MxNx3
    F, C, L = np.shape(LAB)
    if L != 3:
        raise ValueError('lab2rgb: La imagen debe ser MxNx3')
    
    # Paso 1: Convertir de LAB a XYZ
    xyz = lab2xyz(LAB)
    
    # Paso 2: Convertir de XYZ a RGB
    RGB = xyz2rgb(xyz)
    
    return RGB

#---------------------------------------------------------
def imrotate(I, grados):
    """
    Rota una imagen en escala de grises o en color por un ángulo especificado.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 2D (escala de grises) o 3D (color) de numpy.
    grados (float): Ángulo de rotación en grados.

    Retorna:
    numpy.ndarray: Imagen rotada.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imrotate
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> rotated_img = imrotate(img, 45)
    >>> print(rotated_img.shape)
    """
    # Convertir los grados a radianes
    theta = -grados * np.pi / 180
    
    # Obtener las dimensiones de la imagen original
    M, N, C = I.shape  # C es el número de canales (3 para RGB)
    
    # Centro de la imagen original
    pc = np.array([N, M, 1]) / 2
    
    # Matriz de rotación inversa
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    R_inv = np.linalg.inv(R)
    
    # Calcular las nuevas dimensiones de la imagen rotada
    D = np.abs(R)
    z = np.array([N, M, 1])
    zp = np.dot(D, z)  # Nuevas dimensiones sin el término homogéneo
    
    # Dimensiones de la imagen rotada
    Np = int(np.ceil(zp[0]))  # Nueva anchura
    Mp = int(np.ceil(zp[1]))  # Nueva altura
    
    # Centro de la imagen rotada
    pc_p = np.array([Np, Mp, 1]) / 2
    
    # Inicializar la nueva imagen rotada
    I_rotada = np.zeros((Mp, Np, C), dtype=np.uint8)
    
    # Ciclos for para recorrer la imagen rotada
    for xp in range(Np):
        for yp in range(Mp):
            
            # Coordenadas homogéneas del píxel en la imagen rotada
            p_p = np.array([xp, yp, 1])
            
            # Calcular la posición relativa respecto al centro de la imagen rotada
            p_p_rel = p_p - pc_p
            
            # Aplicar la matriz de rotación inversa a las coordenadas relativas
            p_rel = np.dot(R_inv, p_p_rel)
            
            # Ajustar las coordenadas al centro de la imagen original
            p = p_rel + pc
            
            # Redondear las coordenadas al píxel más cercano
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            
            # Verificar si las coordenadas están dentro de los límites de la imagen original
            if 0 <= x < N and 0 <= y < M:
                # Asignar los valores de los píxeles de la imagen original a la imagen rotada
                I_rotada[yp, xp, :] = I[y, x, :]
    
    return I_rotada

#---------------------------------------------------------
def imcrop(I, x, y, w, h):
    """
    Recorta una imagen a un rectángulo especificado.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 2D (escala de grises) o 3D (color) de numpy.
    x (int): Coordenada x del punto de inicio del recorte.
    y (int): Coordenada y del punto de inicio del recorte.
    w (int): Ancho del rectángulo de recorte.
    h (int): Altura del rectángulo de recorte.

    Retorna:
    numpy.ndarray: Imagen recortada.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imcrop
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> cropped_img = imcrop(img, (10, 10, 50, 50))
    >>> print(cropped_img.shape)
    (50, 50, 3)
    """
    if I.ndim not in [2, 3]:
        raise ValueError("La imagen debe ser 2D (escala de grises) o 3D (color)")
    
    height, width = I.shape[:2]
    
    # Validar los límites del recorte
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Los parámetros de recorte están fuera de los límites de la imagen")
    
    if I.ndim == 2:  # Imagen en escala de grises
        return I[y:y+h, x:x+w]
    else:  # Imagen en color
        return I[y:y+h, x:x+w, :]


#---------------------------------------------------------
def imresize(I, S):
    """
    Cambia el tamaño de una imagen a las dimensiones especificadas.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 2D (escala de grises) o 3D (color) de numpy.
    S (int o tuple): Factor de escala o tamaño de la imagen de salida en el formato (nueva_altura, nueva_anchura).

    Retorna:
    numpy.ndarray: Imagen redimensionada.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imresize
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> resized_img = imresize(img, (50, 50))
    >>> print(resized_img.shape)
    (50, 50, 3)
    """
    # Leer tamaño de la imagen original
    N, M, L = I.shape  # N: altura, M: ancho, L: número de canales
    
    # Determinar los factores de escala
    if np.isscalar(S):
        Sx = Sy = S  # Factor de escala en x y en y
    else:
        Sx, Sy = S[0], S[1]  # Factor de escala en x y en y
    
    # Construir la matriz de escalamiento
    S_matrix = np.array([
        [Sx,  0,  0],
        [ 0, Sy,  0],
        [ 0,  0,  1]
    ])
    
    # Calcular el nuevo tamaño de la imagen
    z = np.array([N, M, 1])  # Dimensiones originales en formato homogéneo
    zp = np.dot(S_matrix, z)  # Nuevas dimensiones
    Np = int(np.ceil(zp[0]))  # Nueva altura
    Mp = int(np.ceil(zp[1]))  # Nuevo ancho
    
    # Crear la nueva imagen de salida
    Ip = np.zeros((Np, Mp, L), dtype=np.uint8)
    
    # Método Inverso
    for yp in range(Np):
        for xp in range(Mp):
            # Coordenadas homogéneas del píxel en la imagen escalada
            pp = np.array([xp, yp, 1])
            
            # Calcular la posición en la imagen original
            S_inv = np.array([
                [1/Sx, 0, 0],
                [0, 1/Sy, 0],
                [0, 0, 1]
            ])  # Matriz inversa de escalamiento
            
            p = np.dot(S_inv, pp)  # Coordenadas originales
            
            # Redondear las coordenadas al píxel más cercano
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            
            # Verificar si las coordenadas están dentro de los límites de la imagen original
            if 0 <= x < M and 0 <= y < N:
                # Asignar los valores de los píxeles de la imagen original a la imagen escalada
                Ip[yp, xp, :] = I[y, x, :]
    
    return Ip

#---------------------------------------------------------
def imtranslate(I, translation, mode='same'):
    """
    Traduce (desplaza) una imagen en escala de grises o en color por un desplazamiento especificado.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 2D (escala de grises) o 3D (color) de numpy.
    translation (tuple): Desplazamiento en píxeles en el formato (tx, ty).
    mode (str, opcional): Modo de salida de la imagen. Puede ser 'same' (mantiene las mismas dimensiones) o 'full' (considera el tamaño extendido). El valor por defecto es 'same'.

    Retorna:
    numpy.ndarray: Imagen traducida (desplazada).

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import imtranslate
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> translated_img = imtranslate(img, (10, 20))
    >>> print(translated_img.shape)
    """
    # Obtiene las dimensiones de la imagen original
    N, M, L = I.shape  # N: altura, M: ancho, L: canales
    
    # Desplazamientos en x e y
    tx, ty = translation
    
    # Matriz de translación
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Si el modo es 'full', se considera el tamaño extendido
    if mode == 'full':
        # Cálculo de nuevas dimensiones considerando el desplazamiento
        D = np.abs(T)
        z = np.array([M, N, 1])
        zp = np.dot(D, z)  # Nuevas dimensiones
        Mp = int(np.round(zp[0]))  # Nuevo ancho
        Np = int(np.round(zp[1]))  # Nueva altura
    elif mode == 'same':
        Mp, Np = M, N  # Mantiene las mismas dimensiones que la imagen original
    else:
        raise ValueError("El modo debe ser 'same' o 'full'")

    # Inicializa la nueva imagen con ceros
    Ip = np.zeros((Np, Mp, L), dtype=np.uint8)

    # Recorrer cada píxel de la nueva imagen
    for yp in range(Np):
        for xp in range(Mp):
            # Calcula las coordenadas en la imagen original aplicando la inversa de T
            pp = np.array([xp, yp, 1])
            p = np.dot(np.linalg.inv(T), pp)
            
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))

            # Verifica si las coordenadas están dentro de los límites de la imagen original
            if 0 <= x < M and 0 <= y < N:
                Ip[yp, xp, :] = I[y, x, :]

    # Convierte la imagen resultante a tipo uint8 (si es necesario)
    return Ip


def fitgeotrans(puntos_iniciales, puntos_finales, transformation_type='projective'):
    """
    Calcula una transformación geométrica que mapea puntos de una imagen en movimiento a puntos de una imagen fija.

    Parámetros:
    puntos_iniciales (numpy.ndarray): Coordenadas de los puntos en la imagen en movimiento. Cada fila corresponde a un punto y las columnas a las coordenadas (x, y).
    puntos_finales (numpy.ndarray): Coordenadas de los puntos en la imagen fija. Cada fila corresponde a un punto y las columnas a las coordenadas (x, y).
    transformation_type (str, opcional): Tipo de transformación geométrica. Puede ser 'affine' o 'projective'. El valor por defecto es 'projective'.

    Retorna:
    skimage.transform.ProjectiveTransform o skimage.transform.AffineTransform: Objeto de transformación geométrica que puede ser aplicado a una imagen.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from ip_functions import fitgeotrans
    >>> moving_points = np.array([[0, 0], [1, 0], [0, 1]])
    >>> fixed_points = np.array([[0, 0], [2, 0], [0, 2]])
    >>> tform = fitgeotrans(moving_points, fixed_points, 'affine')
    >>> print(tform)
    """
    # Número de puntos
    n = puntos_iniciales.shape[0]
    
    if transformation_type == 'affine':
        # Transformación afín
        num_params = 6
        A = np.zeros((2 * n, num_params))
        b = np.zeros(2 * n)
        
        for i in range(n):
            x, y = puntos_iniciales[i]
            x_prime, y_prime = puntos_finales[i]
            
            # Ecuación para x'
            A[2*i, :] = [x, y, 1, 0, 0, 0]
            b[2*i] = x_prime
            
            # Ecuación para y'
            A[2*i+1, :] = [0, 0, 0, x, y, 1]
            b[2*i+1] = y_prime
        
        # Resolver el sistema A * h = b
        h = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # La matriz H tiene la forma [h1 h2 h3; h4 h5 h6; 0 0 1]
        H = np.array([[h[0], h[1], h[2]], 
                      [h[3], h[4], h[5]], 
                      [0, 0, 1]])

    elif transformation_type == 'projective':
        # Transformación proyectiva
        num_params = 8
        A = np.zeros((2 * n, num_params))
        b = np.zeros(2 * n)
        
        for i in range(n):
            x, y = puntos_iniciales[i]
            x_prime, y_prime = puntos_finales[i]
            
            # Ecuación para x'
            A[2*i, :] = [x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y]
            b[2*i] = x_prime
            
            # Ecuación para y'
            A[2*i+1, :] = [0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y]
            b[2*i+1] = y_prime
        
        # Resolver el sistema A * h = b
        h = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # La matriz H tiene la forma [h1 h2 h3; h4 h5 h6; h7 h8 1]
        H = np.array([[h[0], h[1], h[2]], 
                      [h[3], h[4], h[5]], 
                      [h[6], h[7], 1]])

    else:
        raise ValueError("Tipo de transformación no soportado. Use 'affine' o 'projective'.")
    
    return H

#---------------------------------------------------------
def imwarp(I, H):
    """
    Aplica una transformación geométrica a una imagen utilizando una matriz de transformación.

    Parámetros:
    I (numpy.ndarray): Imagen representada como un arreglo 3D de numpy.
    H (numpy.ndarray): Matriz de transformación geométrica 3x3.
    
    Retorna:
    numpy.ndarray: Imagen transformada.

    Ejemplo de uso:
    >>> import numpy as np
    >>> from skimage.transform import AffineTransform
    >>> from ip_functions import imwarp
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> tform = AffineTransform(scale=(1.5, 1.5))
    >>> warped_img = imwarp(img, tform)
    >>> print(warped_img.shape)
    """
    # Obtener las dimensiones de la imagen
    N, M, L = I.shape

    # Esquinas de la imagen original (en coordenadas homogéneas)
    corners = np.array([
        [1, 1, 1],
        [M, 1, 1],
        [1, N, 1],
        [M, N, 1]
    ]).T  # Transpuesta para facilitar operaciones matriciales

    # Transformar las esquinas de la imagen
    transformed_corners = H @ corners
    transformed_corners /= transformed_corners[2, :]  # Normalizar

    # Calcular los límites de la nueva imagen
    x_min, y_min = np.min(transformed_corners[:2, :], axis=1)
    x_max, y_max = np.max(transformed_corners[:2, :], axis=1)

    # Calcular el nuevo tamaño de la imagen transformada
    Np = int(np.ceil(y_max - y_min + 1))
    Mp = int(np.ceil(x_max - x_min + 1))

    # Crear una imagen vacía para almacenar la imagen transformada
    imagen_transformada = np.zeros((Np, Mp, L), dtype=np.uint8)

    # Calcular la matriz de transformación inversa
    H_inv = np.linalg.inv(H)

    # Recorrer la imagen transformada píxel por píxel
    for yp in range(Np):
        for xp in range(Mp):
            # Coordenadas ajustadas (en coordenadas homogéneas)
            p_p = np.array([xp + x_min - 1, yp + y_min - 1, 1])

            # Aplicar la transformación inversa
            p = H_inv @ p_p
            x_o = int(round(p[0] / p[2]))
            y_o = int(round(p[1] / p[2]))

            # Verificar si las coordenadas están dentro de los límites de la imagen original
            if 1 <= x_o <= M and 1 <= y_o <= N:
                for c in range(L):  # Recorrer cada canal
                    imagen_transformada[yp, xp, c] = I[y_o - 1, x_o - 1, c]  # Ajustar a 0-indexed

    return imagen_transformada


    
    
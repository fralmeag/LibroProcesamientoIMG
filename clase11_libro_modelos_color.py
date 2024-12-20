import numpy as np
import plotly.graph_objects as go 

def create_rgb_cube(step=21):
    r, g, b = np.meshgrid(np.linspace(0, 1, step),
                          np.linspace(0, 1, step),
                          np.linspace(0, 1, step))

    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    return r, g, b

def create_arrow(start, end, color, name, width=8):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=width),
        name=name
    )

def plot_rgb_cube(r, g, b):
    # Crear el gráfico de dispersión 3D para los puntos del cubo
    scatter = go.Scatter3d(
        x=r, y=g, z=b,
        mode='markers',
        marker=dict(
            size=4,
            color=['rgb({},{},{})'.format(int(255*r[i]), int(255*g[i]), int(255*b[i])) for i in range(len(r))],
            opacity=1
        ),
        name='Puntos de color'
    )

    # Crear los ejes R, G y B
    axis_length = 1.1  # Longitud de los ejes

    r_axis = create_arrow([0, 0, 0], [axis_length, 0, 0], 'red', 'Eje R')
    g_axis = create_arrow([0, 0, 0], [0, axis_length, 0], 'green', 'Eje G')
    b_axis = create_arrow([0, 0, 0], [0, 0, axis_length], 'blue', 'Eje B')

    # Letras en los ejes "R", "G", "B"
    text_r = go.Scatter3d(
        x=[1.2], y=[0], z=[0],
        mode='text',
        text=["R"],
        textposition="middle right",
        textfont=dict(size=40, color="red"),
        showlegend=False
    )

    text_g = go.Scatter3d(
        x=[0], y=[1.2], z=[0],
        mode='text',
        text=["G"],
        textposition="middle right",
        textfont=dict(size=40, color="green"),
        showlegend=False
    )

    text_b = go.Scatter3d(
        x=[0], y=[0], z=[1.2],
        mode='text',
        text=["B"],
        textposition="middle right",
        textfont=dict(size=40, color="blue"),
        showlegend=False
    )

    # Combinar todos los elementos
    data = [scatter, r_axis, g_axis, b_axis, text_r, text_g, text_b]

    # Configurar el diseño
    layout = go.Layout(
        scene = dict(
            xaxis=dict(
                title='Rojo',
                showgrid=True,
                gridwidth=4,  # Aumentar grosor de la grilla
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=5,  # Línea cero más gruesa
                zerolinecolor='red',
                showline=True,
                linewidth=5,  # Línea de los ejes más gruesa
                linecolor='red',
                showbackground=False,
                range=[0, 1.1]
            ),
            yaxis=dict(
                title='Verde',
                showgrid=True,
                gridwidth=4,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=5,
                zerolinecolor='green',
                showline=True,
                linewidth=5,
                linecolor='green',
                showbackground=False,
                range=[0, 1.1]
            ),
            zaxis=dict(
                title='Azul',
                showgrid=True,
                gridwidth=4,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=5,
                zerolinecolor='blue',
                showline=True,
                linewidth=5,
                linecolor='blue',
                showbackground=False,
                range=[0, 1.1]
            ),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(255,255,255,0)'
        ),
        width=800,
        height=800,
        margin=dict(r=10, b=10, l=10, t=40),
        title='Cubo RGB Interactivo con Ejes Coloreados y Grilla Visible',
        paper_bgcolor='rgba(255,255,255,0)'
    )

    # Crear y mostrar la figura
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Crear y visualizar el cubo RGB
r, g, b = create_rgb_cube(step=21)
plot_rgb_cube(r, g, b)

import numpy as np
import plotly.graph_objects as go
import colorsys

# Función para crear las coordenadas del cilindro HSV
def create_hsv_cylinder(n_hues=72, n_saturations=50, n_values=50):
    h = np.linspace(0, 1, n_hues)
    s = np.linspace(0, 1, n_saturations)
    v = np.linspace(0, 1, n_values)

    h, s, v = np.meshgrid(h, s, v)

    x = s * np.cos(h * 2 * np.pi)
    y = s * np.sin(h * 2 * np.pi)
    z = v

    return x.flatten(), y.flatten(), z.flatten(), h.flatten(), s.flatten(), v.flatten()

# Función para crear flechas que representen los ejes
def create_arrow(start, end, color, name, width=5):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=width),
        name=name
    )

# Función para crear y visualizar el cilindro HSV
def plot_hsv_cylinder(x, y, z, h, s, v):
    # Puntos en el cilindro HSV
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=['rgb({},{},{})'.format(*[int(255*c) for c in colorsys.hsv_to_rgb(h[i], s[i], v[i])]) for i in range(len(h))],
            opacity=1
        ),
        name='Puntos de color'
    )

    # Vectores de los ejes H, S y V
    s_arrow = create_arrow([0, 0, 1], [1.2, 0, 1], 'gray', 'Saturación (S)', width=7)  # Eje S radial (sobre la tapa superior)
    h_arrow = go.Scatter3d(
        x=np.cos(np.linspace(0, 2 * np.pi, 100)),
        y=np.sin(np.linspace(0, 2 * np.pi, 100)),
        z=[1]*100,
        mode='lines',
        line=dict(color='red', width=7),
        name='Tono (H)'
    )  # Eje H (perímetro en la tapa superior)
    v_arrow = create_arrow([0, 0, 1], [0, 0, 1.5], 'black', 'Valor (V)', width=7)  # Eje V (vertical desde la tapa superior)

    # Letras en los ejes "H", "S" y "V"
    text_s = go.Scatter3d(
        x=[1.4], y=[0], z=[1],
        mode='text',
        text=["S"],
        textposition="middle right",
        textfont=dict(size=40, color="gray"),
        showlegend=False
    )

    text_h = go.Scatter3d(
        x=[0], y=[1.4], z=[1],
        mode='text',
        text=["H"],
        textposition="middle right",
        textfont=dict(size=40, color="red"),
        showlegend=False
    )

    text_v = go.Scatter3d(
        x=[0], y=[0], z=[1.6],
        mode='text',
        text=["V"],
        textposition="middle right",
        textfont=dict(size=40, color="black"),
        showlegend=False
    )

    # Agregar los datos
    data = [scatter, s_arrow, h_arrow, v_arrow, text_s, text_h, text_v]

    # Configuración del gráfico
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='X (Saturación * cos(Tono))',
                showgrid=True,
                gridwidth=2,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                showline=True,
                linewidth=2,
                linecolor='black',
                showbackground=False,
                range=[-1.5, 1.5]
            ),
            yaxis=dict(
                title='Y (Saturación * sin(Tono))',
                showgrid=True,
                gridwidth=2,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                showline=True,
                linewidth=2,
                linecolor='black',
                showbackground=False,
                range=[-1.5, 1.5]
            ),
            zaxis=dict(
                title='Valor',
                showgrid=True,
                gridwidth=2,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                showline=True,
                linewidth=2,
                linecolor='black',
                showbackground=False,
                range=[0, 1.6]
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
            bgcolor='rgba(255,255,255,0)'
        ),
        width=900,
        height=900,
        margin=dict(r=10, b=10, l=10, t=40),
        title='Cilindro HSV con Ejes H, S y V',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='black')
    )

    # Crear la figura y mostrar
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Crear el cilindro HSV y visualizarlo
x, y, z, h, s, v = create_hsv_cylinder(n_hues=72, n_saturations=50, n_values=50)
plot_hsv_cylinder(x, y, z, h, s, v)

import numpy as np
import plotly.graph_objects as go
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

def create_lab_sphere(n_points=150):
    phi = np.linspace(0, np.pi, n_points)
    theta = np.linspace(0, 2*np.pi, n_points)
    phi, theta = np.meshgrid(phi, theta)

    r = 100
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    L = 50 + z/2
    a = 128 * x / r
    b = 128 * y / r

    return L.flatten(), a.flatten(), b.flatten()

def create_arrow(start, end, color, name):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=10),
        name=name
    )

def plot_lab_space(L, a, b):
    rgb_colors = []
    for i in range(len(L)):
        lab = LabColor(L[i], a[i], b[i])
        try:
            rgb = convert_color(lab, sRGBColor)
            rgb_colors.append(rgb.get_upscaled_value_tuple())
        except ValueError:
            rgb_colors.append((128, 128, 128))

    scatter = go.Scatter3d(
        x=a, y=b, z=L,
        mode='markers',
        marker=dict(
            size=4,
            color=[f'rgb({r},{g},{b})' for r, g, b in rgb_colors],
            opacity=1
        ),
        name='Puntos de color'
    )

    extension_factor = 1.3
    L_arrow = create_arrow([0, 0, 0], [0, 0, 100*extension_factor], 'black', 'L*')
    a_arrow = create_arrow([0, 0, 50], [128*extension_factor, 0, 50], 'red', 'a*')
    b_arrow = create_arrow([0, 0, 50], [0, 128*extension_factor, 50], 'blue', 'b*')

    # Etiquetas de ejes con mayor visibilidad
    labels = go.Scatter3d(
        x=[0, 128*extension_factor, 0],
        y=[0, 0, 128*extension_factor],
        z=[110*extension_factor, 50, 50],
        mode='text',
        text=['L*', 'a*', 'b*'],
        textposition='middle center',
        textfont=dict(size=40, color=['black', 'red', 'blue']),
        name='Etiquetas de ejes'
    )

    data = [scatter, L_arrow, a_arrow, b_arrow, labels]

    layout = go.Layout(
        scene = dict(
            aspectmode='cube',
            xaxis=dict(range=[-170, 170], gridcolor='black', showbackground=False, title=''),
            yaxis=dict(range=[-170, 170], gridcolor='black', showbackground=False, title=''),
            zaxis=dict(range=[0, 170], gridcolor='black', showbackground=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
            bgcolor='rgba(255,255,255,0)'
        ),
        width=1000,
        height=1000,
        margin=dict(r=10, b=10, l=10, t=40),
        title='Espacio de Color CIELAB (Esfera Densa con Etiquetas Muy Visibles)',
        paper_bgcolor='rgba(255,255,255,0)'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Crear y visualizar el espacio CIELAB
L, a, b = create_lab_sphere(n_points=150)
plot_lab_space(L, a, b)

# Nota: Asegúrese de que plotly y colormath estén instalados
# !pip install plotly colormath
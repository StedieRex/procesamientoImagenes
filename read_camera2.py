import cv2
import argparse
import matplotlib as plt
import numpy as np

# Importamos los paquetes necesarios

# Creamos el objeto ArgumentParser
# El objeto 'parser' contendrá la información necesaria
# para analizar los argumentos de la línea de comandos en tipos de datos.
parser = argparse.ArgumentParser()

# Agregamos el argumento 'index_camera' usando add_argument() e incluimos una ayuda.
parser.add_argument("index_camera", help="índice de la cámara para leer", type=int)
args = parser.parse_args()
# si se manda con -1 será una imagen. si es 1, 2, 3, ..., etc es mediante las cámaras que tengamos

# Creamos un objeto VideoCapture para leer desde la cámara (pasamos 0):
capture = cv2.VideoCapture(args.index_camera)

# Verificamos si la cámara se abrió correctamente
if capture.isOpened() is False:
    print("Error al abrir la cámara")
    exit()

# Obtenemos algunas propiedades de VideoCapture (ancho del fotograma, altura del fotograma y fotogramas por segundo (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Imprimimos estos valores:
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

# Arreglos que almacenarán la media de cada fotograma, así se hace más robusta la captura
low_color_h = []
low_color_s = []
low_color_v = []
high_color_h = []
high_color_s = []
high_color_v = []


# Leemos hasta que se complete el video
while capture.isOpened():
    # Capturamos fotograma por fotograma de la cámara
    ret, frame = capture.read()

    if ret is True:
        
        # Se hace para mejorar el manejo de cada fotograma (imagen), así se disminuyen los recursos que se necesitan para procesarla SOLO se REDUCE más NO se COMPRIME
        frame = cv2.resize(frame, (0,0), fx = 0.4, fy = 0.4)
        # VGA 640 680
        # Costo lineal con el tamaño de la imagen
        
        # Transformar espacio de color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        (h, s, v) = cv2.split(frame)
        
        # Se van agregando los mínimos y máximos de cada fotograma
        low_color_h.append(np.min(h))
        low_color_s.append(np.min(s))
        low_color_v.append(np.min(v))
        high_color_h.append(np.max(h))
        high_color_s.append(np.max(s))
        high_color_v.append(np.max(v))
        # Convertir el fotograma capturado de la cámara a escala de grises:
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mostrar el fotograma capturado:
        cv2.imshow('Leyendo valores HSV....', frame)
        
        # Presionar 'q' en el teclado para salir del programa
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Romper el bucle
    else:
        break
cv2.destroyAllWindows()

# Se obtiene el valor de los MÍNIMOS y MÁXIMOS PERO GLOBALES
low_values = np.array([np.min(low_color_h), np.min(low_color_s), np.min(low_color_v)])
high_values = np.array([np.max(high_color_h), np.max(high_color_s), np.max(high_color_v)])

print(low_values)
print(high_values)
'''
De hacerlo de esta manera "sencilla" se nos pueden presentar varios conflictos, uno de ellos es el fallo del
sensor de la cámara, lo cual podría hacer que se detecten valores muy altos o muy bajos, lo cual afectará nuestro rango
y por consecuente la captura final saldrá con ruido
'''
############
# Leemos hasta que se complete el video
while capture.isOpened():
    # Capturamos fotograma por fotograma de la cámara
    ret, frame = capture.read()

    if ret is True:
        # Mostrar el fotograma capturado:
        frame = cv2.resize(frame, (0,0), fx = 0.4, fy = 0.4)
        cv2.imshow('Fotograma de entrada de la cámara', frame)

        # Convertir el fotograma capturado de la cámara a escala de grises:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mostrar el fotograma en escala de grises:
        cv2.imshow('Cámara en escala de grises', gray_frame)
 
        # Presionar 'q' en el teclado para salir del programa
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Romper el bucle
    else:
        break
   
 
#####
# Liberar todo:
capture.release()
cv2.destroyAllWindows()

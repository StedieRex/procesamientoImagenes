# Import the required packages
import cv2
import argparse
import matplotlib as plt
import numpy as np

# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'index_camera' argument using add_argument() including a help.
parser.add_argument("index_camera", help="index of the camera to read from", type=int)
args = parser.parse_args()
#  si se manda con -1 será una imagen. si es 1, 2, 3, ..., etc es mediante las camaras que tengamos

# We create a VideoCapture object to read from the camera (pass 0):
capture = cv2.VideoCapture(args.index_camera)

# Check if camera opened successfully
if capture.isOpened() is False:
    print("Error opening the camera")
    exit()

# Get some properties of VideoCapture (frame width, frame height and frames per second (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Print these values:
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(frame_height))
print("CAP_PROP_FPS : '{}'".format(fps))

# Arreglos que almacenarán la media de cada frame, así se hace más robusta la captura
low_color_h = []
low_color_s = []
low_color_v = []
high_color_h = []
high_color_s = []
high_color_v = []


# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    if ret is True:
        
        # Se hace para mejorar el manejo de cada frame(imagen), así se disminuyen los recuros que se necesitan  para procesarla SOLO se REDUCE más NO se COMPRIME
        frame = cv2.resize(frame, (0,0), fx = 0.4, fy = 0.4)
        #VGA 640 680
        #Costo  lineal con el tamaño de la imagen
        
        #Trasformar espacio de color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        (h, s, v) = cv2.split(frame)
        
        #Se van agregando los minimos y maximos de cada frame
        low_color_h.append(np.min(h))
        low_color_s.append(np.min(s))
        low_color_v.append(np.min(v))
        high_color_h.append(np.max(h))
        high_color_s.append(np.max(s))
        high_color_v.append(np.max(v))
        # Convert the frame captured from the camera to grayscale:
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the captured frame:
        cv2.imshow('Read HSV values....', frame)
        
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
cv2.destroyAllWindows()

#Se obtiene el valor de los MAXIMOS y MINIMO PERO GLOBALES
low_values = np.array[np.min(low_color_h), np.min(low_color_s), np.min(low_color_v)]
high_values = np.array[np.min(high_color_h), np.min(high_color_s), np.min(high_color_v)]

print(low_values)
print(high_values)
'''
De hacerlo de esto modo "sencillo" se nos pueden presentar varios conflictos, uno de ellos es el fallo del
censor de la camara, loc cual pdría hacer que se detecten valores muy altos o muy bajos, loc ual efctara nuestro rango
y por consecuente la captura final saldrá con ruido
'''
############
# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    if ret is True:
        # Display the captured frame:
        frame = cv2.resize(frame, (0,0), fx = 0.4, fy = 0.4)
        cv2.imshow('Input frame from the camera', frame)

        # Convert the frame captured from the camera to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame:
        cv2.imshow('Grayscale input camera', gray_frame)
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
   
 
#####
# Release everything:
capture.release()
cv2.destroyAllWindows()

import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import json

def camara():
    # Valores iniciales para HSV
    h_min = 0
    h_max = 179
    s_min = 0
    s_max = 255
    v_min = 0
    v_max = 255
    
    def update(val):
        nonlocal h_min, h_max, s_min, s_max, v_min, v_max
        h_min = int(hmin.val)
        h_max = int(hmax.val)
        s_min = int(smin.val)
        s_max = int(smax.val)
        v_min = int(vmin.val)
        v_max = int(vmax.val)

    def guardar_valores(event):
        valores_hsv = {
            'h_min': h_min,
            'h_max': h_max,
            's_min': s_min,
            's_max': s_max,
            'v_min': v_min,
            'v_max': v_max
        }
        with open('hsv_values.txt', 'w') as file:
            json.dump(valores_hsv, file)
        print("Valores HSV guardados.")

    def cargar_valores(event):
        nonlocal h_min, h_max, s_min, s_max, v_min, v_max
        try:
            with open('hsv_values.txt', 'r') as file:
                valores_hsv = json.load(file)
            h_min = valores_hsv['h_min']
            h_max = valores_hsv['h_max']
            s_min = valores_hsv['s_min']
            s_max = valores_hsv['s_max']
            v_min = valores_hsv['v_min']
            v_max = valores_hsv['v_max']
            hmin.set_val(h_min)
            hmax.set_val(h_max)
            smin.set_val(s_min)
            smax.set_val(s_max)
            vmin.set_val(v_min)
            vmax.set_val(v_max)
            print("Valores HSV cargados.")
        except FileNotFoundError:
            print("No se encontró un archivo de valores HSV guardados.")

    # Configuración del ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("index_camera", help="Índice de la cámara para leer", type=int)
    args = parser.parse_args()

    # Captura de video
    capture = cv2.VideoCapture(args.index_camera)
    if not capture.isOpened():
        print("Error al abrir la cámara")
        exit()
    
    # Configuración de los sliders y botones
    plt.ion()
    
    # Figura principal con subtramas para la imagen enmascarada y la original
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(left=0.25, bottom=0.5)

    axhmin = plt.axes([0.25, 0.3, 0.65, 0.03], figure=fig)
    axhmax = plt.axes([0.25, 0.25, 0.65, 0.03], figure=fig)
    axsmin = plt.axes([0.25, 0.2, 0.65, 0.03], figure=fig)
    axsmax = plt.axes([0.25, 0.15, 0.65, 0.03], figure=fig)
    axvmin = plt.axes([0.25, 0.1, 0.65, 0.03], figure=fig)
    axvmax = plt.axes([0.25, 0.05, 0.65, 0.03], figure=fig)
    
    hmin = Slider(axhmin, 'H min', 0, 179, valinit=h_min)
    hmax = Slider(axhmax, 'H max', 0, 179, valinit=h_max)
    smin = Slider(axsmin, 'S min', 0, 255, valinit=s_min)
    smax = Slider(axsmax, 'S max', 0, 255, valinit=s_max)
    vmin = Slider(axvmin, 'V min', 0, 255, valinit=v_min)
    vmax = Slider(axvmax, 'V max', 0, 255, valinit=v_max)

    hmin.on_changed(update)
    hmax.on_changed(update)
    smin.on_changed(update)
    smax.on_changed(update)
    vmin.on_changed(update)
    vmax.on_changed(update)

    # Botones de guardado y carga
    guardar_ax = plt.axes([0.8, 0.4, 0.1, 0.04])
    cargar_ax = plt.axes([0.6, 0.4, 0.1, 0.04])
    guardar_btn = Button(guardar_ax, 'Guardar HSV')
    cargar_btn = Button(cargar_ax, 'Cargar HSV')
    
    guardar_btn.on_clicked(guardar_valores)
    cargar_btn.on_clicked(cargar_valores)

    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img1 = ax1.imshow(frame)
    img2 = ax2.imshow(frame)

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            mascara = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mascara)

            # Actualizar la cámara con enmascaramiento
            img1.set_data(result)

            # Actualizar la cámara sin enmascaramiento
            img2.set_data(frame_rgb)
            
            # Refrescar la figura
            fig.canvas.draw_idle()

            plt.pause(0.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saliendo...")
                break
        else:
            break
    
    capture.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    camara()
    print("Fin del programa")

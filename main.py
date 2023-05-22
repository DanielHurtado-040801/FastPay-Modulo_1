import cv2
import time
import json
import os
# pip install requests
import requests
from pprint import pprint
from datetime import datetime
import pytz
import base64

data_plate_recognizer = {}

def analizar(ruta_foto):
    regions = ['mx', 'us-ca'] # Change to your country
    with open(ruta_foto, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': 'Token 2aed27be38a5e7902034ca9c5503bba286b752ee'})
    data_plate_recognizer = response.json()
    pprint(data_plate_recognizer)
    print('--------------------------------')
    placa = data_plate_recognizer['results'][0]['plate']
    return placa
        
#--------------------------------------------------------------------------------------------------------------------------------


# ------------------- LEER EL MODELO ----------------
# Arquitectura
arquitectura = 'model/arquitectura.txt'
# Wheights
model = 'model/MobileNetSSD_deploy.caffemodel'

classes = {
    0: "background", 1: "aeroplane", 2: "bicycle",
    3: "bird", 4: "boat",
    5: "bottle", 6: "bus",
    7: "car", 8: "cat",
    9: "chair", 10: "cow",
    11: "diningtable", 12: "dog",
    13: "horse", 14: "motorbike",
    15: "person", 16: "pottedplant",
    17: "sheep", 18: "sofa",
    19: "train", 20: "tvmonitor"
}

# Cargar el modelo
net = cv2.dnn.readNetFromCaffe(arquitectura, model)

# Inicializar la cámara
input_video = cv2.VideoCapture(1)  # 0 para la cámara predeterminada, puedes cambiarlo si tienes múltiples cámaras

# Contador para las fotos
contador_fotos = 1

detener = False

def detener_programa():
    global detener
    detener = True

while True:
    ret, frame = input_video.read()
    if not ret:
        break
    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))
    # Crear un blob
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

    # ----------- DETECCIONES Y PREDICCIONES -----------
    net.setInput(blob)
    detections = net.forward()
    for detection in detections[0][0]:
        if detection[2] > 0.45:
            label = classes[int(detection[1])]
            if label == "car":
                box = detection[3:7] * [width, height, width, height]
                x_start, y_start, x_end, y_end = map(int, box)
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2,
                            (255, 0, 0), 2)
                cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)
                # Guardar la foto
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") #Usar este mismo formato en el modulo 2
                foto_nombre = f"foto_{timestamp}.jpg"
                time.sleep(3)
                cv2.imwrite(foto_nombre, frame)
                ruta_foto = os.path.abspath(foto_nombre)
                print(f"Se ha tomado la foto: {foto_nombre}")
                time.sleep(3)
                placa = analizar(ruta_foto)

                # URL de la API en Django
                url = 'http://localhost:8000/vehiculo/vehiculo/'  # Reemplaza 'URL_DE_LA_API' con la URL de tu API


                # Obtener la hora actual en Colombia
                tz_colombia = pytz.timezone('America/Bogota')
                hora_actual_colombia = datetime.now(tz_colombia)
                hora_actual_str = hora_actual_colombia.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


                # Leer la imagen
                ruta_imagen = foto_nombre
                imagen = cv2.imread(ruta_imagen)

                files = {'img_placa': open(ruta_imagen, 'rb')}



                # Datos a enviar en la solicitud POST
                files = {
                    'placa': (None, placa.upper()),
                    'hora_ingreso': (None, hora_actual_str),
                    'img_placa': (foto_nombre, open(ruta_imagen, 'rb'))
                }
                print("-------------------------------- POST --------------------------------")
                # Convertir los datos a formato JSON
                #json_data = json.dumps(data)
                print(placa.upper(), hora_actual_str)


                # Realizar la solicitud POST
                response = requests.post(url,  files=files)

                # Obtener la respuesta de la API
                if response.status_code == 200:  # Verificar si la solicitud fue exitosa
                    response_data = response.json()
                    # Procesar la respuesta de la API según tus necesidades
                else:
                    print('Error en la solicitud POST:', response.status_code)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    if detener:
        break

input_video.release()
cv2.destroyAllWindows()

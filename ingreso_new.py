import cv2
import time
import json
import os
import requests
from pprint import pprint
from datetime import datetime
import pytz
import base64
from pprint import pprint


#Modelo que analiza la imagen y reconoce la placa
data_plate_recognizer = {}

# Obtener la hora actual en Colombia
tz_colombia = pytz.timezone('America/Bogota')
hora_actual_colombia = datetime.now(tz_colombia)
hora_actual_str = hora_actual_colombia.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def analizar_carro(ruta_foto):
    regions = ['mx', 'us-ca'] # Change to your country
    with open(ruta_foto, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': 'Token 90052a868e461e82c265234a491257b3c896b769'},
            verify=False  # Desactivar la verificación del certificado SSL
        )

    if response is None:
        print("No se pudo obtener una respuesta.")
        return None

    data_plate_recognizer = response.json()
    print('--------------------------------')

    if 'results' not in data_plate_recognizer or not data_plate_recognizer['results']:
        print("No se detectó ninguna placa.")
        return None

    placa = data_plate_recognizer['results'][0]['plate']
    print(placa)

    # Validación de formato de la placa
    if not placa or not placa.isalnum() or len(placa) != 6 or not placa[:3].isalpha() or not placa[3:].isdigit():
        print("La placa detectada no cumple con el formato esperado (3 letras seguidas de 3 números).")
        return None

    return placa


def analizar_moto(ruta_foto):
    regions = ['mx', 'us-ca']  # Cambiar según tu país
    with open(ruta_foto, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Opcional
            files=dict(upload=fp),
            headers={'Authorization': 'Token 90052a868e461e82c265234a491257b3c896b769'},
            verify=False  # Desactivar la verificación del certificado SSL
        )

    if response is None:
        print("No se pudo obtener una respuesta.")
        return None

    data_plate_recognizer = response.json()
    print('--------------------------------')

    if 'results' not in data_plate_recognizer or not data_plate_recognizer['results']:
        print("No se detectó ninguna placa.")
        return None

    placa = data_plate_recognizer['results'][0]['plate']
    print(placa)

    # Validación del formato de la placa
    if not placa or len(placa) != 6:
        print("La placa detectada no cumple con el formato esperado (6 caracteres).")
        return None

    letras = placa[:3]
    numeros = placa[3:5]
    letra_final = placa[5]

    if not letras.isalpha() or not numeros.isdigit() or not letra_final.isalpha():
        print("La placa detectada no cumple con el formato esperado (3 letras seguidas de 2 números y una letra).")
        return None

    return placa





# ----------- READ DNN MODEL -----------
# Model architecture
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
# Weights
model = "model/MobileNetSSD_deploy.caffemodel"
# Class labels
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

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ----------- READ THE VIDEO AND PREPROCESSING -----------
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))

    # Create a blob
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    # print("blob.shape:", blob.shape)

    # ----------- DETECTIONS AND PREDICTIONS -----------
    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0][0]:
        # print(detection)

        if detection[2] > 0.45:
            label = classes[int(detection[1])]
            # print("Label:", label)
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = map(int, box)

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0),
                        2)
            cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)

            if label == "car":
                print(label)

                # Delay before capturing photo
                time.sleep(3)
                
                # Capture photo
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                photo_name = f"photo_{timestamp}.jpg"
                cv2.imwrite(photo_name, frame)
                print(f"Photo saved as: {photo_name}")

                ruta_foto = os.path.abspath(photo_name)
                placa = analizar_carro(ruta_foto)

                # Delay after capturing photo
                time.sleep(5)

            if label == "motorbike":
                print(label)
                # Delay before capturing photo
                time.sleep(3)

                # Capture photo
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                photo_name = f"photo_{timestamp}.jpg"
                cv2.imwrite(photo_name, frame)
                print(f"Photo saved as: {photo_name}")
                ruta_foto = os.path.abspath(photo_name)
                placa = analizar_moto(ruta_foto)

                # Creación de json para envio a la API
                files = {
                    'placa': (None, placa.upper()),
                    'hora_ingreso': (None, hora_actual_str),
                    'img_placa': (photo_name, open(ruta_foto, 'rb'))
                }
                print(files)


                #-------Validación si el vehiculo esta registrado--------------
                # response = requests.get(url + placa.upper())

                # if response.status_code == 200:  # Verificar si la solicitud fue exitosa, por lo tanto ya esta registrado
                #     print("Placa ya ingresada")
                # else:
                     #Realizar la solicitud POST
                #     response2 = requests.post(url,  files=files)
                #     response_data = response2.json()

                # Delay after capturing photo
                time.sleep(5)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
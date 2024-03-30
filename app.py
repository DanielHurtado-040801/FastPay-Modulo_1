import io
import os
import cv2
import time
import logging
import requests
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Configura el logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Crea un manejador de archivo rotativo para registrar los mensajes de nivel INFO en un archivo
info_log_file = "app_info.log"
info_file_handler = RotatingFileHandler(info_log_file, maxBytes=10*1024*1024, backupCount=5)
info_file_handler.setLevel(logging.INFO)

# Crea un formato para los mensajes de nivel INFO
info_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
info_file_handler.setFormatter(info_formatter)

# Agrega el manejador de archivo para los mensajes de nivel INFO al logger
logger.addHandler(info_file_handler)

# Crea un manejador de archivo rotativo para registrar los mensajes de nivel ERROR en un archivo
error_log_file = "app_error.log"
error_file_handler = RotatingFileHandler(error_log_file, maxBytes=10*1024*1024, backupCount=5)
error_file_handler.setLevel(logging.ERROR)

# Crea un formato para los mensajes de nivel ERROR
error_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
error_file_handler.setFormatter(error_formatter)

# Agrega el manejador de archivo para los mensajes de nivel ERROR al logger
logger.addHandler(error_file_handler)


API_PLATE_TOKEN = os.getenv("API_PLATE_TOKEN")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Config cloudinary
cloudinary.config( 
  cloud_name = CLOUDINARY_CLOUD_NAME, 
  api_key = CLOUDINARY_API_KEY,
  api_secret = CLOUDINARY_API_SECRET,
)

# ----------- READ DNN MODEL -----------
# Model architecture
prototxt = "model/arquitectura.txt"
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

# Class labels
type_vehicule = {
    "car": "Carro",
    "motorbike": "Moto"
}

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def capture_plate(width, height, frame, detection):
    label = classes[int(detection[1])]
    box = detection[3:7] * [width, height, width, height]
    x_start, y_start, x_end, y_end = map(int, box)

    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0),
                2)
    cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)
    time.sleep(3)
    success, buffer = cv2.imencode(".jpg", frame)

    return success, buffer, label

def upload_image(buffer):
    image_in_memory = io.BytesIO(buffer)
    try:
        result = cloudinary.uploader.upload(image_in_memory)
        # Obtiene la URL de la imagen subida
        url_imagen = result["secure_url"]
        logger.info("La imagen se ha subido exitosamente a Cloudinary.")
        return True, url_imagen
    except Exception as e:
        logger.error("Error al subir la imagen a Cloudinary:", str(e))
        return False, None

def analyze_plate(buffer, label):
    regions = ["mx", "us-ca"]  # Cambiar según tu país
    logger.info("La imagen se envio al servicio platerecognizer.")
    files={'upload':buffer.tobytes()}
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),  # Opcional
        files=files,
        headers={'Authorization': f'Token {API_PLATE_TOKEN}'},
        verify=False  # Desactivar la verificación del certificado SSL
    )
    print(response)
    if response is None:
        logger.error("No se pudo obtener una respuesta del servicio platerecognizer")
        return None
    data_plate_recognizer = response.json()
    if "results" not in data_plate_recognizer or not data_plate_recognizer["results"]:
        logger.error("El servicio platerecognizer no detectó ninguna placa.")
        return None

    plate = data_plate_recognizer["results"][0]["plate"]
    logger.info(f"El servicio platerecognizer detecto la placa: {plate}.")

    # Validación del formato de la plate
    if not plate or len(plate) != 6:
        logger.error(f"La placa {plate} detectada no cumple con el formato esperado (6 caracteres).")
        return None
    
    if label == "car":
        # Validación de formato de la placa
        if not plate or not plate.isalnum() or len(plate) != 6 or not plate[:3].isalpha() or not plate[3:].isdigit():
            logger.error(f"La placa {plate} detectada con el tipo de vehiculo {type_vehicule[label]} no cumple con el formato esperado (3 letras seguidas de 3 números).")
            return None
    elif label == "motorbike":
        letras = plate[:3]
        numeros = plate[3:5]
        letra_final = plate[5]

        if not letras.isalpha() or not numeros.isdigit() or not letra_final.isalpha():
            logger.error(f"La placa {plate} detectada con el tipo de vehiculo {type_vehicule[label]} no cumple con el formato esperado (3 letras seguidas de 2 números y una letra).")
            return None
    else:
        logger.error(f"La placa {plate} no corresponde a un carro o una moto")
        return None    
    return plate

def init():
    # ----------- READ THE VIDEO AND PREPROCESSING -----------
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_resized = cv2.resize(frame, (300, 300))

        # Create a blob
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

        # ----------- DETECTIONS AND PREDICTIONS -----------
        net.setInput(blob)
        detections = net.forward()
        
        for detection in detections[0][0]:
            if detection[2] > 0.45:
                success_capture, buffer, label = capture_plate(width, height, frame, detection)
                if success_capture:
                    success_image, url = upload_image(buffer)
                    if success_image:
                        plate = analyze_plate(buffer, label)
                        if plate:
                            payload = {
                            "placa": plate,
                            "ulr_image": url,
                            "type_vehicule": label,
                            }
                            logger.info(payload)
                else:
                    print("Error")
                time.sleep(10)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
init()

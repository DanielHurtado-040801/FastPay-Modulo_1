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
        
analizar('foto.jpg')

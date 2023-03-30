import cv2
import os
import csv
import numpy as np
import datetime

def reconocimiento_facial():
    # Directorio donde se encuentran las imágenes de las personas a reconocer
    input_dir = ""

    # Archivo CSV donde se guardarán los nombres de las personas reconocidas
    output_file = ""

    # Tamaño de la imagen que se utilizará para el reconocimiento facial
    face_size = (200, 200)

    # Umbral de similitud para determinar si una imagen coincide con una persona conocida
    threshold = 6000

    # Cargar las imágenes de las personas a reconocer
    persons = []
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            name = file.split(".")[0]
            persons.append(name)
            path = os.path.join(input_dir, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, face_size)
            cv2.imwrite(path, resized)

    # Crear el modelo de reconocimiento facial a partir de las imágenes
    recognizer = cv2.face.EigenFaceRecognizer_create()
    images = []
    labels = []
    for i, person in enumerate(persons):
        path = os.path.join(input_dir, f"{person}.jpg")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(i)
    recognizer.train(images, np.array(labels))

    # Abrir la cámara
    cap = cv2.VideoCapture(0)

    # Inicializar el archivo CSV de salida
    if not os.path.isfile(output_file):
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "person"])

    # Registrar las personas que ya han sido reconocidas
    recognized_persons = []

    # Procesar cada fotograma de video
    while True:
        # Capturar un fotograma de video
        ret, frame = cap.read()

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el fotograma
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Reconocer cada rostro detectado
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, face_size)
            label, confidence = recognizer.predict(face_resized)

            # Si la imagen coincide con una persona conocida
            if confidence < threshold:
                person = persons[label]
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Si la persona ya ha sido reconocida, saltar
                if person in recognized_persons:
                    continue

                # Agregar la persona al archivo

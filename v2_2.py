#!/bin/python3

import face_recognition
import cv2
import numpy as np
import datetime
import csv
import pandas as pd
import os

# Obtener las imágenes de rostros y nombres de alumnos
carpeta_alumnos = "C:/Users/simon/OneDrive/Escritorio/script_nd/Script_Asistencia/fotos_alumnos"
extensiones_imagenes = ["jpg", "jpeg", "png"]
imagenes = []
nombres = []
for archivo in os.listdir(carpeta_alumnos):
    if archivo.split(".")[-1].lower() in extensiones_imagenes:
        imagen = face_recognition.load_image_file(os.path.join(carpeta_alumnos, archivo))
        nombre = archivo.split(".")[0]
        imagenes.append(imagen)
        nombres.append(nombre)

# Obtener las codificaciones de los rostros de las imágenes cargadas
codificaciones_conocidas = []
for img in imagenes:
    codificaciones_conocidas.append(face_recognition.face_encodings(img)[0])

# Inicializar algunas variables
caras_conocidas = nombres
caras_reconocidas = []
codificaciones_caras_reconocidas = []
fechas_asistencia = []
max_tolerancia = 0.6

# Iniciar captura de video
captura = cv2.VideoCapture(0)

while True:
    # Obtener una imagen de la cámara
    ret, imagen = captura.read()

    # Convertir la imagen de BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Encontrar todas las caras en la imagen actual
    caras_actuales = face_recognition.face_locations(imagen_rgb, model='hog')

    # Codificar las caras actuales
    codificaciones_actuales = face_recognition.face_encodings(imagen_rgb, caras_actuales)

    # Para cada cara en la imagen actual
    for codificacion_actual, ubicacion_actual in zip(codificaciones_actuales, caras_actuales):

        # Comparar la cara actual con las caras conocidas
        coincidencias = face_recognition.compare_faces(codificaciones_conocidas, codificacion_actual, tolerance=max_tolerancia)

        # Si se encuentra una coincidencia, obtener el nombre del alumno
        nombre_actual = "Desconocido"
        if True in coincidencias:
            indice_coincidencia = coincidencias.index(True)
            nombre_actual = caras_conocidas[indice_coincidencia]

            # Si la cara actual no ha sido registrada ya, agregar a la lista de caras reconocidas
            if nombre_actual not in caras_reconocidas:
                caras_reconocidas.append(nombre_actual)
                codificaciones_caras_reconocidas.append(codificacion_actual)
                fechas_asistencia.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Dibujar un cuadro alrededor de la cara actual y poner el nombre correspondiente
        y1, x2, y2, x1 = ubicacion_actual
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(imagen, nombre_actual, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

# Mostrar la imagen resultante
cv2.imshow('Asistencia en clase', imagen)

# Si se presiona la tecla 'q', salir del bucle
if cv2.waitKey(1) & 0xFF == ord('q'):
    # Crear el DataFrame con los datos de la asistencia
    data = {'Nombre': caras_reconocidas, 'Fecha y Hora': fechas_asistencia}
    df = pd.DataFrame(data)

    # Guardar el DataFrame en un archivo CSV
    filename = "alumnos_presentes.csv"
    full_path = os.path.join(os.getcwd(), filename)
    df.to_csv(full_path, index=False)

    # Liberar la cámara y cerrar las ventanas
    captura.release()
    cv2.destroyAllWindows()
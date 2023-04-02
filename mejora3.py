import cv2
import face_recognition
import pandas as pd
import os

# Cargar las imágenes de la carpeta
image_folder = "C:/Users/simon/OneDrive/Escritorio/script_nd/fotos_alumnos"
image_files = os.listdir(image_folder)

# Crear un diccionario vacío para almacenar los nombres ya reconocidos
recognized_names = {}

# Cargar las imágenes y extraer los rostros
known_face_encodings = []
known_face_names = []
for filename in image_files:
    image = face_recognition.load_image_file(os.path.join(image_folder, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(filename.split(".")[0].replace("_", " ").title())

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)

# Crear un DataFrame vacío para almacenar los nombres reconocidos
df = pd.DataFrame(columns=['Nombre'])

# Definir la tipografía y tamaño de letra para el nombre
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.60
font_thickness = 3

# Bucle principal
while True:
    # Capturar un fotograma de la cámara
    ret, frame = video_capture.read()

    # Encontrar todos los rostros en el fotograma
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Comparar los rostros encontrados con los rostros previamente cargados
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        # Si se encuentra una coincidencia, actualizar el nombre
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Verificar si el nombre ya ha sido agregado al diccionario antes de agregarlo nuevamente
            if name not in recognized_names:
                recognized_names[name] = True
                # Agregar el nombre al DataFrame
                df = df.append({'Nombre': name}, ignore_index=True)

    # Mostrar el resultado en la ventana de la cámara
    for (top, right, bottom, left), name in zip(face_locations, df['Nombre']):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 20)
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), -1)
        cv2.putText(frame, name, (left + 6, top - 6), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Mostrar el fotograma resultante en la ventana de la cámara
    cv2.imshow('Video', frame)

    # Si se presiona la tecla 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Guardar el DataFrame en un archivo XLSX
        filename = "alumnos_presentes.xlsx"
        full_path = os.path.join(os.getcwd(), filename)
        df.to_excel(full_path, index=False)
        # Liberar la cámara y cerrar las ventanas
        video_capture.release()
        cv2.destroyAllWindows

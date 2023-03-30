import cv2
import face_recognition
import os
import csv

# Cargar las imágenes de la carpeta especificada
path = 'ruta/de/la/carpeta/de/imagenes'
images = []
classNames = []
myList = os.listdir(path)

# Extraer los nombres de archivo para su uso en el archivo CSV
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Crear una función para codificar las imágenes
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Crear una lista de codificaciones
encodeListKnown = findEncodings(images)

# Crear el archivo CSV para guardar los nombres en la misma carpeta del script
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alumnos.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Nombres'])

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Redimensionar la imagen para acelerar el proceso de reconocimiento facial
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Encontrar todas las caras en la imagen y sus correspondientes codificaciones
    facesCurrFrame = face_recognition.face_locations(imgSmall, model='hog')
    encodesCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        # Comparar la codificación de la cara actual con la lista de codificaciones conocidas
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Encontrar el índice de la cara con la menor distancia
        matchIndex = matches.index(True)


        # Si hay una coincidencia, guardar el nombre en el archivo CSV
        if matches[matchIndex]:
            name = classNames[matchIndex]
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clasificacion.csv')
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name])

        # Dibujar un rectángulo alrededor de la cara
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

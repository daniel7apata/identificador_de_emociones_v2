import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

# Cargar el modelo entrenado
model = load_model("emotion_model.h5")

# Etiquetas de emociones ordenadas alfabéticamente
emotion_labels = ['Alegria', 'Desagrado', 'Enojo', 'Miedo', 'Neutral', 'Sorpresa', 'Tristeza']

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(1)

# Cargar el clasificador de caras preentrenado de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lista para almacenar las emociones detectadas y el tiempo
detected_emotions = []

# Iniciar el cronómetro
start_time = time.time()

# Obtener el directorio del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la carpeta si no existe
# Crear la carpeta "fotogramas_generados" en el mismo directorio que el archivo actual
output_dir = os.path.join(current_dir, "fotogramas_generados")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) de la cara
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predecir la emoción
        prediction = model.predict(roi_gray)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]

        # Dibujar un rectángulo alrededor de la cara y mostrar la emoción
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Guardar la emoción detectada y el tiempo
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time * 1000) % 1000)
        detected_emotions.append(f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}")

        # Recortar la región de interés (ROI) del fotograma
        roi = frame[y:y+h, x:x+w]

        # Formatear el nombre del archivo
        filename = f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}.png"

        # Guardar el fotograma clasificado (solo la ROI)
        frame_path = os.path.join(output_dir, filename)
        cv2.imwrite(frame_path, roi)

    # Mostrar el cronómetro en la esquina inferior derecha
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    cv2.putText(frame, f"Time: {minutes:02}:{seconds:02}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Mostrar el frame procesado
    cv2.imshow('Emotion Recognition', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

# Exportar las emociones detectadas a un archivo .txt
with open("emociones_detectadas.txt", "w") as file:
    for emotion in detected_emotions:
        file.write(emotion + "\n")
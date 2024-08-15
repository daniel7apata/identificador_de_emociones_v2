import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading

# Cargar el modelo entrenado
model = load_model("emotion_model.h5")

# Etiquetas de emociones ordenadas alfabéticamente
emotion_labels = ['Alegria', 'Desagrado', 'Enojo', 'Miedo', 'Neutral', 'Sorpresa', 'Tristeza']

# Cargar el clasificador en cascada de Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Obtener el directorio del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la carpeta "fotogramas_generados" en el mismo directorio que el archivo actual
output_dir = os.path.join(current_dir, "fotogramas_generados")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para procesar video desde archivo mp4
def process_video(file_path, progress_bar, progress_label):
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    detected_emotions = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)
            max_index = np.argmax(prediction[0])
            emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int((elapsed_time * 1000) % 1000)
            detected_emotions.append(f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}")

            roi = frame[y:y+h, x:x+w]
            filename = f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}.png"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, roi)

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        progress_bar['value'] = progress
        progress_label.config(text=f"{progress:.2f}%")
        root.update_idletasks()

    cap.release()
    with open("emociones_detectadas.txt", "w") as file:
        for emotion in detected_emotions:
            file.write(emotion + "\n")

# Función para iniciar la captura de video en vivo
def start_live_capture(camera_index):
    cap = cv2.VideoCapture(camera_index)
    detected_emotions = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)
            max_index = np.argmax(prediction[0])
            emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int((elapsed_time * 1000) % 1000)
            detected_emotions.append(f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}")

            roi = frame[y:y+h, x:x+w]
            filename = f"{minutes:02}_{seconds:02}_{milliseconds:03} - {emotion}.png"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, roi)

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        cv2.putText(frame, f"Time: {minutes:02}:{seconds:02}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    with open("emociones_detectadas.txt", "w") as file:
        for emotion in detected_emotions:
            file.write(emotion + "\n")

# Función para seleccionar archivo mp4
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        progress_bar['value'] = 0
        progress_label.config(text="0%")
        threading.Thread(target=process_video, args=(file_path, progress_bar, progress_label)).start()

# Crear la ventana principal
root = tk.Tk()
root.title("Identificador de Emociones")

# Label para el identificador de emociones
label = tk.Label(root, text="Identificador de Emociones")
label.pack()

# Botón para seleccionar archivo mp4
select_file_button = tk.Button(root, text="Seleccionar archivo MP4", command=select_file)
select_file_button.pack()

# Barra de progreso
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack()

# Label para mostrar el porcentaje de progreso
progress_label = tk.Label(root, text="0%")
progress_label.pack()

# Botón para iniciar la captura de video en vivo
live_capture_button = tk.Button(root, text="Iniciar captura en vivo (para terminar presionar Q)", command=lambda: start_live_capture(camera_var.get()))
live_capture_button.pack()

# Menú desplegable para seleccionar la cámara
camera_var = tk.IntVar()
camera_menu = ttk.Combobox(root, textvariable=camera_var)
camera_menu['values'] = [0, 1, 2]  # Aquí puedes agregar más cámaras si es necesario
camera_menu.current(0)
camera_menu.pack()

# Iniciar la interfaz gráfica
root.mainloop()
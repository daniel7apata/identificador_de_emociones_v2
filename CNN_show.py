import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import queue
import tensorflow as tf
import csv

# --- Configuración y Carga Inicial ---

# Cargar el modelo entrenado
try:
    model = load_model("emotion_model.h5")
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de tener 'emotion_model.h5' en el mismo directorio o especifica la ruta correcta.")
    exit()

# Etiquetas de emociones ordenadas alfabéticamente
emotion_labels = ['Alegria', 'Desagrado', 'Enojo', 'Miedo', 'Neutral', 'Sorpresa', 'Tristeza']

# Cargar el clasificador en cascada de Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Obtener el directorio del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Variables de control
stop_processing = False
roi_coordinates = None # Para almacenar las coordenadas del ROI
processing_thread = None
save_queue = queue.Queue() # Cola para guardar fotogramas en segundo plano

# Variables globales para la barra de progreso de guardado (inicializadas en la GUI)
save_progress_bar = None
save_progress_label = None
total_frames_to_save = 0
saved_frames_count = 0

# Variable global para almacenar las emociones detectadas
detected_emotions_list_global = []

# --- Funciones de Utilidad ---

# Función para guardar fotogramas en segundo plano
def save_frames_worker():
    global saved_frames_count, total_frames_to_save
    while True:
        item = save_queue.get()
        if item is None: # Señal para terminar el worker
            break
        frame_path, roi = item
        try:
            cv2.imwrite(frame_path, roi)
            saved_frames_count += 1
            if save_progress_bar and save_progress_label and total_frames_to_save > 0:
                progress = (saved_frames_count / total_frames_to_save) * 100
                # Usar root.after para actualizar la GUI desde un hilo secundario
                root.after(0, lambda: save_progress_bar.config(value=progress))
                root.after(0, lambda: save_progress_label.config(text=f"{progress:.2f}%"))
        except Exception as e:
            print(f"Error al guardar el fotograma {frame_path}: {e}")
        save_queue.task_done()
    # Resetear la barra de progreso y el contador al finalizar (asegurarse de que se ejecuta en el hilo principal)
    root.after(0, lambda: save_progress_bar.config(value=0))
    root.after(0, lambda: save_progress_label.config(text="0%"))
    saved_frames_count = 0
    total_frames_to_save = 0


# Iniciar el worker de guardado de fotogramas
threading.Thread(target=save_frames_worker, daemon=True).start()


# Función para seleccionar un ROI en la imagen/video
def select_roi_gui(frame):
    global roi_coordinates
    if frame is None:
        return None

    display_frame = frame.copy()

    r = cv2.selectROI("Seleccione ROI y presione ENTER o ESC para cancelar", display_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Seleccione ROI y presione ENTER o ESC para cancelar")

    if r[2] > 0 and r[3] > 0:
        roi_coordinates = r
        print(f"ROI seleccionado: {roi_coordinates}")
        return r
    else:
        roi_coordinates = None
        print("Selección de ROI cancelada o inválida.")
        return None

# --- Funciones de Procesamiento ---

# Función principal para procesar video desde archivo mp4
def process_video_from_file(file_path, progress_bar, progress_label, status_label, emotion_graph_canvas, fig, ax):
    global stop_processing, roi_coordinates, total_frames_to_save, saved_frames_count, detected_emotions_list_global

    status_label.config(text="Preparando video...")
    enable_disable_buttons(False)

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    detected_emotions_local_list = [] # Lista local para el registro de emociones
    emotion_counts = Counter() # Para el gráfico de emociones
    
    # Resetear la barra de progreso de guardado al inicio de un nuevo procesamiento
    total_frames_to_save = 0
    saved_frames_count = 0
    root.after(0, lambda: save_progress_bar.config(value=0))
    root.after(0, lambda: save_progress_label.config(text="0%"))


    response = tk.messagebox.askyesno("Seleccionar ROI", "¿Desea seleccionar un área de interés (ROI) en el video? Esto se aplicará al primer fotograma.")
    if response:
        ret, first_frame = cap.read()
        if not ret:
            status_label.config(text="Error: No se pudo leer el primer fotograma para selección de ROI.")
            enable_disable_buttons(True)
            return
        roi_coordinates = select_roi_gui(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar el video al principio
    else:
        roi_coordinates = None # Si el usuario no quiere ROI, se establece explícitamente a None

    # --- El cronómetro empieza AQUÍ, después de definir el ROI ---
    start_time = time.time()
    status_label.config(text="Procesando video...")

    # Lista temporal para guardar los ROIs de caras detectadas en este frame
    current_frame_faces_rois = []

    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        minutes = int((current_frame_time_ms / 1000) // 60)
        seconds = int((current_frame_time_ms / 1000) % 60)
        milliseconds = int(current_frame_time_ms % 1000)

        frame_to_process = frame

        # Dibuja el ROI amarillo SÓLO si roi_coordinates NO ES None Y ES VÁLIDO
        if roi_coordinates:
            x_roi, y_roi, w_roi, h_roi = roi_coordinates
            # Asegurarse de que el ROI sea válido (no 0 ancho/alto) antes de recortar
            if w_roi > 0 and h_roi > 0 and y_roi >= 0 and x_roi >= 0 and \
               (y_roi + h_roi) <= frame.shape[0] and (x_roi + w_roi) <= frame.shape[1]:
                frame_to_process = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 255), 2) # Dibuja el rectángulo amarillo
            else:
                frame_to_process = frame # Si el ROI es inválido, procesar el frame completo
                print("Advertencia: ROI inválido. Procesando frame completo.")
                # No dibujar el rectángulo amarillo si el ROI es inválido o se procesa el frame completo
                roi_coordinates = None # Resetear a None para evitar futuros intentos de ROI inválido

        gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_frame_faces_rois.clear() # Limpiar para este frame

        for (x, y, w, h) in faces:
            roi_face = gray[y:y+h, x:x+w]
            roi_face = cv2.resize(roi_face, (48, 48))
            roi_face = roi_face.astype('float32') / 255.0
            roi_face = np.expand_dims(roi_face, axis=0)
            roi_face = np.expand_dims(roi_face, axis=-1)

            prediction = model.predict(roi_face, verbose=0)
            max_index = np.argmax(prediction[0])
            emotion = emotion_labels[max_index]

            # Dibujar rectángulo de detección de cara y texto en el frame original
            if roi_coordinates: # Si hay un ROI activo, las coordenadas de la cara son relativas a ese ROI
                cv2.rectangle(frame, (x + x_roi, y + y_roi), (x + w + x_roi, y + h + y_roi), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x + x_roi, y + y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else: # Si no hay ROI (modo por defecto), las coordenadas de la cara son absolutas
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            detected_emotions_local_list.append({
                "time": f"{minutes:02}:{seconds:02}:{milliseconds:03}",
                "emotion": emotion
            })
            emotion_counts[emotion] += 1

            # Recortar el ROI de la cara del frame original para el guardado
            if roi_coordinates:
                roi_to_save = frame[y + y_roi : y + h + y_roi, x + x_roi : x + w + x_roi]
            else:
                roi_to_save = frame[y:y+h, x:x+w]
            current_frame_faces_rois.append((emotion, roi_to_save))

        total_frames_to_save += len(current_frame_faces_rois)

        output_dir = os.path.join(current_dir, "fotogramas_generados")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for emotion, roi_to_save in current_frame_faces_rois:
            filename = f"{minutes:02}_{seconds:02}_{milliseconds:03}_{emotion}.png"
            frame_path = os.path.join(output_dir, filename)
            save_queue.put((frame_path, roi_to_save))


        frame_count += 1
        progress = (frame_count / total_frames) * 100
        progress_bar['value'] = progress
        progress_label.config(text=f"{progress:.2f}%")
        root.update_idletasks()

        update_emotion_graph(emotion_counts, ax, emotion_graph_canvas)

        cv2.imshow('Emotion Recognition - Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_processing = True

    cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Procesamiento de video completado.")
    enable_disable_buttons(True)
    detected_emotions_list_global.extend(detected_emotions_local_list)
    save_emotions_to_csv(detected_emotions_local_list, "video")
    save_results_button.config(state="normal")

# Función para iniciar la captura de video en vivo
def start_live_capture_thread(camera_index, status_label, emotion_graph_canvas, fig, ax):
    global stop_processing, roi_coordinates, total_frames_to_save, saved_frames_count, detected_emotions_list_global

    current_live_frame = None

    def on_key_press_live(key, frame_for_roi):
        nonlocal current_live_frame
        if key == ord('r') or key == ord('R'):
            if frame_for_roi is not None:
                select_roi_gui(frame_for_roi.copy())
                print("Re-selección de ROI solicitada.")
            else:
                print("No hay frame disponible para re-seleccionar ROI.")
        elif key == ord('d') or key == ord('D'):
            global roi_coordinates
            roi_coordinates = None # Establecer ROI a None para modo por defecto
            print("ROI restaurado a modo por defecto (imagen completa).")


    stop_processing = False
    status_label.config(text="Iniciando captura en vivo...")
    enable_disable_buttons(False)

    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)

    if not cap.isOpened():
        status_label.config(text=f"Error: No se pudo abrir la cámara {camera_index}.")
        enable_disable_buttons(True)
        return

    detected_emotions_local_list = []
    emotion_counts = Counter()
    
    total_frames_to_save = 0
    saved_frames_count = 0
    root.after(0, lambda: save_progress_bar.config(value=0))
    root.after(0, lambda: save_progress_label.config(text="0%"))


    ret, frame_preview = cap.read()
    if ret:
        tk.messagebox.showinfo("Previsualización de Cámara", "Se mostrará una previsualización de la cámara. Presione 'q' para continuar o 'esc' para cancelar.")
        while True:
            cv2.imshow('Previsualizacion de Camara (Presione Q para continuar, ESC para cancelar)', frame_preview)
            key_preview = cv2.waitKey(1) & 0xFF # Usar una variable diferente para evitar conflicto
            if key_preview == ord('q'):
                break
            elif key_preview == 27: # ESC
                cap.release()
                cv2.destroyAllWindows()
                status_label.config(text="Captura en vivo cancelada.")
                enable_disable_buttons(True)
                return
            ret_prev, frame_preview = cap.read()
            if not ret_prev:
                break
        cv2.destroyWindow('Previsualizacion de Camara (Presione Q para continuar, ESC para cancelar)')

        response_roi = tk.messagebox.askyesno("Seleccionar ROI", "¿Desea seleccionar un área de interés (ROI) para la captura en vivo? Esto se aplicará a la primera imagen.")
        if response_roi:
            roi_coordinates = select_roi_gui(frame_preview)
        else:
            roi_coordinates = None # Si el usuario no quiere ROI, se establece explícitamente a None
    else:
        status_label.config(text="Error: No se pudo obtener previsualización de la cámara.")
        enable_disable_buttons(True)
        cap.release()
        return

    # --- El cronómetro empieza AQUÍ, después de definir el ROI ---
    start_time = time.time()
    status_label.config(text="Captura en vivo iniciada. Presione 'q' para detener, 'r' para re-seleccionar ROI o 'd' para restaurar ROI por defecto.")
    
    current_frame_faces_rois = []

    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret:
            break

        current_live_frame = frame.copy()

        current_time_live = time.time() - start_time
        minutes = int(current_time_live // 60)
        seconds = int(current_time_live % 60)
        milliseconds = int((current_time_live * 1000) % 1000)

        frame_to_process = frame

        # Dibuja el ROI amarillo SÓLO si roi_coordinates NO ES None Y ES VÁLIDO
        if roi_coordinates:
            x_roi, y_roi, w_roi, h_roi = roi_coordinates
            # Asegurarse de que el ROI sea válido (no 0 ancho/alto) antes de recortar
            if w_roi > 0 and h_roi > 0 and y_roi >= 0 and x_roi >= 0 and \
               (y_roi + h_roi) <= frame.shape[0] and (x_roi + w_roi) <= frame.shape[1]:
                frame_to_process = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 255), 2) # Dibuja el rectángulo amarillo
            else:
                frame_to_process = frame # Si el ROI es inválido, procesar el frame completo
                print("Advertencia: ROI inválido. Procesando frame completo.")
                # No dibujar el rectángulo amarillo si el ROI es inválido o se procesa el frame completo
                roi_coordinates = None # Resetear a None para evitar futuros intentos de ROI inválido

        gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_frame_faces_rois.clear()

        for (x, y, w, h) in faces:
            roi_face = gray[y:y+h, x:x+w]
            roi_face = cv2.resize(roi_face, (48, 48))
            roi_face = roi_face.astype('float32') / 255.0
            roi_face = np.expand_dims(roi_face, axis=0)
            roi_face = np.expand_dims(roi_face, axis=-1)

            prediction = model.predict(roi_face, verbose=0)
            max_index = np.argmax(prediction[0])
            emotion = emotion_labels[max_index]

            if roi_coordinates:
                cv2.rectangle(frame, (x + x_roi, y + y_roi), (x + w + x_roi, y + h + y_roi), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x + x_roi, y + y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            detected_emotions_local_list.append({
                "time": f"{minutes:02}:{seconds:02}:{milliseconds:03}",
                "emotion": emotion
            })
            emotion_counts[emotion] += 1

            if roi_coordinates:
                roi_to_save = frame[y + y_roi : y + h + y_roi, x + x_roi : x + w + x_roi]
            else:
                roi_to_save = frame[y:y+h, x:x+w]
            current_frame_faces_rois.append((emotion, roi_to_save))
        
        total_frames_to_save += len(current_frame_faces_rois)

        output_dir_live = os.path.join(current_dir, "fotogramas_generados_live")
        if not os.path.exists(output_dir_live):
            os.makedirs(output_dir_live)
        for emotion, roi_to_save in current_frame_faces_rois:
            filename = f"{minutes:02}_{seconds:02}_{milliseconds:03}_{emotion}.png"
            frame_path = os.path.join(output_dir_live, filename)
            save_queue.put((frame_path, roi_to_save))


        cv2.putText(frame, f"Tiempo: {minutes:02}:{seconds:02}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow('Emotion Recognition - Live', frame)

        update_emotion_graph(emotion_counts, ax, emotion_graph_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') or key == ord('R') or key == ord('d') or key == ord('D'):
            on_key_press_live(key, current_live_frame)

    cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Captura en vivo finalizada.")
    enable_disable_buttons(True)
    detected_emotions_list_global.extend(detected_emotions_local_list)
    save_emotions_to_csv(detected_emotions_local_list, "live")
    save_results_button.config(state="normal")

# Función para guardar el registro de emociones en un archivo CSV
def save_emotions_to_csv(emotions_data, source_type):
    if not emotions_data:
        print("No hay datos de emociones para guardar en CSV.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if source_type == "live":
        filename = os.path.join(current_dir, f"emociones_detectadas_live_{timestamp}.csv")
    else:
        filename = os.path.join(current_dir, f"emociones_detectadas_video_{timestamp}.csv")

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['time', 'emotion']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in emotions_data:
                writer.writerow(row)
        print(f"Emociones guardadas automáticamente en: {filename}")
        tk.messagebox.showinfo("Guardado Automático CSV", f"Registro de emociones guardado automáticamente en:\n{os.path.basename(filename)}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
        tk.messagebox.showerror("Error al guardar CSV", f"No se pudo guardar el archivo CSV: {e}")

# --- Funciones de la Interfaz Gráfica ---

# Función para habilitar/deshabilitar botones
def enable_disable_buttons(enable):
    select_file_button.config(state="normal" if enable else "disabled")
    live_capture_button.config(state="normal" if enable else "disabled")
    stop_button.config(state="normal" if not enable else "disabled")
    save_results_button.config(state="normal" if enable and detected_emotions_list_global else "disabled")


# Función para seleccionar archivo mp4 (inicia el hilo de procesamiento)
def select_file():
    global stop_processing, processing_thread, detected_emotions_list_global
    if processing_thread and processing_thread.is_alive():
        tk.messagebox.showwarning("Proceso en curso", "Ya hay un procesamiento en curso. Por favor, espere a que termine o deténgalo.")
        return

    stop_processing = False
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")], initialdir=current_dir)
    if file_path:
        detected_emotions_list_global.clear()
        
        progress_bar['value'] = 0
        progress_label.config(text="0%")
        ax.clear()
        ax.set_title("Distribución de Emociones")
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Emoción")
        emotion_graph_canvas.draw_idle()

        processing_thread = threading.Thread(target=process_video_from_file, args=(file_path, progress_bar, progress_label, status_label, emotion_graph_canvas, fig, ax))
        processing_thread.start()
        enable_disable_buttons(False)
        stop_button.config(state="normal")
        save_results_button.config(state="disabled")

# Función para detener el procesamiento
def stop_processing_video():
    global stop_processing
    stop_processing = True
    status_label.config(text="Deteniendo procesamiento...")

# Función para iniciar la captura de video en vivo (inicia el hilo de procesamiento)
def start_live_capture_handler():
    global processing_thread, detected_emotions_list_global
    if processing_thread and processing_thread.is_alive():
        tk.messagebox.showwarning("Proceso en curso", "Ya hay un procesamiento en curso. Por favor, espere a que termine o deténgalo.")
        return

    detected_emotions_list_global.clear()
    
    processing_thread = threading.Thread(target=start_live_capture_thread, args=(camera_var.get(), status_label, emotion_graph_canvas, fig, ax))
    processing_thread.start()
    enable_disable_buttons(False)
    stop_button.config(state="normal")
    save_results_button.config(state="disabled")

# Función para guardar los resultados (ahora solo para JSON o si se quiere guardar manualmente a CSV/JSON)
def save_results_manually():
    if not detected_emotions_list_global:
        tk.messagebox.showinfo("Guardar resultados", "No hay emociones detectadas para guardar.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile="emociones_detectadas_manual.json",
        initialdir=current_dir,
        title="Guardar resultados de emociones manualmente"
    )
    if file_path:
        try:
            if file_path.lower().endswith('.csv'):
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['time', 'emotion']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in detected_emotions_list_global:
                        writer.writerow(row)
                tk.messagebox.showinfo("Guardado", f"Emociones detectadas guardadas manualmente en: {file_path}")
            else:
                with open(file_path, "w", encoding='utf-8') as f:
                    json.dump(detected_emotions_list_global, f, indent=4)
                tk.messagebox.showinfo("Guardado", f"Emociones detectadas guardadas manualmente en: {file_path}")
        except Exception as e:
            tk.messagebox.showerror("Error al guardar", f"No se pudo guardar el archivo: {e}")


# Función para actualizar el gráfico de emociones (barras horizontales, ordenadas)
def update_emotion_graph(emotion_counts, ax, canvas):
    ax.clear()
    if not emotion_counts:
        ax.set_title("Distribución de Emociones")
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Emoción")
        canvas.draw_idle()
        return

    sorted_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)
    top_7_emotions = sorted_emotions[:7]

    emotions = [item[0] for item in top_7_emotions]
    counts = [item[1] for item in top_7_emotions]

    ax.barh(emotions, counts, color='skyblue')
    ax.set_title("Distribución de Emociones Detectadas")
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Emoción")
    ax.invert_yaxis()
    plt.tight_layout()
    canvas.draw_idle()


# --- Configuración de la Ventana Principal de Tkinter ---
root = tk.Tk()
root.title("Identificador de Emociones")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=1)

controls_panel_frame = ttk.Frame(root, padding="10")
controls_panel_frame.grid(row=0, column=0, sticky="nsew")
controls_panel_frame.columnconfigure(0, weight=1)

label = ttk.Label(controls_panel_frame, text="Identificador de Emociones", font=("Helvetica", 16, "bold"))
label.grid(row=0, column=0, pady=10)

button_frame = ttk.Frame(controls_panel_frame)
button_frame.grid(row=1, column=0, pady=10)

select_file_button = ttk.Button(button_frame, text="Seleccionar archivo MP4", command=select_file)
select_file_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

live_capture_button = ttk.Button(button_frame, text="Iniciar captura en vivo", command=start_live_capture_handler)
live_capture_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

stop_button = ttk.Button(button_frame, text="Detener procesamiento", command=stop_processing_video, state="disabled")
stop_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

save_results_button = ttk.Button(button_frame, text="Guardar Resultados (Manual)", command=save_results_manually, state="disabled")
save_results_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

camera_frame = ttk.Frame(controls_panel_frame)
camera_frame.grid(row=2, column=0, pady=5)
ttk.Label(camera_frame, text="Seleccionar Cámara:").pack(side="left")
camera_var = tk.IntVar(value=0)
camera_menu = ttk.Combobox(camera_frame, textvariable=camera_var, values=[0, 1, 2, 3, 4])
camera_menu.pack(side="left", padx=5)


progress_frame = ttk.Frame(controls_panel_frame)
progress_frame.grid(row=3, column=0, sticky="ew", pady=10)
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(side="left", fill="x", expand=True, padx=5)
progress_label = ttk.Label(progress_frame, text="0%")
progress_label.pack(side="right", padx=5)

save_progress_frame = ttk.Frame(controls_panel_frame)
save_progress_frame.grid(row=4, column=0, sticky="ew", pady=5)
ttk.Label(save_progress_frame, text="Guardando fotogramas:").pack(side="left", padx=5)
save_progress_bar = ttk.Progressbar(save_progress_frame, orient="horizontal", length=150, mode="determinate")
save_progress_bar.pack(side="left", fill="x", expand=True, padx=5)
save_progress_label = ttk.Label(save_progress_frame, text="0%")
save_progress_label.pack(side="right", padx=5)


status_label = ttk.Label(controls_panel_frame, text="Esperando...", relief="sunken", anchor="w")
status_label.grid(row=5, column=0, sticky="ew", pady=5)


graph_frame = ttk.LabelFrame(root, text="Distribución de Emociones Detectadas")
graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

graph_frame.columnconfigure(0, weight=1)
graph_frame.rowconfigure(0, weight=1)

fig, ax = plt.subplots(figsize=(4.8, 4))
ax.set_title("Distribución de Emociones")
ax.set_xlabel("Frecuencia")
ax.set_ylabel("Emoción")
emotion_graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
emotion_graph_canvas_widget = emotion_graph_canvas.get_tk_widget()
emotion_graph_canvas_widget.grid(row=0, column=0, sticky="nsew")


def on_closing():
    global stop_processing
    stop_processing = True
    save_queue.put(None)
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=2)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
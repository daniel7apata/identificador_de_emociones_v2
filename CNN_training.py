import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
dataPath = os.path.join(os.getcwd(), "datos_entrenamiento")
peopleList = os.listdir(dataPath)

# Parameters
img_size = (48, 48)
batch_size = 32
epochs = 120

# Data preparation
labels = []
faces_data = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    for filename in os.listdir(personPath):
        img = cv2.imread(os.path.join(personPath, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        faces_data.append(img)
        labels.append(label)
    label += 1

faces_data = np.array(faces_data).reshape(-1, img_size[0], img_size[1], 1)
labels = to_categorical(labels, num_classes=len(peopleList))

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(peopleList), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(datagen.flow(faces_data, labels, batch_size=batch_size), epochs=epochs)

# Obtener la ruta del directorio donde se encuentra el archivo actual
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta completa para guardar el modelo
model_path = os.path.join(current_directory, "emotion_model.h5")

# Save model
model.save("emotion_model.h5")

print("Modelo guardado como emotion_model.h5")
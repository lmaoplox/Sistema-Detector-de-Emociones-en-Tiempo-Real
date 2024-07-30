# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Definir rutas (ajusta estas rutas según la ubicación de tu dataset)
train_dir = 'C:\\Users\\Usuario\\Desktop\\Universidad\\3ro\\Arquitectura de Computadores\\Proyecto\\dataset\\train'
val_dir = 'C:\\Users\\Usuario\\Desktop\\Universidad\\3ro\\Arquitectura de Computadores\\Proyecto\\dataset\\test'

# Aumento de datos y generadores de imágenes
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

# Construir el modelo
model = Sequential()

# Primera capa de convolución
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa de convolución
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa de convolución
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar las capas
model.add(Flatten())

# Capa completamente conectada
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(7, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size
)

# Evaluar el modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Pérdida de Validación: {loss}')
print(f'Precisión de Validación: {accuracy}')

# Guardar el modelo (ajusta la ruta para guardar el modelo)
model.save('C:\\Users\\Usuario\\Desktop\\Universidad\\3ro\\Arquitectura de Computadores\\Proyecto\\modelo\\emotion_recognition_model.h5')

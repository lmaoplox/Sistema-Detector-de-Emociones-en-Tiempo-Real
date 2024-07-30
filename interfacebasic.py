# interface.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo
model = tf.keras.models.load_model('C:\\Users\\Usuario\\Desktop\\Universidad\\3ro\\Arquitectura de Computadores\\Proyecto\\modelo\\emotion_recognition_model.h5')

def emotion_analysis(emotions):
    objects = ('enojado', 'disgusto', 'miedo', 'feliz', 'triste', 'sorpresa', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('porcentaje')
    plt.title('emoción')

    plt.show()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = image.load_img(file_path, color_mode='grayscale', target_size=(48, 48))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255

        predictions = model.predict(img)
        emotion_analysis(predictions[0])

root = tk.Tk()
root.title("Sistema de Reconocimiento de Emociones")
root.geometry("400x200")

btn = tk.Button(root, text="Abrir Imagen", command=open_file)
btn.pack(pady=20)

root.mainloop()

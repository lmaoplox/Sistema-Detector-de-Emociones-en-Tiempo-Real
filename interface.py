import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import stat

# Cargar el modelo
model = tf.keras.models.load_model(r'C:\Users\Usuario\Desktop\Universidad\3ro\Arquitectura de Computadores\Proyecto\modelo\emotion_recognition_model.h5')

def emotion_analysis(emotions):
    objects = ('enojado', 'disgusto', 'miedo', 'feliz', 'triste', 'sorpresa', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('porcentaje')
    plt.title('emoción')

    plt.show()

# Función para capturar y procesar imágenes de la cámara
def process_camera_feed():
    cap = cv2.VideoCapture(0)

    # Ruta al archivo Haar Cascade
    face_cascade_path = r'C:\Users\Usuario\Desktop\Universidad\3ro\Arquitectura de Computadores\Proyecto\xml\haarcascade_frontalface_alt2.xml'
    
    if not os.path.isfile(face_cascade_path):
        print("Error: El archivo Haar Cascade no se encuentra en la ruta especificada.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Error: No se pudo cargar el clasificador Haar Cascade.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Usar el clasificador de detección de rostros de OpenCV
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extraer la región del rostro
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Predecir la emoción
            predictions = model.predict(roi_gray)
            max_index = np.argmax(predictions[0])
            emotion = ('enojado', 'disgusto', 'miedo', 'feliz', 'triste', 'sorpresa', 'neutral')[max_index]
            confidence = predictions[0][max_index]

            # Mostrar la emoción y la confianza
            cv2.putText(frame, f'{emotion} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Mostrar el cuadro
        cv2.imshow('Sistema de Reconocimiento de Emociones', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejemplo de código de interfaz (interfaz de texto básica)
import tkinter as tk

def start_camera():
    process_camera_feed()

root = tk.Tk()
root.title("Sistema de Reconocimiento de Emociones")
root.geometry("400x200")

btn = tk.Button(root, text="Iniciar Cámara", command=start_camera)
btn.pack(pady=20)

root.mainloop()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os

# Cargar el modelo
model = tf.keras.models.load_model(r'C:\Users\Usuario\Desktop\Universidad\3cer Año\Arquitectura de Computadores\Proyecto\modelo\emotion_recognition_model.h5')

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
    cap = cv2.VideoCapture(0)  # Abrir la cámara. 0 significa la cámara por defecto.

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Ruta al archivo Haar Cascade
    face_cascade_path = r'C:\Users\Usuario\Desktop\Universidad\3cer Año\Arquitectura de Computadores\Proyecto\xml\haarcascade_frontalface_default.xml'
    
    if not os.path.isfile(face_cascade_path):
        print("Error: El archivo Haar Cascade no se encuentra en la ruta especificada.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Error: No se pudo cargar el clasificador Haar Cascade.")
        return

    while True:
        ret, frame = cap.read()  # Leer un cuadro de la cámara.
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
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

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir del bucle.
            break

    cap.release()  # Liberar la cámara.
    cv2.destroyAllWindows()  # Cerrar todas las ventanas.

# Iniciar la captura de video
process_camera_feed()

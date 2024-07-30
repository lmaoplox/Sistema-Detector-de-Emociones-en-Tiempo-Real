# Sistema de Reconocimiento de Emociones

Este proyecto tiene como objetivo desarrollar un sistema de reconocimiento de emociones utilizando una Red Neuronal Convolucional (CNN) entrenada con el dataset FER-2013. El sistema puede identificar y clasificar diversas emociones humanas a partir de imágenes faciales.

## Características
- Entrenamiento de un modelo CNN utilizando el dataset FER-2013.
- Detección de emociones en tiempo real usando una cámara web.

## Tecnologías Utilizadas
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Instalación
1. Clona el repositorio:
    ```bash
    git clone https://github.com/tuusuario/sistema-reconocimiento-emociones.git
    cd sistema-reconocimiento-emociones
    ```

2. Crea un entorno virtual:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows, usa `.venv\Scripts\activate`
    ```

3. Instala los paquetes requeridos:
    ```bash
    pip install -r requirements.txt
    ```

4. Descarga el dataset FER-2013 y colócalo en el directorio `dataset`.

## Uso
### Entrenamiento del Modelo
Para entrenar el modelo CNN, ejecuta:
```bash
python train_model.py
```
### Interfaz
Para usar el modelo, ejecuta:
```bash
python interface.py
```

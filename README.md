# Detección de Objetos en Vía Pública

**Redes Neuronales - Semestre 2026-2**

# Descripción
Este proyecto implementa técnicas de visión por computadora y aprendizaje automático para detectar objetos presentes en imágenes de la vía pública, como vehículos, personas y señales de tránsito.

El análisis fue desarrollado en Python mediante una libreta de Jupyter Notebook.

# Objetivo
Desarrollar un modelo capaz de identificar objetos en escenarios urbanos con el fin de apoyar aplicaciones como:

    - Seguridad vial
    - Monitoreo urbano
    - Vehículos autónomos
    - Gestión del tráfico

# Descripción del Dataset
El dataset utlizado es "tesi Computer Vision Dataset" utilizado para detección de objetos mediante vision por computadora alojado en la plataforma Roboflow. Esta diseñado para ntrenar modelos capaces de identificar en imagenes mediante bounding boxes

## Contenido esperado
Al estar en Roboflow, este tipo de datasets contiene:

- Imágenes etiquetadas manualmente
- Clases de objetos definidas por el usuario:
    - Clases en el dataset:
        -   bus
        -   Construction
        -   Cyclist
        -   Pedestrian
        -   Vehicle
- Coordenadas de cajas delimitadoras (bounding boxes)
- División en conjuntos de entrenamiento, validación y prueba
- Opciones de exportación a formatos YOLO, COCO, VOC, etc.

---
## Requisitos

- Tener instalado:
    - Python3 v13.13.3
    - Pip

1. Crear entorno virtual
    - [Mac]
    ```bash
    python3 -m venv venv
    ```
    -[Windows]
    ```bash
    python -m venv venv
    ```
2. Activar entorno virtual
    - [Mac]
    ```bash
    source venv/bin/activate
    ```
    -[Windows]
    ```bash
    venv\Scripts\activate
    ```

3. Instalar dependencias
    - [Mac]
    ```bash
    pip install -r requirements.txt
    ```
    -[Windows]
    ```bash
    pip install -r requirements.txt
    ```


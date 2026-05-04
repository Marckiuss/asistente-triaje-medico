# Asistente Inteligente de Triaje Médico
Este proyecto consiste en el desarrollo de un asistente de triaje médico automatizado basado en Inteligencia Artificial. El sistema analiza síntomas descritos en lenguaje natural por el usuario para sugerir un nivel de urgencia, la especialidad médica recomendada y posibles condiciones clínicas relacionadas.

## Objetivo del Proyecto
Proporcionar una herramienta de apoyo que optimice la clasificación inicial de pacientes en entornos de salud, integrando modelos avanzados de NLP (Natural Language Processing) y sistemas de recuperación de información (RAG) para reducir la carga en los servicios de urgencias.

## Arquitectura de la Solución
El sistema se orquestra mediante Docker Compose y consta de tres pilares: 
**Backend (API)**: Construido con FastAPI, sirve las predicciones del modelo BERT y gestiona el motor de búsqueda semántica.
**Frontend (Web)**: Interfaz interactiva en Streamlit que permite al usuario chatear con el asistente y visualizar fuentes médicas.
**Inteligencia Artificial Híbrida**:
   **Clasificador**: Modelo BERT Base (bert_base_en_uncased) entrenado para identificar 24 tipos de patologías con un F1-Score del 98%.  
   **Motor RAG**: Sistema de recuperación que utiliza embeddings multilingües y **ChromaDB** para extraer fragmentos de guías clínicas reales. 

## Stack Tecnológico
* **Lenguaje:** Python 3.11 (Principal) y R (Análisis Estadístico/EDA).
* **Deep Learning:** Keras / TensorFlow.
* **NLP:** Hugging Face Transformers & PEFT.
* **MLOps:** MLflow para tracking y DVC para el versionado de datos pesados.
* **Despliegue:** Docker y Docker Compose.

## MLOps y Explicabilidad
Siguiendo los estándares de la industria, se han implementado capas de auditoría en la IA:  
* **Seguimiento con MLflow**: Se utiliza mlflow.keras.autolog() para registrar automáticamente cada intento de entrenamiento, capturando métricas de accuracy e hiperparámetros.  
* **Transparencia con LIME**: El sistema incluye un pipeline de explicabilidad que desglosa qué síntomas (ej. "ardor", "pecho") han tenido mayor peso en la decisión de la IA.

## Estructura del Repositorio
* `/api`: Código del servidor FastAPI y modelos exportados.
* `/web`: Código de la interfaz de usuario con Streamlit.
* `/src`: Contiene los módulos principales de lógica interna:  
      * `rag.py`: Implementación del sistema de recuperación de información médica utilizando ChromaDB.  
      * `predictor.py`: Pipeline de inferencia que gestiona el modelo BERT y el mapeo de especialidades clínicas.
* `/data`: Almacén de datasets y base de datos vectorial.
* `/notebooks`: Documentación del proceso de entrenamiento de la IA y análisis estadístico realizado en R
* `/analysis`: Código fuente en R y visualizaciones generadas durante la fase de análisis exploratorio (EDA)

## Gestión de Datos y Modelos (DVC)
Debido a que el proyecto maneja aproximadamente 8GB de datos binarios (modelos y bases de datos vectoriales), se utiliza DVC para desacoplar estos archivos del historial de Git:
* Los archivos pesados residen en un almacenamiento remoto.
* Git solo rastrea los punteros .dvc (como models.dvc y vector_db.dvc).

## Instalación y Despliegue
Para replicar el entorno de forma exacta, siga estos pasos:
   * 1 - Clonar el repositorio.
   * 2 - Descargar datos pesados (DVC):
            *dvc pull*
   * 3 - Despliegue unificado con Docker:
            *docker-compose up --build*
   * 4 - Acceda a la interfaz web en `http://localhost:8501`.

## Aviso Legal
Este sistema es un **prototipo académico** y no debe utilizarse como sustituto de un diagnóstico médico profesional. En caso de emergencia real, acuda inmediatamente a un centro de salud.

---
**Autores:** Marc Ferrero y Jesús Riestra.  
*Proyecto Final - Ciclo Formativo de Grado Superior en IA y Big Data.*
---
**Autores:** Marc Ferrero y Jesús Riestra.
*Proyecto Final - Ciclo Formativo de Grado Superior en IA y Big Data.*
```

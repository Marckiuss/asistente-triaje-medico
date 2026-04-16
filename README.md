# Asistente Inteligente de Triaje Médico
Este proyecto consiste en el desarrollo de un asistente de triaje médico automatizado basado en Inteligencia Artificial. El sistema analiza síntomas descritos en lenguaje natural por el usuario para sugerir un nivel de urgencia, la especialidad médica recomendada y posibles condiciones clínicas relacionadas.

## Objetivo del Proyecto
Proporcionar una herramienta de apoyo que optimice la clasificación inicial de pacientes en entornos de salud, integrando modelos avanzados de NLP (Natural Language Processing) y sistemas de recuperación de información (RAG).

## Arquitectura de la Solución
El proyecto está dividido en tres componentes principales:
* **Backend (API):** Desarrollado con **FastAPI**, encargado de servir las predicciones del modelo y gestionar la lógica del sistema RAG.
* **Frontend (Web):** Interfaz interactiva construida con **Streamlit** para la entrada de síntomas y visualización de resultados.
* **Inteligencia Artificial:** Fine-tuning de modelos Transformer (Mistral 7B/BioBERT) mediante técnicas de **LoRA/QLoRA** y una base de datos vectorial con **ChromaDB**.

## Stack Tecnológico
* **Lenguaje:** Python (Principal) y R (Análisis Estadístico/EDA).
* **Deep Learning:** Keras / TensorFlow.
* **NLP:** Hugging Face Transformers & PEFT.
* **MLOps:** MLflow (tracking) y DVC (versionado de datos).
* **Despliegue:** Docker y Docker Compose.

## Estructura del Repositorio
* `/api`: Código del servidor FastAPI y modelos exportados.
* `/web`: Código de la interfaz de usuario con Streamlit.
* `/data`: Almacén de datasets y base de datos vectorial.
* `/notebooks`: Experimentos de entrenamiento y análisis en R.
* `/mlops`: Configuraciones de seguimiento y versionado.

## Gestión de Datos y Modelos
Debido al tamaño de los datasets y los modelos de Deep Learning (como Mistral 7B), este repositorio utiliza:
* **DVC (Data Version Control):** Para gestionar y versionar los datos en `/data` sin saturar el historial de Git.
* **MLflow:** Para el seguimiento de experimentos y registro de versiones de modelos.
* **Gitignore:** Configurado para evitar la subida accidental de archivos binarios pesados y entornos virtuales al repositorio.

## Instalación y Uso
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Levantar los servicios: `docker-compose up --build`.

---
**Autores:** Marc Ferrero y Jesús Riestra.
*Proyecto Final - Ciclo Formativo de Grado Superior en IA y Big Data.*
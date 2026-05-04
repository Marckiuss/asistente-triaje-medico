import tensorflow as tf
import numpy as np
import pickle
from deep_translator import GoogleTranslator
import os
import keras
import keras_nlp


class ClinicalPredictor:
    # Diccionario REFACTORIZADO: Ahora usa el estándar de listas para soportar múltiples especialistas nativamente
    MAPEO_CLINICO = {
        "Acne": {
            "especialistas": ["Dermatólogo"],
            "urgencia": "Rutinario",
            "es": "Acné",
        },
        "Arthritis": {
            "especialistas": ["Reumatólogo"],
            "urgencia": "Prioritario",
            "es": "Artritis",
        },
        "Bronchial Asthma": {
            "especialistas": ["Neumólogo", "Alergólogo"],
            "urgencia": "Urgente",
            "es": "Asma Bronquial",
        },
        "Cervical spondylosis": {
            "especialistas": ["Traumatólogo", "Fisioterapeuta"],
            "urgencia": "Rutinario",
            "es": "Espondilosis Cervical",
        },
        "Chicken pox": {
            "especialistas": ["Pediatra", "Infectólogo"],
            "urgencia": "Prioritario",
            "es": "Varicela",
        },
        "Common Cold": {
            "especialistas": ["Médico de Cabecera"],
            "urgencia": "Rutinario",
            "es": "Resfriado Común",
        },
        "Dengue": {
            "especialistas": ["Infectólogo"],
            "urgencia": "Urgente",
            "es": "Dengue",
        },
        "Dimorphic Hemorrhoids": {
            "especialistas": ["Proctólogo", "Digestivo"],
            "urgencia": "Rutinario",
            "es": "Hemorroides",
        },
        "Fungal infection": {
            "especialistas": ["Dermatólogo"],
            "urgencia": "Rutinario",
            "es": "Infección por Hongos",
        },
        "Hypertension": {
            "especialistas": ["Cardiólogo"],
            "urgencia": "Prioritario",
            "es": "Hipertensión",
        },
        "Impetigo": {
            "especialistas": ["Dermatólogo"],
            "urgencia": "Rutinario",
            "es": "Impétigo",
        },
        "Jaundice": {
            "especialistas": ["Hepatólogo", "Digestivo"],
            "urgencia": "Urgente",
            "es": "Ictericia",
        },
        "Malaria": {
            "especialistas": ["Infectólogo"],
            "urgencia": "Urgente",
            "es": "Malaria",
        },
        "Migraine": {
            "especialistas": ["Neurólogo"],
            "urgencia": "Rutinario",
            "es": "Migraña",
        },
        "Pneumonia": {
            "especialistas": ["Neumólogo"],
            "urgencia": "Urgente",
            "es": "Neumonía",
        },
        "Psoriasis": {
            "especialistas": ["Dermatólogo"],
            "urgencia": "Rutinario",
            "es": "Psoriasis",
        },
        "Typhoid": {
            "especialistas": ["Infectólogo"],
            "urgencia": "Urgente",
            "es": "Tifoidea",
        },
        "Varicose Veins": {
            "especialistas": ["Angiólogo", "Cirujano Vascular"],
            "urgencia": "Rutinario",
            "es": "Varices",
        },
        "allergy": {
            "especialistas": ["Alergólogo"],
            "urgencia": "Rutinario",
            "es": "Alergia",
        },
        "diabetes": {
            "especialistas": ["Endocrinólogo"],
            "urgencia": "Prioritario",
            "es": "Diabetes",
        },
        "drug reaction": {
            "especialistas": ["Alergólogo", "Urgencias"],
            "urgencia": "Urgente",
            "es": "Reacción Alérgica a Medicamento",
        },
        "gastroesophageal reflux disease": {
            "especialistas": ["Digestivo"],
            "urgencia": "Rutinario",
            "es": "Reflujo Gastroesofágico",
        },
        "peptic ulcer disease": {
            "especialistas": ["Digestivo"],
            "urgencia": "Prioritario",
            "es": "Úlcera Péptica",
        },
        "urinary tract infection": {
            "especialistas": ["Urólogo", "Médico de Cabecera"],
            "urgencia": "Prioritario",
            "es": "Infección de Orina",
        },
    }

    def __init__(
        self,
        model_path: str = "asistente_triaje_medico.keras",
        encoder_path: str = "label_encoder.pkl",
    ):
        self.translator = GoogleTranslator(source="es", target="en")

        # Calculamos la carpeta raíz del proyecto (subiendo un nivel desde src/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Forzamos la búsqueda en la carpeta models
        ruta_modelo = os.path.join(base_dir, "models", model_path)
        ruta_encoder = os.path.join(base_dir, "models", encoder_path)

        # Cargamos el modelo y el encoder
        try:
            # CAMBIO CLAVE: Usamos 'keras.models' en lugar de 'tf.keras.models'
            self.model = keras.models.load_model(ruta_modelo)
            with open(ruta_encoder, "rb") as f:
                self.label_encoder = pickle.load(f)
            print("Motor BERT de KerasNLP cargado correctamente")
            self.is_loaded = True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.is_loaded = False

    # Obtiene los detalles de triaje a partir del nombre de la enfermedad en inglés
    def obtener_triaje(self, enfermedad_en: str) -> dict:
        return self.MAPEO_CLINICO.get(
            enfermedad_en,
            {
                "especialistas": ["Indeterminado"],
                "urgencia": "Indeterminada",
                "es": "Desconocida",
            },
        )

    # Recibe los síntomas en texto libre y devuelve un diccionario con la especialidad médica sugerida y el nivel de urgencia
    def predict(self, user_input: str):
        # AHORA devuelve un diccionario vacío {} en lugar de 0.0 en caso de error
        if not self.is_loaded:
            return {
                "especialistas": ["SISTEMA NO DISPONIBLE (Fallo de IA)"],
                "urgencia": "INDETERMINADA",
            }, {}

        try:
            # Traducción a inglés
            texto_en = self.translator.translate(user_input)

            # Predicción con el modelo de Keras
            logits = self.model.predict([texto_en], verbose=0)
            probabilidades = tf.nn.softmax(logits, axis=1).numpy()[0]

            # Cálculo probabilidades iterando la lista de especialistas
            probabilidades_por_especialidad = {}

            for idx, prob in enumerate(probabilidades):
                if prob > 0:  # Filtro rápido
                    enfermedad_en = self.label_encoder.inverse_transform([idx])[0]
                    info_clinica = self.obtener_triaje(enfermedad_en)

                    # Iteramos limpiamente sobre la lista nativa de especialistas
                    for esp in info_clinica["especialistas"]:
                        if esp in probabilidades_por_especialidad:
                            probabilidades_por_especialidad[esp] += float(prob)
                        else:
                            probabilidades_por_especialidad[esp] = float(prob)

            # Extracción del TOP 3 de especialidades
            especialidades_ordenadas = sorted(
                probabilidades_por_especialidad.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_3_dict = {}
            for esp, prob in especialidades_ordenadas[:3]:
                top_3_dict[esp] = min(prob, 1.0)

            # Mapear los detalles para la API
            idx_ganador_absoluto = np.argmax(probabilidades)
            enfermedad_ganadora_en = self.label_encoder.inverse_transform(
                [idx_ganador_absoluto]
            )[0]
            detalles_triaje = self.obtener_triaje(enfermedad_ganadora_en)

            return detalles_triaje, top_3_dict

        except Exception as e:
            print(f"[CRÍTICO] Fallo en el pipeline de inferencia BERT: {e}")
            print(
                f"[ERROR MLOPS] No se encontró el modelo o falló traducción. Detalle: {e}",
                flush=True,
            )
            return {
                "especialistas": ["ERROR EN PROCESAMIENTO"],
                "urgencia": "INDETERMINADA",
            }, {}

import tensorflow as tf
import numpy as np
import pickle
from deep_translator import GoogleTranslator
import keras_hub
import os

class ClinicalPredictor:
    # Diccionario para traducir enfermedades y obtener especialista y urgencia
    MAPEO_CLINICO = {
        "Acne": {"especialista": "Dermatólogo", "urgencia": "Rutinario", "es": "Acné"},
        "Arthritis": {"especialista": "Reumatólogo", "urgencia": "Prioritario", "es": "Artritis"},
        "Bronchial Asthma": {"especialista": "Neumólogo / Alergólogo", "urgencia": "Urgente", "es": "Asma Bronquial"},
        "Cervical spondylosis": {"especialista": "Traumatólogo / Fisioterapeuta", "urgencia": "Rutinario", "es": "Espondilosis Cervical"},
        "Chicken pox": {"especialista": "Pediatra / Infectólogo", "urgencia": "Prioritario", "es": "Varicela"},
        "Common Cold": {"especialista": "Médico de Cabecera", "urgencia": "Rutinario", "es": "Resfriado Común"},
        "Dengue": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Dengue"},
        "Dimorphic Hemorrhoids": {"especialista": "Proctólogo / Digestivo", "urgencia": "Rutinario", "es": "Hemorroides"},
        "Fungal infection": {"especialista": "Dermatólogo", "urgencia": "Rutinario", "es": "Infección por Hongos"},
        "Hypertension": {"especialista": "Cardiólogo", "urgencia": "Prioritario", "es": "Hipertensión"},
        "Impetigo": {"especialista": "Dermatólogo", "urgencia": "Rutinario", "es": "Impétigo"},
        "Jaundice": {"especialista": "Hepatólogo / Digestivo", "urgencia": "Urgente", "es": "Ictericia"},
        "Malaria": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Malaria"},
        "Migraine": {"especialista": "Neurólogo", "urgencia": "Rutinario", "es": "Migraña"},
        "Pneumonia": {"especialista": "Neumólogo", "urgencia": "Urgente", "es": "Neumonía"},
        "Psoriasis": {"especialista": "Dermatólogo", "urgencia": "Rutinario", "es": "Psoriasis"},
        "Typhoid": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Tifoidea"},
        "Varicose Veins": {"especialista": "Angiólogo / Cirujano Vascular", "urgencia": "Rutinario", "es": "Varices"},
        "allergy": {"especialista": "Alergólogo", "urgencia": "Rutinario", "es": "Alergia"},
        "diabetes": {"especialista": "Endocrinólogo", "urgencia": "Prioritario", "es": "Diabetes"},
        "drug reaction": {"especialista": "Alergólogo / Urgencias", "urgencia": "Urgente", "es": "Reacción Alérgica a Medicamento"},
        "gastroesophageal reflux disease": {"especialista": "Digestivo", "urgencia": "Rutinario", "es": "Reflujo Gastroesofágico"},
        "peptic ulcer disease": {"especialista": "Digestivo", "urgencia": "Prioritario", "es": "Úlcera Péptica"},
        "urinary tract infection": {"especialista": "Urólogo / Médico de Cabecera", "urgencia": "Prioritario", "es": "Infección de Orina"},
    }

    def __init__(self, model_path: str = "asistente_triaje_medico.keras", encoder_path: str = "label_encoder.pkl"):
        self.translator = GoogleTranslator(source="es", target="en")

        # 1. Calculamos la raíz del proyecto (subiendo un nivel desde src/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 2. Forzamos la búsqueda DENTRO de la carpeta models
        ruta_modelo = os.path.join(base_dir, "models", "asistente_triaje_medico.keras")
        ruta_encoder = os.path.join(base_dir, "models", "label_encoder.pkl")

        try:
            self.model = tf.keras.models.load_model(ruta_modelo)
            with open(ruta_encoder, "rb") as f:
                self.label_encoder = pickle.load(f)
            print("Motor Keras cargado correctamente")
            self.is_loaded = True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.is_loaded = False

    def obtener_triaje(self, enfermedad_en: str) -> dict:
        return self.MAPEO_CLINICO.get(
            enfermedad_en,
            {"especialista": "Médico de Cabecera", "urgencia": "Rutinario", "es": enfermedad_en}
        )

    def predict(self, user_input: str):
        if not self.is_loaded:
            return {
                "especialista": "SISTEMA NO DISPONIBLE (Fallo de IA)", 
                "urgencia": "INDETERMINADA"
            }, 0.0
        try:
            # 1. Traducción a inglés
            texto_en = self.translator.translate(user_input)
            
            # 2. Predicción con el modelo de Keras
            logits = self.model.predict([texto_en], verbose=0)
            probabilidades = tf.nn.softmax(logits, axis=1).numpy()[0]
            
            # 3. Extraer el ganador y su nivel de confianza
            idx_ganador = np.argmax(probabilidades)
            enfermedad_en = self.label_encoder.inverse_transform([idx_ganador])[0]
            confianza_real = float(probabilidades[idx_ganador])
            
            # 4. Mapear al especialista y urgencia
            detalles_triaje = self.obtener_triaje(enfermedad_en)
            return detalles_triaje, confianza_real
            
        except Exception as e:
            # Capturamos cualquier fallo en tiempo de ejecución (ej. fallo de red del traductor)
            print(f"[CRÍTICO] Fallo en el pipeline de inferencia: {e}")
            print(f"[ERROR MLOPS] No se encontró el modelo. Detalle: {e}", flush=True)
            return {
                "especialista": "ERROR EN PROCESAMIENTO", 
                "urgencia": "INDETERMINADA"
            }, 0.0
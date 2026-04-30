import tensorflow as tf
import numpy as np
import pickle
from deep_translator import GoogleTranslator

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

    def __init__(self, model_path: str = "models/asistente_triaje_medico.keras", encoder_path: str = "models/label_encoder.pkl"):
        self.translator = GoogleTranslator(source="es", target="en")
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print("Motor Mistral/Keras cargado correctamente")
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
            enfermedad_en = "Common Cold"
            confianza_real = 0.50
        else:
            # Traducimos a inglés
            texto_en = self.translator.translate(user_input)
            
            # Predicción con modelo
            logits = self.model.predict([texto_en], verbose=0)
            probabilidades = tf.nn.softmax(logits, axis=1).numpy()[0]
            
            # Ganador
            idx_ganador = np.argmax(probabilidades)
            enfermedad_en = self.label_encoder.inverse_transform([idx_ganador])[0]
            confianza_real = float(probabilidades[idx_ganador])
            
        detalles_triaje = self.obtener_triaje(enfermedad_en)
        return detalles_triaje, confianza_real
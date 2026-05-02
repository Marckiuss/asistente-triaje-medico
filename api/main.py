from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.rag import MedicalRAG
from src.predictor import ClinicalPredictor

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="Asistente de Triaje Médico - API",
    description="Backend para la clasificación de síntomas y sugerencia de urgencias.",
    version="1.0.0",
)

# Arrancamos los motores
print("Inicializando componentes del sistema...")
rag_engine = MedicalRAG()
predictor_engine = ClinicalPredictor()

class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
def home():
    return {"status": "online", "message": "Servidor de Triaje Médico activo"}

@app.post("/predict")
async def predict_triage(request: SymptomRequest):
    user_input = request.symptoms.lower()

    # 1. Recuperamos contexto médico estructurado (Lista de fuentes)
    # Ahora esto es una lista: [{"texto": "...", "fuente": "...", "pagina": "..."}, ...]
    fuentes_recuperadas = rag_engine.retrieve_context(user_input)

    # 2. Preparamos el texto plano para el predictor (si el modelo lo requiere como string)
    # Unimos solo los contenidos de texto para el análisis del modelo
    contexto_texto_plano = "\n\n".join([f['texto'] for f in fuentes_recuperadas]) if fuentes_recuperadas else ""

    # 3. Predicción de IA y Triaje
    # Nota: Asegúrate de si tu predictor necesita el contexto_texto_plano como argumento
    detalles_triaje, confianza_real = predictor_engine.predict(user_input)

    # 4. Respuesta final unificada
    return {
        "especialidad_sugerida": detalles_triaje["especialista"],
        "nivel_urgencia": detalles_triaje["urgencia"],
        "mensaje": f"Análisis completado para: '{user_input}'",
        # Enviamos la lista completa de fuentes para que Marc la use en el frontend
        "fuentes": fuentes_recuperadas, 
        "instrucciones": f"Recomendación del sistema: Consulte con un {detalles_triaje['especialista']} lo antes posible.",
        "confianza_modelo": {
            detalles_triaje["especialista"]: confianza_real,
            "Otros (Ruido)": 1 - confianza_real,
        },
    }

if __name__ == "__main__":
    # Importante: En Docker usamos 0.0.0.0, pero aquí mantenemos tu config local
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
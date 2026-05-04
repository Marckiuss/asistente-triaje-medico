from fastapi import FastAPI, HTTPException
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
    try:
        user_input = request.symptoms.lower()

        # Recuperamos contexto médico estructurado (Lista de fuentes) en una lista de diccionarios con 'texto' y 'fuente'
        fuentes_recuperadas = rag_engine.retrieve_context(user_input)

        # Predicción de IA y Triaje
        detalles_triaje, top_3_probabilidades = predictor_engine.predict(user_input)

        # Traducción y preparación de variables
        lista_especialistas = detalles_triaje.get("especialistas", ["Indeterminado"])
        texto_especialistas = " / ".join(lista_especialistas)

        # Creamos un mensaje seguro si el modelo no sabe qué hacer
        if "Indeterminado" in lista_especialistas:
            instrucciones_finales = "Atención: El sistema no ha podido determinar una especialidad clara con los síntomas proporcionados. Acuda a un centro médico para una valoración humana."
        else:
            instrucciones_finales = f"Recomendación del sistema: Consulte con {texto_especialistas} lo antes posible."

        # Respuesta final unificada
        return {
            "especialidad_sugerida": texto_especialistas,
            "nivel_urgencia": detalles_triaje["urgencia"],
            "mensaje": f"Análisis completado para: '{user_input}'",
            "fuentes": fuentes_recuperadas,
            "instrucciones": instrucciones_finales,
            "confianza_modelo": top_3_probabilidades,
        }

    except Exception as e:
        print(f"Error crítico en el servidor: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}"
        )


if __name__ == "__main__":
    # Levantamos el servidor en 0.0.0.0 para que sea accesible desde la red de Docker
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

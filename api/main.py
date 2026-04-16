from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Inicializamos la aplicación FastAPI
# FastAPI genera documentación automática (Swagger) por defecto 
app = FastAPI(
    title="Asistente de Triaje Médico - API",
    description="Backend para la clasificación de síntomas y sugerencia de urgencias.",
    version="1.0.0"
)

# Definimos el esquema de los datos de entrada usando Pydantic
class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
def home():
    """Endpoint de comprobación de salud del servidor."""
    return {"status": "online", "message": "Servidor de Triaje Médico activo"}

@app.post("/predict")
async def predict_triage(request: SymptomRequest):
    """
    Endpoint principal para recibir síntomas y devolver un triaje. [cite: 236]
    En esta fase de MVP, devolvemos una respuesta simulada (Mock).
    """
    # Aquí es donde integraremos el modelo .keras que generará Jesús [cite: 237]
    user_input = request.symptoms.lower()
    
    # Lógica de respuesta temporal (Mock) para validar la conexión con el frontend
    return {
        "especialidad_sugerida": "Medicina General (MVP)",
        "nivel_urgencia": "Pendiente de evaluación técnica",
        "mensaje": f"Recibido análisis de: '{user_input}'",
        "instrucciones": "Consulte con un profesional sanitario si los síntomas persisten."
    }

if __name__ == "__main__":
    # Ejecución local del servidor
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
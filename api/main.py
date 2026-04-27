import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="Asistente de Triaje Médico - API",
    description="Backend para la clasificación de síntomas y sugerencia de urgencias.",
    version="1.0.0",
)


# Definimos el esquema de los datos de entrada usando Pydantic
class SymptomRequest(BaseModel):
    symptoms: str


# Bloque completo de Configuración RAG y conexión a ChromaDB
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(directorio_actual)
CHROMA_PATH = os.path.join(directorio_raiz, "data", "vector_db")

print("Cargando modelo de embeddings para búsqueda RAG...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Conectamos con la base de datos vectorial
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


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
    # mod - Aquí va el modelo .keras de Jesús
    user_input = request.symptoms.lower()

    resultados = vector_db.similarity_search(user_input, k=3)

    contexto_medico = ""
    if resultados:
        # Unimos los textos encontrados
        contexto_medico = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    else:
        contexto_medico = "No se encontró contexto clínico en las guías."

    # mod - Lógica de respuesta temporal (Mock) para validar la conexión con el frontend
    return {
        "especialidad_sugerida": "Medicina General (MVP)",
        "nivel_urgencia": "Pendiente de evaluación técnica",
        "mensaje": f"Recibido análisis de: '{user_input}'",
        "contexto_recuperado": contexto_medico,
        "instrucciones": "Consulte con un profesional sanitario si los síntomas persisten.",
    }


if __name__ == "__main__":
    # Ejecución local del servidor
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

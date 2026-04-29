import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import random  # Añade este import arriba del todo

# Diccionario Clínico de referencia para el triaje (enfermedad en inglés -> especialista, urgencia, nombre en español)
MAPEO_CLINICO = {
    "Acne": {"especialista": "Dermatólogo", "urgencia": "Rutinario", "es": "Acné"},
    "Arthritis": {
        "especialista": "Reumatólogo",
        "urgencia": "Prioritario",
        "es": "Artritis",
    },
    "Bronchial Asthma": {
        "especialista": "Neumólogo / Alergólogo",
        "urgencia": "Urgente",
        "es": "Asma Bronquial",
    },
    "Cervical spondylosis": {
        "especialista": "Traumatólogo / Fisioterapeuta",
        "urgencia": "Rutinario",
        "es": "Espondilosis Cervical",
    },
    "Chicken pox": {
        "especialista": "Pediatra / Infectólogo",
        "urgencia": "Prioritario",
        "es": "Varicela",
    },
    "Common Cold": {
        "especialista": "Médico de Cabecera",
        "urgencia": "Rutinario",
        "es": "Resfriado Común",
    },
    "Dengue": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Dengue"},
    "Dimorphic Hemorrhoids": {
        "especialista": "Proctólogo / Digestivo",
        "urgencia": "Rutinario",
        "es": "Hemorroides",
    },
    "Fungal infection": {
        "especialista": "Dermatólogo",
        "urgencia": "Rutinario",
        "es": "Infección por Hongos",
    },
    "Hypertension": {
        "especialista": "Cardiólogo",
        "urgencia": "Prioritario",
        "es": "Hipertensión",
    },
    "Impetigo": {
        "especialista": "Dermatólogo",
        "urgencia": "Rutinario",
        "es": "Impétigo",
    },
    "Jaundice": {
        "especialista": "Hepatólogo / Digestivo",
        "urgencia": "Urgente",
        "es": "Ictericia",
    },
    "Malaria": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Malaria"},
    "Migraine": {"especialista": "Neurólogo", "urgencia": "Rutinario", "es": "Migraña"},
    "Pneumonia": {"especialista": "Neumólogo", "urgencia": "Urgente", "es": "Neumonía"},
    "Psoriasis": {
        "especialista": "Dermatólogo",
        "urgencia": "Rutinario",
        "es": "Psoriasis",
    },
    "Typhoid": {"especialista": "Infectólogo", "urgencia": "Urgente", "es": "Tifoidea"},
    "Varicose Veins": {
        "especialista": "Angiólogo / Cirujano Vascular",
        "urgencia": "Rutinario",
        "es": "Varices",
    },
    "allergy": {"especialista": "Alergólogo", "urgencia": "Rutinario", "es": "Alergia"},
    "diabetes": {
        "especialista": "Endocrinólogo",
        "urgencia": "Prioritario",
        "es": "Diabetes",
    },
    "drug reaction": {
        "especialista": "Alergólogo / Urgencias",
        "urgencia": "Urgente",
        "es": "Reacción Alérgica a Medicamento",
    },
    "gastroesophageal reflux disease": {
        "especialista": "Digestivo",
        "urgencia": "Rutinario",
        "es": "Reflujo Gastroesofágico",
    },
    "peptic ulcer disease": {
        "especialista": "Digestivo",
        "urgencia": "Prioritario",
        "es": "Úlcera Péptica",
    },
    "urinary tract infection": {
        "especialista": "Urólogo / Médico de Cabecera",
        "urgencia": "Prioritario",
        "es": "Infección de Orina",
    },
}


def obtener_triaje(enfermedad_en):
    info = MAPEO_CLINICO.get(
        enfermedad_en,
        {
            "especialista": "Médico de Cabecera",
            "urgencia": "Rutinario",
            "es": enfermedad_en,
        },
    )
    return info


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

    # --- SIMULACIÓN TRANSITORIA (Hasta la Fase 2) ---
    # Como aún no hemos enchufado la red neuronal de Jesús, simulamos que
    # la IA ha detectado una enfermedad para probar que el Mapeo funciona bien.
    enfermedad_simulada_en = random.choice(list(MAPEO_CLINICO.keys()))

    # --- APLICAMOS LA LÓGICA DE JESÚS ---
    detalles_triaje = obtener_triaje(enfermedad_simulada_en)

    # --- RESPUESTA DINÁMICA ---
    return {
        "especialidad_sugerida": detalles_triaje["especialista"],
        "nivel_urgencia": detalles_triaje["urgencia"],
        "mensaje": f"Recibido análisis de: '{user_input}'",
        "contexto_recuperado": contexto_medico,
        "instrucciones": f"Recomendación del sistema: Consulte con un {detalles_triaje['especialista']} lo antes posible.",
        "confianza_modelo": {
            detalles_triaje["especialista"]: 0.85,
            "Medicina General": 0.10,
            "Urgencias": 0.05,
        },
    }


if __name__ == "__main__":
    # Ejecución local del servidor
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

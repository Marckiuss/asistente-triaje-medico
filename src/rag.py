import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class MedicalRAG:
    def __init__(self):
        # Definimos la ruta a la base de datos vectorial
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        directorio_raiz = os.path.dirname(directorio_actual)
        self.chroma_path = os.path.join(directorio_raiz, "data", "vector_db")
        
        print("Cargando modelo de embeddings para búsqueda RAG...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Conectamos con la base de datos vectorial
        self.vector_db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)

    def retrieve_context(self, user_input: str, k: int = 3) -> str:
        """Busca en los PDFs y devuelve el texto formateado."""
        resultados = self.vector_db.similarity_search(user_input, k=k)
        
        if resultados:
            return "\n\n---\n\n".join([doc.page_content for doc in resultados])
        else:
            return "No se encontró contexto clínico en las guías."
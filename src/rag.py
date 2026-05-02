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
        self.vector_db = Chroma(
            persist_directory=self.chroma_path, embedding_function=self.embeddings
        )

    def retrieve_context(self, user_input: str, k: int = 3):
        """Busca en los PDFs y devuelve una lista de diccionarios con texto y fuente."""
        # similarity_search devuelve una lista de objetos 'Document' que contienen .page_content y .metadata
        resultados = self.vector_db.similarity_search(user_input, k=k)

        if not resultados:
            return []

        contexto_estructurado = []
        for doc in resultados:
            # Extraemos la fuente de los metadatos (inyectados por LangChain al crear la DB)
            ruta_completa = doc.metadata.get("source", "Fuente desconocida")
            ruta_normalizada = ruta_completa.replace("\\", "/")
            # Extraemos solo el nombre del archivo
            nombre_archivo = os.path.basename(ruta_normalizada)

            contexto_estructurado.append(
                {
                    "texto": doc.page_content,
                    "fuente": nombre_archivo,
                    "pagina": doc.metadata.get(
                        "page", "N/A"
                    ),  # Si usaste PyPDFLoader, trae la página
                }
            )

        return contexto_estructurado

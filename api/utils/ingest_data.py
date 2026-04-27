import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Configuración de Rutas Dinámicas ---
# Esto busca automáticamente la raíz de tu proyecto para no fallar con las rutas
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))

DATA_PATH = os.path.join(directorio_raiz, "data", "raw", "guias_clinicas")
CHROMA_PATH = os.path.join(directorio_raiz, "data", "vector_db")

def create_vector_db():
    print(f"Buscando PDFs en: {DATA_PATH}...")
    
    # --- 2. Carga de Documentos ---
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("No se encontraron PDFs. Revisa la ruta de la carpeta.")
        return
        
    print(f"Se han cargado {len(documents)} páginas en total.")

    # --- 3. Troceado del texto (Chunking) ---
    # Partimos los PDFs en trozos de 1000 caracteres. 
    # El solapamiento (overlap) de 200 evita que se corten frases médicas por la mitad.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Textos divididos en {len(chunks)} fragmentos (chunks).")

    # --- 4. Modelo de Embeddings (El traductor a números) ---
    # Usamos un modelo open-source ligero y multilingüe (ideal para español)
    print("Cargando modelo de embeddings (esto puede tardar unos segundos la primera vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- 5. Creación de la Base de Datos Vectorial ---
    print(f"Guardando base de datos vectorial en: {CHROMA_PATH}...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print("¡Base de datos vectorial ChromaDB creada con éxito!")

if __name__ == "__main__":
    create_vector_db()
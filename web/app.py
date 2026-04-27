import streamlit as st
import requests

# --- Configuración de la página ---
st.set_page_config(page_title="Triaje IA", page_icon="🏥", layout="centered")

st.title("🏥 Asistente Inteligente de Triaje")
st.caption(
    "Esta herramienta utiliza Inteligencia Artificial para sugerir la especialidad médica y el nivel de urgencia basándose en tus síntomas. **Nota: Prototipo académico.**"
)

# --- Inicializar el historial de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de triaje. ¿Podrías describirme tus síntomas detalladamente?",
        }
    ]

# --- Mostrar historial de mensajes ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Si el mensaje del asistente tiene contexto RAG, lo mostramos en un desplegable
        if message.get("rag_context"):
            with st.expander("📚 Ver fuentes clínicas (Sistema RAG)"):
                st.info(message["rag_context"])

# --- Entrada de Usuario ---
if prompt := st.chat_input(
    "Describe tus síntomas aquí (ej: dolor en el pecho que se irradia...)"
):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando guías clínicas y analizando síntomas..."):
            try:
                API_URL = "http://127.0.0.1:8000/predict"
                payload = {"symptoms": prompt}

                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    # 1. Preparamos el texto principal (Limpio y directo)
                    respuesta_principal = f"""
**🩺 Análisis Preliminar:**
* **Especialidad Sugerida:** {result['especialidad_sugerida']}
* **Nivel de Urgencia:** {result['nivel_urgencia']}

⚠️ **Aviso legal:** {result['instrucciones']}
"""
                    st.markdown(respuesta_principal)

                    # 2. Preparamos el contexto RAG (Limpiando los saltos de línea del PDF)
                    contexto_crudo = result.get("contexto_recuperado", "")
                    contexto_limpio = ""

                    if (
                        contexto_crudo
                        and contexto_crudo
                        != "No se encontró contexto clínico en las guías."
                    ):
                        # Truco para PDFs: Guardamos los párrafos reales (\n\n), limpiamos las líneas rotas (\n) y restauramos.
                        texto_temp = contexto_crudo.replace("\n\n", "@@MARCADOR@@")
                        texto_temp = texto_temp.replace("\n", " ")
                        contexto_limpio = texto_temp.replace("@@MARCADOR@@", "\n\n")

                        with st.expander("📚 Ver fuentes clínicas (Sistema RAG)"):
                            st.info(contexto_limpio)

                    # 3. Guardamos en el historial tanto la respuesta como el contexto limpio
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": respuesta_principal,
                            "rag_context": contexto_limpio,  # Se guarda para poder pintarlo al recargar
                        }
                    )

                else:
                    st.error(f"Error en la API (Código {response.status_code})")

            except Exception as e:
                st.error(
                    f"No se pudo conectar con el servidor IA. ¿Está encendida la API? (Error: {e})"
                )

# --- Menú lateral ---
st.sidebar.title("Opciones")
if st.sidebar.button("🗑️ Nueva Consulta"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de triaje. ¿Podrías describirme tus síntomas detalladamente?",
        }
    ]
    st.rerun()

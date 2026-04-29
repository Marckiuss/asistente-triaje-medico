import streamlit as st
import requests

# --- Configuración de la página ---
st.set_page_config(page_title="Triaje IA", page_icon="🏥", layout="centered")

st.title("🏥 Asistente Inteligente de Triaje")
st.caption(
    "Esta herramienta utiliza Inteligencia Artificial para sugerir la especialidad médica y el nivel de urgencia basándose en tus síntomas. **Nota: Prototipo académico.**"
)

# --- Inicializamos el historial de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de triaje. ¿Podrías describirme tus síntomas detalladamente?",
        }
    ]

# --- Mostramos historial de mensajes ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Si hay gráficos en el historial, los pintamos
        if message.get("confianza"):
            st.caption("**Confianza de la predicción:**")
            for esp, prob in message["confianza"].items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"*{esp}*")
                with col2:
                    st.progress(prob, text=f"{int(prob * 100)}%")
            st.write("---")

        # Si el mensaje del asistente tiene contexto RAG, lo mostramos en un desplegable
        if message.get("rag_context"):
            with st.expander("Ver fuentes clínicas (Sistema RAG)"):
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

                    # Preparamos el texto inicial
                    respuesta_principal = f"""
                        **Análisis Preliminar:**
                        * **Especialidad Sugerida:** {result['especialidad_sugerida']}
                        * **Nivel de Urgencia:** {result['nivel_urgencia']}

                        **Aviso legal:** {result['instrucciones']}
"""
                    st.markdown(respuesta_principal)

                    # Dibujo de Gráficos de Confianza
                    if "confianza_modelo" in result:
                        st.caption("**Probabilidad por Especialidad Médica:**")
                        for especialidad, probabilidad in result[
                            "confianza_modelo"
                        ].items():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"*{especialidad}*")
                            with col2:
                                # Creamos una barra de progreso nativa de Streamlit
                                st.progress(
                                    probabilidad, text=f"{int(probabilidad * 100)}%"
                                )
                        st.write("---")

                    # Preparamos el contexto RAG
                    contexto_crudo = result.get("contexto_recuperado", "")
                    contexto_limpio = ""

                    if (
                        contexto_crudo
                        and contexto_crudo
                        != "No se encontró contexto clínico en las guías."
                    ):
                        # Limpieza en la devolución del RAG para que no se rompan los párrafos y el texto quede bien formado.
                        texto_temp = contexto_crudo.replace("\n\n", "@@MARCADOR@@")
                        texto_temp = texto_temp.replace("\n", " ")
                        contexto_limpio = texto_temp.replace("@@MARCADOR@@", "\n\n")

                        with st.expander("Ver fuentes clínicas (Sistema RAG)"):
                            st.info(contexto_limpio)

                    # Guardamos en el historial tanto la respuesta como el contexto limpio
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": respuesta_principal,
                            "rag_context": contexto_limpio,  # Se guarda para poder pintarlo al recargar
                            "confianza": result.get("confianza_modelo"),
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

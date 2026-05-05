import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Triaje IA", page_icon="🏥", layout="centered")

st.title("🏥 Asistente Inteligente de Triaje")
st.caption(
    "Esta herramienta utiliza Inteligencia Artificial para sugerir la especialidad médica y el nivel de urgencia basándose en tus síntomas. **Nota: Prototipo académico.**"
)

# Inicializamos el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente de triaje. ¿Podrías describirme tus síntomas detalladamente?",
        }
    ]

# Mostramos historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("confianza"):
            st.caption("**Top 3 Especialidades Sugeridas:**")

            # Convertimos las claves y valores a listas para poder asignar posiciones
            esps = list(message["confianza"].keys())
            probs = list(message["confianza"].values())

            # Creamos las 3 columnas para mostrar las especialidades y sus probabilidades
            cols = st.columns(3)
            medallas = ["🥇", "🥈", "🥉"]

            for i, col in enumerate(cols):
                with col:
                    if i < len(esps):
                        medalla = medallas[i] if i < 3 else ""
                        st.metric(
                            label=f"{medalla} {esps[i]}",
                            value=f"{probs[i]*100:.1f}%",
                        )
                        st.progress(probs[i])
            st.write("---")

        # Recorremos la lista de fuentes en el historial
        if message.get("fuentes"):
            with st.expander("📚 Ver referencias médicas consultadas"):
                for doc in message["fuentes"]:
                    st.markdown(
                        f"**Fuente:** :blue[{doc['fuente']}] | **Pág:** {doc.get('pagina', 'N/A')}"
                    )
                    st.info(doc["texto"])
                    st.divider()

# Entrada del Usuario
if prompt := st.chat_input("Describe tus síntomas aquí..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consultando guías clínicas y analizando síntomas..."):
            try:
                # URL del contenedor API en la red de Docker
                API_URL = "http://api:8000/predict"
                payload = {"symptoms": prompt}

                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    respuesta_principal = f"""
                        **Análisis Preliminar:**
                        * **Especialidad Sugerida:** {result['especialidad_sugerida']}
                        * **Nivel de Urgencia:** {result['nivel_urgencia']}

                        **Aviso legal:** {result['instrucciones']}
                    """
                    st.markdown(respuesta_principal)

                    # Gráficos de Confianza (Top-3)
                    if "confianza_modelo" in result:
                        st.caption("**Top 3 Especialidades Sugeridas:**")

                        esps = list(result["confianza_modelo"].keys())
                        probs = list(result["confianza_modelo"].values())

                        cols = st.columns(3)
                        medallas = ["🥇", "🥈", "🥉"]

                        for i, col in enumerate(cols):
                            with col:
                                if i < len(esps):
                                    medalla = medallas[i] if i < 3 else ""
                                    st.metric(
                                        label=f"{medalla} {esps[i]}",
                                        value=f"{probs[i]*100:.1f}%",
                                    )
                                    st.progress(probs[i])
                        st.write("---")

                    # Procesamos la lista de fuentes
                    fuentes_api = result.get("fuentes", [])
                    if fuentes_api:
                        with st.expander("📚 Ver referencias médicas consultadas"):
                            for doc in fuentes_api:
                                st.markdown(
                                    f"**Fuente:** :blue[{doc['fuente']}] | **Pág:** {doc.get('pagina', 'N/A')}"
                                )
                                st.info(doc["texto"])
                                st.divider()

                    # Guardamos en el historial
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": respuesta_principal,
                            "fuentes": fuentes_api,  # Guardamos la lista estructurada
                            "confianza": result.get("confianza_modelo"),
                        }
                    )
                else:
                    st.error(f"Error en la API (Código {response.status_code})")

            except Exception as e:
                st.error(f"Error de conexión: {e}")

# Menú lateral
st.sidebar.title("Opciones")
if st.sidebar.button("🗑️ Nueva Consulta"):
    st.session_state.messages = [{"role": "assistant", "content": "Hola..."}]
    st.rerun()

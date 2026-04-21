import streamlit as st
import requests

# Configuración de la página (Título e Icono)
st.set_page_config(page_title="Triaje IA", page_icon="")

st.title("Asistente de Triaje Médico")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para sugerir la especialidad médica y el nivel de urgencia basándose en tus síntomas. 
**Nota:** Esto es un prototipo con fines académicos.
""")

# --- Entrada de Datos ---
with st.form("triage_form"):
    symptoms = st.text_area("Describe tus síntomas detalladamente:", placeholder="Ej: Me duele el pecho y tengo dificultad para respirar...")
    submitted = st.form_submit_button("Analizar Síntomas")

# --- Lógica de Conexión con la API ---
if submitted:
    if symptoms.strip() == "":
        st.warning("Por favor, introduce tus síntomas antes de continuar.")
    else:
        with st.spinner("Analizando información clínica..."):
            try:
                # Llamada al endpoint de FastAPI que creamos antes
                # mod - Esta url cambiará cuando implementemos Docker
                API_URL = "http://127.0.0.1:8000/predict"
                payload = {"symptoms": symptoms}
                
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # --- Visualización de Resultados ---
                    st.divider()
                    st.subheader("Resultado del Análisis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Especialidad Sugerida", result["especialidad_sugerida"])
                    with col2:
                        st.metric("Nivel de Urgencia", result["nivel_urgencia"])
                    
                    st.info(f"**Detalles:** {result['mensaje']}")
                    st.warning("**Aviso legal:** " + result["instrucciones"])
                else:
                    st.error(f"Error en la API (Código {response.status_code})")
            
            except Exception as e:
                st.error(f"No se pudo conectar con el servidor de IA. ¿Está encendida la API? (Error: {e})")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("**Estado del Sistema:** MVP v1.0")
st.sidebar.write("**Desarrollado por:** Marc y Jesús")
import streamlit as st
import requests

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="🔍",
    layout="centered"
)

# Título y descripción
st.title("🔍 Análisis de Sentimientos")
st.markdown("""
    ¡Bienvenido! Esta aplicación analiza el sentimiento de las reviews de ecommerce enfocados en productos para mascotas a través de un modelo de inteligencia artificial.
    Escribe tu texto en el cuadro de abajo y haz clic en **Analizar Sentimiento**.
""")

# Entrada de texto
text_input = st.text_area(
    "Escribe un texto para analizar:",
    placeholder="Ej: 'Mi perro ama este alimento, ¡es increíble!'",
    height=150
)

# Botón de análisis
if st.button("Analizar Sentimiento"):
    if text_input.strip():  # Verifica que el texto no esté vacío
        try:
            # Llamada a la API
            response = requests.post(
                "http://127.0.0.1:8000/predict",  # URL de la API
                json={"text": text_input},        # Datos enviados a la API
                timeout=10                        # Tiempo de espera
            )
            response.raise_for_status()  # Lanza una excepción si hay un error HTTP
            
            # Obtener el resultado
            sentiment = response.json().get("sentiment", "Error")
            
            # Mostrar el resultado
            if sentiment == "Positivo":
                st.success(f"🎉 **Sentimiento Predicho:** {sentiment}")
            elif sentiment == "Negativo":
                st.error(f"😞 **Sentimiento Predicho:** {sentiment}")
            else:
                st.warning(f"🤔 **Sentimiento Predicho:** {sentiment}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al conectar con la API: {e}")
    else:
        st.warning("⚠️ Por favor, ingresa un texto para analizar.")

# Sección de información adicional
st.markdown("---")
st.markdown("### ℹ️ Información Adicional")
st.markdown("""
    - **Modelo Utilizado:** Transformers(Mecanismo de atención y) fine-tuneado para análisis de sentimientos.
    - **Vectorización:** Modelo prentrenado BERT
    - **API:** FastAPI en `http://127.0.0.1:8000`.
    - **Desarrollado por:** [Diego Casasbuenas]
""")
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
#Prob
import matplotlib.pyplot as plt
import pandas as pd

# Configurar la página
st.set_page_config(page_title="Clasificador de Dígitos Manuscritos", layout="centered")

st.title("🖍️ Clasificador de Dígitos Manuscritos (CNN)")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    modelo = load_model('modelo_cnn_def.keras')  # Asegúrate de tener el modelo guardado
    return modelo

modelo = cargar_modelo()

# Función para preprocesar la imagen
def preprocesar_imagen(imagen, img_size=28):
    imagen = imagen.convert('L')  # Convertir a escala de grises
    imagen = imagen.resize((img_size, img_size))  # Redimensionar
    imagen_array = img_to_array(imagen)
    imagen_array = imagen_array / 255.0  # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir batch dimension
    return imagen_array

# Cargar imagen
imagen_subida = st.file_uploader("Sube una imagen de un dígito manuscrito (png o jpg)", type=["png", "jpg", "jpeg"])

if imagen_subida is not None:
    # Abrir la imagen que fue subida usando PIL (Python Imaging Library)
    imagen = Image.open(imagen_subida)
    # Mostrar la imagen subida en la interfaz de Streamlit con un título y un tamaño de 150px de ancho
    st.image(imagen, caption="Imagen subida", width=150)

    # Procesar imagen y predecir
    if st.button("🔍 Predecir Dígito"):
        # Preprocesar la imagen (convertirla a escala de grises, redimensionar, normalizar, etc.)
        imagen_preprocesada = preprocesar_imagen(imagen)
        # Realizar la predicción utilizando el modelo previamente entrenado
        # 'imagen_preprocesada' tiene la forma necesaria para el modelo
        prediccion = modelo.predict(imagen_preprocesada)
        # Extraer la clase predicha (la clase con la mayor probabilidad)
        clase_predicha = np.argmax(prediccion)
        # Extraer la clase predicha (la clase con la mayor probabilidad)
        confianza = np.max(prediccion)
        # Mostrar el resultado de la predicción, la clase y la confianza en porcentaje
        st.success(f"Predicción: **{clase_predicha}** con confianza de {confianza*100:.2f}%")
        # Mostrar las probabilidades de predicción para cada clase en un gráfico de barras
        st.subheader("Probabilidades por clase:")
        # 'prediccion' es un array de probabilidades por clase, por lo que se toma el primer elemento
        st.bar_chart(prediccion[0])
        st.subheader("Probabilidades por clase:")

        # Debido al eje X con los datos girados
        fig, ax = plt.subplots()
        clases = range(10)  # Dígitos del 0 al 9
        ax.bar(clases, prediccion[0])  # Gráfico de barras
        ax.set_xticks(clases)  # Etiquetas del eje x
        ax.set_xlabel("Clase (dígito)")
        ax.set_ylabel("Probabilidad")
        ax.set_title("Probabilidades por clase")

        st.pyplot(fig)  # Mostrar en Streamlit

        # En tabla
        st.subheader("Probabilidades por clase: (tabla)")
        st.dataframe(
            pd.DataFrame({
                "Clase": range(10),
                "Probabilidad": prediccion[0]
            }).set_index("Clase")
        )
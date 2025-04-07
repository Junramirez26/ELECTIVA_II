# Importar librerías necesarias
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("Gráfica de datos con Streamlit")

# Crear lista para almacenar los datos de voltaje y tiempo
list_voltaje = []
Tiempo = []


# Crear datos
Slider_Voltaje = st.slider("Voltaje (V)", 5, 50, 10, 1)

for Valores_voltaje in range(Slider_Voltaje):
    Valores_voltaje = rd.randint(5, 100)
    list_voltaje.append(Valores_voltaje)

# Crear los intervalos de tiempo de 0 a 10 segundos
Tiempo = np.linspace(0, 10, Slider_Voltaje)

# Crear un DataFrame para almacenar los datos
data = {"Tiempo (s)": Tiempo, "Voltaje (V)" : list_voltaje}
df = pd.DataFrame(data)

# Mostrar datos en una tabla
st.write("Datos de medición:")
st.dataframe(df)

# Graficar los datos
fig, ax = plt.subplots()
ax.plot(df["Tiempo (s)"], df["Voltaje (V)"], marker="o", linestyle="-")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Voltaje (V)")
ax.set_title("Voltaje vs. Tiempo")
st.pyplot(fig)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
filename = 'modelo-reg-redneuronal.pkl'
modelo, min_max_scaler,variables = pickle.load(open(filename, 'rb'))


import streamlit as st

st.title('Predicción de ventas')

uploaded_file = st.file_uploader("Sube el archivo CSV con los datos a predecir",type=["csv"])

if uploaded_file is None:
    st.stop()

data = pd.read_csv(uploaded_file)
st.dataframe(data.head())

if not st.button("Predecir"):
    st.stop()

data_preparada=data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma','Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
data_preparada.head()

data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()

data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
data_preparada.head()
Y_pred = modelo.predict(data_preparada)
data['Predicción']=Y_pred
data.head()

data
st.warning("El modelo tiene un error del 4%")

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar resultados",
    data=csv,
    file_name="resultados_prediccion.csv",
    mime="text/csv"
)



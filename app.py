import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# Carrega os dados
df = pd.read_csv("data.csv")
df['condominio'] = df['total'] - df['rent']

# Lista de preditores
features = ['area', 'bedrooms', 'garage']

# Cria dicionário de modelos por tipo
modelos_por_tipo = {}
tipos = df['type'].dropna().unique()

for tipo in tipos:
    df_tipo = df[df['type'] == tipo].dropna(subset=features + ['rent'])
    X = df_tipo[features]
    y = df_tipo['rent']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    modelos_por_tipo[tipo] = modelo

# Interface do Streamlit
st.title("🔢 Simulador de Aluguel por Tipo de Imóvel")

tipo_escolhido = st.selectbox("Escolha o tipo de imóvel:", tipos)
area = st.number_input("Área (m²):", min_value=10, max_value=1000, value=60)
bedrooms = st.number_input("Quantidade de quartos:", min_value=0, max_value=10, value=2)
garage = st.number_input("Quantidade de vagas de garagem:", min_value=0, max_value=5, value=1)

if st.button("Calcular aluguel estimado"):
    modelo = modelos_por_tipo[tipo_escolhido]
    entrada = [[area, bedrooms, garage]]
    aluguel = modelo.predict(entrada)[0]
    st.success(f"💰 Aluguel estimado: R$ {aluguel:.2f}")

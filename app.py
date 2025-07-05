import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Título da aplicação
st.set_page_config(page_title="Simulador de Aluguel", layout="centered")
st.title("🔢 Simulador de Aluguel por Tipo de Imóvel")

# Carrega os dados
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("data.csv")
        df['condominio'] = df['total'] - df['rent']
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o CSV: {e}")
        return pd.DataFrame()

df = carregar_dados()

# Verifica se os dados foram carregados corretamente
if df.empty:
    st.stop()

# Lista de features e criação dos modelos por tipo
features = ['area', 'bedrooms', 'garage']
modelos_por_tipo = {}

tipos = df['type'].dropna().unique()

for tipo in tipos:
    df_tipo = df[df['type'] == tipo].dropna(subset=features + ['rent'])
    if len(df_tipo) >= 10:  # Garante dados suficientes
        X = df_tipo[features]
        y = df_tipo['rent']
        modelo = LinearRegression()
        modelo.fit(X, y)
        modelos_por_tipo[tipo] = modelo

# Interface de simulação
tipo_escolhido = st.selectbox("Escolha o tipo de imóvel:", sorted(modelos_por_tipo.keys()))

area = st.number_input("Área (m²):", min_value=10, max_value=1000, value=60)
bedrooms = st.number_input("Quantidade de quartos:", min_value=0, max_value=10, value=2)
garage = st.number_input("Quantidade de vagas de garagem:", min_value=0, max_value=5, value=1)

if st.button("Calcular aluguel estimado"):
    modelo = modelos_por_tipo.get(tipo_escolhido)
    entrada = [[area, bedrooms, garage]]
    aluguel = modelo.predict(entrada)[0]
    st.success(f"💰 Aluguel estimado para um {tipo_escolhido.lower()}: R$ {aluguel:.2f}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import cargar_datos, crear_matriz_usuario_item, entrenar_modelo_softimpute, recomendar_peliculas

st.set_page_config(page_title="Recomendador de Películas", layout="wide")
st.title("🎬 Sistema de Recomendación con Matrix Completion")

# Cargar datos (cache para no repetir)
@st.cache_data
def obtener_datos_y_modelo():
    ratings, movies = cargar_datos('u.data', 'u.item')
    matriz_original = crear_matriz_usuario_item(ratings)
    matriz_completada = entrenar_modelo_softimpute(matriz_original)
    return ratings, movies, matriz_original, matriz_completada

ratings_df, movies_df, matriz_original, matriz_completada = obtener_datos_y_modelo()

st.sidebar.header("Instrucciones")
st.sidebar.write("1. Califica al menos 10 películas usando los sliders.")
st.sidebar.write("2. Haz clic en 'Recomendar'.")
st.sidebar.write("3. El sistema te mostrará 10 películas que podrían gustarte.")

# Mostrar heatmap de NAs (opcional, como pide el PDF)
if st.sidebar.checkbox("Mostrar heatmap de matriz original"):
    st.subheader("Matriz usuario-película (valores faltantes en amarillo)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matriz_original.isnull().sample(frac=0.1, axis=1), cbar=False, cmap='viridis')
    st.pyplot(fig)

# Selección de películas para calificar
st.subheader("Paso 1: Califica estas 10 películas")
# Elegir 10 películas populares para mostrar
pelis_populares_ids = ratings_df['item_id'].value_counts().head(20).index.tolist()
pelis_seleccionadas = movies_df[movies_df['item_id'].isin(pelis_populares_ids)].sample(10, random_state=42)

ratings_usuario = {}
cols = st.columns(2)
for idx, (_, row) in enumerate(pelis_seleccionadas.iterrows()):
    with cols[idx % 2]:
        rating = st.slider(
            f"{row['title']}",
            min_value=1, max_value=5, value=3, step=1,
            key=f"rate_{row['item_id']}"
        )
        ratings_usuario[row['item_id']] = rating

if st.button("🎯 ¡Recomiéndame películas!"):
    if len(ratings_usuario) < 5:
        st.warning("Por favor califica al menos 5 películas para obtener mejores recomendaciones.")
    else:
        with st.spinner("Calculando recomendaciones..."):
            recomendaciones = recomendar_peliculas(ratings_usuario, matriz_original, movies_df, top_n=10)
        st.subheader("🎥 Tus 10 recomendaciones")
        st.dataframe(recomendaciones.style.format({'rating_predicho': '{:.2f}'}))

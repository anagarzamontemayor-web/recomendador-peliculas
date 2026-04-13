import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import cargar_datos, crear_matriz_usuario_item, entrenar_modelo_iterative, recomendar_peliculas

st.set_page_config(page_title="Recomendador de Películas", layout="wide")
st.title("🎬 Sistema de Recomendación con Matrix Completion")

@st.cache_data(show_spinner="🔄 Cargando datos y entrenando modelo (versión ligera)...")
def cargar_modelo_recomendador():
    print("Iniciando carga de datos...")
    ratings, movies = cargar_datos('u.data', 'u.item')
    print("Datos cargados. Creando matriz...")
    matriz_original = crear_matriz_usuario_item(ratings)
    print(f"Matriz creada: {matriz_original.shape}. Entrenando modelo...")
    matriz_completada = entrenar_modelo_iterative(matriz_original)
    print("Modelo entrenado. Devolviendo datos.")
    return ratings, movies, matriz_original, matriz_completada

ratings_df, movies_df, matriz_original, matriz_completada = cargar_modelo_recomendador()

st.sidebar.header("Instrucciones")
st.sidebar.write("1. Elige al menos 5 películas de la lista.")
st.sidebar.write("2. Califica cada una con el slider.")
st.sidebar.write("3. Haz clic en 'Recomendar'.")
st.sidebar.write("4. El sistema te mostrará 10 películas que podrían gustarte.")

if st.sidebar.checkbox("Mostrar heatmap de matriz original"):
    st.subheader("Matriz usuario-película (valores faltantes en amarillo)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matriz_original.isnull().sample(frac=0.1, axis=1), cbar=False, cmap='viridis')
    st.pyplot(fig)

st.subheader("🎞️ Paso 1: Selecciona al menos 5 películas para calificar")

# Usar las películas que realmente están en la matriz completada (las 50 entrenadas)
pelis_populares_ids = list(matriz_completada.columns)
pelis_populares_df = movies_df[movies_df['item_id'].isin(pelis_populares_ids)].copy()
pelis_populares_df = pelis_populares_df.sort_values('title')
titulo_a_id = dict(zip(pelis_populares_df['title'], pelis_populares_df['item_id']))
opciones_titulos = pelis_populares_df['title'].tolist()

seleccion_titulos = st.multiselect(
    "Elige películas de la lista (mínimo 5):",
    options=opciones_titulos,
    default=opciones_titulos[:5]
)

ratings_usuario = {}
if seleccion_titulos:
    cols = st.columns(2)
    for idx, titulo in enumerate(seleccion_titulos):
        item_id = titulo_a_id[titulo]
        with cols[idx % 2]:
            rating = st.slider(
                f"{titulo}",
                min_value=1, max_value=5, value=3, step=1,
                key=f"rate_{item_id}"
            )
            ratings_usuario[item_id] = rating

# Botón fuera del condicional para que siempre se vea
col1, col2, col3 = st.columns([1,2,1])
with col2:
    boton = st.button("🎯 ¡Recomiéndame películas!", use_container_width=True)

if boton:
    if len(seleccion_titulos) < 5:
        st.warning(f"⚠️ Seleccionaste {len(seleccion_titulos)} películas. Necesitas al menos 5.")
    else:
        with st.spinner("Calculando recomendaciones..."):
            st.write(f"📊 Películas calificadas: {len(ratings_usuario)}")
            recomendaciones = recomendar_peliculas(ratings_usuario, matriz_completada, movies_df, top_n=10)
            if recomendaciones.empty:
                st.error("❌ No se pudieron generar recomendaciones. Prueba seleccionando otras películas de la lista.")
            else:
                st.subheader("🎥 Tus 10 recomendaciones")
                st.dataframe(recomendaciones.style.format({'rating_predicho': '{:.2f}'}))

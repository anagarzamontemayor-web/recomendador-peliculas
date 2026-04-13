import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def cargar_datos(ruta_ratings='u.data', ruta_items='u.item'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_ratings = os.path.join(script_dir, ruta_ratings)
    ruta_items = os.path.join(script_dir, ruta_items)
    
    ratings = pd.read_csv(ruta_ratings, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(ruta_items, sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['item_id', 'title'])
    return ratings, movies

def crear_matriz_usuario_item(ratings):
    matriz = ratings.pivot(index='user_id', columns='item_id', values='rating')
    matriz = matriz.astype(float)
    return matriz

def entrenar_modelo_iterative(matriz, n_usuarios=50, n_peliculas=50, max_iter=1):
    """
    Entrena sobre una muestra reducida y devuelve la matriz completada.
    """
    usuarios_activos = matriz.count(axis=1).sort_values(ascending=False).head(n_usuarios).index
    pelis_populares = matriz.count().sort_values(ascending=False).head(n_peliculas).index
    
    matriz_reducida = matriz.loc[usuarios_activos, pelis_populares]
    matriz_limpia = matriz_reducida.dropna(how='all').dropna(axis=1, how='all')
    
    if matriz_limpia.empty:
        raise ValueError("Matriz vacía tras limpieza.")
    
    print(f"Entrenando con matriz de {matriz_limpia.shape[0]} usuarios x {matriz_limpia.shape[1]} películas...")
    datos = matriz_limpia.values.astype(np.float64)
    
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    completada = imputer.fit_transform(datos)
    
    matriz_completada = pd.DataFrame(completada, 
                                     index=matriz_limpia.index, 
                                     columns=matriz_limpia.columns)
    print("Entrenamiento completado.")
    return matriz_completada

def recomendar_peliculas(ratings_usuario, matriz_completada, movies_df, top_n=10):
    """
    Usa la matriz ya completada (cacheada) para recomendar.
    Se asume que 'ratings_usuario' son calificaciones del nuevo usuario.
    Para predecir, se usa el promedio de los ratings de los 10 usuarios más similares
    (similitud coseno) sobre la matriz completada.
    """
    # Filtrar películas que están en la matriz completada
    pelis_validas = [p for p in ratings_usuario.keys() if p in matriz_completada.columns]
    if len(pelis_validas) < 3:
        return pd.DataFrame(columns=['title', 'rating_predicho'])
    
    # Crear vector del nuevo usuario solo con las películas en la matriz
    nuevo_vector = pd.Series(index=matriz_completada.columns, dtype=float)
    for p in pelis_validas:
        nuevo_vector[p] = ratings_usuario[p]
    
    # Calcular similitud coseno con todos los usuarios existentes
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Rellenar NaN con 0 para el cálculo de similitud (en la matriz completada no hay NaN)
    matriz_llena = matriz_completada.fillna(0)
    nuevo_lleno = nuevo_vector.fillna(0).values.reshape(1, -1)
    
    similitudes = cosine_similarity(nuevo_lleno, matriz_llena.values).flatten()
    # Excluir al propio usuario (si existiera, pero aquí no)
    usuarios_similares = np.argsort(similitudes)[::-1][:10]  # top 10 más similares
    
    # Predecir rating para cada película no calificada como promedio ponderado por similitud
    pelis_no_calificadas = [c for c in matriz_completada.columns if c not in ratings_usuario]
    predicciones = {}
    for peli in pelis_no_calificadas:
        ratings_similares = matriz_completada.iloc[usuarios_similares][peli].values
        sims = similitudes[usuarios_similares]
        # Evitar división por cero
        if np.sum(np.abs(sims)) > 0:
            pred = np.dot(ratings_similares, sims) / np.sum(np.abs(sims))
        else:
            pred = np.nanmean(ratings_similares)
        predicciones[peli] = pred
    
    # Ordenar y tomar top_n
    predicciones_series = pd.Series(predicciones).sort_values(ascending=False).head(top_n)
    recomendaciones_df = pd.DataFrame({'item_id': predicciones_series.index, 'rating_predicho': predicciones_series.values})
    recomendaciones_df = recomendaciones_df.merge(movies_df, on='item_id', how='left')
    return recomendaciones_df[['title', 'rating_predicho']]

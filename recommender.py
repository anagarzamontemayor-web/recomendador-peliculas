import pandas as pd
import numpy as np
from fancyimpute import SoftImpute

def cargar_datos(ruta_ratings='u.data', ruta_items='u.item'):
    # Cargar ratings
    ratings = pd.read_csv(ruta_ratings, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # Cargar títulos de películas (solo columnas id y título)
    movies = pd.read_csv(ruta_items, sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['item_id', 'title'])
    return ratings, movies

def crear_matriz_usuario_item(ratings):
    matriz = ratings.pivot(index='user_id', columns='item_id', values='rating')
    return matriz

def entrenar_modelo_softimpute(matriz):
    completada = SoftImpute().fit_transform(matriz.values)
    completada_df = pd.DataFrame(completada, index=matriz.index, columns=matriz.columns)
    return completada_df

def recomendar_peliculas(ratings_usuario, matriz_original, movies_df, top_n=10):
    """
    ratings_usuario: dict {item_id: rating}
    """
    nuevo_user_id = matriz_original.index.max() + 1
    nueva_fila = pd.Series(index=matriz_original.columns, dtype=float)
    for item_id, rating in ratings_usuario.items():
        if item_id in nueva_fila.index:
            nueva_fila[item_id] = rating
    
    matriz_ampliada = matriz_original.copy()
    matriz_ampliada.loc[nuevo_user_id] = nueva_fila
    
    matriz_ampliada_completada = SoftImpute().fit_transform(matriz_ampliada.values)
    matriz_ampliada_completada = pd.DataFrame(matriz_ampliada_completada, 
                                              index=matriz_ampliada.index, 
                                              columns=matriz_ampliada.columns)
    predicciones_usuario = matriz_ampliada_completada.loc[nuevo_user_id]
    pelis_calificadas = set(ratings_usuario.keys())
    predicciones_filtradas = predicciones_usuario.drop(labels=pelis_calificadas, errors='ignore')
    recomendaciones = predicciones_filtradas.sort_values(ascending=False).head(top_n)
    recomendaciones_df = pd.DataFrame({'item_id': recomendaciones.index, 'rating_predicho': recomendaciones.values})
    recomendaciones_df = recomendaciones_df.merge(movies_df, on='item_id', how='left')
    return recomendaciones_df[['title', 'rating_predicho']]

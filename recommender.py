import pandas as pd
import numpy as np
import os
from fancyimpute import SoftImpute

def cargar_datos(ruta_ratings='u.data', ruta_items='u.item'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_ratings = os.path.join(script_dir, ruta_ratings)
    ruta_items = os.path.join(script_dir, ruta_items)
    
    ratings = pd.read_csv(ruta_ratings, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(ruta_items, sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['item_id', 'title'])
    return ratings, movies

def crear_matriz_usuario_item(ratings):
    matriz = ratings.pivot(index='user_id', columns='item_id', values='rating')
    # Asegurar que todos los valores son float
    matriz = matriz.astype(float)
    return matriz

def entrenar_modelo_softimpute(matriz):
    # Eliminar filas y columnas completamente vacías
    matriz_limpia = matriz.dropna(how='all').dropna(axis=1, how='all')
    if matriz_limpia.empty:
        raise ValueError("La matriz está completamente vacía después de limpiar filas/columnas sin ratings.")
    
    # Convertir a numpy float64 explícitamente
    datos = matriz_limpia.values.astype(np.float64)
    
    # Aplicar SoftImpute
    completada = SoftImpute().fit_transform(datos)
    
    # Reconstruir DataFrame con los índices/columnas originales
    matriz_completada = pd.DataFrame(completada, 
                                     index=matriz_limpia.index, 
                                     columns=matriz_limpia.columns)
    return matriz_completada

def recomendar_peliculas(ratings_usuario, matriz_original, movies_df, top_n=10):
    nuevo_user_id = matriz_original.index.max() + 1
    nueva_fila = pd.Series(index=matriz_original.columns, dtype=float)
    for item_id, rating in ratings_usuario.items():
        if item_id in nueva_fila.index:
            nueva_fila[item_id] = float(rating)
    matriz_ampliada = matriz_original.copy()
    matriz_ampliada.loc[nuevo_user_id] = nueva_fila
    
    # Limpiar filas/columnas completamente vacías antes de SoftImpute
    matriz_ampliada_limpia = matriz_ampliada.dropna(how='all').dropna(axis=1, how='all')
    datos = matriz_ampliada_limpia.values.astype(np.float64)
    completada = SoftImpute().fit_transform(datos)
    
    matriz_ampliada_completada = pd.DataFrame(completada, 
                                              index=matriz_ampliada_limpia.index, 
                                              columns=matriz_ampliada_limpia.columns)
    
    # El nuevo usuario puede haber sido eliminado si solo calificó películas que luego fueron eliminadas
    if nuevo_user_id not in matriz_ampliada_completada.index:
        # Reintegrar el usuario con sus predicciones originales (menos preciso pero evita error)
        return pd.DataFrame(columns=['title', 'rating_predicho'])
    
    predicciones_usuario = matriz_ampliada_completada.loc[nuevo_user_id]
    pelis_calificadas = set(ratings_usuario.keys())
    predicciones_filtradas = predicciones_usuario.drop(labels=pelis_calificadas, errors='ignore')
    recomendaciones = predicciones_filtradas.sort_values(ascending=False).head(top_n)
    
    recomendaciones_df = pd.DataFrame({'item_id': recomendaciones.index, 'rating_predicho': recomendaciones.values})
    recomendaciones_df = recomendaciones_df.merge(movies_df, on='item_id', how='left')
    return recomendaciones_df[['title', 'rating_predicho']]

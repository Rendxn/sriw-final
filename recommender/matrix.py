import pandas as pd
import numpy as np

columns: list[str] = ['uri', 'name', 'description', 'thumbnail', 'style',
                      'manufacturer', 'layout']

cat_columns: list[str] = ['style', 'manufacturer', 'layout']


def get_encoded_cars(data: list) -> tuple[pd.Index, np.ndarray]:
    df = pd.DataFrame(data, columns=columns)
    df.set_index(['uri', 'name', 'description',
                 'thumbnail'], inplace=True)

    df = pd.get_dummies(df, columns=cat_columns)
    return df.index, df.to_numpy()


def get_weighted_matrix(encoded_matrix: np.ndarray, weight_scores: np.ndarray):
    weighted_matrix = (encoded_matrix.T * weight_scores).T
    # No hay necesidad de normalizar, pues todas las variables son indicadoras.
    return weighted_matrix

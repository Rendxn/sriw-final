import numpy as np


def get_content_recommendation(weighted_profile: np.ndarray, encoded_matrix: np.ndarray):
    # Calculamos las distancias que hay entre los carros y el perfil
    distances = np.linalg.norm(
        encoded_matrix - weighted_profile, axis=1, ord=2)

    # Retornamos los índices de los carros en orden ascendente.
    return np.argsort(distances)

import numpy as np


# Calc the distance between 2 califications
def get_distance(my_profile: np.ndarray, other_profile: np.ndarray):

    distances = np.linalg.norm(
        my_profile - other_profile, ord=2)

    return distances


def get_content_recommendation(weighted_profile: np.ndarray, encoded_matrix: np.ndarray):
    # Calculamos las distancias que hay entre los carros y el perfil
    distances = np.linalg.norm(
        encoded_matrix - weighted_profile, axis=1, ord=2)

    # Retornamos los índices de los carros en orden ascendente.
    return np.argsort(distances)

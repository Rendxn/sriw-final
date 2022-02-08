import json
import numpy as np


def save_scores(scores: dict[str, int]) -> list[dict[str, int]]:
    with open('resources/data/scores.json', 'r+') as f:
        data = json.load(f)
        data.append(scores)
        f.seek(0)
        json.dump(data, f, indent=4)
        return data


def get_scores() -> list[dict[str, int]]:
    with open('resources/data/scores.json', 'r+') as f:
        scores = json.load(f)
        return scores


def get_extended_score(scores: dict[str, int], imputate_with: int = 0, length: int = 1000) -> list[dict[str, int]]:
    extended_score = np.zeros((1, length))
    np.put(extended_score, [int(ind) for ind in scores.keys()], [
           int(val) for val in scores.values()])
    return extended_score


def get_profile(weighted_matrix: np.ndarray):
    return np.sum(a=weighted_matrix, axis=0)


def save_profile_json(profile: np.ndarray):
    profile_lst = profile.tolist()
    with open('resources/data/profiles.json', 'r+') as f:
        data = json.load(f)
        data.append(profile_lst)
        f.seek(0)
        json.dump(data, f, indent=4)
        return profile_lst


def get_weighted_profile(profile: np.ndarray, weight_scores: np.ndarray):
    # No hay necesidad de normalizar, pues todas las variables son indicadoras [0,1].
    return np.true_divide(profile, np.sum(weight_scores))

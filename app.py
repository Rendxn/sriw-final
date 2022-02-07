from random import sample
from flask import Flask, jsonify, render_template, redirect, request
from cars.cars import fetch_cars_from_dbpedia
from recommender.matrix import get_encoded_cars, get_weighted_matrix
from recommender.profile import get_extended_score, get_profile, get_weighted_profile, save_scores
from recommender.recommender import get_content_recommendation

cars = fetch_cars_from_dbpedia()
encoded_cars_index, encoded_cars = get_encoded_cars(cars)

app = Flask(__name__)
app.cars = cars
app.encoded_cars = encoded_cars

cars_sample: list = []


@app.route('/')
def home():
    # enumerate para preservar el index original
    # pues este retorna una tupla (index, value)
    cars_sample = sample(list(enumerate(cars)), 5)
    return render_template('index.html', cars=cars_sample)


@app.route('/car/<id>')
def get_car(id: int):
    return jsonify(encoded_cars.iloc[int(id)].to_dict())


@app.route('/recommend-car', methods=['POST'])
def recommend_car():
    if request.method != 'POST':
        return redirect('/')
    else:
        data = request.form.to_dict()
        scores = {}
        items = data.items()
        for index, value in items:
            scores[int(index)] = int(value)
        # Guardamos el perfil del usuario en el json
        # `resources/data/profiles.json`
        save_scores(scores)

        extended_scores = get_extended_score(
            scores, imputate_with=0, length=len(encoded_cars))
        weighted_matrix = get_weighted_matrix(
            encoded_matrix=encoded_cars, weight_scores=extended_scores)
        profile = get_profile(weighted_matrix)
        weighted_profile = get_weighted_profile(profile, extended_scores)
        indexes = get_content_recommendation(weighted_profile, encoded_cars)

        return render_template('recommendation.html', recommended=indexes.tolist(), cars=cars)


@app.route('/test')
def test():
    scores = get_extended_score(scores={
        "275": 5,
        "82": 6,
        "463": 8,
        "459": 5,
        "72": 9
    }, imputate_with=0, length=len(encoded_cars))
    weighted_matrix = get_weighted_matrix(
        encoded_matrix=encoded_cars, weight_scores=scores)
    profile = get_profile(weighted_matrix)
    weighted_profile = get_weighted_profile(profile, scores)
    indexes = get_content_recommendation(weighted_profile, encoded_cars)
    return jsonify(indexes.tolist())

from random import sample
from flask import Flask, jsonify, render_template, redirect, request
from cars.cars import fetch_cars_from_dbpedia
from recommender.matrix import get_encoded_cars, get_weighted_matrix
from recommender.profile import *
from recommender.recommender import *
import numpy as np

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
        # Get and build my scores
        data = request.form.to_dict()
        my_scores = {}
        items = data.items()
        for index, value in items:
            my_scores[int(index)] = int(value)
        my_extended_scores = get_extended_score(
            my_scores, imputate_with=0, length=len(encoded_cars))

        # Get other people scores
        people_scores = get_scores()
        people_extended = np.array([get_extended_score(
            p, imputate_with=0, length=len(encoded_cars)) for p in people_scores])
        # Solver problems with shapes
        people_extended_flat = people_extended.reshape(
            -1, people_extended.shape[-1])

        # Calc distances
        distances = [get_distance(my_extended_scores, p)
                     for p in people_extended_flat]
        distances_norm = np.array([
            0 if dist == 0 else 1 / dist for dist in distances])

        # Begin the best part reccomend!
        sum_row = np.sum((people_extended_flat.T * distances_norm).T, axis=0)

        # Read other weighted profiles
        # weightedProfiles = get_weighted_profiles_json()

        # return render_template('recommendation.html', recommended=indexes.tolist(), cars=cars)

        # save_scores(my_scores)
        return jsonify(sum_row.tolist())


@ app.route('/test')
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

from random import sample
from flask import Flask, render_template
from cars.cars import fetch_cars_from_dbpedia
from recommender.matrix import content_matrix

app = Flask(__name__)


cars = fetch_cars_from_dbpedia()
matrix_index, matrix = content_matrix(cars)

app.cars = cars
app.content_matrix = matrix


@app.route('/')
def home():
    return render_template('index.html', cars=sample(cars, 5))


@app.route('/recommend-car', methods=['POST'])
def recommend_car():
    return render_template('recommendation.html')

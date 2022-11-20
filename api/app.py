from flask import Flask
from flask import request
from joblib import load
import os
import glob

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow
#     - query parameter
#     - get call / methods
#     - post call / methods **

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y
    return {'sum': z}


@app.route("/find_best_model")
def find_best_model():
    best_model = ''
    best_f1 = 0
    processed_files_path = '../results/'
    processed_files = glob.glob(os.path.join(processed_files_path, "*.txt"))
    for filename in processed_files:
        with open(filename, 'r') as file:
            str = file.read()
            lines = str.split('\n')
            f1 = float(lines[1].split(':')[1])
            model = lines[2][15:]
        if f1 > best_f1:
            best_f1 = f1
            best_model = model.split('/')[-1]

    return best_model


@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    model_name = request.json.get('model_name')
    if model_name == None:
        model_name = find_best_model()
    print("done loading")
    model_path = f"../models/{model_name}"
    model = load(model_path)
    predicted = model.predict([image])
    return {"y_predicted": int(predicted[0])}

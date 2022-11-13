from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "../models/SVC(C=0.7, gamma=0.001)"
model = load(model_path)


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


@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    image2 = request.json['image2']
    print("done loading")
    predicted = model.predict([image1, image2])
    return {"same-digit": 'true' if predicted[0] == predicted[0] else 'false'}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

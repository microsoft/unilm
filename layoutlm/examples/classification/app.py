#!flask/bin/python
import os
from flask import Flask, request, jsonify, abort
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
from examples.classification.predict_api import predict
from examples.classification.train import do_training
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_label():
    if not request.json or not 'img' in request.json:
        abort(400)
    response = predict(request.json['img'])
    return jsonify(response), 201
@app.route('/train', methods=['POST'])
def train_label():
    if not request.json or not 'img' in request.json or not 'label' in request.json:
        abort(400)
    response = do_training(request.json['img'], request.json['label'])
    return jsonify({}), 201

if __name__ == '__main__':
    app.run(debug=True)
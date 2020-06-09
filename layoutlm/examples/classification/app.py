#!flask/bin/python
from flask import Flask, request
from examples.classification.predict_api import predict
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_label():
    if not request.json or not 'img' in request.json:
        abort(400)
    response = predict(request.json['img'])
    return jsonify(response), 201

if __name__ == '__main__':
    app.run(debug=True)
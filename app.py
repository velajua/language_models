from __future__ import print_function

import os
import sys
import json
import requests
import numpy as np

from dill import load
from envyaml import EnvYAML
from functools import lru_cache

from google.cloud import storage

from flask import Flask, request, jsonify, render_template
# from model_deployment.model_deployment import deploy_models

FILE_PREF = '' if "language_models" in os.getcwd() else '/tmp/'
MODELS = EnvYAML("config.yaml")["MODELS"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


app = Flask(__name__)
app.json_encoder = NpEncoder


def get_blob(model_name):
    path = f'{model_name}.dill'
    bucket = storage.Client().get_bucket(
        'my_model_deployment') 
    blob_of_file = bucket.blob(path)
    return blob_of_file


def get_model(model_name):
    blob = get_blob(model_name)
    blob.download_to_filename(f"{FILE_PREF}{model_name}.dill")
    with open(f"{FILE_PREF}{model_name}.dill", "rb") as f:
        model = load(f)
    return model


@lru_cache(maxsize=1)
def load_models():
    dicto_models = {}
    for model in MODELS:
        try:
            model_loaded = get_model(model)
            dicto_models[model] = model_loaded
        except Exception:
            print(f"model {model} not in storage")
            if os.path.exists(model + '.dill'):
                with open(f"{model}.dill", "rb") as f:
                    dicto_models[model] = load(f)
            elif os.path.exists(FILE_PREF + model + '.dill'):
                with open(f"{FILE_PREF}{model}.dill", "rb") as f:
                    dicto_models[model] = load(f)
    print('models loaded:', dicto_models)
    return dicto_models


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/model_deployment')
def model_deployment():
    # deploy_models()
    return '200'


@app.route('/predict_proxy', methods=['POST'])
def predict_proxy():
    form_data = request.form
    modified_data = {}
    for key, value in form_data.items():
        if key == 'model_name':
            modified_data[key] = (
                'entity_match_model' if value == '0' else
                'russian_translation_model' if value == '1' else
                'summarize_data_model')
        else:
            modified_data[key] = value
    response = requests.post(
        request.url_root + 'predict', json=modified_data,
        headers={'Content-Type': 'application/json'})
    return response.json()


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        data = request.get_json()
        if data.get('model_name',
                    '') not in MODELS and not data.get(
                        'data', ''):
            return {'404': 'Request Incomplete'}
        current_model = model[data['model_name']]
        pred = current_model.predict(data['data'])
        return jsonify({"body": pred})
    else:
        return {'405': 'Method not allowed'}


if __name__ == '__main__' or __name__ == 'app':
    model = load_models()
    app.run(host='0.0.0.0', port=8080)

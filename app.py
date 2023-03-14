from __future__ import print_function
import sys

import os
import json
import requests
import numpy as np

from dill import load
from envyaml import EnvYAML
from functools import lru_cache

from typing import Union, Dict

from google.cloud import storage
from flask import Flask, request, jsonify, render_template

# from model_deployment.model_deployment import deploy_models

FILE_PREF = '' if "language_models" in os.getcwd() else '/tmp/'
MODELS = EnvYAML("config.yaml")["MODELS"]
global model

class NpEncoder(json.JSONEncoder):
    """
    A custom encoder for serializing numpy types to json.
    """
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


def get_blob(model_name: str) -> storage.Blob:
    """
    Retrieve a Google Cloud Storage blob for a given model name.

    Args:
        model_name: A string representing the name of the model.

    Returns:
        A Blob object representing the model file in GCS.
    """
    path = f'{model_name}.dill'
    bucket = storage.Client().get_bucket('my_model_deployment')
    blob_of_file = bucket.blob(path)
    return blob_of_file


def get_model(model_name: str) -> object:
    """
    Download a model file from GCS and load it into memory.

    Args:
        model_name: A string representing the name of the model.

    Returns:
        A loaded model object.
    """
    blob = get_blob(model_name)
    blob.download_to_filename(f"{FILE_PREF}{model_name}.dill")
    with open(f"{FILE_PREF}{model_name}.dill", "rb") as f:
        model = load(f)
    return model


@lru_cache(maxsize=1)
def load_models() -> Dict[str, object]:
    """
    Load all the models specified in the MODELS config file into memory.

    Returns:
        A dictionary containing all the loaded models.
    """
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
def homepage() -> requests.Response:
    """
    Renders the homepage template.

    Returns:
        A rendered HTML page.
    """
    return render_template('home.html')


@app.route('/model_deployment')
def model_deployment() -> str:
    """
    Endpoint to deploy the trained models.

    Returns:
        A '200' status code indicating the success of the operation.
    """
    global model
    # deploy_models()
    # model = load_models()
    return '200'


@app.route('/predict_proxy', methods=['POST'])
def predict_proxy() -> Union[Dict[str, Union[str, int]], Dict[str, str]]:
    """
    Endpoint to proxy the prediction request to the appropriate model.

    Returns:
        A JSON response containing the prediction.
    """
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
    print('sending to predict:', modified_data, file=sys.stderr)
    response = requests.request('POST',
        '/'.join(request.base_url.split('/')[:-1]) + '/predict',
        json=modified_data, params=modified_data)
    return jsonify(response.json())


@app.route('/predict', methods=['GET', 'POST'])
def get_prediction() -> Union[Dict[str, Union[str, int]], Dict[str, str]]:
    global model
    """
    Endpoint to get the prediction from the appropriate model.

    Returns:
        A JSON response containing the prediction or an error message.
    """
    if request.method == 'POST':
        data = request.get_json()
    elif request.method == 'GET':
        data = request.args.to_dict()
    if (data and data.get('model_name', '') not in MODELS
            and not data.get('data', '')):
        return {'404': 'Request Incomplete'}
    current_model = model[data['model_name']]
    pred = current_model.predict(data['data'])
    print('prediction:', pred, file=sys.stderr)
    return jsonify({"body": pred})


if __name__ == '__main__' or __name__ == 'app':
    model = load_models()
    app.run(host='0.0.0.0', port=8080)

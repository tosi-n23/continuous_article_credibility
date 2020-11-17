from flask import Blueprint, request, abort, jsonify
# from ..longformer_model.prediction import predict # for running pytest
from longformer_model.prediction import predict # for hosting api

import os

from .config import get_logger
from . import __version__ as api_version

_logger = get_logger(logger_name=__name__)


article_credibility_app = Blueprint('article_credibility_app', __name__)


@article_credibility_app.route("/")
def instructions():
    return 'MISINFORMATION CLASSIFICATION' \
           '\n' \
           '\nPOST a JSON file with article body to classify credibility of content'




@article_credibility_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@article_credibility_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({
            # 'model_version': _version,
                        'api_version': api_version})


@article_credibility_app.route('/v2/predict/distilroberta_cls', methods=['POST'])
def classify():
    # if request.method == 'POST':
    # Step 1: Extract POST data from request body as JSON
    json_data = request.get_json(force=True)
    # _logger.debug(f'Inputs: {json_data}')
    # _logger.info(f'Inputs: {json_data}')
    # print(type(json_data))

    # Check if required fields are in query
    if isinstance(json_data, dict):
        print('yes datatype confirmed')
        # if any(key not in json_data for key in ('_id', 'title', 'domain', 'body')):
        if any(key not in json_data for key in ('_id', 'title', 'sentiment', 'source_cred', 'body')):
            abort(400)

        sentiment = json_data.get('sentiment')

        def flatten_dict(d):
            def expand(key, value):
                if isinstance(value, dict):
                    return [ (key, v) for k, v in flatten_dict(value).items() ]
                else:
                    return [ (key, value) ]

            items = [ item for k, v in d.items() for item in expand(k, v) ]

            return dict(items)

        sentiment = flatten_dict(sentiment)

        sentiment = max(sentiment, key=sentiment.get)

        print(sentiment)

        concat_feat = json_data.get('title') + ' ' + sentiment + ' ' + json_data.get('source_cred') + ' ' + json_data.get('body')
        # concat_feat = json_data.get('title') + ' ' + sentiment(json_data.get('body')) + ' ' + source_cred(json_data.get('domain')) + ' ' + json_data.get('body')
        print(concat_feat)

        # Step 2: Model prediction
        result, confidence = predict(content=concat_feat)
        
        # _logger.debug(f'Outputs: {result}')
        # _logger.info(f'Outputs: {result, confidence}')
        print(result)

        # Step 3: Return the response as JSON
        return jsonify({'_id': json_data['_id'],
                        'predictions': result,
                        'confidence': confidence})
                        #'version': version})
    else:
        abort(400)
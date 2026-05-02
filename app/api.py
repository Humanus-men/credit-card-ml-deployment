from flask import Flask, request, jsonify
import logging
import os
from pathlib import Path
from app.model_handler import ModelHandler

app = Flask(__name__)

# Логирование в JSON
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Корень проекта — на один уровень выше папки app/ (родительская директория для app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_V1_PATH = os.environ.get(
    "MODEL_V1_PATH",
    str(PROJECT_ROOT / "models" / "model_v1.pkl")
)
MODEL_V2_PATH = os.environ.get(
    "MODEL_V2_PATH",
    str(PROJECT_ROOT / "models" / "model_v2.pkl")
)
PREPROCESSOR_PATH = os.environ.get(
    "PREPROCESSOR_PATH",
    str(PROJECT_ROOT / "models" / "preprocessor.pkl")
)

# Инициализация обработчиков
model_v1 = ModelHandler(MODEL_V1_PATH, PREPROCESSOR_PATH)
model_v2 = ModelHandler(MODEL_V2_PATH, PREPROCESSOR_PATH)

@app.before_request
def log_request():
    app.logger.info(f"Request: {request.method} {request.path}")

@app.route('/predict', methods=['POST'])
def predict():
    # Выбор версии модели через заголовок X-Model-Version
    model_version = request.headers.get('X-Model-Version', 'v1')
    if model_version == 'v1':
        model = model_v1
    elif model_version == 'v2':
        model = model_v2
    else:
        return jsonify({'error': 'Invalid model version. Use v1 or v2'}), 400

    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Request body must be JSON'}), 400

        result = model.predict(data)
        result['model_version'] = model_version
        app.logger.info(f"Prediction: {result}")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
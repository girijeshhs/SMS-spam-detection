from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_spam, train_and_save_model

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = predict_spam(text)
    return jsonify({'spam': bool(result)})

@app.route('/train', methods=['POST'])
def train():
    train_and_save_model()
    return jsonify({'status': 'Model retrained'})


# Health check route
@app.route('/', methods=['GET'])
def home():
    return 'Spam Detector API is running! Use /predict (POST) to check spam.'

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

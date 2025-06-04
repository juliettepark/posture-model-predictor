from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model_filename = 'retry_fourtypes_posture_model.pkl'
model = pickle.load(open(model_filename, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()['data']
        data = json.loads(data)
        print(data)

        # flatten into a 1-row by as many columns as we need
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="localhost",port=4000,debug=True)
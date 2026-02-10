from flask import Flask, request, jsonify
import joblib
from utils import preprocess_input

app = Flask(__name__)

# Load the best model (Random Forest)
model = joblib.load('models/rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Preprocess using helper functions
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            'heart_stroke_risk': int(prediction),
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
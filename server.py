from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS class
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open('random_forest_model.pkl', 'rb') as model_file:
    model, label_encoder = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Input validation
        if 'features' not in data:
            raise ValueError("Missing 'features' in the input data")

        features = data['features']

        # Make the prediction
        prediction = model.predict([features])[0]

        return jsonify({'prediction': prediction})

    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"})

    except ValueError as e:
        return jsonify({'error': str(e)})

    except Exception as e:
        # Log the exception for further analysis
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'})

if __name__ == '__main__':
    # Set debug=True for development, but disable in production
    app.run(debug=False,host='0.0.0.0')

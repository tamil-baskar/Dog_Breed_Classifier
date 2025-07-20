from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import traceback
import sys
from predictor import DogBreedPredictor

print(f"\n=== Starting Dog Breed Classifier ===")
print(f"Python version: {sys.version}")

# Set the template folder explicitly
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the predictor
predictor = DogBreedPredictor()

print(f"\n=== Predictor initialized ===")
print(f"Device: {predictor.device}")
print(f"Number of classes: {len(predictor.class_to_idx)}")

@app.route('/')
def index():
    print("\n=== Serving index page ===")
    print(f"Template folder: {app.template_folder}")
    print(f"Current directory: {os.getcwd()}")
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("\n=== Prediction request received ===")
    try:
        if 'image' not in request.files:
            print("Error: No image file provided")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("Error: No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save the image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"Image saved to: {filepath}")

            try:
                # Use the exact same predictor logic
                result = predictor.predict(filepath)
                
                # Format the response to match your test script output
                return jsonify({
                    'top_prediction': result['top_prediction'],
                    'all_predictions': result['all_predictions']
                })
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return jsonify({
                    'error': str(e)
                }), 500
                
    except Exception as e:
        print(f"\n=== Error during prediction ===")
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({
            'error': 'An error occurred during prediction',
            'debug': str(e)
        }), 500

if __name__ == '__main__':
    print("\n=== Starting Flask server ===")
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=True)

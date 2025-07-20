# Dog Breed Classifier

A web-based application that uses deep learning to classify dog breeds from images.

## Features

- Upload dog images through a user-friendly interface
- Real-time breed prediction with confidence scores
- Displays top prediction and all possible breed predictions
- Modern, responsive web interface

## Prerequisites

- Python 3.8 or higher
- Git
- Web browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tamil-baskar/Dog_Breed_Classifier.git
cd Dog_Breed_Classifier
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the "Choose File" button or drag and drop a dog image into the drop zone
2. Wait for the prediction to process
3. View the predicted breed and confidence score
4. See all possible breed predictions with their percentages

## Project Structure

```
.
├── app.py              # Main Flask application
├── predictor.py        # Dog breed prediction logic
├── breed_classifier.pth  # Trained model weights
├── class_to_idx.json    # Class mapping
├── requirements.txt     # Python dependencies
├── static/            # Static files
└── templates/         # HTML templates
```

## Model Details

- Architecture: ResNet18
- Input Size: 224x224 pixels
- Minimum Confidence Threshold: 94%
- Runs on CPU

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

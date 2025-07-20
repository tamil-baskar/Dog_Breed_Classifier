import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

class DogBreedPredictor:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cpu")
        print(f"Using device: CPU")

        # Load class-to-index mapping
        with open("class_to_idx.json", "r") as f:
            self.class_to_idx = json.load(f)

        # Reverse the mapping to get index-to-class
        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Load the model
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.idx_to_class))
        self.model.load_state_dict(torch.load("breed_classifier.pth", map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image_path):
        try:
            # Load and preprocess the input image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

            # Perform prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class_idx = torch.argmax(probabilities).item()
                breed = self.idx_to_class[predicted_class_idx]

            # Get the confidence score
            confidence_score = probabilities[predicted_class_idx].item() * 100
            
            # Prepare response
            if confidence_score > 94:
                result = {
                    "top_prediction": {
                        "breed": breed,
                        "confidence": f"{confidence_score:.2f}%"
                    },
                    "all_predictions": []
                }
                
                # Add all predictions with their percentages
                for idx, prob in enumerate(probabilities):
                    result["all_predictions"].append({
                        "breed": self.idx_to_class[idx],
                        "percentage": f"{prob.item() * 100:.2f}%"
                    })
            else:
                result = {
                    "top_prediction": {
                        "breed": "Cannot recognize",
                        "confidence": f"{confidence_score:.2f}%"
                    },
                    "all_predictions": []
                }

            return result

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

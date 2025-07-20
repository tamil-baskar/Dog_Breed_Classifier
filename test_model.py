import torch
from model import get_model

print("Testing model loading...")

try:
    # Load the model
    model = get_model(num_classes=120)
    print("Model created successfully")
    
    # Try loading the state dict
    try:
        model.load_state_dict(torch.load('breed_classifier.pth', map_location='cpu'))
        print("Model state dict loaded successfully")
    except Exception as e:
        print(f"Error loading model state dict: {str(e)}")
        
    model.eval()
    print("Model ready for inference")
    
except Exception as e:
    print(f"Error: {str(e)}")

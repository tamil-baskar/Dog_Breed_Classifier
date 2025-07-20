import torch
import torch.nn as nn
import torchvision.models as models

class BreedClassifier(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        
        # Use ResNet18 as base model
        self.base_model = models.resnet18(weights=None)
        
        # Freeze the base model layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the last layer with our classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Initialize model with the correct number of classes
def get_model(num_classes=21):
    model = BreedClassifier(num_classes)
    return model
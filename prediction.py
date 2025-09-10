import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import argparse
import os

# Import utilities
from utils import (
    get_device, CIFAR10_CLASSES, get_inference_transforms,
    smart_resize_with_padding, load_model_state_dict,
    compile_model_if_available, CIFAR10Net
)

# Create a regrouped model class for inference
class RegroupedCIFAR10Net(nn.Module):
    def __init__(self, num_classes=3):
        super(RegroupedCIFAR10Net, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

device = get_device()
print(f'Using device: {device}')

# CIFAR-10 classes
classes = CIFAR10_CLASSES
regrouped_classes = ['Vehicles', 'Animals', 'Other']

# Cached model for repeated inference
_cached_model = None
_cached_model_path = None

def detect_model_type(model_path, device):
    """Detect if model is original CIFAR-10 (10 classes) or regrouped (3 classes)"""
    state_dict = load_model_state_dict(model_path, device)
    
    # Check the shape of the final layer to determine model type
    if 'fc2.weight' in state_dict:
        output_size = state_dict['fc2.weight'].shape[0]
        if output_size == 10:
            return 'original', classes
        elif output_size == 3:
            return 'regrouped', regrouped_classes
    
    # Default to original if we can't determine
    return 'original', classes

def load_model_cached(model_path, device):
    """Load model with caching for repeated inference"""
    global _cached_model, _cached_model_path
    
    if _cached_model is None or _cached_model_path != model_path:
        print(f"Loading model from '{model_path}'...")
        
        # Detect model type
        model_type, model_classes = detect_model_type(model_path, device)
        print(f"Detected model type: {model_type} ({len(model_classes)} classes)")
        
        # Load state dict using utility function
        state_dict = load_model_state_dict(model_path, device)
        
        # Load appropriate model architecture
        if model_type == 'regrouped':
            model = RegroupedCIFAR10Net(num_classes=3).to(device)
        else:
            model = CIFAR10Net().to(device)
        
        model.load_state_dict(state_dict)
        
        model.eval()
        
        # Compile model for faster inference if available
        model = compile_model_if_available(model)
        
        _cached_model = model
        _cached_model_path = model_path
        print("Model loaded and cached")
    
    return _cached_model

def predict_image(model_path, image_path, device=device):
    """Run inference on a single image with model caching"""
    # Detect model type and get appropriate classes
    model_type, model_classes = detect_model_type(model_path, device)
    
    # Load the trained model (cached)
    model = load_model_cached(model_path, device)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Smart resize with padding to preserve aspect ratio
    image = smart_resize_with_padding(image, target_size=32)
    
    # Apply normalization (same as training)
    transform = get_inference_transforms()
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"Predicted class: {model_classes[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top predictions (up to 3 or total classes, whichever is smaller)
    num_top = min(3, len(model_classes))
    top_prob, top_classes = torch.topk(probabilities, num_top)
    print(f"\nTop {num_top} predictions:")
    for i in range(num_top):
        class_idx = top_classes[0][i].item()
        prob = top_prob[0][i].item()
        print(f"{i+1}. {model_classes[class_idx]}: {prob:.4f}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Prediction')
    parser.add_argument('--model-path', type=str, default='models/cifar10_model.pth',
                       help='Path to trained model file (default: models/cifar10_model.pth)')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to image for prediction')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found. Please train the model first or specify correct path.")
        exit(1)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        exit(1)
    
    print(f"Running prediction on '{args.image_path}' using model '{args.model_path}'...")
    predicted_class, confidence = predict_image(args.model_path, args.image_path, device)
    
    # Explicit cleanup to reduce exit delay
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
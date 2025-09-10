from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import os
import traceback

# Import utilities and model classes
from utils import (
    get_device, CIFAR10_CLASSES, get_inference_transforms,
    smart_resize_with_padding, load_model_state_dict,
    compile_model_if_available, CIFAR10Net
)

app = Flask(__name__)

# Global variables for model caching
device = get_device()
_cached_models = {}  # Dictionary to cache different models

# CIFAR-10 classes
classes = CIFAR10_CLASSES
regrouped_classes = ['Vehicles', 'Animals', 'Other']

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

def detect_model_type(model_path):
    """Detect if model is original CIFAR-10 (10 classes) or regrouped (3 classes)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    
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

def load_model_cached(model_path):
    """Load model with caching for repeated inference"""
    global _cached_models
    
    if model_path not in _cached_models:
        print(f"Loading model from '{model_path}'...")
        
        # Detect model type
        model_type, model_classes = detect_model_type(model_path)
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
        
        _cached_models[model_path] = {
            'model': model,
            'type': model_type,
            'classes': model_classes
        }
        print(f"Model loaded and cached: {model_type}")
    
    return _cached_models[model_path]

def predict_from_image(image, model_path):
    """Run inference on a PIL Image"""
    # Load the cached model
    model_info = load_model_cached(model_path)
    model = model_info['model']
    model_classes = model_info['classes']
    model_type = model_info['type']
    
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
    
    # Get top predictions (up to 3 or total classes, whichever is smaller)
    num_top = min(3, len(model_classes))
    top_prob, top_classes = torch.topk(probabilities, num_top)
    
    top_predictions = []
    for i in range(num_top):
        class_idx = top_classes[0][i].item()
        prob = top_prob[0][i].item()
        top_predictions.append({
            'class': model_classes[class_idx],
            'confidence': float(prob)
        })
    
    return {
        'predicted_class': model_classes[predicted_class],
        'confidence': float(confidence),
        'model_type': model_type,
        'top_predictions': top_predictions
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(device)})

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    models_dir = 'models'
    available_models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                model_path = os.path.join(models_dir, file)
                try:
                    model_type, model_classes = detect_model_type(model_path)
                    available_models.append({
                        'name': file,
                        'path': model_path,
                        'type': model_type,
                        'classes': len(model_classes)
                    })
                except Exception as e:
                    # Skip models that can't be loaded
                    continue
    
    return jsonify({'models': available_models})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint that accepts image file or base64 encoded image"""
    try:
        # Get model type from request (default to original)
        model_type = request.form.get('model_type', 'original').lower()
        
        # Default model paths
        if model_type == 'regrouped' or model_type == 'finetuned':
            default_model_path = 'models/cifar10_regrouped_model.pth'
        else:
            default_model_path = 'models/cifar10_model.pth'
        
        # Allow custom model path
        model_path = request.form.get('model_path', default_model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            return jsonify({
                'error': f'Model not found: {model_path}',
                'available_models': [f['name'] for f in list_models().json['models']]
            }), 404
        
        # Handle image input
        image = None
        
        # Option 1: File upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Option 2: Base64 encoded image
        elif 'image_base64' in request.form:
            try:
                image_data = base64.b64decode(request.form['image_base64'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
        
        # Option 3: Image URL/path (for testing)
        elif 'image_path' in request.form:
            image_path = request.form['image_path']
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Cannot load image from path: {str(e)}'}), 400
        
        else:
            return jsonify({
                'error': 'No image provided. Use "image" (file), "image_base64" (base64 string), or "image_path" (file path)'
            }), 400
        
        # Run prediction
        result = predict_from_image(image, model_path)
        result['model_path'] = model_path
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple images"""
    try:
        # Get model configuration
        model_type = request.form.get('model_type', 'original').lower()
        
        if model_type == 'regrouped' or model_type == 'finetuned':
            default_model_path = 'models/cifar10_regrouped_model.pth'
        else:
            default_model_path = 'models/cifar10_model.pth'
        
        model_path = request.form.get('model_path', default_model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            return jsonify({
                'error': f'Model not found: {model_path}',
                'available_models': [f['name'] for f in list_models().json['models']]
            }), 404
        
        # Process multiple images
        results = []
        
        # Handle multiple file uploads
        if 'images' in request.files:
            files = request.files.getlist('images')
            for i, file in enumerate(files):
                if file.filename == '':
                    continue
                
                try:
                    image = Image.open(file.stream).convert('RGB')
                    result = predict_from_image(image, model_path)
                    result['image_index'] = i
                    result['filename'] = file.filename
                    results.append(result)
                except Exception as e:
                    results.append({
                        'image_index': i,
                        'filename': file.filename,
                        'error': f'Failed to process image: {str(e)}'
                    })
        
        if not results:
            return jsonify({'error': 'No valid images provided'}), 400
        
        return jsonify({
            'model_path': model_path,
            'results': results,
            'processed_count': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print(f"Starting CIFAR-10 Image Classification Server")
    print(f"Using device: {device}")
    print(f"Available models will be loaded from: models/")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5100, debug=True)

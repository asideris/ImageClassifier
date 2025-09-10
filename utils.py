import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import multiprocessing

# Device configuration
def get_device():
    """Get the appropriate device (GPU or CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# CIFAR-10 constants
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Data transforms
def get_train_transforms():
    """Get training data transforms with augmentation"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_test_transforms():
    """Get test/validation data transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_inference_transforms():
    """Get transforms for single image inference"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

# Data loading configuration
def get_dataloader_config():
    """Get optimized data loading configuration"""
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use available CPUs but cap at 8
    pin_memory = torch.cuda.is_available()  # Use pin_memory for GPU
    return {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': True if num_workers > 0 else False
    }

# Image preprocessing utilities
def smart_resize_with_padding(image, target_size=32):
    """Resize image to target size while preserving aspect ratio using padding"""
    # Get original dimensions
    width, height = image.size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image in center
    new_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))  # Gray padding
    
    # Calculate position to center the image
    x = (target_size - new_width) // 2
    y = (target_size - new_height) // 2
    
    new_image.paste(image, (x, y))
    return new_image

# MLflow configuration
def setup_mlflow(experiment_name, tracking_uri="file:./mlruns"):
    """Setup MLflow experiment tracking"""
    import mlflow
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)

# Directory setup
def ensure_models_directory(models_dir='models'):
    """Create models directory if it doesn't exist"""
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

# Model utilities
def load_model_state_dict(model_path, device):
    """Load model state dict and handle compiled model keys"""
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove '_orig_mod.' prefix if present (from compiled models)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    return state_dict

def compile_model_if_available(model):
    """Compile model for faster inference if PyTorch compilation is available"""
    if hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print("Model compiled for faster inference")
            return compiled_model
        except Exception as e:
            print(f"Model compilation not available: {e}")
    return model

# CNN Model Definition
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
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
        self.fc2 = nn.Linear(512, 10)
        
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
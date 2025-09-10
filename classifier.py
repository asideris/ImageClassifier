import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch
import argparse
import os
from PIL import Image
import multiprocessing
import gc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# MLflow setup
mlflow.set_experiment("CIFAR10-Classification")
mlflow.set_tracking_uri("file:./mlruns")

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Data preprocessing and loading
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Optimized data loading configuration
num_workers = min(multiprocessing.cpu_count(), 8)  # Use available CPUs but cap at 8
pin_memory = torch.cuda.is_available()  # Use pin_memory for GPU

# Dataset loading will be done per mode to avoid unnecessary loading

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

# Initialize model, loss function, and optimiser
model = CIFAR10Net().to(device)

# Compile model for faster inference (PyTorch 2.0+)
if hasattr(torch, 'compile') and torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("Model compiled for faster inference")
    except Exception as e:
        print(f"Model compilation not available: {e}")

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

# Training function with mixed precision and optimization options
def train_model(model, trainloader, criterion, optimiser, num_epochs=20, use_amp=True, accumulate_grad_steps=1):
    model.train()
    train_losses = []
    train_accuracies = []
    
    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Log hyperparameters
    mlflow.log_param("batch_size", trainloader.batch_size)
    mlflow.log_param("learning_rate", optimiser.param_groups[0]['lr'])
    mlflow.log_param("weight_decay", optimiser.param_groups[0]['weight_decay'])
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("optimizer", optimiser.__class__.__name__)
    mlflow.log_param("model_architecture", "CIFAR10Net")
    mlflow.log_param("device", str(device))
    mlflow.log_param("mixed_precision", use_amp and torch.cuda.is_available())
    mlflow.log_param("gradient_accumulation_steps", accumulate_grad_steps)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accumulate_grad_steps
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % accumulate_grad_steps == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
            else:
                # Regular forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulate_grad_steps
                
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % accumulate_grad_steps == 0:
                    optimiser.step()
                    optimiser.zero_grad()
            
            # Statistics
            running_loss += loss.item() * accumulate_grad_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Calculate epoch accuracy
        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(trainloader)
        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Accuracy: {epoch_acc:.2f}%')
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
        mlflow.log_metric("learning_rate", optimiser.param_groups[0]['lr'], step=epoch)
        
        # Step the scheduler
        scheduler.step()
    
    return train_losses, train_accuracies

# Testing function
def test_model(model, testloader, log_to_mlflow=True):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {overall_accuracy:.2f}%')
    
    # Log test accuracy to MLflow
    if log_to_mlflow:
        mlflow.log_metric("test_accuracy", overall_accuracy)
    
    # Per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]}: {class_acc:.2f}%')
            # Log per-class accuracy to MLflow
            if log_to_mlflow:
                mlflow.log_metric(f"test_accuracy_{classes[i]}", class_acc)
    
    return overall_accuracy


# Smart resize function that preserves aspect ratio
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


# Cached model for repeated inference
_cached_model = None
_cached_model_path = None

def load_model_cached(model_path, device):
    """Load model with caching for repeated inference"""
    global _cached_model, _cached_model_path
    
    if _cached_model is None or _cached_model_path != model_path:
        print(f"Loading model from '{model_path}'...")
        model = CIFAR10Net().to(device)
        
        # Load state dict and handle compiled model keys
        state_dict = torch.load(model_path, map_location=device)
        
        # Remove '_orig_mod.' prefix if present (from compiled models)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Compile model for faster inference if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except:
                pass
        
        _cached_model = model
        _cached_model_path = model_path
        print("Model loaded and cached")
    
    return _cached_model

# Inference function for single images
def inference(model_path, image_path, device):
    """Run inference on a single image with model caching"""
    # Load the trained model (cached)
    model = load_model_cached(model_path, device)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Smart resize with padding to preserve aspect ratio
    image = smart_resize_with_padding(image, target_size=32)
    
    # Apply normalization (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"Predicted class: {classes[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 3 predictions
    top3_prob, top3_classes = torch.topk(probabilities, 3)
    print("\nTop 3 predictions:")
    for i in range(3):
        class_idx = top3_classes[0][i].item()
        prob = top3_prob[0][i].item()
        print(f"{i+1}. {classes[class_idx]}: {prob:.4f}")
    
    return predicted_class, confidence


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'], required=True,
                       help='Mode: train (train model), test (test existing model), inference (classify single image)')
    parser.add_argument('--model-path', type=str, default='cifar10_model.pth',
                       help='Path to model file (default: cifar10_model.pth)')
    parser.add_argument('--image-path', type=str,
                       help='Path to image for inference mode')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--accumulate-grad-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Load CIFAR-10 training dataset
        print("Loading training dataset...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        print(f"Training dataset loaded: {len(trainset)} samples")
        
        # Load CIFAR-10 test dataset for evaluation
        print("Loading test dataset...")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        print(f"Test dataset loaded: {len(testset)} samples")
        testloader = DataLoader(testset, batch_size=100, shuffle=False, 
                               num_workers=num_workers, pin_memory=pin_memory,
                               persistent_workers=True if num_workers > 0 else False)
        
        # Create optimized data loaders with custom batch size
        train_batch_size = args.batch_size
        trainloader_custom = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, 
                                       num_workers=num_workers, pin_memory=pin_memory, 
                                       persistent_workers=True if num_workers > 0 else False)
        
        # Training mode
        with mlflow.start_run():
            print("Starting CIFAR-10 training...")
            print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
            print(f"Training with batch size: {train_batch_size}")
            print(f"Mixed precision: {not args.no_amp and torch.cuda.is_available()}")
            print(f"Gradient accumulation steps: {args.accumulate_grad_steps}")
            
            # Log model parameters count
            mlflow.log_param("total_parameters", sum(p.numel() for p in model.parameters()))
            
            # Train the model
            train_losses, train_accuracies = train_model(
                model, trainloader_custom, criterion, optimiser, 
                num_epochs=args.epochs, 
                use_amp=not args.no_amp,
                accumulate_grad_steps=args.accumulate_grad_steps
            )
            
            # Test the model
            test_accuracy = test_model(model, testloader)
            
            # Save the model
            torch.save(model.state_dict(), args.model_path)
            print(f"Model saved as '{args.model_path}'")
            
            # Log the trained model to MLflow
            mlflow.pytorch.log_model(model, "model")
            
            # Log artifacts to MLflow
            mlflow.log_artifact(args.model_path)
            
            print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
    
    elif args.mode == 'test':
        # Testing mode
        if not os.path.exists(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found. Please train the model first or specify correct path.")
            exit(1)
        
        # Load CIFAR-10 test dataset only
        print("Loading test dataset...")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        print(f"Test dataset loaded: {len(testset)} samples")
        testloader = DataLoader(testset, batch_size=100, shuffle=False, 
                               num_workers=num_workers, pin_memory=pin_memory,
                               persistent_workers=True if num_workers > 0 else False)
        
        print(f"Loading model from '{args.model_path}'...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Testing model on CIFAR-10 test dataset...")
        test_accuracy = test_model(model, testloader, log_to_mlflow=False)
        print(f"Final test accuracy: {test_accuracy:.2f}%")
        
        # Explicit cleanup to reduce exit delay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    elif args.mode == 'inference':
        # Inference mode
        if not args.image_path:
            print("Error: --image-path is required for inference mode")
            exit(1)
        
        if not os.path.exists(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found. Please train the model first or specify correct path.")
            exit(1)
        
        if not os.path.exists(args.image_path):
            print(f"Error: Image file '{args.image_path}' not found.")
            exit(1)
        
        print(f"Running inference on '{args.image_path}' using model '{args.model_path}'...")
        predicted_class, confidence = inference(args.model_path, args.image_path, device)
        
        # Explicit cleanup to reduce exit delay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

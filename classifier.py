import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import mlflow
import mlflow.pytorch
import argparse
import gc

# Import utilities
from utils import (
    get_device, CIFAR10_CLASSES, get_train_transforms, get_test_transforms,
    get_dataloader_config, setup_mlflow, ensure_models_directory,
    compile_model_if_available, CIFAR10Net
)

# Set device
device = get_device()
print(f'Using device: {device}')

# Create models directory if it doesn't exist
models_dir = ensure_models_directory()

# MLflow setup
setup_mlflow("CIFAR10-Classification")

# CIFAR-10 classes
classes = CIFAR10_CLASSES

# Data preprocessing and loading
transform_train = get_train_transforms()
transform_test = get_test_transforms()

# Optimized data loading configuration
dataloader_config = get_dataloader_config()
num_workers = dataloader_config['num_workers']
pin_memory = dataloader_config['pin_memory']

# Dataset loading will be done per mode to avoid unnecessary loading


# Initialize model, loss function, and optimiser
model = CIFAR10Net().to(device)
model = compile_model_if_available(model)

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



# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                       help='Mode: train (train model), test (test existing model)')
    parser.add_argument('--model-path', type=str, default='models/cifar10_model.pth',
                       help='Path to model file (default: models/cifar10_model.pth)')
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
        testloader = DataLoader(testset, batch_size=100, shuffle=False, **dataloader_config)
        
        # Create optimized data loaders with custom batch size
        train_batch_size = args.batch_size
        trainloader_custom = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, **dataloader_config)
        
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
        testloader = DataLoader(testset, batch_size=100, shuffle=False, **dataloader_config)
        
        print(f"Loading model from '{args.model_path}'...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Testing model on CIFAR-10 test dataset...")
        test_accuracy = test_model(model, testloader, log_to_mlflow=False)
        print(f"Final test accuracy: {test_accuracy:.2f}%")
        
        # Explicit cleanup to reduce exit delay
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    

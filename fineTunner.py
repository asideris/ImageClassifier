import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
import warnings
import logging

# Suppress MLflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
logging.getLogger("mlflow").setLevel(logging.ERROR)

import mlflow
import mlflow.pytorch
import argparse
import gc

# Import utilities
from utils import (
    get_device, get_train_transforms, get_test_transforms,
    get_dataloader_config, setup_mlflow, ensure_models_directory,
    load_model_state_dict, CIFAR10Net
)

device = get_device()
print(f'Using device: {device}')

models_dir = ensure_models_directory()

setup_mlflow("CIFAR10-FineTuning-Regrouped")

class RegroupedCIFAR10Dataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.class_mapping = self._create_class_mapping()
        self.new_classes = ['Vehicles', 'Animals', 'Other']
        
    def _create_class_mapping(self):
        return {
            0: 0,  # plane -> Vehicles
            1: 0,  # car -> Vehicles  
            2: 1,  # bird -> Animals
            3: 1,  # cat -> Animals
            4: 1,  # deer -> Animals
            5: 1,  # dog -> Animals
            6: 1,  # frog -> Animals
            7: 1,  # horse -> Animals
            8: 0,  # ship -> Vehicles
            9: 0,  # truck -> Vehicles
        }
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, original_label = self.original_dataset[idx]
        new_label = self.class_mapping[original_label]
        return image, new_label
    
    def get_class_names(self):
        return self.new_classes

class ModifiedCIFAR10Net(nn.Module):
    def __init__(self, original_model_path, num_classes=3, freeze_conv=True):
        super(ModifiedCIFAR10Net, self).__init__()
        
        original_model = CIFAR10Net()
        state_dict = load_model_state_dict(original_model_path, device)
        original_model.load_state_dict(state_dict)
        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.conv2 = original_model.conv2
        self.bn2 = original_model.bn2
        self.pool1 = original_model.pool1
        self.dropout1 = original_model.dropout1
        
        self.conv3 = original_model.conv3
        self.bn3 = original_model.bn3
        self.conv4 = original_model.conv4
        self.bn4 = original_model.bn4
        self.pool2 = original_model.pool2
        self.dropout2 = original_model.dropout2
        
        self.conv5 = original_model.conv5
        self.bn5 = original_model.bn5
        self.conv6 = original_model.conv6
        self.bn6 = original_model.bn6
        self.pool3 = original_model.pool3
        self.dropout3 = original_model.dropout3
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        if freeze_conv:
            self._freeze_conv_layers()
    
    def _freeze_conv_layers(self):
        conv_layers = [
            self.conv1, self.bn1, self.conv2, self.bn2,
            self.conv3, self.bn3, self.conv4, self.bn4,
            self.conv5, self.bn5, self.conv6, self.bn6
        ]
        
        for layer in conv_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

def train_regrouped_model(model, trainloader, criterion, optimizer, num_epochs=5, use_amp=True):
    model.train()
    train_losses = []
    train_accuracies = []
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    mlflow.log_param("regrouped_batch_size", trainloader.batch_size)
    mlflow.log_param("regrouped_learning_rate", optimizer.param_groups[0]['lr'])
    mlflow.log_param("regrouped_num_epochs", num_epochs)
    mlflow.log_param("regrouped_optimizer", optimizer.__class__.__name__)
    mlflow.log_param("regrouped_num_classes", 3)
    mlflow.log_param("frozen_conv_layers", True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("regrouped_trainable_parameters", trainable_params)
    mlflow.log_param("regrouped_frozen_parameters", total_params - trainable_params)
    
    print(f"Training with {trainable_params}/{total_params} trainable parameters ({100*trainable_params/total_params:.1f}%)")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/50:.4f}')
                running_loss = 0.0
        
        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(trainloader)
        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Accuracy: {epoch_acc:.2f}%')
        
        mlflow.log_metric("regrouped_train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("regrouped_train_accuracy", epoch_acc, step=epoch)
        mlflow.log_metric("regrouped_learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
    
    return train_losses, train_accuracies

def test_regrouped_model(model, testloader, class_names):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f'\nRegrouped Test Accuracy: {overall_accuracy:.2f}%')
    
    mlflow.log_metric("regrouped_test_accuracy", overall_accuracy)
    
    print('\nPer-class accuracy for regrouped classes:')
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_acc:.2f}%')
            mlflow.log_metric(f"regrouped_test_accuracy_{class_names[i]}", class_acc)
    
    return overall_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Fine-tuner for Regrouped Classes')
    parser.add_argument('--model-path', type=str, default='models/cifar10_model.pth',
                       help='Path to pretrained CIFAR-10 model (default: models/cifar10_model.pth)')
    parser.add_argument('--output-path', type=str, default='models/cifar10_regrouped_model.pth',
                       help='Output path for fine-tuned model (default: models/cifar10_regrouped_model.pth)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of fine-tuning epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for fine-tuning (default: 0.001)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Pretrained model file '{args.model_path}' not found.")
        exit(1)
    
    print("Loading CIFAR-10 dataset...")
    transform_train = get_train_transforms()
    transform_test = get_test_transforms()
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    
    print("Creating regrouped datasets...")
    regrouped_trainset = RegroupedCIFAR10Dataset(trainset)
    regrouped_testset = RegroupedCIFAR10Dataset(testset)
    
    dataloader_config = get_dataloader_config()
    
    trainloader = DataLoader(regrouped_trainset, batch_size=args.batch_size, shuffle=True, **dataloader_config)
    testloader = DataLoader(regrouped_testset, batch_size=100, shuffle=False, **dataloader_config)
    
    print(f"Loading pretrained model from '{args.model_path}'...")
    model = ModifiedCIFAR10Net(args.model_path, num_classes=3, freeze_conv=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    with mlflow.start_run():
        print("Starting fine-tuning for regrouped CIFAR-10 classification...")
        print(f"Original classes -> New classes:")
        print(f"  Vehicles: plane, car, ship, truck")
        print(f"  Animals: bird, cat, deer, dog, frog, horse")
        print(f"  Other: (none in this mapping)")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Mixed precision: {not args.no_amp and torch.cuda.is_available()}")
        
        mlflow.log_param("base_model_path", args.model_path)
        mlflow.log_param("fine_tuning_approach", "regrouped_classification")
        
        train_losses, train_accuracies = train_regrouped_model(
            model, trainloader, criterion, optimizer,
            num_epochs=args.epochs,
            use_amp=not args.no_amp
        )
        
        print("Testing fine-tuned model...")
        test_accuracy = test_regrouped_model(model, testloader, regrouped_trainset.get_class_names())
        
        torch.save(model.state_dict(), args.output_path)
        print(f"Fine-tuned model saved as '{args.output_path}'")
        
        # Create input example for MLflow signature
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            model.eval()
            dummy_output = model(dummy_input)
        
        mlflow.pytorch.log_model(
            model, 
            "regrouped_model",
            input_example=dummy_input.cpu().numpy(),
            signature=mlflow.models.infer_signature(
                dummy_input.cpu().numpy(), 
                dummy_output.cpu().numpy()
            )
        )
        mlflow.log_artifact(args.output_path)
        
        print(f"\nFine-tuning completed!")
        print(f"  Final accuracy: {test_accuracy:.2f}%")
        print(f"  Model saved to: {args.output_path}")
        print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

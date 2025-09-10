# CIFAR-10 Image Classifier

A high-performance PyTorch-based image classifier for CIFAR-10 dataset with multiple operation modes and extensive optimizations.

## Project Structure

- **`classifier.py`**: Main training and testing script for the original 10-class CIFAR-10 classification
- **`prediction.py`**: Standalone prediction/inference script for classifying individual images
- **`fineTunner.py`**: Fine-tuning script that adapts the trained model to predict regrouped classes (Vehicles, Animals, Other)
- **`utils.py`**: Shared utilities, constants, and the CIFAR10Net model definition

## Features

- **Multiple Modes**: Train, test, inference, and fine-tuning capabilities
- **Optimized Performance**: Mixed precision training, model compilation, and optimized data loading
- **Smart Image Processing**: Aspect ratio preservation with intelligent padding
- **MLflow Integration**: Comprehensive experiment tracking
- **Model Caching**: Efficient inference with model caching in prediction script
- **Fine-tuning Support**: Adapt trained models for new class groupings with transfer learning
- **Flexible Configuration**: Extensive command-line options
- **Modular Design**: Shared utilities and clean separation of concerns

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:asideris/ImageClassifier.git
cd ImageClassifier

# Install dependencies
pip install torch torchvision matplotlib numpy mlflow pillow
```

### Usage

#### Training Mode (classifier.py)
```bash
# Basic training
python classifier.py --mode train

# Advanced training with optimizations
python classifier.py --mode train --batch-size 256 --epochs 50 --accumulate-grad-steps 2
```

#### Testing Mode (classifier.py)
```bash
# Test existing model
python classifier.py --mode test --model-path models/cifar10_model.pth
```

#### Image Prediction (prediction.py)
```bash
# Classify a single image
python prediction.py --image-path your_image.jpg --model-path models/cifar10_model.pth
```

#### Fine-tuning for Regrouped Classes (fineTunner.py)
```bash
# Basic fine-tuning for regrouped classes (Vehicles, Animals, Other)
python fineTunner.py --model-path models/cifar10_model.pth

# Advanced fine-tuning with custom parameters
python fineTunner.py --model-path models/cifar10_model.pth \
    --epochs 10 --batch-size 64 --learning-rate 0.0005 \
    --output-path models/my_regrouped_model.pth
```

## Command Line Options

### classifier.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operation mode: train, test | Required |
| `--model-path` | Path to model file | models/cifar10_model.pth |
| `--epochs` | Number of training epochs | 20 |
| `--batch-size` | Training batch size | 128 |
| `--no-amp` | Disable mixed precision training | False |
| `--accumulate-grad-steps` | Gradient accumulation steps | 1 |

### prediction.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to trained model file | models/cifar10_model.pth |
| `--image-path` | Path to image for prediction | Required |

### fineTunner.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to pretrained CIFAR-10 model | models/cifar10_model.pth |
| `--output-path` | Output path for fine-tuned model | models/cifar10_regrouped_model.pth |
| `--epochs` | Number of fine-tuning epochs | 5 |
| `--batch-size` | Batch size for training | 128 |
| `--learning-rate` | Learning rate for fine-tuning | 0.001 |
| `--no-amp` | Disable mixed precision training | False |

## Performance Optimizations

- **Mixed Precision Training**: ~50% memory reduction, 30-50% speed improvement
- **Model Compilation**: 20-40% faster inference with PyTorch 2.0+
- **Optimized Data Loading**: Multi-threaded with persistent workers
- **Smart Image Preprocessing**: Aspect ratio preservation with LANCZOS resampling
- **Model Caching**: Eliminates reloading overhead for repeated inference
- **Fine-tuning with Layer Freezing**: Selective parameter training for efficient transfer learning

See [optimizations.md](optimizations.md) for detailed performance analysis.

## Fine-tuning Features

The `fineTunner.py` script provides specialized fine-tuning for class regrouping:

- **Class Regrouping**: Adapts the 10-class CIFAR-10 model to predict broader categories:
  - **Vehicles**: plane, car, ship, truck
  - **Animals**: bird, cat, deer, dog, frog, horse
  - **Other**: (reserved for future extensions)
- **Transfer Learning**: Freezes convolutional layers and trains only fully connected layers
- **Reduced Parameters**: Trains only ~6% of total parameters for efficient fine-tuning
- **Specialized Tracking**: Separate MLflow experiments for regrouped classification
- **Performance Monitoring**: Tracks accuracy improvements for the new class structure

## Model Architecture

- **CNN Architecture**: 6 convolutional layers with batch normalization
- **Regularization**: Dropout and weight decay
- **Activation**: ReLU activation functions
- **Pooling**: Max pooling for spatial reduction
- **Classification**: 10-class output for CIFAR-10 categories

## Classification Categories

### Original CIFAR-10 Classes (classifier.py)
The model classifies images into 10 categories:
- Plane
- Car  
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Regrouped Classes (fineTunner.py)
The fine-tuned model classifies images into 3 broader categories:
- **Vehicles**: plane, car, ship, truck
- **Animals**: bird, cat, deer, dog, frog, horse
- **Other**: (reserved for future class additions)

## Results

- **Test Accuracy**: ~84% on CIFAR-10 test set
- **Training Time**: ~25 minutes for 20 epochs (with optimizations)
- **Inference Speed**: <50ms per image (after model loading)
- **Prediction Accuracy**: Individual image classification with confidence scores and top-3 predictions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- mlflow
- pillow
- CUDA (optional, for GPU acceleration)

## License

MIT License

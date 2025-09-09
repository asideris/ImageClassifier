# CIFAR-10 Image Classifier

A high-performance PyTorch-based image classifier for CIFAR-10 dataset with multiple operation modes and extensive optimizations.

## Features

- **Multiple Modes**: Train, test, and inference capabilities
- **Optimized Performance**: Mixed precision training, model compilation, and optimized data loading
- **Smart Image Processing**: Aspect ratio preservation with intelligent padding
- **MLflow Integration**: Comprehensive experiment tracking
- **Model Caching**: Efficient inference with model caching
- **Flexible Configuration**: Extensive command-line options

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

#### Training Mode
```bash
# Basic training
python classifier.py --mode train

# Advanced training with optimizations
python classifier.py --mode train --batch-size 256 --epochs 50 --accumulate-grad-steps 2
```

#### Testing Mode
```bash
# Test existing model
python classifier.py --mode test --model-path cifar10_model.pth
```

#### Inference Mode
```bash
# Classify a single image
python classifier.py --mode inference --image-path your_image.jpg --model-path cifar10_model.pth
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operation mode: train, test, inference | Required |
| `--model-path` | Path to model file | cifar10_model.pth |
| `--image-path` | Path to image for inference | Required for inference |
| `--epochs` | Number of training epochs | 20 |
| `--batch-size` | Training batch size | 128 |
| `--no-amp` | Disable mixed precision training | False |
| `--accumulate-grad-steps` | Gradient accumulation steps | 1 |

## Performance Optimizations

- **Mixed Precision Training**: ~50% memory reduction, 30-50% speed improvement
- **Model Compilation**: 20-40% faster inference with PyTorch 2.0+
- **Optimized Data Loading**: Multi-threaded with persistent workers
- **Smart Image Preprocessing**: Aspect ratio preservation with LANCZOS resampling
- **Model Caching**: Eliminates reloading overhead for repeated inference

See [optimizations.md](optimizations.md) for detailed performance analysis.

## Model Architecture

- **CNN Architecture**: 6 convolutional layers with batch normalization
- **Regularization**: Dropout and weight decay
- **Activation**: ReLU activation functions
- **Pooling**: Max pooling for spatial reduction
- **Classification**: 10-class output for CIFAR-10 categories

## CIFAR-10 Classes

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

## Results

- **Test Accuracy**: ~84% on CIFAR-10 test set
- **Training Time**: ~25 minutes for 20 epochs (with optimizations)
- **Inference Speed**: <50ms per image (after model loading)

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
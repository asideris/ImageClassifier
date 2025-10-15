# CIFAR-10 Image Classifier

A PyTorch-based image classifier for CIFAR-10 dataset with multiple operation modes and extensive optimizations.

## Project Structure

- **`classifier.py`**: Main training and testing script for the original 10-class CIFAR-10 classification
- **`prediction.py`**: Standalone prediction/inference script for classifying individual images
- **`fineTunner.py`**: Fine-tuning script that adapts the trained model to predict regrouped classes (Vehicles, Animals, Other)
- **`server.py`**: Flask API server for image classification with REST endpoints
- **`utils.py`**: Shared utilities, constants, and the CIFAR10Net model definition

## Features

- **Multiple Modes**: Train, test, inference, and fine-tuning capabilities
- **Optimized Performance**: Mixed precision training, model compilation, and optimized data loading
- **Smart Image Processing**: Aspect ratio preservation with intelligent padding
- **MLflow Integration**: Experiment tracking
- **Model Caching**: Efficient inference with model caching in prediction script
- **Fine-tuning Support**: Adapt trained models for new class groupings with transfer learning
- **Flexible Configuration**: Extensive command-line options
- **Modular Design**: Shared utilities and clean separation of concerns
- **REST API**: Flask-based web service for remote image classification
- **Model Auto-Detection**: Automatically detects and loads appropriate model architecture

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:asideris/ImageClassifier.git
cd ImageClassifier

# Install dependencies
pip install torch torchvision matplotlib numpy mlflow pillow flask
```

### Usage

#### MLflow

You can start MLflow UI in a terminal as follows:

```bash
mlflow ui --port 5000
```

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

#### API Server (server.py)

```bash
# Start the Flask API server
python server.py

# The server will be available at http://localhost:5100
```

## Command Line Options

### classifier.py Options

| Option                    | Description                      | Default                  |
| ------------------------- | -------------------------------- | ------------------------ |
| `--mode`                  | Operation mode: train, test      | Required                 |
| `--model-path`            | Path to model file               | models/cifar10_model.pth |
| `--epochs`                | Number of training epochs        | 20                       |
| `--batch-size`            | Training batch size              | 128                      |
| `--no-amp`                | Disable mixed precision training | False                    |
| `--accumulate-grad-steps` | Gradient accumulation steps      | 1                        |

### prediction.py Options

| Option         | Description                  | Default                  |
| -------------- | ---------------------------- | ------------------------ |
| `--model-path` | Path to trained model file   | models/cifar10_model.pth |
| `--image-path` | Path to image for prediction | Required                 |

### server.py API Endpoints

| Endpoint         | Method | Description             |
| ---------------- | ------ | ----------------------- |
| `/health`        | GET    | Health check endpoint   |
| `/models`        | GET    | List available models   |
| `/predict`       | POST   | Single image prediction |
| `/predict_batch` | POST   | Batch image prediction  |

#### API Usage Examples

**Health Check:**

```bash
curl http://localhost:5100/health
```

**List Available Models:**

```bash
curl http://localhost:5100/models
```

**Single Image Prediction (file upload):**

```bash
# Using original 10-class model
curl -X POST -F "image=@your_image.jpg" -F "model_type=original" http://localhost:5100/predict

# Using regrouped 3-class model
curl -X POST -F "image=@your_image.jpg" -F "model_type=regrouped" http://localhost:5100/predict

# Using custom model path
curl -X POST -F "image=@your_image.jpg" -F "model_path=models/custom_model.pth" http://localhost:5100/predict
```

**Batch Prediction:**

```bash
curl -X POST -F "images=@image1.jpg" -F "images=@image2.jpg" -F "model_type=original" http://localhost:5100/predict_batch
```

### fineTunner.py Options

| Option            | Description                       | Default                            |
| ----------------- | --------------------------------- | ---------------------------------- |
| `--model-path`    | Path to pretrained CIFAR-10 model | models/cifar10_model.pth           |
| `--output-path`   | Output path for fine-tuned model  | models/cifar10_regrouped_model.pth |
| `--epochs`        | Number of fine-tuning epochs      | 5                                  |
| `--batch-size`    | Batch size for training           | 128                                |
| `--learning-rate` | Learning rate for fine-tuning     | 0.001                              |
| `--no-amp`        | Disable mixed precision training  | False                              |

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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- mlflow
- pillow
- flask (for API server)
- CUDA (optional, for GPU acceleration)

## Authors and acknowledgment

- Archie (Human)
- Claude Code (AI)

### Dataset Acknowledgement

This project uses the CIFAR-10 dataset:

**CIFAR-10 Dataset**
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. We acknowledge and thank Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for making this dataset publicly available for research and educational purposes.

### Important Note on Dataset Quality

**Data Leakage in CIFAR-10**: Research by Barz and Denzler (2020) discovered that approximately 3.3% of images in the CIFAR-10 test set have duplicates in the training set. This data leakage means:

- **Performance metrics may be inflated**: Models have effectively "seen" some test examples during training
- **True generalization ability may be lower**: Reported accuracy scores may overestimate real-world performance
- **Comparison validity**: Results should be interpreted with this limitation in mind

**Reference**: Björn Barz and Joachim Denzler, "Do We Train on Test Data? Purging CIFAR of Near-Duplicates," *Journal of Imaging* 6, no. 6 (2020): 41.

Users should be aware of this issue when evaluating model performance and comparing results with other work using CIFAR-10.

## License

This project is open source and available under the [MIT Licence](https://opensource.org/licenses/MIT).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Note

I am excited about how technology can help humans prosper and improve well-being for both people and the wider world. In this direction, the synergy between humans and AI shows great potential despite possible risks and ethical challenges.
This project has been co-developed with the help of an AI coding assistant as part of my practice in interacting and collaborating with AI entities (agents, chatbots, platforms, tools, etc.). My goal is to learn firsthand what works and what does not, and how this synergy can make us more productive and knowledgeable without losing our creativity, agency, curiosity, or ethical responsibility.

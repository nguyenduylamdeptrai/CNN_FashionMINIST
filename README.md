# 👁️ CNN Architecture & Feature Map Visualization

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Overview
This project bridges the gap between deep learning theory (CS231n, DeepLearning.AI) and practice. It involves building a Custom Convolutional Neural Network (CNN) from scratch, manually calculating tensor shape transformations, and visually inspecting the internal **Feature Maps** to understand exactly what the network is "seeing" (e.g., edges, textures) when looking at an image.

## 📂 Project Structure
- `train.ipynb`: The core notebook containing dataset loading, the custom CNN architecture (`Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`), manual spatial shape calculations, and the training loop.
- `visualize.py`: A dedicated script used to hook into the trained model, extract the output of the first Convolutional layer, and plot the intermediate Feature Maps using Matplotlib.
- `main.py`: A standalone inference script to predict the class of a custom input image.
- `fashion_cnn.pth`: The saved weights of the trained CNN model.
- `test_tshirt.jpg`: A sample image used to test the inference and visualization scripts.

## 🚀 Quick Start

### 1. Run Inference
To test the model's prediction on a sample image:
```bash
python main.py
```
### 2. Visualize Feature Maps
To see how the CNN's filters process the input image:
```bash
python visualize.py
```

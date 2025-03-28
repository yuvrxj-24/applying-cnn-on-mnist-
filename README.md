CNN Classification on MNIST Dataset

📌 Project Overview

This project demonstrates a Convolutional Neural Network (CNN) built with TensorFlow and Keras for classifying handwritten digits from the MNIST dataset. The model is trained and evaluated using various performance metrics, including accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

📂 Dataset

MNIST Dataset: Consists of 70,000 grayscale images (28x28 pixels) of handwritten digits (0-9).

Train Set: 60,000 images

Test Set: 10,000 images

📖 Steps in the Project

Data Preprocessing

Normalize images (scale pixel values between 0 and 1)

Reshape data to match CNN input requirements

Convert labels to one-hot encoding

Split training data into 80% training and 20% validation

CNN Architecture

Conv2D (32 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2 pool size)

Conv2D (64 filters, 3x3 kernel, ReLU activation)

MaxPooling2D (2x2 pool size)

Flatten layer

Dense (128 neurons, ReLU activation)

Output layer (10 neurons, Softmax activation)

Model Compilation & Training

Optimizer: Adam

Loss function: Categorical Crossentropy

Metrics: Accuracy

Epochs: 10

Batch Size: 64

Evaluation & Metrics

Confusion Matrix

Precision, Recall, F1-score

ROC Curve and AUC

Visualizations using Seaborn & Matplotlib

📊 Results

Achieved high accuracy on the MNIST dataset.

Evaluated model performance using multiple metrics.

Plotted confusion matrix and ROC curve.

🔧 Requirements

Python

TensorFlow & Keras

NumPy

Matplotlib

Seaborn

scikit-learn

OpenCV (for image handling)

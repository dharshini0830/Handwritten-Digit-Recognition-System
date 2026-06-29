# Handwritten Digit Recognition System

## Project Overview

The Handwritten Digit Recognition System is a Deep Learning-based application that recognizes handwritten digits (0–9) from images using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and deployed through a Streamlit web application, allowing users to upload or draw handwritten digits and receive instant predictions with confidence scores.

---

# Objectives

* Recognize handwritten digits accurately using Deep Learning.
* Build a CNN model trained on the MNIST dataset.
* Provide an easy-to-use web interface for digit prediction.
* Demonstrate image classification using Computer Vision.

---

# Features

* ✅ Handwritten Digit Recognition
* ✅ CNN-Based Deep Learning Model
* ✅ MNIST Dataset
* ✅ Image Upload / Drawing Canvas
* ✅ Confidence Score Display
* ✅ Real-Time Prediction
* ✅ Interactive Streamlit Dashboard
* ✅ User-Friendly Interface

---

# Technologies Used

### Programming Language

* Python

### Libraries

* TensorFlow
* Keras
* NumPy
* Pillow
* Streamlit
* Matplotlib

### Deep Learning

* Convolutional Neural Network (CNN)

---

# Dataset

### MNIST Handwritten Digits Dataset

The project uses the MNIST dataset, one of the most popular benchmark datasets for image classification.

* Total Images: 70,000
* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Classes: Digits 0–9

---

# Methodology

## Step 1 – Load Dataset

Load the MNIST handwritten digit dataset provided by TensorFlow.

## Step 2 – Data Preprocessing

* Normalize pixel values
* Reshape images to 28 × 28 × 1
* Convert labels into categorical format

## Step 3 – CNN Model

Build a Convolutional Neural Network consisting of:

* Convolution Layer
* Max Pooling Layer
* Flatten Layer
* Dense Layer
* Dropout Layer
* Softmax Output Layer

## Step 4 – Model Training

Train the CNN model using the Adam optimizer and categorical cross-entropy loss function.

## Step 5 – Model Evaluation

Evaluate the trained model on the testing dataset.

## Step 6 – Deployment

Deploy the trained model using Streamlit for real-time digit prediction.

---

# Model Performance

| Parameter     | Value   |
| ------------- | ------- |
| Algorithm     | CNN     |
| Dataset       | MNIST   |
| Classes       | 10      |
| Image Size    | 28 × 28 |
| Test Accuracy | ~98–99% |

---

# Project Structure

Handwritten Digit Recognition/

├── train_model.py

├── app.py

├── digit_model.keras

├── README.md

└── screenshots/

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Handwritten-Digit-Recognition.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

## Train the Model

```bash
python train_model.py
```

This generates:

* digit_model.keras

## Launch the Streamlit App

```bash
streamlit run app.py
```

---

# How It Works

1. User uploads or draws a handwritten digit.
2. The image is converted to grayscale and resized to 28 × 28 pixels.
3. The trained CNN model processes the image.
4. The model predicts the digit.
5. The predicted digit and confidence score are displayed.

---

# Screenshot
---

## Prediction Result

*(Insert screenshot here)*

![Prediction](screenshots/prediction.png)

---

# Future Enhancements

* Support handwritten letters (A–Z)
* Real-time webcam digit recognition
* Mobile application deployment
* Multi-digit recognition
* Explainable AI (XAI)
* TensorFlow Lite deployment for Android

---

# Developed By

**Dharshini Mary J**

Artificial Intelligence and Data Science (AIDS)

Deep Learning & Computer Vision Project

---

# Handwritten-Digit-Recognition-System

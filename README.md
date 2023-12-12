# Bank Document Classification using VGG16
This project involves training a VGG16 model for bank document classification, as well as using the trained model for detecting and classifying bank document images.


# Overview

The goal of this project is to classify bank document images into different categories such as checks, deposit slips, account statements, etc., using a deep learning approach. It utilizes the VGG16 convolutional neural network architecture for training and inference.

# Model Training

The training script uses a bank document dataset to fine-tune the pre-trained VGG16 model for document classification. The model is trained to distinguish between various types of bank documents.

# Document Classification

The detect_documents.py script applies the trained VGG16 model to input images, providing classification results for bank documents.

#Customization

Adjust the training script to fit your dataset structure and classes.

Fine-tune hyperparameters for better model performance.

Modify the document detection script to integrate with your application or system.

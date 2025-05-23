# 🩺 Liver Fibrosis Detection using Ultrasound Images

This project uses deep learning to detect and classify liver fibrosis stages (F0–F4) from ultrasound images. It assists medical professionals with non-invasive diagnosis and stage classification.

## 🧠 Project Overview

- **Goal:** Classify liver fibrosis into stages F0, F1, F2, F3, F4
- **Input:** Preprocessed ultrasound images
- **Output:** Predicted fibrosis stage
- **Model:** CNN (Convolutional Neural Network)

## 📊 Dataset

- Sourced from Kaggle: [Liver Histopathology Dataset](https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images)
- Images resized to 128x128 and normalized
- Data augmentation applied to improve generalization

## 🔍 Preprocessing & EDA

- Converted images to RGB, resized, normalized
- Augmented with flip, zoom, rotation
- Visualized class distribution and sample images

## 🧪 Model Summary

- **Architecture:** Conv2D → MaxPool → Dense → Softmax
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy

## ✅ Results

| Stage | Accuracy (Example) |
|-------|--------------------|
| F0    | 94%                |
| F1    | 91%                |
| F2    | 90%                |
| F3    | 89%                |
| F4    | 92%                |

*Note: Replace these with your actual results.*


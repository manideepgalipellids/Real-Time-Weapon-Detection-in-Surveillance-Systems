# Real-Time Weapon Detection in Surveillance Systems

## Project Overview

This project presents a deep learning-based real-time weapon detection system for surveillance applications. The proposed system automatically detects and classifies dangerous objects into three categories:

* Knife
* Pistol
* No Weapon

The project uses a Custom Convolutional Neural Network (CNN) and MobileNetV2 Transfer Learning model for image classification. The best-performing model is deployed for real-time webcam-based testing.

---

## Student Information

**Student Name:** Galipelli Manideep
**University:** University of Europe for Applied Sciences
**Course:** PS26 - DSC01 Machine Learning 90
**Professor:** Raja Hashim Ali

---

## Objectives

The main objective of this project is to develop an intelligent surveillance-based weapon detection system capable of identifying dangerous objects in real time to improve public safety and reduce manual monitoring efforts.

---

## Dataset Information

Two public datasets were used:

### Guns and Knives Dataset

Used for:

* Knife images
* Pistol images

Dataset Link:
https://www.kaggle.com/datasets/iqmansingh/guns-knives-object-detection

### Human Detection Dataset

Used for:

* No Weapon images

Dataset Link:
https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Image Resizing and Normalization
4. Train/Validation/Test Split
5. Data Augmentation
6. CNN Model Development
7. Transfer Learning using MobileNetV2
8. Model Training
9. Model Evaluation
10. Confusion Matrix and ROC Curve
11. Error Analysis
12. Real-Time Deployment

---

## Model Performance

### Custom CNN Model

Accuracy: 84.81%

### MobileNetV2 Model

Accuracy: ~94%

MobileNetV2 outperformed the custom CNN model and was selected for deployment.

---

## Repository Structure

```text
notebook/              -> Kaggle notebook
deployment/            -> Real-time deployment files
output_figures/        -> Generated figures and graphs
reports/               -> Proposal and project reports
dataset_links.txt      -> Dataset information
README.md              -> Project documentation
```

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install tensorflow==2.19.0 opencv-python numpy matplotlib seaborn scikit-learn
```

### Step 2: Download Model

Download:

```text
final_weapon_detection_model.keras
```

and place it inside the deployment folder.

### Step 3: Run Real-Time Detection

```bash
py -3.11 weapon_detection_live.py
```

### Controls

Press:

```text
Q
```

to quit webcam detection.

---

## Output Figures

The project generates:

* Class Distribution
* Sample Images
* Augmented Images
* CNN Confusion Matrix
* CNN Accuracy/Loss Curve
* MobileNetV2 Confusion Matrix
* MobileNetV2 Accuracy/Loss Curve
* ROC Curve
* Error Analysis

---

## Future Improvements

* Integration with CCTV systems
* YOLO-based object localization
* Alert notification system
* Edge device deployment
* Improved multi-weapon detection

---

## Conclusion

The proposed system successfully detects knives and pistols in surveillance imagery and demonstrates the feasibility of real-time AI-powered weapon detection for public safety applications.

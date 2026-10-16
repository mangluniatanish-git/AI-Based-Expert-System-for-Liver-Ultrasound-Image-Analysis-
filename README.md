AI-Based-Expert-System-for-Liver-Ultrasound-Image-Analysis

A Python-based medical image analysis research prototype that processes liver ultrasound images, extracts quantitative image features, and classifies them into **Normal, Benign, or Malignant** categories using classical machine learning.

The project combines image processing, segmentation, feature extraction, machine learning, and a desktop graphical user interface to demonstrate an end-to-end AI-assisted ultrasound screening workflow.

> **Disclaimer:** This is an educational and research prototype, not a clinically validated diagnostic system. Its predictions must not be used to diagnose or treat patients. All results require evaluation by a qualified medical professional.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technology Stack](#technology-stack)
- [How the System Works](#how-the-system-works)
- [Machine Learning Approach](#machine-learning-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Training the Model](#training-the-model)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Project Overview

Liver Ultrasound AI is designed to demonstrate how traditional computer vision and machine learning techniques can be applied to liver ultrasound image analysis.

The system accepts an ultrasound image, performs preprocessing, identifies an approximate liver region and potential candidate regions, extracts numerical features, and passes those features to a trained Support Vector Machine (SVM) classifier.

The desktop dashboard displays the predicted class, class probabilities when available, image-processing results, quantitative feature comparisons, and a downloadable PDF report.

The project uses **classical machine learning rather than a CNN or deep-learning architecture**.

## Objectives

- Process liver ultrasound images using computer vision techniques.
- Reduce image noise and improve local contrast.
- Identify an approximate liver region of interest (ROI).
- Detect potential candidate regions using image-processing heuristics.
- Extract quantitative intensity, texture, edge, and shape features.
- Classify images into Normal, Benign, or Malignant categories.
- Present analysis results through an interactive desktop application.
- Generate PDF reports containing quantitative analysis results.
- Explore an additional learning-based reliability assessment for flagging uncertain predictions.

## Technology Stack

| Technology    | Purpose                                                   |
| ------------- | --------------------------------------------------------- |
| Python        | Main programming language                                 |
| OpenCV        | Image processing and denoising                            |
| NumPy         | Numerical operations and feature arrays                   |
| scikit-image  | Image analysis and texture features                       |
| scikit-learn  | Machine learning, feature scaling, and SVM classification |
| Joblib        | Saving and loading trained model artifacts                |
| Pillow (PIL)  | Image loading and display support                         |
| Tkinter / ttk | Desktop graphical user interface                          |
| ReportLab     | PDF report generation                                     |

The exact dependencies required depend on the imports used in the current project files and should be listed in `requirements.txt`.

## How the System Works

The application follows a sequential image-processing and machine-learning pipeline.

### 1. Image Input

The user selects a liver ultrasound image through the graphical interface. The application loads the image and displays its details.

### 2. Image Preprocessing

The image is prepared for further analysis using operations such as:

- Resizing the image while preserving its aspect ratio.
- Converting the image to grayscale.
- Applying median filtering and Non-Local Means denoising.
- Normalizing image intensity values.
- Applying CLAHE (Contrast-Limited Adaptive Histogram Equalization) to improve local contrast.

### 3. Liver Region of Interest (ROI)

Thresholding, contour analysis, and morphological operations are used to estimate the liver region.

The ROI helps restrict subsequent feature extraction to a relevant portion of the image. Its accuracy depends on image quality and the segmentation algorithm.

### 4. Candidate Region Detection

Image-processing techniques are used to identify potential lesion-like regions. These operations may include thresholding, morphological filtering, contour analysis, and region scoring.

Candidate detection is heuristic-based and should not be interpreted as a clinically validated lesion detector.

### 5. Feature Extraction

The system converts image information into a numerical feature vector. The features include:

- **Intensity features:** Mean intensity and intensity variation.
- **Texture features:** Gray-Level Co-occurrence Matrix (GLCM) properties such as contrast, energy, and homogeneity.
- **Edge features:** Edge density and related image characteristics.
- **Shape features:** Area, circularity, solidity, extent, and aspect ratio where applicable.
- **Candidate-region features:** Measurements describing a potential candidate and its relationship to the surrounding liver region.

The current application displays a feature vector of 20 values. The exact feature definitions and ordering are determined by the implementation in `feature_extraction.py`.

### 6. Machine Learning Classification

The extracted features are passed through the fitted feature scaler and then to the trained SVM classifier using an RBF kernel.

The classifier predicts one of the available classes:

- Normal
- Benign
- Malignant

The dashboard displays the predicted class and class probability estimates when available.

A predicted class does not establish that the image is medically normal or that a lesion is benign or malignant.

### 7. Results and Reporting

The application presents the analysis through the desktop dashboard, including:

- Input ultrasound image.
- Annotated output image.
- Predicted class and available probabilities.
- Quantitative feature comparison against normal reference values.
- Detection details and model information.
- PDF report generation.

## Machine Learning Approach

The project uses supervised classical machine learning.

### Selected Model

**Support Vector Machine (SVM) with an RBF kernel**

An SVM learns decision boundaries from labelled training examples. The RBF kernel allows it to model nonlinear relationships between extracted features and class labels.

Feature scaling is important because the extracted features have different numerical ranges.

The project also includes code for dataset preparation and model training. Other classical classifiers may be compared during experimentation, depending on the current implementation of `train_model.py`.

### Learning-Based Reliability Assessment

A potential extension is a learning-based agent that evaluates whether a prediction is suitable for display or requires additional review.

This component can assess factors such as image quality, model uncertainty, and validated confidence thresholds. Uncertain results can be flagged as **Review Required**.

High model confidence does not guarantee correctness, and low confidence does not automatically mean that an image is abnormal. Any reliability thresholds must be evaluated on independent validation data.

## Project Structure

The main files and directories are organized as follows:

```text
Liver-Ultrasound-AI/
│
├── liver_ui.py
├── pipeline.py
├── preprocessing.py
├── segmentation.py
├── feature_extraction.py
├── predict.py
├── dataset_builder.py
├── train_model.py
├── requirements.txt
│
├── features_dataset.npz
│
├── models/
│   └── Trained model artifacts
│
├── models_old/
│   └── Previous model artifacts
│
├── predictions/
│   └── Prediction outputs
│
├── reports/
│   └── Generated PDF reports
│
└── README.md
```

The directory names above describe the intended organization. The exact contents of `models/`, `predictions/`, and `reports/` depend on the files generated during execution.

### Main Files

- `liver_ui.py` — Implements the desktop dashboard and user interactions.
- `pipeline.py` — Coordinates the image-processing and analysis stages.
- `preprocessing.py` — Contains image preprocessing operations.
- `segmentation.py` — Implements ROI and candidate-region detection.
- `feature_extraction.py` — Extracts numerical features from images.
- `dataset_builder.py` — Prepares the feature dataset for model training.
- `train_model.py` — Trains and evaluates machine-learning models.
- `predict.py` — Provides prediction-related functionality.
- `features_dataset.npz` — Stores the extracted feature dataset.
- `models/` — Stores trained model artifacts and related files.
- `requirements.txt` — Lists Python package dependencies.

## Installation

### Prerequisites

- Windows, Linux, or macOS.
- Python installed on the system.
- The project source files and required dataset.
- A terminal such as PowerShell or the VS Code terminal.

Use a Python version supported by the project's dependencies. Python 3.11 is a reasonable starting point if your installed packages do not support Python 3.14.

### Step 1: Open the Project Directory

Open the project folder in VS Code and open a terminal.

For example, in PowerShell:

```powershell
cd "E:\Project_files - Copy"
```

Replace the path with your actual project directory.

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects.

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 3: Install Dependencies

With the virtual environment activated, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is incomplete, update it to include the packages actually imported by the application before installing.

## Running the Application

From the project root, run:

```powershell
python liver_ui.py
```

The application should open the Liver Ultrasound AI dashboard.

### Basic Usage

1. Open the application.
2. Select an ultrasound image using **Choose Image**.
3. Click **Analyze** to start the analysis pipeline.
4. Inspect the annotated output and prediction probabilities.
5. Review the quantitative feature comparisons and model information.
6. Download the PDF report if report generation completes successfully.

For research and educational use only. Predictions should not be interpreted as clinical diagnoses.

## Training the Model

If you want to retrain the classifier after improving the dataset, follow the project's actual training workflow.

A typical sequence is:

1. Organize labelled images into the appropriate classes.
2. Verify the labels and remove duplicates or corrupted images.
3. Generate or update the feature dataset using `dataset_builder.py`.
4. Train and evaluate the model using `train_model.py`.
5. Save the trained model and associated scaler or metadata.
6. Ensure the application loads the newly generated model artifacts.

Run the scripts with:

```powershell
python dataset_builder.py
python train_model.py
```

**Important:** Check the argument requirements and execution flow in these scripts before running them. The commands above assume they can be executed directly without additional arguments.

Use patient-level separation when the dataset contains multiple images from the same patient. Otherwise, similar images can appear in both training and test sets, producing overly optimistic evaluation results.

After retraining, evaluate the model on an independent test set and compare per-class precision, recall, macro-F1, and the confusion matrix. Do not assume that balancing the dataset alone will guarantee better accuracy.

## Features

- Interactive desktop dashboard.
- Ultrasound image selection and preview.
- Image preprocessing and enhancement.
- Approximate liver ROI detection.
- Candidate-region analysis.
- Handcrafted quantitative feature extraction.
- SVM-based three-class classification.
- Prediction probability visualization.
- Comparison of extracted features with normal reference values.
- Annotated analysis output.
- PDF report generation.
- Model information and training artifact display.
- Expandable image inspection for closer visual review, if enabled in the current UI.

## Dataset Information

The previously used feature dataset contains 735 samples distributed across three classes.

| Class           | Number of samples |
| --------------- | ----------------: |
| Benign          |               200 |
| Malignant       |               435 |
| Normal          |               100 |
| **Total** |     **735** |

These counts describe the previously used dataset and should be updated if the dataset is changed.

The class distribution is imbalanced, particularly because the malignant class has more samples than the normal class. This can affect model performance and its tendency to favour particular classes.

The number of samples alone does not establish dataset quality. Image provenance, label reliability, patient-level separation, and independent testing are also important.

## Limitations

- **Limited dataset:** The available sample count may not represent the full diversity of liver ultrasound images.
- **Class imbalance:** Unequal class counts may affect classification performance.
- **Handcrafted features:** Intensity, texture, edge, and shape features may not capture all clinically relevant patterns.
- **Segmentation errors:** Thresholding and contour-based methods can select irrelevant structures or miss important regions.
- **Image quality variation:** Noise, acquisition settings, contrast, and scanning conditions can affect results.
- **False predictions:** Normal images may be classified as benign or malignant, and abnormal images may be misclassified.
- **Uncertain probabilities:** Model probability estimates are not necessarily calibrated or clinically meaningful.
- **Limited generalization:** Performance on the training dataset does not guarantee performance on images from other scanners, institutions, or patient populations.
- **No clinical validation:** The application is not established as a safe or accurate clinical diagnostic tool.

## Future Improvements

- Expand the dataset with verified, diverse, and correctly labelled ultrasound images.
- Address class imbalance using training-only sampling strategies and appropriate class weighting.
- Improve ROI segmentation and validate candidate detection against expert annotations.
- Compare the existing SVM against alternative models using identical patient-level splits.
- Add feature selection and systematic hyperparameter tuning.
- Evaluate sensitivity, specificity, per-class recall, macro-F1, and calibration.
- Implement a reliability assessment component that flags unsuitable or uncertain predictions.
- Add image-quality checks and an explicit review-required state.
- Compare MATLAB processing with the Python pipeline using identical algorithms and parameters.
- Conduct independent validation before considering any clinical use.

## Disclaimer

Liver Ultrasound AI is an educational and research prototype developed to demonstrate image processing and classical machine learning concepts.

It is not a substitute for a radiologist, medical examination, or professional clinical judgment. The software has not been established as a clinically validated diagnostic device. Its outputs may be incorrect, incomplete, or misleading.

Do not use this application as the sole basis for medical decisions.

# Traffic Sign Detection and Classification System

Computer Vision project focused on detecting and classifying traffic signs using classical image processing and machine learning techniques. The system performs segmentation, feature extraction, dimensionality reduction, and classification to recognize traffic signs under different environmental conditions. :contentReference[oaicite:0]{index=0}

## Features

- Traffic sign segmentation using HSV color space
- Morphological image processing and geometric filtering
- Hybrid feature extraction:
  - Shape descriptors
  - Color descriptors
  - HOG (Histogram of Oriented Gradients)
- PCA dimensionality reduction
- Multiple classifier evaluation:
  - Gaussian SVM
  - Linear SVM
  - Bagging Ensemble
  - Logistic Regression
  - LDA
- Validation on both internal and external datasets

## Pipeline

1. Image segmentation (HSV + morphology)
2. ROI extraction
3. Feature extraction (Shape + Color + HOG)
4. PCA dimensionality reduction
5. Feature normalization
6. Model training and evaluation
7. Traffic sign prediction

## Dataset

The dataset contains **2,775 images** distributed across 11 classes, including:

- Stop
- Speed Limit
- Mandatory Direction
- Prohibition Signs
- Bicycle Zone
- Pedestrian Crossing
- No Parking
- No Signal (custom added class)
- And others

## Results

### Feature Importance

| Features | Accuracy |
|-----------|-----------|
| Shape + HOG | 96.9% |
| Shape + HOG + Color | 97.1% |

### Classifier Comparison

| Model | Accuracy |
|---------|----------|
| Gaussian SVM | 97.5% |
| Bagging Ensemble | 96.7% |
| Logistic Regression | 96.5% |
| LDA | 96.3% |
| Linear SVM | 94.8% |

The **Gaussian SVM** was selected as the final model due to its superior overall performance. :contentReference[oaicite:1]{index=1}

## Technologies

- MATLAB
- Computer Vision Toolbox
- Image Processing Toolbox
- Machine Learning Toolbox

## Future Improvements

- Adaptive segmentation thresholds
- Better handling of difficult lighting conditions
- Larger and more balanced datasets
- Quantitative evaluation on external datasets
- Deep learning approaches (CNNs / Transfer Learning)

## Authors

- Alexander de Jong Roca
- Enric Segarra

## Project Goal

Develop a robust traffic sign recognition system capable of detecting and classifying traffic signs in real-world environments using traditional computer vision and machine learning techniques.

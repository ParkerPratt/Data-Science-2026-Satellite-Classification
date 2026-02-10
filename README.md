# Satellite Image Classification Using Deep Learning

## Overview

This project investigates deep learning methods for classifying objects in satellite imagery (7 classes). The goal was to build a robust image classification pipeline and evaluate how preprocessing, data augmentation, and training strategies affect predictive performance. 

The project was completed as part of USU's 2026 Data Science Competition which I placed 2nd in with an F1 score of .8632

---

## Objectives

* Train a high-performance image classification model on satellite imagery
* Improve generalization using data augmentation and preprocessing
* Evaluate model performance using appropriate classification metrics
* Explore training strategies and model combinations to improve performance

---

## Methods

### Data Preprocessing

Several preprocessing steps were applied:

* Image resizing and scaling
* Normalization and standardization
* Data augmentation to improve generalization

These steps helped reduce overfitting and improved model robustness to variation in satellite imagery.

---

### Model Architecture

The primary model used was:

* **Pretrained ConvNeXt-Tiny convolutional neural network**

Transfer learning was used to leverage pretrained weights and accelerate convergence.

---

### Training Strategy

The training pipeline included:

* Fine-tuning of pretrained network layers
* Cosine annealing learning rate scheduler
* Hyperparameter tuning and iterative experimentation
* Monitoring validation performance to guide training

The cosine annealing schedule improved convergence behavior and training stability.

---

### Ensembling

I ensembled the ConvNext Tiny model with a previously trained ResNet 18 model (weights imported at the end of the notebook)

In this dataset, the ensemble approach did not outperform the best single ConvNeXt model, likely due to:

* Similar error patterns between models
* Limited model diversity

This  helped confirm that architecture and preprocessing improvements had a larger impact than ensembling for this task.

---

### Evaluation

Models were evaluated using:

* F1 score (primary metric)
* Validation accuracy
* Training and validation loss trends

F1 score was used as the primary metric due to class imbalance and the importance of balancing precision and recall in satellite classification tasks.

---

## Results

Key observations:

* Data augmentation significantly improved generalization.
* Transfer learning with ConvNeXt-Tiny accelerated training and improved performance.
* Learning rate scheduling improved convergence stability.
* Ensemble modeling did not outperform the best single model in this setting.

(Insert training curves, prediction examples, or performance tables here.)

---

## Tools and Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Jupyter Notebook

---

## Future Work

Potential extensions include:

* Comparing additional architectures (ResNet, EfficientNet, detection pipelines)
* Larger-scale hyperparameter sweeps
* More diverse ensembles to test variance reduction
* A larger number of Epochs (only 18 were run before the final submission)

---

## Author

Parker Pratt
Mathematics and Data Science, Utah State University

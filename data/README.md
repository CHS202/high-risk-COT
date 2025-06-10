# Model Outputs and Data Analysis

This repository contains the results, models, and data for analyzing high-risk cases, training different types of machine learning models, and evaluating their performance based on clinical features.

## Directory Structure

The repository is organized into several directories containing various datasets, model outputs, and pretrained models:

### 1. `high-risk-cases/`
- **Description**: Contains the high-risk cases identified in the test set results. These are the cases considered to have a higher probability of being critical, based on model predictions or other clinical factors.
- **Contents**: Data files related to high-risk cases (e.g., CSV files).

### 2. `model_multi/{model_name}/`
- **Description**: Contains the outputs of various models trained for multi-feature analysis. Each subfolder inside `{model_name}` corresponds to a specific model and contains the predictions and other evaluation results for that model.
- **Example**: `model_multi/SVM/`, `model_multi/KNN/`, `model_multi/NN/`
- **Contents**: Model output files such as predictions, pretrained models.

### 3. `model_multi_all/`
- **Description**: Contains the pretrained Neural Network (NN) model and its outputs using both **clinical relevant features** and **high-risk features**.
- **Contents**: Pretrained NN model files, predictions, and evaluation results.

### 4. `model_multi_high-risk/`
- **Description**: Contains the pretrained Neural Network (NN) model and its outputs using **only high-risk features** (derived from the dataset).
- **Contents**: Pretrained NN model files, predictions, and evaluation results based on high-risk features only.

### 5. `model_multi_norsbi/`
- **Description**: Contains the pretrained Neural Network (NN) model and its outputs using the dataset **without the RSBI feature** (Rapid Shallow Breathing Index).
- **Contents**: Pretrained NN model files, predictions, and evaluation results without the RSBI feature.

### 6. `preped/`
- **Description**: Contains preprocessed feature files that have been cleaned and transformed for use in training models.
- **Contents**: CSV or other data files with preprocessed features.

### 7. `raw/`
- **Description**: Contains the raw data before preprocessing. This is the original dataset that has not yet undergone cleaning or transformation.
- **Contents**: Raw data files, such as CSVs or other formats, that are used for initial analysis.

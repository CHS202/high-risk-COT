import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import pickle
import os
from keras.models import load_model
import h5py


def load_and_prepare_data():
    """Load and preprocess the dataset."""
    try:
        # Load the dataset
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")

        # Drop 'stay_id' column if it exists
        if 'stay_id' in df.columns:
            df = df.drop('stay_id', axis=1)

        # Separate features and target
        X = df.drop('outcome', axis=1)
        y = df['outcome']  # Assuming binary outcome: 0=failure, 1=success

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def generate_shap_and_pdp_plots(model, X_train, X_test, y_test, model_name, save_path, threshold=0.9):
    """Generate SHAP force plots, SHAP summary plots, and PDP plots for one success and one failure sample."""
    try:
        # Prepare background data for SHAP
        background_data = shap.sample(X_train, 1000, random_state=42)
        # background_data = X_train
        background_data = pd.DataFrame(background_data, columns=X_train.columns)
        print(background_data.shape)

        # Custom predict function for positive class probability
        def predict_proba_positive(X):
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            # print(f"predict_proba_positive input shape: {X_np.shape}")
            if model_name == 'SVM':
                # Ensure X is a numpy array (KernelExplainer passes numpy arrays)
                X_df = pd.DataFrame(X, columns=X_train.columns)
                proba = model.predict_proba(X_df)
                # Return probabilities for both classes (SHAP expects [n_samples, n_classes])
            elif model_name == 'NN':
                proba = model.predict(X)
                # Ensure NN output is [n_samples, n_classes]
                if proba.shape[1] == 1:  # If binary output is [n_samples, 1], convert to [n_samples, 2]
                    proba = np.hstack([1 - proba, proba])
            else:
                proba = model.predict_proba(X)
            # print(f"predict_proba_positive output shape: {proba.shape}")
            return proba

        # Initialize SHAP explainer
        shap_explainer = shap.KernelExplainer(predict_proba_positive, background_data)

        # Compute SHAP values for background data
        shap_values_background = shap_explainer.shap_values(background_data)

        # Diagnostic prints to debug shapes
        print(f"--- Debugging {model_name} ---")
        print(f"Type of shap_values_background: {type(shap_values_background)}")
        print(f"Shape of background_data: {background_data.shape}")

        # Handle SHAP values based on their structure
        if isinstance(shap_values_background, np.ndarray) and len(shap_values_background.shape) == 3:
            # Case: 3D array [n_samples, n_features, n_classes]
            print(f"Shape of shap_values_background: {shap_values_background.shape}")
            # Extract SHAP values for the positive class (index 1)
            shap_values_positive = shap_values_background[:, :, 1]
            print(f"Shape of shap_values_positive: {shap_values_positive.shape}")
        elif isinstance(shap_values_background, list) and len(shap_values_background) >= 2:
            # Case: List of arrays for each class
            shap_values_positive = shap_values_background[1]
            print(f"Shape of shap_values_background[1]: {shap_values_positive.shape}")
        else:
            raise ValueError(f"Unexpected shap_values_background structure for {model_name}")

        # Verify shape compatibility
        if shap_values_positive.shape == background_data.shape:
            # SHAP Summary Plot for positive class
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_positive, background_data, show=False)
            plt.title(f'SHAP Summary Plot ({model_name}, Positive Class)')
            summary_plot_path = os.path.join(save_path, f'shap_summary_{model_name.lower()}_threshold_{threshold}.png')
            plt.savefig(summary_plot_path)
            plt.close()
            print(f"SHAP summary plot saved to {summary_plot_path}")
        else:
            raise ValueError(f"Shape mismatch: shap_values_positive {shap_values_positive.shape} vs background_data {background_data.shape}")

        # PDP Plots for top features
        shap_values_mean = np.abs(shap_values_positive).mean(axis=0)
        feature_importance = pd.Series(shap_values_mean, index=X_train.columns)
        top_features = feature_importance.sort_values(ascending=False).head(3).index.tolist()
        print(f"Top features for PDP: {top_features}")

        # Map feature names to indices for PDP
        feature_indices = []
        for feature in top_features:
            if feature not in X_train.columns:
                raise ValueError(f"Feature '{feature}' not in X_train.columns: {X_train.columns.tolist()}")
            feature_indices.append(X_train.columns.get_loc(feature))
        print(f"Feature indices for PDP: {feature_indices}")

    except Exception as e:
        print(f"Error generating plots for {model_name}: {e}")


        # Define paths for data and model storage
BASE_PATH = 'data'
DATA_PATH = os.path.join(BASE_PATH, 'preped', 'features_multivariate_norsbi.csv')
NN_MODEL_PATH = r'D:\Courses\medical-data\final-project\data\model_multi_norsbi\NN'

# Create directories if they don't exist
os.makedirs(NN_MODEL_PATH, exist_ok=True)

# Set custom threshold
THRESHOLD = {
    'SVM': 0.85,
    'KNN': 0.91,
    'NN': 0.5
}

# Load and prepare data
X_train, X_test, y_train, y_test = load_and_prepare_data()
# print(X_train.head())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Adjust this path if running from a different location
print(os.path.exists(os.path.join(NN_MODEL_PATH, 'NN_classifier_balanced.keras')))
try:
    with h5py.File(os.path.join(NN_MODEL_PATH, 'NN_classifier_balanced.keras'), 'r') as f:
        print("File is valid HDF5")
except Exception as e:
    print(f"File validation error: {e}")
nn_classifier = load_model(os.path.join(NN_MODEL_PATH, 'NN_classifier_balanced.keras'))

# Generate SHAP force plots for SVM:
generate_shap_and_pdp_plots(
    nn_classifier, X_train_scaled, X_test_scaled, y_test, 'NN', NN_MODEL_PATH, THRESHOLD.get('NN')
)
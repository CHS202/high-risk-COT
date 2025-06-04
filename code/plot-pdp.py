import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.utils import class_weight
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from scikeras.wrappers import KerasClassifier
import seaborn as sns
from keras.models import load_model

# --- 1. Configuration & Data Loading ---
FILE_PATH = r'data\preped\high-risk_features_v1_multivariate.csv'  # Ensure this path is correct
TARGET_VARIABLE = 'outcome'  # The column name of your target variable
POSITIVE_CLASS_LABEL = 1  # Positive class (e.g., 'success')
NEGATIVE_CLASS_LABEL = 0  # Negative class (e.g., 'failed')
output_dir = r'D:\Courses\medical-data\final-project\data\model_multi_high-risk\NN'

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

# Load your data
try:
    features_prep_df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {FILE_PATH}")
    print("Please ensure the file path is correct and the data is accessible.")
    exit()

# Drop 'stay_id' column if it exists
if 'stay_id' in features_prep_df.columns:
    features_prep_df = features_prep_df.drop('stay_id', axis=1)
    print("Dropped 'stay_id' column.")

# --- 2. Data Preparation ---
if TARGET_VARIABLE not in features_prep_df.columns:
    print(f"Error: Target variable '{TARGET_VARIABLE}' not found in DataFrame columns: {features_prep_df.columns.tolist()}")
    exit()

X = features_prep_df.drop(TARGET_VARIABLE, axis=1)
y = features_prep_df[TARGET_VARIABLE]

# Ensure target variable is numeric (0 and 1)
if not pd.api.types.is_numeric_dtype(y):
    print(f"Warning: Target variable '{TARGET_VARIABLE}' is not numeric. Attempting to map common string labels.")
    unique_labels = y.unique()
    if len(unique_labels) == 2:
        label_map = {}
        if 'failed' in unique_labels and 'success' in unique_labels:
            label_map = {'failed': POSITIVE_CLASS_LABEL, 'success': NEGATIVE_CLASS_LABEL}
        elif 'yes' in unique_labels and 'no' in unique_labels:
            label_map = {'yes': POSITIVE_CLASS_LABEL, 'no': NEGATIVE_CLASS_LABEL}
        if label_map:
            print(f"Mapping labels: {label_map}")
            y = y.map(label_map)
            if y.isnull().any():
                print("Error: Null values found in target variable after mapping. Please check your labels.")
                exit()
        else:
            print("Error: Could not automatically map non-numeric target labels. Please encode them manually to 0 and 1.")
            exit()
    else:
        print("Error: Target variable must be binary (have exactly two unique classes).")
        exit()

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_1, X_val, y_train_1, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# # Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")

# --- 3. Handle Class Imbalance ---
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: class_weights_values[i] for i in range(len(class_weights_values))}
print(f"Calculated class weights: {class_weights}")

# --- 4. Neural Network Model Definition ---
n_features = X_train_scaled.shape[1]

def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(n_features,)),  # Input shape required here
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Wrap the model with KerasClassifier
model = KerasClassifier(model=create_model)

# --- 5. Model Training ---
print("\nTraining the Neural Network...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Fit the KerasClassifier
model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("\nEvaluating the model on the test set...")
# Use the underlying Keras model for evaluation
loss, accuracy, auc = model.model_.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Make predictions (probabilities)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
y_pred_classes = (y_pred_proba > 0.5).astype(int)  # Convert to class labels

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.clf()

# Classification Report
print("\nClassification Report:")
target_names = [f'Class {NEGATIVE_CLASS_LABEL} (e.g., Failed)', f'Class {POSITIVE_CLASS_LABEL} (e.g., Success)']
print(classification_report(y_test, y_pred_classes, target_names=target_names))

# Balanced Accuracy
bal_acc = balanced_accuracy_score(y_test, y_pred_classes)
print(f"Balanced Accuracy: {bal_acc:.4f}")

# AUC-ROC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC (from probabilities): {roc_auc:.4f}")

# --- 7. Calculate Youden's Index ---
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    youden_index = sensitivity + specificity - 1
    print(f"\nSensitivity (Recall for Class {POSITIVE_CLASS_LABEL}): {sensitivity:.4f}")
    print(f"Specificity (for Class {NEGATIVE_CLASS_LABEL}): {specificity:.4f}")
    print(f"Youden's Index: {youden_index:.4f}")
else:
    print("\nCould not calculate Youden's Index because the confusion matrix is not 2x2.")

print("F1 score:", f1_score(y_test, y_pred_classes))
print("Weighted F1 score:", f1_score(y_test, y_pred_classes, average='weighted'))

# --- Plot Training History ---
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.clf()

# Note: With KerasClassifier, history is not directly returned by fit. 
# If you need history, you can train the underlying model separately first.
# For simplicity, we'll skip plotting history here unless you modify the code further.

# --- Plot PartialDependenceDisplay ---
feature_names = X_train.columns.tolist()
# Determine the number of features and layout
n_features = len(feature_names)
n_cols = 3  # Number of columns, adjustable based on preference
n_rows = int(np.ceil(n_features / n_cols))  # Number of rows needed

# Create figure and axes with custom size and spacing
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), gridspec_kw={'wspace': 1, 'hspace': 2.5})
axes_flat = axes.ravel()  # Flatten the axes array for easier handling

# Generate the PDP using the custom axes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
pdp_display = PartialDependenceDisplay.from_estimator(
    model, X_train_scaled, features=feature_names, ax=axes_flat[:n_features]
)

# Hide any extra axes if the grid is larger than the number of features
for ax in axes_flat[n_features:]:
    ax.set_visible(False)

# Add a title to the entire figure
fig.suptitle("Partial Dependence Plots", fontsize=16)

# --- 5. MODIFIED PART: Customize axes - Relabel X-axis ticks to original scale ---
print("Relabeling X-axis ticks to original scale...")
for i in range(n_features):
    ax = axes_flat[i]
    feature_name = feature_names[i]

    ax.grid(True) # Add grid for better readability
    # The xlabel is already set by from_estimator with the feature name.
    # If you want to ensure it or change font size:
    ax.set_xlabel(feature_name, fontsize=12)
    ax.tick_params(axis='x', labelsize=10)


    # Get current tick locations (these are in the SCALED space)
    scaled_ticks = ax.get_xticks()

    # We need to inverse-transform these ticks for THIS specific feature.
    # To do this with a scaler like StandardScaler that was fit on all features,
    # we create a temporary dataset where all other features are at their mean (scaled)
    # value (which is 0 for StandardScaler), and the current feature takes on the tick values.

    try:
        # Find the column index of the current feature in the scaled DataFrame
        feature_col_idx = X_train_scaled.columns.get_loc(feature_name)

        # Create a 'background' sample for inverse transform.
        # Using the mean of the scaled data (0 for StandardScaler) for other features.
        # The shape should be (number_of_ticks, total_number_of_features_in_scaler)
        temp_scaled_data = np.zeros((len(scaled_ticks), scaler.n_features_in_))
        
        # Substitute the scaled_ticks into the column for the current feature
        temp_scaled_data[:, feature_col_idx] = scaled_ticks

        # Perform inverse transformation using the fitted scaler
        original_ticks_all_features = scaler.inverse_transform(temp_scaled_data)
        
        # Extract the inverse-transformed ticks for the current feature
        original_ticks_for_feature = original_ticks_all_features[:, feature_col_idx]
        
        # Set the new tick labels, formatted appropriately
        # Heuristic: if original data for the feature is integer and has few unique values, format as integer.
        if pd.api.types.is_integer_dtype(X_train[feature_name]) and \
           X_train[feature_name].nunique() < 10 and \
           (original_ticks_for_feature % 1 == 0).all(): # Check if all ticks are whole numbers
             ax.set_xticklabels([f"{int(round(tick))}" for tick in original_ticks_for_feature])
        else:
             ax.set_xticklabels([f"{tick:.2f}" for tick in original_ticks_for_feature])
        
        # print(f"Relabeled ticks for {feature_name}")

    except AttributeError as e:
        print(f"AttributeError for feature '{feature_name}': {e}. This might happen if 'scaler.n_features_in_' is not available or scaler is not what's expected. Using scaled ticks.")
        ax.set_xticklabels([f"{tick:.2f}" for tick in scaled_ticks]) # Fallback
    except ValueError as e:
        print(f"ValueError during inverse transform for feature '{feature_name}': {e}. Check shape or scaler compatibility. Using scaled ticks.")
        ax.set_xticklabels([f"{tick:.2f}" for tick in scaled_ticks]) # Fallback
    except Exception as e:
        print(f"An unexpected error occurred while inverse-transforming ticks for feature '{feature_name}': {e}. Using scaled ticks.")
        ax.set_xticklabels([f"{tick:.2f}" for tick in scaled_ticks]) # Fallback

# Save the plot to the specified output directory
plt.savefig(os.path.join(output_dir, "PartialDependenceDisplay.png"))
plt.clf()  # Clear the figure to free memory

y_pred_proba_val = model.predict_proba(X_val_scaled)[:, 1]
# --- Calibrate Predictions ---
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_pred_proba_val, y_val)
y_pred_proba_test_calibrated = iso_reg.transform(y_pred_proba)

# --- Plot Calibration Plot ---
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
fig, ax = plt.subplots(figsize=(8, 6))

# Plot original NN calibration curve
CalibrationDisplay.from_predictions(y_test, y_pred_proba, color='blue', ax=ax, name="Original NN")

# Customize plot
ax.set_title("Calibration Plot", fontsize=16)
ax.grid(True)

# ax.lines[0].set_color('balck')  # Reference line
# ax.lines[1].set_color('blue')  # Original NN

ax.lines[0].set_linewidth(1)
ax.lines[1].set_linewidth(2)
plt.savefig(os.path.join(output_dir, "calibration_plot.png"))

# Plot calibrated NN calibration curve
CalibrationDisplay.from_predictions(y_test, y_pred_proba_test_calibrated, color='red', ax=ax, name="Calibrated NN")

# Customize lines: calibration curve (red), reference line (black)
# different color for original NN and calibrated NN

# ax.lines[2].set_color('red')  # Calibrated NN
ax.lines[2].set_linewidth(2)

# Save the plot
plt.savefig(os.path.join(output_dir, "calibration_plot_vs.png"))
plt.clf()

# --- Save the Model ---
# Save the KerasClassifier instance
with open(os.path.join(output_dir, "NN_classifier_balanced.pkl"), 'wb') as file:
    pickle.dump(model, file)

# Save the underlying Keras model
model.model_.save(os.path.join(output_dir, "NN_classifier_balanced.keras"))

# Save additional outputs
np.savetxt(os.path.join(output_dir, "y_test_NN.csv"), y_test, delimiter=',')
np.savetxt(os.path.join(output_dir, "y_pred_proba_NN.csv"), y_pred_proba, delimiter=',')
np.savetxt(os.path.join(output_dir, "X_train_NN.csv"), X_train, delimiter=',')
np.savetxt(os.path.join(output_dir, "X_test_NN.csv"), X_test, delimiter=',')
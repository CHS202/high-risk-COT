import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

all_features_df = pd.read_csv(r'data\raw\all_features_v1.csv')

# filter high-risk patients by high_risk column
high_risk_df = all_features_df[all_features_df['high_risk'] == 1]
print(len(high_risk_df))

clinical_features_df = pd.read_csv(r'data\raw\feature_extracted_all.csv')
# get high-risk patients index
high_risk_index = clinical_features_df[clinical_features_df['stay_id'].isin(high_risk_df['stay_id'])].index

preprocessed_features_df = pd.read_csv(r'data\preped\features_multivariate.csv')
# filter high-risk patients by high_risk column
high_risk_df = preprocessed_features_df.iloc[high_risk_index]
print(len(high_risk_df))
print(high_risk_df.columns)

# load NN model
nn_model = load_model(r'data\model_multi\NN\NN_classifier_balanced.keras')

# split train test
X = high_risk_df.drop('outcome', axis=1)
y = high_risk_df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# predict    
y_pred = nn_model.predict(X_test_scaled)
# set threshold to 0.5
y_pred_class = np.where(y_pred > 0.5, 1, 0)
print(y_pred_class)

# plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('High-Risk Patients in Test Set')
plt.show()

# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}")

# calculate F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred_class)
print(f"F1 score: {f1:.4f}")

# calculate sensitivity
from sklearn.metrics import recall_score
sensitivity = recall_score(y_test, y_pred_class, pos_label=1)
print(f"Sensitivity: {sensitivity:.4f}")

# calculate specificity
specificity = recall_score(y_test, y_pred_class, pos_label=0)
print(f"Specificity: {specificity:.4f}")

# calculate Youden's Index
youden_index = sensitivity + specificity - 1
print(f"Youden's Index: {youden_index:.4f}")
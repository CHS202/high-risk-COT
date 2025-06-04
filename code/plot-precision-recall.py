# plot Precision-Recall curve

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_auc_score

BASE_PATH = 'data'
DATA_PATH = os.path.join(BASE_PATH, 'preped', 'high-risk_features_v1_multivariate.csv')
NN_MODEL_PATH = os.path.join(BASE_PATH, 'model_multi_high-risk', 'NN')

# load y_test_NN.csv, y_pred_proba_NN.csv
y_test = np.loadtxt(os.path.join(NN_MODEL_PATH, 'y_test_NN.csv'), delimiter=',')
y_pred_proba = np.loadtxt(os.path.join(NN_MODEL_PATH, 'y_pred_proba_NN.csv'), delimiter=',')

print(y_test)
print(y_pred_proba)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
print(precision)
print(recall)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(NN_MODEL_PATH, 'precision_recall_curve.png'))
plt.clf()

# plot ROC curve
from sklearn.metrics import roc_curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"NN (AUC = {roc_auc:.2f})",)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
# legend
plt.legend(loc='lower right')
plt.savefig(os.path.join(NN_MODEL_PATH, 'roc_curve.png'))
plt.clf()
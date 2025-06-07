import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

df = pd.read_csv("features_multivariate.csv")
X = df.drop(columns=["outcome"])
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型初始化（rusboost 先空著）
models = {
    "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "gentleboost": GradientBoostingClassifier(loss="exponential", n_estimators=100, random_state=42),
    "logitboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "rusboost": None
}

# rusboost：對多數類別做下抽樣後訓練 AdaBoost
minority_class_size = y_train.value_counts().min()
X_majority = X_train[y_train == 0]
y_majority = y_train[y_train == 0]
X_minority = X_train[y_train == 1]
y_minority = y_train[y_train == 1]

X_majority_downsampled, y_majority_downsampled = resample(
    X_majority, y_majority,
    replace=False,
    n_samples=minority_class_size,
    random_state=42
)

X_rus = pd.concat([X_majority_downsampled, X_minority])
y_rus = pd.concat([y_majority_downsampled, y_minority])

rus_model = AdaBoostClassifier(n_estimators=100, random_state=42)
rus_model.fit(X_rus, y_rus)
models["rusboost"] = rus_model

results = []
for name, model in models.items():
    if name != "rusboost":
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # 最佳 threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]

    y_pred = (y_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    youden = sensitivity + specificity - 1

    results.append({
        "Model": name,
        "AUC": round(auc, 4),
        "F1-score": round(f1, 4),
        "Threshold": round(best_threshold, 4),
        "Sensitivity (TPR)": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Youden index": round(youden, 4)
    })

pd.set_option("display.max_columns", None)

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv("model_results.csv", index=False)

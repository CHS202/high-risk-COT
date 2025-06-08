# -*- coding: utf-8 -*-
import time
from pathlib import Path

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer


set_config(transform_output="pandas")

if __name__ == "__main__":
    # Configuration
    SAVE_MODEL = False
    SHOW_ROC = False
    SAVE_PREDICTIONS = False
    SHAP_ANALYSIS = True
    LIME_ANALYSIS = False
    PDP_ANALYSIS = False

    # Variables
    RANDOM_SEED = 42
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    MODEL_SAVE_DIR = Path("models")
    PREDICTION_SAVE_DIR = Path("predictions")
    EXPLAIN_SAVE_DIR = Path("explanations")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Data type
    DATA_TYPE = "multivariate"  # "knn", "multi"

    # Model parameters
    # kernel, logistic_regression, subspace_knn, random_forest
    MODEL_TYPE = "random_forest"

    match DATA_TYPE:
        case "knn":
            train_pd = pd.read_csv("train_data/features_prepared.csv")
        case "multivariate":
            train_pd = pd.read_csv("train_data/features_multivariate.csv")
        case _:
            raise ValueError(f"Unknown DATA_TYPE: {DATA_TYPE}")

    X = train_pd.drop(columns=["outcome"])
    y = train_pd["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    # get class names and feature names
    class_names = train_pd["outcome"].unique()
    feature_names = X.columns.tolist()
    num_of_features = len(feature_names)
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of features: {num_of_features}")

    match MODEL_TYPE:
        case "kernel":
            pipeline = Pipeline([("model", SVC(kernel="rbf", probability=True))])
        case "logistic_regression":
            pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
        case "subspace_knn":
            pipeline = Pipeline(
                [
                    (
                        "model",
                        BaggingClassifier(
                            KNeighborsClassifier(n_neighbors=5),
                            bootstrap=False,  # no sample bootstrapping, only feature subspace
                            n_estimators=100,
                            max_samples=0.8,
                            max_features=0.8,
                            random_state=RANDOM_SEED,
                        ),
                    )
                ]
            )
        case "random_forest":
            pipeline = Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=None,
                            min_samples_split=3,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            )
        case _:
            raise ValueError(f"Unknown Model_type: {MODEL_TYPE}")

    print(f"Starting training with {MODEL_TYPE} model...")
    # Train the model
    pipeline.fit(X_train, y_train)
    # Evaluate the model
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    sensitivity = tpr
    specificity = 1 - fpr
    youden_index = tpr - fpr

    # Find best threshold
    best_threshold_index = youden_index.argmax()
    best_threshold = thresholds[best_threshold_index]
    best_sensitivity = sensitivity[best_threshold_index]
    best_specificity = specificity[best_threshold_index]
    best_youden_index = youden_index[best_threshold_index]

    # Make predictions with the best threshold
    y_pred = (
        pipeline.predict_proba(X_test)[:, 1] > thresholds[best_threshold_index]
    ).astype(int)
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    print(
        "=============== Results ===============\n"
        f"Model Type: {MODEL_TYPE}\n"
        f"Timestamp: {TIMESTAMP}\n"
        f"AUC: {auc:.4f}\n"
        f"Best F1 Score: {f1:.4f}\n"
        f"Best Threshold: {best_threshold:.4f}\n"
        f"Best Sensitivity: {best_sensitivity:.4f}\n"
        f"Best Specificity: {best_specificity:.4f}\n"
        f"Best Youden's Index: {best_youden_index:.4f}\n"
        "======================================="
    )

    if SAVE_MODEL:
        # Create metadata
        metadata = {
            "model_type": MODEL_TYPE,
            "timestamp": TIMESTAMP,
            "auc": auc,
            "best f1_score": f1,
            "best thresholds": best_threshold,
            "best sensitivity": best_sensitivity,
            "best specificity": best_specificity,
            "best youden_index": best_youden_index,
        }

        MODEL_SAVE_PATH = MODEL_SAVE_DIR / f"model_{MODEL_TYPE}_{TIMESTAMP}.joblib"
        joblib.dump(
            {
                "model": pipeline,
                "metadata": metadata,
            },
            MODEL_SAVE_PATH,
        )
        print(f"Model saved to {MODEL_SAVE_PATH}")

    if SAVE_PREDICTIONS:
        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(
            PREDICTION_SAVE_DIR / f"y_test_{DATA_TYPE}_{MODEL_TYPE}.csv", index=False
        )
        pred_probabilities = pipeline.predict_proba(X_test)[:, 1]

        pred_proba_df = pd.DataFrame(
            pred_probabilities, columns=["predicted_probabilities"]
        )
        pred_proba_df.to_csv(
            PREDICTION_SAVE_DIR / f"y_pred_proba_{DATA_TYPE}_{MODEL_TYPE}.csv",
            index=False,
        )

    if SHOW_ROC:
        plt.plot(fpr, tpr, label="ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    if SHAP_ANALYSIS:

        def predict_positive_proba(X):
            if not isinstance(X, pd.DataFrame):
                # raise ValueError("Input must be a pandas DataFrame")
                X = pd.DataFrame(X, columns=X_train.columns)
            return pipeline.predict_proba(X)[:, 1]

        # Sample background data where it includes both positive and negative samples
        NUM_OF_BACKGROUND_SAMPLES = 3
        X_background = pd.concat(
            [
                X_test[y_test == 1].sample(1, random_state=RANDOM_SEED),
                X_test[y_test == 0].sample(1, random_state=RANDOM_SEED),
                X_test.sample(NUM_OF_BACKGROUND_SAMPLES - 2, random_state=RANDOM_SEED),
            ]
        )
        y_background = y_test.loc[X_background.index]

        # Select index of a positive/negative sample
        positive_sample_pos_index = X_background.index.get_loc(
            X_background[y_background == 1].index[0]
        )
        negative_sample_pos_index = X_background.index.get_loc(
            X_background[y_background == 0].index[0]
        )

        print(positive_sample_pos_index, negative_sample_pos_index)
        assert positive_sample_pos_index < len(
            X_background
        ) and negative_sample_pos_index < len(
            X_background
        ), f"Sample index out of range, ({positive_sample_pos_index}, {negative_sample_pos_index}) should be less than {len(X_background)}"

        # Create a SHAP explainer based on the model type
        if MODEL_TYPE in ["kernel", "subspace_knn"]:
            explainer = shap.KernelExplainer(
                predict_positive_proba,
                X_background,
                link="logit",
                feature_names=X_test.columns,
            )
        elif MODEL_TYPE in ["logistic_regression"]:
            explainer = shap.LinearExplainer(
                pipeline.named_steps["model"],
                X_background,
                feature_names=X_test.columns,
            )
        elif MODEL_TYPE in ["random_forest"]:
            explainer = shap.TreeExplainer(
                pipeline.named_steps["model"],
                X_background,
                feature_names=X_test.columns,
            )
        else:
            raise ValueError(f"Unknown Model_type: {MODEL_TYPE}")

        # Calculate SHAP violin plot
        # shap_values = explainer(background)
        # shap.plots.violin(
        #     shap_values,
        #     features=background,
        #     feature_names=background.columns,
        #     plot_type="violin",
        # )

        # Calculate SHAP force plot
        shap_values = explainer(X_background)

        print(shap_values.shape)
        print(shap_values)

        # Plot SHAP force plot for a positive sample
        shap.plots.force(
            shap_values[..., 0][positive_sample_pos_index],
            feature_names=X_background.columns,
            matplotlib=True,
            text_rotation=15,
            figsize=(15, 5),
            show=False,
        )
        plt.title(f"SHAP Force Plot for Positive Sample ({MODEL_TYPE})")
        plt.show()

        # Plot SHAP force plot for a negative sample
        shap.plots.force(
            shap_values[..., 0][negative_sample_pos_index],
            feature_names=X_background.columns,
            matplotlib=True,
            text_rotation=15,
            figsize=(15, 5),
            show=False,
        )
        plt.title(f"SHAP Force Plot for Negative Sample ({MODEL_TYPE})")
        plt.show()

    if LIME_ANALYSIS:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=class_names,
            mode="classification",
        )

        # Select a failure sample to explain
        positive_sample_pos_index = X_test[y_test == 1].index[0]
        positive_sample = X_test.loc[positive_sample_pos_index]

        explain = lime_explainer.explain_instance(
            data_row=positive_sample,
            predict_fn=pipeline.predict_proba,
            num_features=num_of_features,
        )
        fig = explain.as_pyplot_figure()
        plt.title("LIME Explanation for Success Sample")
        plt.show()

        # Select a success sample to explain
        negative_sample_pos_index = X_test[y_test == 0].index[0]
        negative_sample = X_test.loc[negative_sample_pos_index]
        explain = lime_explainer.explain_instance(
            data_row=negative_sample,
            predict_fn=pipeline.predict_proba,
            num_features=num_of_features,
        )
        fig = explain.as_pyplot_figure()
        plt.title("LIME Explanation for Failure Sample")
        plt.show()

        # html = explain.as_html()
        # with open(
        #     EXPLAIN_SAVE_DIR / f"lime_{DATA_TYPE}_{MODEL_TYPE}.html",
        #     "w",
        #     encoding="utf-8",
        # ) as f:
        #     f.write(html)

    if PDP_ANALYSIS:
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            pipeline,
            X_train,
            features=[0, 1],  # Change this to the feature indices you want to plot
            feature_names=feature_names,
            ax=ax,
            grid_resolution=50,
        )
        plt.title("Partial Dependence Plot")
        plt.show()

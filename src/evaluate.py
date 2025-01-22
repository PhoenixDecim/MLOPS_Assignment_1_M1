from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def evaluate_model(model, X_test, y_test):
    # Predict the target values
    y_pred = model.predict(X_test)
    # Check if model supports predicting probability values
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr",
                                average="weighted")
    except AttributeError:
        roc_auc = None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    class_labels = pd.unique(y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels,
                                  columns=class_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix_df)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [accuracy, precision, recall, f1],
    }
    if roc_auc is not None:
        metrics["Metric"].append("ROC-AUC")
        metrics["Score"].append(roc_auc)
    evaluation_df = pd.DataFrame(metrics)
    return evaluation_df

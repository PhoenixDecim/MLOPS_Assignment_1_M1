from sklearn.metrics import accuracy_score
import pandas as pd


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    evaluation_df = pd.DataFrame({"Metric": ["Accuracy"], "Score": [accuracy]})
    return evaluation_df

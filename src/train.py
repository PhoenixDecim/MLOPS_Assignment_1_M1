from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_name):
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=200)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

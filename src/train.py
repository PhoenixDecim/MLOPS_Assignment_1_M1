from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, model_name, **model_params):
    if model_name == "logistic_regression":
        max_iter = model_params["max_iter"]
        model = LogisticRegression(max_iter=max_iter)
    elif model_name == "random_forest":
        n_estimators = model_params["n_estimators"]
        random_state = model_params["random_state"]
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    model.fit(X_train, y_train)
    return model

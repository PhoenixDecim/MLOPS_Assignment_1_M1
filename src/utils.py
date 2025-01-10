import joblib
import pandas as pd


def save_model(model, path):
    joblib.dump(model, path)
    return True


def load_model(path):
    model = joblib.load(path)
    return model


def load_dataset(path):
    iris_df = pd.read_csv(path)
    return iris_df

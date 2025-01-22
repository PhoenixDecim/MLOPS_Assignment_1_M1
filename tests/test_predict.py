import unittest
from src.predict import predict
from src.utils import load_dataset, load_model
from src.preprocess import preprocess_data


class TestPredict(unittest.TestCase):
    def test_predict_logistic_regression(self):
        model = load_model("models/logistic_regression_model.joblib")
        data = load_dataset("data/iris.csv")
        X, y = preprocess_data(data)
        predictions = predict(model, X)
        self.assertGreater(
            len(predictions), 0,
            "No predictions made for logistic regression model"
        )
        self.assertEqual(
            len(predictions),
            len(X),
            "Number of predictions doesn't match the number of inputs",
        )
        self.assertTrue(
            all(isinstance(p, str) for p in predictions),
            "Predictions are not strings"
        )

    def test_predict_random_forest(self):
        model = load_model("models/random_forest_model.joblib")
        data = load_dataset("data/iris.csv")
        X, y = preprocess_data(data)
        predictions = predict(model, X)
        self.assertGreater(
            len(predictions), 0, "No predictions made for random forest model"
        )
        self.assertEqual(
            len(predictions),
            len(X),
            "Number of predictions doesn't match the number of inputs",
        )
        self.assertTrue(
            all(isinstance(p, str) for p in predictions),
            "Predictions are not strings"
        )


if __name__ == "__main__":
    unittest.main()

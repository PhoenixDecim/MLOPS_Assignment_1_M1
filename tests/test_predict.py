import unittest
from src.predict import predict
from src.utils import load_dataset, load_model
from src.preprocess import preprocess_data


class TestPredict(unittest.TestCase):
    def test_predict(self):
        model = load_model("models/logistic_regression_model.joblib")
        data = load_dataset("data/iris.csv")
        X, y = preprocess_data(data)
        predictions = predict(model, X)
        self.assertGreater(len(predictions), 0, "No predictions made")
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

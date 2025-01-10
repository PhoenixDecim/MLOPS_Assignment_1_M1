from src.preprocess import preprocess_data
from src.utils import load_dataset
import unittest


class TestPreprocess(unittest.TestCase):
    def test_preprocess_data(self):
        data = load_dataset("data/iris.csv")
        X, y = preprocess_data(data)
        assert X is not None, "Features are empty"
        assert y is not None, "Target is empty"
        self.assertGreater(len(X), 0, "No features after preprocessing")
        self.assertGreater(len(y), 0, "No target data after preprocessing")


if __name__ == "__main__":
    unittest.main()

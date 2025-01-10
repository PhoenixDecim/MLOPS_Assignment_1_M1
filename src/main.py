from sklearn.model_selection import train_test_split
from utils import save_model, load_model, load_dataset
from train import train_model
from evaluate import evaluate_model
from preprocess import preprocess_data
from predict import predict
import pandas as pd

path = "models/logistic_regression_model.joblib"
print("\n1. Loading Iris dataset\n")
iris_df = load_dataset("data/iris.csv")
print(iris_df.head(5))
print("\n2. Preprocessing the data")
X, y = preprocess_data(iris_df)
print(X.head(5))
print("\n3. Splitting the data in train and test sets: 80-20")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Size of X_train: {X_train.shape}")
print(f"Size of X_test: {X_test.shape}")
print(f"Size of y_train: {y_train.shape}")
print(f"Size of y_test: {y_test.shape}")
print("\n4. Training the Logistic Regression model")
model = train_model(X_train, y_train)
print("\n5. Saving the model")
save_status = save_model(model, path)
print("\n6. Evaluating the model\n")
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
print("\n7. Loading the model")
loaded_model = load_model(path)
print("\n8. Making predictions using the model\n")
predictions = predict(loaded_model, X_test)
results_df = pd.DataFrame(X_test, columns=X.columns)
results_df["True Variety"] = y_test
results_df["Predicted Variety"] = predictions
print(results_df)

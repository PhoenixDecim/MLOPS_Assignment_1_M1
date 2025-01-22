# MLOPS Assignment 1
## M1
![CI](https://github.com/PhoenixDecim/MLOPS_Assignment_1_M1/actions/workflows/ci_cd_pipeline.yml/badge.svg?branch=main)
### Description
A simple machine learning project that includes the following steps:
- load a dataset
- pre-process the data
- train a ML model
- evaluate the model
- make predictions using the model.

### Setup
- Install the requirements
```bash
pip install -r requirements.txt
```
### Run
- Run the `main.py` that does all the steps mentioned above.
```bash
cd ml_iris
python3 src/main.py
```
### Test
- Run the tests using the following command.
```bash
cd ml_iris
python3 -m unittest discover -s tests -p "test_*.py"
```
### Prediction API
- Start the API Server using the following command.
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```
- Post a request to the server.
```bash
# request - logistic_regression
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2, \"model_name\": \"logistic_regression\"}"
# response
{"model":"logistic_regression","prediction":"Setosa"}

# request - random_forest
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2, \"model_name\": \"random_forest\"}"
# response
{"model":"logistic_regression","prediction":"Setosa"}
```

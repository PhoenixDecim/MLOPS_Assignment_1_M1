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
python3 src/main.py --model <model_name>
# model name can be either logistic_regression or random_forest
```
### Test
- Run the tests using the following command.
```bash
cd ml_iris
python3 -m unittest discover -s tests -p "test_*.py"
```

# MLOPS Assignment 1
## M1
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
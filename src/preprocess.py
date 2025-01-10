def preprocess_data(data):
    # drop the missing values
    data = data.dropna()
    # select the features as X and target as y
    X = data.drop(columns=["variety"])
    y = data["variety"]
    return X, y

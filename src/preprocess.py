from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(data):
    # drop the missing values
    print(
        "The dataset has {} rows and {} columns.".format(data.shape[0],
                                                         data.shape[1])
    )
    null_count = data.isnull().sum()
    total_nulls = null_count.sum()
    if total_nulls == 0:
        print("No null values found in the dataset.")
    elif total_nulls > 0 and total_nulls < len(data):
        print(f"{total_nulls} null values found, removing them.")
        data = data.dropna()
    else:
        raise ValueError("All values are null in the dataset.")
    # select the features as X and target as y
    X = data.drop(columns=["variety"])
    y = data["variety"]
    scaler = StandardScaler()
    print("Applying standard scaler to the input values.\n")
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y

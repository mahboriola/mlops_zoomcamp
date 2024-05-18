from typing import Union
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np

# Load and Prepare Data
def load_and_prep_data(data_path: str) -> pd.DataFrame:
    data = pd.read_parquet(data_path)

    num_rows, num_cols = data.shape

    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")

    data["duration"] = data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
    data["duration"] = data["duration"].dt.total_seconds() / 60

    print(f"Duration STD: {data['duration'].std()}")

    data =  data[(data["duration"] >= 1) & (data["duration"] <= 60)]

    print(f"Remaining data after dropping outliers (%) {data.shape[0] / num_rows * 100}")

    return data


def vectorize_data(data: pd.DataFrame, training: bool = True, vectorizer: DictVectorizer = None) -> Union[np.ndarray, DictVectorizer]:
    data = data.astype(str)
    data_dict = data.to_dict(orient="records")
    
    if training:
        vectorizer = DictVectorizer()
        vectorized_data = vectorizer.fit_transform(data_dict)
    elif not training and vectorizer is None:
        raise ValueError("You need to pass a vectorizer when training is False")
    else:
        vectorized_data = vectorizer.transform(data_dict)
    
    num_cols = vectorized_data.shape[1]

    print(f"Number of columns: {num_cols}")

    return vectorized_data, vectorizer


def evaluate_model(model: LinearRegression, X: np.ndarray, y: pd.Series) -> float:
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)

    return rmse


def main():
    print("Training Data")
    train_data = load_and_prep_data("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet")

    X_train, dv = vectorize_data(train_data[["PULocationID", "DOLocationID"]])
    y_train = train_data["duration"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_rmse = evaluate_model(model, X_train, y_train)

    print(f"Train RMSE: {train_rmse}")

    print("\n\nTest Data")
    test_data = load_and_prep_data("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet")

    X_test, _ = vectorize_data(test_data[["PULocationID", "DOLocationID"]], training=False, vectorizer=dv)
    y_test = test_data["duration"]

    test_rmse = evaluate_model(model, X_test, y_test)

    print(f"Test RMSE: {test_rmse}")


if __name__ == "__main__":
    main()
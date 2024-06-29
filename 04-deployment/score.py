import sys
import pickle
import pandas as pd

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(input_file, dv, model, output_file, year, month):
    df = read_data(input_file)
    
    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df.ride_id, 'prediction': y_pred})
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(df_result['prediction'].mean())


def run():
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2])  # 3

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    apply_model(input_file, dv, model, output_file, year, month)


if __name__ == "__main__":
    run()
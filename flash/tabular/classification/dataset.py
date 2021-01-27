import os

import pandas as pd
from sklearn.model_selection import train_test_split

from flash.core.data import download_data


def titanic_data_download(path: str, predict_size: float = 0.1):
    if not os.path.exists(path):
        os.makedirs(path)

    path_data = os.path.join(path, "titanic.csv")
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", path_data)

    if set(os.listdir(path)) != {"predict.csv", "titanic.csv"}:
        assert predict_size > 0 and predict_size < 1
        df = pd.read_csv(path_data)
        df_train, df_predict = train_test_split(df, test_size=predict_size)
        df_train.to_csv(path_data)
        df_predict = df_predict.drop(columns=["Survived"])
        df_predict.to_csv(os.path.join(path, "predict.csv"))

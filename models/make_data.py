import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from const import INPUT_DIR


def make_forecast_train_data(query: str):
    df_train_val = pd.read_csv(f"{INPUT_DIR}/train.csv")

    # categorical features
    categorical_features = []
    for i in df_train_val.columns:
        if df_train_val[i].dtypes == "O":
            df_train_val[i] = df_train_val[i].astype("category")
            categorical_features.append(i)

    x_train_val = df_train_val.drop([query], axis=1)
    y_train_val = df_train_val[query]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=3655)

    print("set dataset")
    lgb_train = lgbm.Dataset(x_train, y_train, categorical_feature=categorical_features, free_raw_data=False)
    lgb_eval = lgbm.Dataset(
        x_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features, free_raw_data=False
    )
    num_class = len(df_train_val[query].unique())

    return lgb_train, lgb_eval, num_class


def make_forecast_test_data():
    x_test = pd.read_csv(f"{INPUT_DIR}/test.csv", index_col=0)

    categorical_features = []
    for i in x_test.columns:
        if x_test[i].dtypes == "O":
            x_test[i] = x_test[i].astype("category")
            categorical_features.append(i)

    # categorical features
    x_test[categorical_features] = x_test[categorical_features].astype("category")

    return x_test

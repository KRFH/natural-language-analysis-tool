import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# LangChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

# langchaintools
from langchaintools import mltools

import os
from app import OUTPUT_DIR, INPUT_DIR

with open("/Users/kai/Desktop/api/openai.txt", mode="r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()


def make_forecast_train_data(query: str):
    df_train_val = pd.read_csv(f"{INPUT_DIR}/{query}", index_col=0)

    # categorical features
    categorical_features = []
    for i in df_train_val.columns:
        if df_train_val[i].dtypes == "O":
            df_train_val[i] = df_train_val[i].astype("category")
            categorical_features.append(i)

    x_train_val = df_train_val.drop(["target"], axis=1)
    y_train_val = df_train_val["target"]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=3655)

    print("set dataset")
    lgb_train = lgbm.Dataset(x_train, y_train, categorical_feature=categorical_features, free_raw_data=False)
    lgb_eval = lgbm.Dataset(
        x_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features, free_raw_data=False
    )
    num_class = len(df_train_val["target"].unique())

    return lgb_train, lgb_eval, num_class


def make_forecast_test_data(query: str):
    x_test = pd.read_csv(f"{INPUT_DIR}/{query}", index_col=0)

    categorical_features = []
    for i in x_test.columns:
        if x_test[i].dtypes == "O":
            x_test[i] = x_test[i].astype("category")
            categorical_features.append(i)

    # categorical features
    x_test[categorical_features] = x_test[categorical_features].astype("category")

    return x_test


def run_mltools():
    # Toolの設定
    tools = [mltools.LgbmtrainTool(), mltools.LgbminferenceTool()]

    # 通常のLangChainの設定
    llm = OpenAI(temperature=0)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    agent.run("train.csvを使ってLightGBMの学習を行なったあとtest.csvのデータを推論してください")

    # 推論結果の読み込み
    test_pred = pd.read_csv(f"/{OUTPUT_DIR}/inference.csv", index_col=0)
    test_pred = np.where(test_pred.values.reshape(-1) > 0.5, 1, 0)

    test_target = pd.read_csv(f"/{INPUT_DIR}/test_target.csv", index_col=0)
    test_target = test_target.values.reshape(-1)

    score = accuracy_score(test_target, test_pred)
    print(f"予測精度:{score*100}%")

    return True


if __name__ == "__main__":
    run_mltools()

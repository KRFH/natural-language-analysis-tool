import pickle
import pandas as pd
import lightgbm as lgbm
from const import OUTPUT_DIR
from langchain.tools import BaseTool
from models.make_data import make_forecast_train_data, make_forecast_test_data


class LgbmtrainTool(BaseTool):
    name = "lgbm_train_tool"
    description = """useful to receive csv file name and learn LightGBM"""

    def _run(self, query: str) -> str:
        """Use the tool."""
        # global lgbm
        lgb_train, lgb_eval, num_class = make_forecast_train_data(query)
        # number of classes of the objective variable
        if num_class == 2:
            print("Binary Classification")
            params = {"task": "train", "boosting_type": "gbdt", "objective": "binary", "metric": "auc"}
        elif num_class <= 50:
            print("Multi Classification")
            params = {
                "task": "train",
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": num_class,
            }
        else:
            print("Regression")
            params = {"task": "train", "boosting_type": "gbdt", "objective": "regression", "metric": "rmse"}

        lgbm_model = lgbm.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=10, verbose=True),  # early_stopping用コールバック関数
                lgbm.log_evaluation(1),
            ],  # コマンドライン出力用コールバック関数
        )

        file = f"{OUTPUT_DIR}/trained_model.pkl"
        pickle.dump(lgbm_model, open(file, "wb"))

        result = "LightGBMの学習が完了しました"
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class LgbminferenceTool(BaseTool):
    name = "lgbm_inference_tool"
    description = """useful for receiving csv file name and making inferences in LightGBM"""

    def _run(self, query: str) -> str:
        x_test = make_forecast_test_data(query)

        file = f"{OUTPUT_DIR}/trained_model.pkl"
        lgbm_model = pickle.load(open(file, "rb"))

        y_pred = lgbm_model.predict(x_test, num_interation=lgbm_model.best_iteration)
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(f"{OUTPUT_DIR}/inference.csv")

        result = "LightGBMの推論が完了しました"
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

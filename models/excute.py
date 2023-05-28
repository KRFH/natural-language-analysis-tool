# LangChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

# langchaintools
from models.langchaintools import mltools
from models.langchaintools import preprocessingtools as ppt

import os
from const import OUTPUT_DIR, INPUT_DIR, API_PATH
from layouts import preprocessed_result_layouts, created_dataset_layouts
from dash import html

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open(API_PATH, mode="r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()


def run_mltools(query: str, target: str, num_class: int):
    # Toolの設定
    tools = [
        mltools.LgbmtrainTool(),
        mltools.LgbminferenceTool(),
        ppt.DropColumnTool(),
        ppt.OnehotEncodingTool(),
        ppt.LabelEncodingTool(),
        ppt.TargetEncodingTool(),
        ppt.Fill0Tool(),
        ppt.FillMeansTool(),
        ppt.FillMedianTool(),
        ppt.MakeDatasetTool(),
        # PreprocessingTool(),
    ]
    # 通常のLangChainの設定
    llm = OpenAI(temperature=0)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    prompt = """
        {input_}一度toolを使ったら必ず終了してください．
        目的変数は{target_}です．
        """.format(
        input_=query, target_=target
    )

    results = agent.run(prompt)
    layouts = []

    if ("deleted" in results) or ("encod" in results) or ("fill" in results):
        layouts = preprocessed_result_layouts(results)

    if "dataset" in results:
        layouts = created_dataset_layouts(results)

    if "learning" in results:
        layouts = [html.Div(results)]
    if "inference" in results:
        layouts = [html.Div(results)]

    # # 推論結果の読み込み
    # test_pred = pd.read_csv(f"/{OUTPUT_DIR}/inference.csv", index_col=0)
    # test_pred = test_pred.values.reshape(-1)
    # test_target = pd.read_csv(f"/{INPUT_DIR}/test_target.csv", index_col=0)
    # test_target = test_target.values.reshape(-1)

    # # Calculate metrics
    # if num_class == 2:
    #     test_pred = np.where(test_pred > 0.5, 1, 0)
    #     results = {
    #         "metrics": {
    #             "accuracy": accuracy_score(test_target, test_pred).round(1),
    #             "precision": precision_score(test_target, test_pred).round(1),
    #             "recall": recall_score(test_target, test_pred).round(1),
    #             "F1-score": f1_score(test_target, test_pred).round(1),
    #             "AUC-ROC": roc_auc_score(test_target, test_pred).round(1),
    #         },
    #         "confusion_matrix_data": confusion_matrix(test_target, test_pred),
    #     }
    # elif num_class <= 50:
    #     pass
    # else:
    #     results = {
    #         "metrics": {
    #             "MAE": mean_absolute_error(test_target, test_pred).round(1),
    #             "MSE": mean_squared_error(test_target, test_pred).round(1),
    #             "RMSE": np.sqrt(mean_squared_error(test_target, test_pred)).round(1),
    #             "R-squared": r2_score(test_target, test_pred).round(1),
    #         },
    #         "actual_vs_predicted_data": {"actual": test_target, "predicted": test_pred},
    #         "residuals_data": {"residuals": test_target - test_pred},
    #     }

    print(f"results:{results}")

    return layouts


# if __name__ == "__main__":
#     run_mltools()

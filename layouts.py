from dash import html, dcc, dash_table
from io import StringIO
import pandas as pd
import numpy as np
from const import INPUT_DIR, OUTPUT_DIR
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def file_upload_and_stored():
    return [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        # Data store and info section
        dcc.Store(id="stored-dataframe"),
        dcc.Store(id="stored-dataframe-eda"),
        html.Div(id="dataframe-info"),
    ]


def preprocess_and_eda():
    return [
        html.H2("Data understanding and preparation"),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="データ前処理",
                    id="preprocessing-tab",
                    children=[
                        # Preprocessing query input section
                        html.Div(
                            [
                                dcc.Input(
                                    id="query-input-preprocessing",
                                    placeholder="クエリを入力してください（例：'age'の欠損値を平均値で補完）",
                                    style={"width": "100%", "height": "50px"},
                                ),
                                html.Button("Submit", id="submit-button-preprocessing"),
                            ],
                            id="preprocessing-input-container",
                        ),
                        # Preprocessing query result section
                        dcc.Loading(html.Div(id="query-result-preprocessing")),
                    ],
                ),
                dcc.Tab(
                    label="EDA",
                    id="eda-tab",
                    children=[
                        # EDA query input section
                        html.Div(
                            [
                                dcc.Input(
                                    id="query-input-eda",
                                    placeholder="クエリを入力してください（例：男性で３０歳以上４０歳未満で生き残った人は？）",
                                    style={"width": "100%", "height": "50px"},
                                ),
                                html.Button("Submit", id="submit-button-eda"),
                            ]
                        ),
                        # EDA query result section
                        dcc.Loading(html.Div(id="query-result-eda")),
                        # EDA plots section
                        dcc.Loading(html.Div(id="eda-plots")),
                    ],
                ),
            ],
        ),
    ]


def modeling_evaluaion():
    return [
        html.H2("Modeling and Evaluation"),
        html.Div(
            [
                dcc.Input(
                    id="query-input-modeling",
                    placeholder="クエリを入力してください（例：train.csvを使ってLightGBMの学習を行なったあとtest.csvのデータを推論してください）",
                    style={"width": "100%", "height": "50px"},
                ),
                html.Button("Submit", id="submit-button-modeling"),
            ],
            id="modeling-input-container",
        ),
        dcc.Loading(html.Div(id="query-result-modeling")),
    ]


def preprocessed_result_layouts(results):
    df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    info_str = info_buffer.getvalue()
    return [
        html.Div("----------------------------------------------------------------"),
        html.Div(results),
        html.Label("データフレームの情報:"),
        html.Pre(info_str),
        html.Label("データフレームの要約統計:"),
        dash_table.DataTable(
            id="describe-table",
            columns=[{"name": i, "id": i} for i in df.reset_index().describe().columns],
            data=df.describe().reset_index().to_dict("records"),
        ),
        html.Label("データフレームの最初の10行:"),
        dash_table.DataTable(
            id="head-table",
            columns=[{"name": i, "id": i} for i in df.head(10).columns],
            data=df.head(10).to_dict("records"),
        ),
    ]


def created_dataset_layouts(results):
    return [
        html.Div("----------------------------------------------------------------"),
        html.P("Create Dataset"),
        html.Div(results),
    ]


def generate_missing_value_plot(df):
    """Function to generate missing value plot"""
    missing_values = df.isnull().sum()
    missing_values_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_values_percent}).reset_index()

    missing_plot = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in missing_df.columns],
        data=missing_df.to_dict("records"),
    )

    return missing_plot


def generate_numerical_distribution_plot(df):
    """Function to generate numerical distribution plot"""
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numerical_columns):
        fig = make_subplots(cols=len(numerical_columns), rows=1, subplot_titles=numerical_columns)

        for i, col in enumerate(numerical_columns, start=1):
            fig.add_trace(go.Histogram(x=df[col], nbinsx=20, histnorm="probability"), col=i, row=1)

        fig.update_layout(title_text="Numerical Distributions")
        layout = dcc.Graph(figure=fig)
    else:
        layout = html.P("データがありません")

    return layout


def generate_categorical_distribution_plot(df):
    """Function to generate categorical distribution plot"""
    categorical_columns = df.select_dtypes(include=["object", "bool"]).columns
    if len(categorical_columns):
        fig = make_subplots(cols=len(categorical_columns), rows=1, subplot_titles=categorical_columns)

        for i, col in enumerate(categorical_columns, start=1):
            fig.add_trace(go.Histogram(x=df[col], histnorm="probability"), col=i, row=1)

        fig.update_layout(title_text="Categorical Distributions")
        layout = dcc.Graph(figure=fig)
    else:
        layout = html.P("データがありません")

    return layout


def generate_correlation_plot(df):
    """Function to generate correlation plot"""
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numerical_columns):
        corr_matrix = df[numerical_columns].corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr_matrix,
                x=numerical_columns,
                y=numerical_columns,
                colorscale="RdBu",
            )
        )
        fig.update_layout(title_text="Numerical Column Correlations")
        layout = dcc.Graph(figure=fig)
    else:
        layout = html.P("データがありません")

    return layout


def learning_result_layouts(results):
    df = pd.read_csv(f"{OUTPUT_DIR}/importance.csv")

    # Create the subplot figure
    fig = make_subplots(rows=1, cols=2)

    # Add the "Split Importance" bar graph
    fig.add_trace(
        go.Bar(x=df["split_importance"], y=df["Unnamed: 0"], orientation="h", name="Split Importance"), row=1, col=1
    )
    # Add the "Gain Importance" bar graph
    fig.add_trace(
        go.Bar(x=df["gain_importance"], y=df["Unnamed: 0"], orientation="h", name="Gain Importance"), row=1, col=2
    )
    # Set the layout for the subplots
    fig.update_layout(title="Feature Importance", height=500, width=800)

    # Set the individual axes titles for each subplot
    fig.update_xaxes(title="Split Importance", row=1, col=1)
    fig.update_xaxes(title="Gain Importance", row=1, col=2)
    fig.update_yaxes(title="Feature", row=1, col=1)

    return [
        html.Div("----------------------------------------------------------------"),
        html.Div(results),
        dcc.Graph(figure=fig),
    ]


def inference_result_layouts(results, num_class, target):
    # 推論結果の読み込み
    test_pred = pd.read_csv(f"{OUTPUT_DIR}/inference.csv", index_col=0)
    test_pred = test_pred.values.reshape(-1)
    test_target = pd.read_csv(f"{INPUT_DIR}/test.csv")[target]
    test_target = test_target.values.reshape(-1)

    # Create the Plotly layout
    if num_class == 2:
        # Binary classification
        # test_pred = np.where(test_pred > 0.5, 1, 0)
        binary_results = {
            "metrics": {
                "accuracy": accuracy_score(test_target, test_pred).round(1),
                "precision": precision_score(test_target, test_pred).round(1),
                "recall": recall_score(test_target, test_pred).round(1),
                "F1-score": f1_score(test_target, test_pred).round(1),
                "AUC-ROC": roc_auc_score(test_target, test_pred).round(1),
            },
            "confusion_matrix_data": confusion_matrix(test_target, test_pred),
        }
        confusion_matrix_fig = ff.create_annotated_heatmap(
            binary_results["confusion_matrix_data"],
            colorscale="Blues",
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
        )
        layouts = [
            html.Div("----------------------------------------------------------------"),
            html.Div(results),
            html.Div(
                [
                    html.H2("Binary Classification Results"),
                    html.Table(
                        [html.Tr([html.Td(key), html.Td(value)]) for key, value in binary_results["metrics"].items()]
                    ),
                    dcc.Graph(id="confusion-matrix", figure=confusion_matrix_fig),
                ]
            ),
        ]

        return layouts

    elif num_class <= 50:
        # Multiclass classification
        return [
            html.Div("----------------------------------------------------------------"),
            html.Div(results),
        ]

    else:
        # Regression
        regression_results = {
            "metrics": {
                "MAE": mean_absolute_error(test_target, test_pred).round(1),
                "MSE": mean_squared_error(test_target, test_pred).round(1),
                "RMSE": np.sqrt(mean_squared_error(test_target, test_pred)).round(1),
                "R-squared": r2_score(test_target, test_pred).round(1),
            },
            "actual_vs_predicted_data": {"actual": test_target, "predicted": test_pred},
            "residuals_data": {"residuals": test_target - test_pred},
        }

        actual_vs_predicted_fig = go.Figure()
        actual_vs_predicted_fig.add_trace(
            go.Scatter(
                x=regression_results["actual_vs_predicted_data"]["actual"],
                y=regression_results["actual_vs_predicted_data"]["predicted"],
                mode="markers",
            )
        )
        actual_vs_predicted_fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")

        residuals_fig = go.Figure()
        residuals_fig.add_trace(go.Histogram(x=regression_results["residuals_data"]["residuals"]))
        residuals_fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
        layouts = [
            html.Div("----------------------------------------------------------------"),
            html.Div(results),
            html.Div(
                [
                    html.H2("Regression Results"),
                    html.Table(
                        [
                            html.Tr([html.Td(key), html.Td(value)])
                            for key, value in regression_results["metrics"].items()
                        ]
                    ),
                    dcc.Graph(id="actual-vs-predicted", figure=actual_vs_predicted_fig),
                    dcc.Graph(id="residuals-plot", figure=residuals_fig),
                ],
            ),
        ]

        return layouts

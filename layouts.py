from dash import html, dcc, dash_table
from io import StringIO
import pandas as pd
from const import INPUT_DIR


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

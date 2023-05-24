from dash import html, dcc


def file_upload_and_stored():
    return [
        html.H2("Upload csv file"),
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

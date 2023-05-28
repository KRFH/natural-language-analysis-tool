import os
import base64
import io
from io import StringIO
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from const import API_PATH, INPUT_DIR

from models.langchaintools.preprocessing import chat_tool_with_pandas_df
from utils import (
    generate_categorical_distribution_plot,
    generate_missing_value_plot,
    generate_numerical_distribution_plot,
    generate_correlation_plot,
    generate_classification_result_layout,
    generate_regression_result_layout,
    df_to_csv_data,
    save_input_data,
)
from layouts import file_upload_and_stored, preprocess_and_eda, modeling_evaluaion
from models.excute import run_mltools


# Set the API key environment variable
with open(API_PATH, mode="r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout for the Dash application
app.layout = html.Div(
    # Title
    [
        dcc.Loading(
            html.Div(
                id="output_layouts",
                children=[
                    html.H1("Interactive Dataset Natural Language Query and Analysis Tool"),
                ],
            ),
        ),
        html.Div(style={"height": "200px"}),
        html.Div(
            id="chat-space-footer",
            children=[
                dcc.Tabs(
                    id="tabs",
                    children=[
                        dcc.Tab(
                            label="File upload",
                            id="file_upload",
                            children=[
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div([" Drag and Drop or Select a CSV File"]),
                                    style={
                                        "width": "100%",
                                        "height": "55px",
                                        "lineHeight": "55px",
                                        "borderWidth": "1px",
                                        "borderStyle": "solid",
                                        "borderRadius": "5px",
                                        "textAlign": "left",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                # Data store and info section
                                dcc.Store(id="stored-dataframe"),
                                dcc.Store(id="stored-dataframe-eda"),
                            ],
                        ),
                        dcc.Tab(
                            label="Data Science",
                            id="data_science",
                            children=[
                                # Preprocessing query input section
                                html.Div(
                                    [
                                        html.Button("Submit", id="submit-button"),
                                        dcc.Input(
                                            id="query-input",
                                            placeholder="クエリを入力してください（例：'age'の欠損値を平均値で補完）",
                                            style={"width": "90%", "height": "50px", "margin": "10px"},
                                        ),
                                    ],
                                    id="input-container",
                                ),
                                html.Div(
                                    [
                                        html.H4("Set target column", style={"margin-right": "20px"}),
                                        dcc.Dropdown(
                                            id="target_column",
                                            multi=False,
                                            persistence=True,
                                            persistence_type="session",
                                            style={"width": "300px"},
                                        ),
                                    ],
                                    style={"display": "flex"},
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="EDA",
                            id="eda",
                            children=[
                                # EDA query input section
                                html.Div(
                                    [
                                        html.Button("Submit", id="submit-button-eda"),
                                        dcc.Input(
                                            id="query-input-eda",
                                            placeholder="クエリを入力してください（例：男性で３０歳以上４０歳未満で生き残った人は？）",
                                            style={"width": "90%", "height": "50px", "margin": "10px"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            style={
                "position": "fixed",
                "bottom": "0",
                "width": "100%",
                "height": "200px",
                "line-height": "20px",
                "background-color": "#f9f9f9",
                "text-align": "left",
            },
        ),
    ]
)


@app.callback(
    Output("target_column", "options"),
    Input("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def update_dropdown_target_col(data):
    if data is None:
        raise PreventUpdate

    cols = pd.DataFrame(data).columns.tolist()

    return cols


@app.callback(
    Output("output_layouts", "children", allow_duplicate=True),
    Input("target_column", "value"),
    State("output_layouts", "children"),
    prevent_initial_call=True,
)
def update_set_target_col(target_column, layouts):
    layouts = layouts + [html.Div(f"Set target column: {target_column}")]

    return layouts


# Callback to store the uploaded dataframe and display its information
@app.callback(
    Output("stored-dataframe", "data", allow_duplicate=True),
    Output("output_layouts", "children", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("output_layouts", "children"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_uploaded_dataframe(contents, layouts, filename):
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            if "csv" in filename:
                # Assume that the user uploaded a CSV file
                uploaded_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), index_col=0)
                info_buffer = StringIO()
                uploaded_df.info(buf=info_buffer)
                info_str = info_buffer.getvalue()

                layouts = layouts + [
                    html.Label("データフレームの情報:"),
                    html.Pre(info_str),
                    html.Label("データフレームの要約統計:"),
                    dash_table.DataTable(
                        id="describe-table",
                        columns=[{"name": i, "id": i} for i in uploaded_df.reset_index().describe().columns],
                        data=uploaded_df.describe().reset_index().to_dict("records"),
                    ),
                    html.Label("データフレームの最初の10行:"),
                    dash_table.DataTable(
                        id="head-table",
                        columns=[{"name": i, "id": i} for i in uploaded_df.head(10).columns],
                        data=uploaded_df.head(10).to_dict("records"),
                    ),
                ]

                uploaded_df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

            else:
                return html.Div(["Invalid file type. Please upload a CSV file."])
        except Exception as e:
            return html.Div([f"Error processing file: {str(e)}"])
        return uploaded_df.to_dict("records"), layouts
    return None, layouts


# Callback to update preprocessing query results and store the result dataframe
@app.callback(
    Output("output_layouts", "children", allow_duplicate=True),
    Output("stored-dataframe", "data", allow_duplicate=True),
    Input("submit-button", "n_clicks"),
    State("output_layouts", "children"),
    State("query-input", "value"),
    State("stored-dataframe", "data"),
    State("target_column", "value"),
    prevent_initial_call=True,
)
def update_query_results(n_clicks, layouts, query, df, target):
    if n_clicks is None or df is None:
        return "", None

    if query and target:
        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        num_class = len(df[target].unique())
        results = run_mltools(query, target, num_class)
        result_df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        store_data = result_df.to_dict("records")
        layouts = layouts + results
        return layouts, store_data

    return layouts + [html.Div("クエリが入力されていません。")], df


# Callback to update EDA query results and store the result dataframe
@app.callback(
    Output("output_layouts", "children", allow_duplicate=True),
    Output("stored-dataframe-eda", "data", allow_duplicate=True),
    Input("submit-button-eda", "n_clicks"),
    State("output_layouts", "children"),
    State("query-input-eda", "value"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def update_eda_query_results(n_clicks, layouts, query, df):
    if n_clicks is None or df is None:
        return "", None

    if query:
        df = pd.DataFrame(df)
        query += "そのdataframeを表示するpandas codeは？"
        result_df = chat_tool_with_pandas_df(df, query)
        # Generate missing value plot
        missing_plot = generate_missing_value_plot(result_df)
        # Generate numerical distribution plot
        numerical_plot = generate_numerical_distribution_plot(result_df)
        # Generate categorical distribution plot
        categorical_plot = generate_categorical_distribution_plot(result_df)
        # Generate correlation plot
        correlation_plot = generate_correlation_plot(result_df)

        layouts = (
            layouts
            + [
                html.Li("EDAクエリ結果"),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in result_df.columns],
                    data=result_df.to_dict("records"),
                ),
            ]
            + [
                html.Li("EDAプロット"),
                html.Label("欠損値の確認"),
                missing_plot,
                html.Label("数値データの分布"),
                numerical_plot,
                html.Label("カテゴリデータの分布"),
                categorical_plot,
                html.Label("数値データの相関"),
                correlation_plot,
            ]
        )

        store_data = result_df.to_dict("records")
        # layouts = layouts+
        return layouts, store_data

    return "クエリが入力されていません。", None


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

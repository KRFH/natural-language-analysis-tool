import os
import base64
import io
from io import StringIO
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from const import API_PATH
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
    [html.H1("Interactive Dataset Natural Language Query and Analysis Tool")]
    +
    # File upload section
    file_upload_and_stored()
    # set target column
    + [
        html.H2("Set target column"),
        dcc.Dropdown(id="target_column", multi=False),
    ]
    # Tabs for Preprocessing and EDA sections
    + preprocess_and_eda()
    # create dataset
    + [
        html.H2("Create Dataset"),
        # html.Button("Create Dataset and Download", id="split-dataset"),
        html.Div(id="train-test-csv-files"),
    ]
    # modeling and evaluation
    + modeling_evaluaion()
)


@app.callback(
    Output("target_column", "options"),
    Input("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def update_radioitem_target_col(data):
    if data is None:
        raise PreventUpdate

    cols = pd.DataFrame(data).columns.tolist()

    return cols


# Callback to store the uploaded dataframe and display its information
@app.callback(
    Output("stored-dataframe", "data", allow_duplicate=True),
    Output("dataframe-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_uploaded_dataframe(contents, filename):
    layout = []
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
                layout = [
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
            else:
                return html.Div(["Invalid file type. Please upload a CSV file."])
        except Exception as e:
            return html.Div([f"Error processing file: {str(e)}"])
        return uploaded_df.to_dict("records"), layout

    return None, layout


# Callback to update preprocessing query results and store the result dataframe
@app.callback(
    Output("query-result-preprocessing", "children"),
    Output("stored-dataframe", "data", allow_duplicate=True),
    Input("submit-button-preprocessing", "n_clicks"),
    State("query-input-preprocessing", "value"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def update_preprocessing_results(n_clicks, query, df):
    if n_clicks is None or df is None:
        return "", None

    if query:
        df = pd.DataFrame(df)
        query += "そのpythoncodeは？"
        result_df = chat_tool_with_pandas_df(df, query)
        store_data = result_df.to_dict("records")
        info_buffer = StringIO()
        result_df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()
        layout = [
            html.Li("前処理クエリ結果"),
            html.Label("データフレームの情報:"),
            html.Pre(info_str),
            html.Label("データフレームの要約統計:"),
            dash_table.DataTable(
                id="describe-table",
                columns=[{"name": i, "id": i} for i in result_df.reset_index().describe().columns],
                data=result_df.describe().reset_index().to_dict("records"),
            ),
            html.Label("データフレームの最初の10行:"),
            dash_table.DataTable(
                id="head-table",
                columns=[{"name": i, "id": i} for i in result_df.head(10).columns],
                data=result_df.head(10).to_dict("records"),
            ),
        ]
        return layout, store_data

    return "クエリが入力されていません。", None


# Callback to update EDA query results and store the result dataframe
@app.callback(
    Output("query-result-eda", "children"),
    Output("stored-dataframe-eda", "data", allow_duplicate=True),
    Input("submit-button-eda", "n_clicks"),
    State("query-input-eda", "value"),
    State("stored-dataframe", "data"),
    prevent_initial_call=True,
)
def update_eda_query_results(n_clicks, query, df):
    if n_clicks is None or df is None:
        return "", None

    if query:
        df = pd.DataFrame(df)
        query += "そのdataframeを表示するpandas codeは？"
        result_df = chat_tool_with_pandas_df(df, query)
        store_data = result_df.to_dict("records")
        return [
            html.Li("EDAクエリ結果"),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in result_df.columns],
                data=result_df.to_dict("records"),
            ),
        ], store_data

    return "クエリが入力されていません。", None


# Callback to generate EDA plots from DataFrame
@app.callback(
    Output("eda-plots", "children"),
    Input("stored-dataframe", "data"),
    Input("stored-dataframe-eda", "data"),
)
def generate_eda_plots(data_default, data):
    if not data_default:
        raise PreventUpdate
    if not data:
        data = data_default

    df = pd.DataFrame(data)

    # Generate missing value plot
    missing_plot = generate_missing_value_plot(df)
    # Generate numerical distribution plot
    numerical_plot = generate_numerical_distribution_plot(df)
    # Generate categorical distribution plot
    categorical_plot = generate_categorical_distribution_plot(df)
    # Generate correlation plot
    correlation_plot = generate_correlation_plot(df)

    layout = [
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

    return layout


# Create a callback function to split the dataset into train and test sets
@app.callback(
    Output("train-test-csv-files", "children"),
    # Input("split-dataset", "n_clicks"),
    Input("stored-dataframe", "data"),
    Input("target_column", "value"),
    prevent_initial_call=False,
)
def split_dataset_into_train_and_test(data, target_column):
    if not data:
        return [html.P("データがありません")]

    if not target_column:
        return [html.P("ターゲットが選択されていません")]

    else:
        df = pd.DataFrame(data)
        x = df.drop(target_column, axis=1)
        y = df[target_column]
        print(f"全データ数:{len(x)}")

        dataset_info = save_input_data(x, y)

        _csv_data = df_to_csv_data(df)

        _href = f"data:text/csv;charset=utf-8;base64,{_csv_data}"

        layout = [html.Pre(f"{key}: {value}") for key, value in dataset_info.items()] + [
            html.A(
                "Download_dataset.csv",
                id="download-link",
                download="dataset.csv",
                href=_href,
                target="_blank",
            ),
        ]

    return layout


# Callback to update modeling query results and output the modeling result
@app.callback(
    Output("query-result-modeling", "children"),
    Input("submit-button-modeling", "n_clicks"),
    State("query-input-modeling", "value"),
    State("stored-dataframe", "data"),
    State("target_column", "value"),
    prevent_initial_call=True,
)
def update_modeling_results(n_clicks, query, data, target_column):
    if n_clicks is None or data is None:
        return "", None

    if query:
        df = pd.DataFrame(data)
        num_class = len(df[target_column].unique())
        results = run_mltools(query, num_class)

        if num_class == 2:
            layout = generate_classification_result_layout(results)

        elif num_class <= 50:
            pass
        else:
            layout = generate_regression_result_layout(results)

        return layout

    return "クエリが入力されていません。", None


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

import os
import base64
import io
from io import StringIO
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from chat import chat_tool_with_pandas_df

# Open the API key file and set the environment variable
with open("/Users/kai/Desktop/api/openai.txt", mode="r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

# Define the layout for the Dash application
app = dash.Dash(__name__)

# Create the layout with upload, input, and display sections
app.layout = html.Div(
    [
        # Title
        html.H1("Dataset Query Tool"),
        # Upload section
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
            # Allow only one file to be uploaded
            multiple=False,
        ),
        dcc.Store(id="stored-dataframe"),
        html.Div(id="dataframe-info"),
        # Query input section
        html.Label("クエリ入力"),
        html.Div(
            [
                dcc.Input(
                    id="query-input",
                    placeholder="クエリを入力してください（例：男性で３０歳以上４０歳未満で生き残った人は？）",
                    style={"width": "100%", "height": "50px"},
                ),
                html.Button("Submit", id="submit-button"),
            ]
        ),
        # Query result section
        html.Label("クエリ結果"),
        dcc.Loading(html.Div(id="query-result")),
        dcc.Store(id="query-result-dataframe"),
        # EDA plots section
        html.Label("EDAプロット"),
        dcc.Loading(html.Div(id="eda-plots")),
    ]
)


# Callback function to display query results and store the result dataframe
@app.callback(
    Output("query-result", "children"),
    Output("query-result-dataframe", "data"),
    Input("submit-button", "n_clicks"),
    [
        State("query-input", "value"),
        State("stored-dataframe", "data"),
    ],
)
def update_output(n_clicks, query, df):
    if n_clicks is None:
        return "", None
    if df is None:
        return "csv ファイルをアップロードしてください", None
    if query:
        df = pd.DataFrame(df)
        result_df = chat_tool_with_pandas_df(df, query).reset_index()
        store_data = result_df.to_dict("records")
        return [
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in result_df.columns],
                data=result_df.to_dict("records"),
            ),
        ], store_data

    return "クエリが入力されていません。", None


# Callback function to store the uploaded dataframe and display its information
@app.callback(
    Output("stored-dataframe", "data"),
    Output("dataframe-info", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def update_output(contents, filename):
    layout = []
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            if "csv" in filename:
                # Assume that the user uploaded a CSV file
                uploaded_df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8")), index_col=0
                )
                info_buffer = StringIO()
                uploaded_df.info(buf=info_buffer)
                info_str = info_buffer.getvalue()
                layout = [
                    html.Label("データフレームの情報:"),
                    html.Pre(info_str),
                    html.Label("データフレームの要約統計:"),
                    dash_table.DataTable(
                        id="describe-table",
                        columns=[
                            {"name": i, "id": i}
                            for i in uploaded_df.reset_index().describe().columns
                        ],
                        data=uploaded_df.describe().reset_index().to_dict("records"),
                    ),
                    html.Label("データフレームの最初の10行:"),
                    dash_table.DataTable(
                        id="head-table",
                        columns=[
                            {"name": i, "id": i} for i in uploaded_df.head(10).columns
                        ],
                        data=uploaded_df.head(10).to_dict("records"),
                    ),
                ]
            else:
                return html.Div(["Invalid file type. Please upload a CSV file."])
        except Exception as e:
            return html.Div([f"Error processing file: {str(e)}"])
        return uploaded_df.to_dict("records"), layout

    return None, layout


# Callback function to generate plots from DataFrame
@app.callback(
    Output("eda-plots", "children"),
    Input("query-result-dataframe", "data"),
)
def generate_eda_plots(data):
    if not data:
        raise PreventUpdate

    df = pd.DataFrame(data)

    # Check for missing values
    missing_values = df.isnull().sum()
    missing_values_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_values_percent}
    ).reset_index()

    # Separate columns into numerical and categorical
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=["object", "bool"]).columns

    # Create a subplot for numerical distributions
    fig_num_dist = make_subplots(
        cols=len(numerical_columns), rows=1, subplot_titles=numerical_columns
    )

    # Plot histograms for numerical columns
    for i, col in enumerate(numerical_columns, start=1):
        fig_num_dist.add_trace(
            go.Histogram(x=df[col], nbinsx=20, histnorm="probability"), col=i, row=1
        )

    fig_num_dist.update_layout(title_text="Numerical Distributions")

    # Create a subplot for categorical distributions
    fig_cat_dist = make_subplots(
        cols=len(categorical_columns), rows=1, subplot_titles=categorical_columns
    )

    # Plot bar charts for categorical columns
    for i, col in enumerate(categorical_columns, start=1):
        fig_cat_dist.add_trace(
            go.Histogram(x=df[col], histnorm="probability"), col=i, row=1
        )

    fig_cat_dist.update_layout(title_text="Categorical Distributions")

    # Plot heatmap for numerical column correlations
    corr_matrix = df[numerical_columns].corr()
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr_matrix,
            x=numerical_columns,
            y=numerical_columns,
            colorscale="RdBu",
        )
    )
    fig_corr.update_layout(title_text="Numerical Column Correlations")

    layout = [
        html.Label("欠損値の確認"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in missing_df.columns],
            data=missing_df.to_dict("records"),
        ),
        html.Label("数値データの分布"),
        dcc.Graph(figure=fig_num_dist),
        html.Label("カテゴリデータの分布"),
        dcc.Graph(figure=fig_cat_dist),
        html.Label("数値データの相関"),
        dcc.Graph(figure=fig_corr),
    ]

    return layout


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

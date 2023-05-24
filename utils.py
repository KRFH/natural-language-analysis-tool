import base64
import pandas as pd
from dash import dash_table, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from const import INPUT_DIR


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


def generate_classification_result_layout(results):
    confusion_matrix_fig = ff.create_annotated_heatmap(
        results["confusion_matrix_data"],
        colorscale="Blues",
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
    )
    layout = [
        html.Div(
            [
                html.H2("Classification Results"),
                html.Table([html.Tr([html.Td(key), html.Td(value)]) for key, value in results["metrics"].items()]),
                dcc.Graph(id="confusion-matrix", figure=confusion_matrix_fig),
            ]
        ),
    ]
    return layout


def generate_regression_result_layout(results):
    actual_vs_predicted_fig = go.Figure()
    actual_vs_predicted_fig.add_trace(
        go.Scatter(
            x=results["actual_vs_predicted_data"]["actual"],
            y=results["actual_vs_predicted_data"]["predicted"],
            mode="markers",
        )
    )
    actual_vs_predicted_fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")

    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Histogram(x=results["residuals_data"]["residuals"]))
    residuals_fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
    layout = [
        html.Div(
            [
                html.H2("Regression Results"),
                html.Table([html.Tr([html.Td(key), html.Td(value)]) for key, value in results["metrics"].items()]),
                dcc.Graph(id="actual-vs-predicted", figure=actual_vs_predicted_fig),
                dcc.Graph(id="residuals-plot", figure=residuals_fig),
            ],
        ),
    ]

    return layout


def df_to_csv_data(df):
    """
    Create a function to convert the DataFrames into downloadable CSV files
    """
    csv_data = df.to_csv(index=True, encoding="utf-8")
    base64_data = base64.b64encode(csv_data.encode()).decode("utf-8")
    return base64_data


def save_input_data(x, y):
    train, test, train_target, test_target = train_test_split(x, y, test_size=0.2, random_state=3655)

    train["target"] = train_target
    train.to_csv(f"{INPUT_DIR}/train.csv")
    test.to_csv(f"{INPUT_DIR}/test.csv")
    train_target.to_csv(f"{INPUT_DIR}/train_target.csv")
    test_target.to_csv(f"{INPUT_DIR}/test_target.csv")

    info = {
        "Number of Training Examples ": train.shape[0],
        "Number of Test Examples ": test.shape[0],
        "Training X Shape ": train.shape,
        "Training y Shape ": train_target.shape[0],
        "Test X Shape ": test.shape,
        "Test y Shape ": test_target.shape[0],
        "train columns": train.columns,
        "test columns": test.columns,
    }
    return info

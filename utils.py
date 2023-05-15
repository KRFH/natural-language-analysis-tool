import base64
import pandas as pd
from dash import dash_table, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def df_to_csv_data(df):
    """
    Create a function to convert the DataFrames into downloadable CSV files
    """
    csv_data = df.to_csv(index=True, encoding="utf-8")
    base64_data = base64.b64encode(csv_data.encode()).decode("utf-8")
    return base64_data

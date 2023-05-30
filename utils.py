import base64
from sklearn.model_selection import train_test_split
from const import INPUT_DIR, TEST_SIZE, RANDOM_STATE


def df_to_csv_data(df):
    """
    Create a function to convert the DataFrames into downloadable CSV files
    """
    csv_data = df.to_csv(index=True, encoding="utf-8")
    base64_data = base64.b64encode(csv_data.encode()).decode("utf-8")
    return base64_data


def save_input_data(x, y):
    train, test, train_target, test_target = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

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


def text_processing(text: str):
    text = text.strip()
    text = text.replace('"', "")
    text = text.replace("'", "")

    return text

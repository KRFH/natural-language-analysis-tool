# natural-language-analysis-tool
This repository is a dashboard application that uses OpenAI's GPT and Langchain to receive queries in Natural language and perform data operations on Pandas data frames. The application supports data upload, preprocessing, exploratory data analysis (EDA), and generation of training data for model learning.

## Features
- CSV file upload: Users can upload CSV files to the application. The uploaded data is displayed as information about the data frame, summary statistics, and the first 10 rows.
- Data preprocessing: Users can enter a preprocessing query in Natural language and apply the corresponding data operation to a Pandas data frame.
- Exploratory data analysis (EDA): Users can enter an EDA query in Natural language and apply the corresponding data operation to a Pandas data frame. EDA plots are also generated, displaying missing values, distributions of numerical data, distributions of categorical data, and correlations of numerical data.
- Generation of training data for model learning: Users can split the data frame into training and test sets and download them as CSV files.

## Usage
Clone the repository and install the necessary packages.

`git clone https://github.com/KRFH/natural-language-analysis-tool.git`
`pip install -r requirements.txt`

Start the application.

`python app.py`

Use the application by accessing it in your browser at http://127.0.0.1:8050/.

## Implementation Details
This application uses GPT and Langchain to receive data operation queries in Natural language and apply the corresponding data operations to a Pandas data frame. This allows Natural language speakers to perform data operations in a natural format.

## Contribution
Contributions to the project, such as bug reports, feature requests, and pull requests, are welcome. This project is open source and aims to be improved by the community. Follow these steps to contribute:

* Fork the repository and make a copy to your account.
* Create a new branch and commit changes.
* Push the branch and create a pull request.
## Contact
If you have any questions or suggestions, please create a new issue in the GitHub Issues section. Contributions, such as pull requests and bug reports, are also welcome.

## Prerequisites
Python 3.6 or higher
Dash 2.9.0 or higher

## Demo
![demo](https://user-images.githubusercontent.com/75525727/236665532-e3d3f2c1-e973-40ca-9f8f-c8f941dc645c.gif)



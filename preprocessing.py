from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def translate_to_english(text: str) -> str:
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Translate the following Japanese text to English: {text}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text)


def get_agent(df):
    return create_pandas_dataframe_agent(
        OpenAI(temperature=0),
        df,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
    )


def chat_tool_with_pandas_df(df, query):
    translated_query = translate_to_english(query)
    translated_query += "The operation must be applied directly to the original dataframe."
    translated_query += "The answer must be just python code"
    agent = get_agent(df)
    result = agent.run(translated_query)
    print(query)
    print(translated_query)
    print(result)
    if "inplace" in result:
        pass
    else:
        df = eval(result)

    return df

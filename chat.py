import json
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def translate_to_english(text: str) -> str:
    # Initialize the OpenAI LLM with a temperature of 0
    llm = OpenAI(temperature=0)

    # Create a prompt template for the translation task
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Translate the following Japanese text to English: {text}",
    )

    # Create a chain with the LLM and prompt
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the input text and return the translated text
    translated_text = chain.run(text)
    return translated_text


def chat_tool_with_pandas_df(df, query):
    query = query + "そのインデックスは？"

    # Translate the query to English
    translated_query = translate_to_english(query)
    print(query, translated_query)

    # Create an agent for querying the pandas DataFrame
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0),
        df,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
    )

    # Run the agent with the translated query
    raw_result = agent.run(translated_query)
    print(raw_result)

    # Create a prompt template for formatting the result in Python list format
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["query", "raw_result"],
        template="Q: {query}\nA: {raw_result}\nAnswer in python list format, using as few characters as possible. No explanatory text is required.",
    )

    # Create a chain to run the prompt with the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the translated query and raw_result as inputs
    formatted_result = chain.run({"query": translated_query, "raw_result": raw_result})
    formatted_result = json.loads(formatted_result.replace("\n", ""))
    print(formatted_result)

    # Return the relevant rows of the DataFrame based on the formatted result
    return df.loc[formatted_result, :]

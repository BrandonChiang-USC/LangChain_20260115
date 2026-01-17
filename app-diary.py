from typing import Annotated
from pydantic import Field
from agent_framework import ai_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from configparser import ConfigParser
# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(
    model=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
    base_url=config["AzureOpenAI"]["ENDPOINT"],
    api_key=config["AzureOpenAI"]["KEY"],
)

diary_vector_db = FAISS.load_local(
    "diary_vector_db", 
    embedding_function,
    "index",
    allow_dangerous_deserialization=True,
)


@ai_function(
    name="weather_tool", description="Retrieves all wather and temperature information with date."
)
def get_weather() -> str:
    """get all weather and temperature information with date."""
    print("[Function Call] get_weather")
    return_string = ""
    # get merged_output.csv file
    # read the csv file and extract weather and temperature information with date
    # for each line in the csv file, extract the date, weather and temperature information
    # and append to the return_string

    import csv
    with open("merged_output.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row["日期"]
            weather = row["氣候"]
            temperature = row["溫度"]
            return_string += f"日期: {date}, 氣候: {weather}, 溫度: {temperature}\n"

    return return_string

@ai_function(
    name="social_events_tool", description="Retrieves all social events information with date."
)
def get_social_events() -> str:
    """get all social events information with date."""
    print("[Function Call] get_social_events")
    return_string = ""
    # get merged_output.csv file
    # read the csv file and extract social events information with date
    # for each line in the csv file, extract the date and social events information
    # and append to the return_string

    import csv
    with open("merged_output.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row["日期"]
            social_events = row["社會記事"]
            return_string += f"日期: {date}, 社會記事: {social_events}\n"

    return return_string

@ai_function(
    name="diary_analysis_tool", description="Analyzes all diary entries and provides summary information."
)
def diary_analysis() -> str:
    """analyze all diary entries and provide summary information."""
    print("[Function Call] diary_analysis")
    return_string = ""
    # get merged_output.csv file
    # read the csv file and extract diary entries
    # for each line in the csv file, extract the diary entry and the date
    # merge all diary entries with date into a single string
    # and append to the return_string

    import csv
    with open("merged_output.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row["日期"]
            diary_content = row["日記內容"]
            return_string += f"日期: {date}, 日記內容: {diary_content}\n"
    
    return return_string

@ai_function(
    name="diary_rag_tool", description="Retrieve specific diary entry based on user question."
)
def diary_rag(
    question: Annotated[str, Field(description="the user question to find specific diary entry.")],
) -> str:
    """retrieve specific diary entry based on user question."""
    print("[Function Call] diary_rag")
    top3answer = diary_vector_db.similarity_search_with_score(question, 3)
    result = ""
    for doc, score in top3answer:
        result += f"Score: {score}\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n\n"
    return result


import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

system_role = """
你是一個非常了解將中正的歷史學家。
你會依據函數回傳的結果，來回答使用者的問題。
1.當使用者詢問天氣或溫度相關的問題時，請呼叫 weather_tool 函數來取得資訊，然後將結果回覆給使用者。
2.當使用者詢問「社會記事」相關的問題時，請呼叫 social_events_tool 函數來取得資訊，然後將結果回覆給使用者。
3.當使用者詢問跟所有日記有關的整體問題(例如全部的日記在講些什麼/摘要/總結/提到什麼事情幾次/平均幾點睡覺等等)，請呼叫 diary_analysis_tool 函數來取得資訊，然後將結果回覆給使用者。
4.當使用者詢問的問題是屬於特定一篇日記中的單一事件時(哪一天遇到誰/是哪一天很早睡等等)，請呼叫 diary_rag_tool 函數來取得資訊，然後將結果回覆給使用者。
"""

agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    instructions=system_role,
    tools=[
        get_weather,
        get_social_events,
        diary_analysis,
        diary_rag,
    ]
)

async def main():
    # question = "那十天的溫度大約在幾度到幾度之間？"
    # question = "那十天有幾天有下雨？"
    question = "社會記事一共講了幾次列強未平？"
    # question = "這十天的日記內容大約在講些什麼？請幫我做個總結。"
    # question = "蔣公這十天有幾天很晚睡覺？"
    # question = "蔣公有某一天很早睡，是哪一天？"
    # question = "蔣公哪一天開北閥會議?"
    print(f"問: {question}")
    result = await agent.run(
        question,
        temperature=0.3,
    )
    print(f"答: {result}")

asyncio.run(main())
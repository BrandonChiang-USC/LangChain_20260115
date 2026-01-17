import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_community.vectorstores import FAISS
import pandas as pd

# Load dataset
diary_data = pd.read_csv("merged_output.csv")

from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(
    model=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
    base_url=config["AzureOpenAI"]["ENDPOINT"],
    api_key=config["AzureOpenAI"]["KEY"],
)

metadatas = []
for i, row in diary_data.iterrows():
    metadata = {
        "日期": row["日期"],
        "提要": row["提要"],
        "社會記事": row["社會記事"],
        "氣候": row["氣候"],
        "溫度": row["溫度"],
    }
    metadatas.append(metadata)

diary_data["日記內容"] = diary_data["日記內容"].astype(str)
vector_db = FAISS.from_texts(
    texts=diary_data["日記內容"].tolist(),
    embedding=embedding_function,
    metadatas=metadatas,
)
vector_db.save_local("diary_vector_db", "index")

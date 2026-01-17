import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_community.vectorstores import FAISS
import pandas as pd

# Load dataset
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

# Embedding function - SentenceTransformer - all-MiniLM-L6-v2
# from langchain_huggingface import HuggingFaceEmbeddings

# embedding_function = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# Embedding function - AzureOpenAI - text-embedding-ada-002

from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(
    model=config["AzureOpenAI"]["Embedding_DEPLOYMENT_NAME"],
    base_url=config["AzureOpenAI"]["ENDPOINT"],
    api_key=config["AzureOpenAI"]["KEY"],
)


metadatas = []
for i, row in animal_data.iterrows():
    metadatas.append(
        {
            "Animal Name": row["animal_name"],
            "Source URL": row["source"],
            # "Media URL": row["media_link"],
            # "Wikipedia URL": row["wikipedia_link"],
        }
    )

animal_data["text"] = animal_data["text"].astype(str)

faiss = FAISS.from_texts(animal_data["text"].to_list(), embedding_function, metadatas)

# export the model
faiss.save_local("faiss_db_openai", "index")


# import the vector-db from disk

faiss_openai = FAISS.load_local(
    "faiss_db_openai",
    embedding_function,
    "index",
    allow_dangerous_deserialization=True
)

faiss_openai.similarity_search_with_score("What is ship of the desert?", 3)
faiss_openai.similarity_search_with_score("What is the earth?", 3)
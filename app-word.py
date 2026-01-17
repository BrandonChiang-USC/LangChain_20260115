from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configparser import ConfigParser
from langchain_core.output_parsers import StrOutputParser

# Get word data by Docx2txtLoader
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("rent_contract.docx")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Set up config parser
config = ConfigParser()
config.read("config.ini")

# Embedding - Google Generative AI - gemini-embedding-001
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=config["Gemini"]["API_KEY"],
)

db_gemini = FAISS.from_documents(docs, embeddings)

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=config["Gemini"]["API_KEY"],
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
請用繁體中文回答。
"""
)

output_parser = StrOutputParser()

chain = prompt | llm_gemini | output_parser

# query = "如果我想終止租約，我應該要多久以前通知房東？"
# query = "租房子簽約時，應該要帶什麼證件？"
query = "房間裡可以放鞭炮嗎？"
results = db_gemini.similarity_search_with_score(query, 5)
print("Retrieved related content :")
print(results[0][0].page_content)
print(results[1][0].page_content)
print(results[2][0].page_content)
# print(results[3][0].page_content)
# print(results[4][0].page_content)
print("====================================================")

llm_result = chain.invoke(
    {
        "input": query,
        "context": [
            results[0][0],
            results[1][0],
            results[2][0],
            # results[3][0],
            # results[4][0]
        ],
    }
)

print("Question:", query)
print("LLM Answer:", llm_result.lstrip(" "))
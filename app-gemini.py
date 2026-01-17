# import the vector-db from disk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
faiss = FAISS.load_local(
    "faiss_db", embedding_function, "index", allow_dangerous_deserialization=True
)

from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=config["Gemini"]["API_KEY"]
)


# question = "What is ship of the desert?"
question = "What is 赤馬紅羊?"
print(f"Q: {question}")
top1answer = faiss.similarity_search_with_score(question, 1)

if top1answer[0][1] > 1.0:
    # print("A: No good match found.")
    user_input = question
    role_description = """
    你是一個AI助理，請用繁體中文回答。
    """
    messages = [
        ("system", role_description),
        ("human", user_input),
    ]
    response_gemini = llm_gemini.invoke(messages)
    print(f"A: {response_gemini.content}")
else:
    print(f"A: {top1answer[0][0].page_content}")
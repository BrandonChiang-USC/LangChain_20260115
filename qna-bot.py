# import the vector-db from disk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
faiss = FAISS.load_local(
    "faiss_db", embedding_function, "index", allow_dangerous_deserialization=True
)

# question = "What is ship of the desert?"
question = "What is ITRI?"
print(f"Q: {question}")
top1answer = faiss.similarity_search_with_score(question, 1)

if top1answer[0][1] > 1.0:
    print("A: No good match found.")
else:
    print(f"A: {top1answer[0][0].page_content}")
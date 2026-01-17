from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display
from configparser import ConfigParser
import base64

config = ConfigParser()
config.read("config.ini")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=config["Gemini"]["API_KEY"],
    max_tokens=8192,
)


def image4LangChain(image_url):
    if "http" in image_url:
        return {"url":image_url}
    else:
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return {"url":f"data:image/jpeg;base64,{image_data}"}


user_messages = []
# append user input question
user_input = "圖片中的生物是什麼？請詳細描述。"
user_messages.append({"type": "text", "text": user_input + "請使用繁體中文回答。"})
# append images
# image_url = "https://i.ibb.co/KyNtMw5/IMG-20240321-172354614-AE.jpg"
image_url = "cat.jpg"

user_messages.append(
    {
        "type": "image_url",
        "image_url": image4LangChain(image_url),
    }
)

human_messages = HumanMessage(content=user_messages)
result = llm.invoke([human_messages])

print("Q: " + user_input)
print("A: " + result.content)

# Display the image
display(Image(url=image_url))
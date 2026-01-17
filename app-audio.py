from google import genai
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")
client = genai.Client(
    api_key=config["Gemini"]["API_KEY"]
)

#upload the audio file
audio_file_name = "tts-audio.mp3"
print(f"Uploading file: {audio_file_name}")
myfile = client.files.upload(file=audio_file_name)
print("Complete upload!")

question = """
請問這個聲音檔裡面在講什麼, 請用100字以內的中文說明摘要
"""
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents=[
        question, 
        myfile
    ]
)

print(response.text)
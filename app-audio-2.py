from google import genai
from google.genai import types
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

with open("outputaudio8.wav", "rb") as f:
    audio_bytes = f.read()

client = genai.Client(
    api_key=config["Gemini"]["API_KEY"]
)

question = f"""
請完整輸出這個聲音檔案的內容文字。
並適當地加上標點符號與換行。
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        question,
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav",
        ),
    ],
)

print(f"Q:{question}")
print(f"A:{response.text}")
# Python Flask Web
import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from google import genai
from google.genai import types
from configparser import ConfigParser
import time

# Config Parser
config = ConfigParser()
config.read("config.ini")

client = genai.Client(
    api_key=config["Gemini"]["API_KEY"],
)

UPLOAD_FOLDER = "static/data"
ALLOWED_EXTENSIONS = set(
    ["mp4", "mov", "avi", "webm", "wmv", "3gp", "flv", "mpg", "mpeg"]
)
video_cloud_file = None
video_file_gemini = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    print("Submit!")
    if request.method == "POST":
        if "file1" not in request.files:
            print("No file part")
            return render_template("index.html")
        file = request.files["file1"]
        if file.filename == "":
            print("No selected file")
            return render_template("index.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            print(filename)
            upload_to_gemini(filename)
            result = "檔案已上傳成功! 並提供給Gemini處理完畢. 可以開始問問題囉!"
        return render_template(
            "index.html",
            prediction=result,
            filename=filename,
        )
    else:
        return render_template("index.html", prediction="Method not allowed")


@app.route("/call_gemini", methods=["POST"])
def call_gemini():
    if request.method == "POST":
        print("POST!")
        data = request.form
        print(data["message"])
        prompt = data["message"]
        global video_cloud_file
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, video_cloud_file],
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                system_instruction="請用繁體中文回答以下問題。",
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    )
                ]
            ),
        )
        print(response)
        return response.text


def upload_to_gemini(filename):
    print(f"Uploading file...")
    global video_cloud_file
    video_cloud_file = client.files.upload(file=f"static/data/{filename}")
    print(f"Completed upload")
    while video_cloud_file.state.name == "PROCESSING":
        print(f"{video_cloud_file.state.name} Waiting for video to be processed.")
        time.sleep(1)
        video_cloud_file = client.files.get(name=video_cloud_file.name)
    if video_cloud_file.state.name == "FAILED":
        raise ValueError(video_cloud_file.state.name)
    print(f"Video processing complete: " + video_cloud_file.uri)

if __name__ == "__main__":
    app.run()
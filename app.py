# Python Flask Web
import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
import PIL.Image
import json

UPLOAD_FOLDER = "static/images"
ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png", "bmp", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

filename = ""


@app.route("/")
def home():
    # delete the previous image incase it exists
    if os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], "gemini_output.jpg")):
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], "gemini_output.jpg"))
    global filename
    filename = "images/question-man.jpg"
    return render_template(
        "index.html", user_image=url_for("static", filename=filename)
    )


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
            global filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            print(filename)
        return render_template(
            "index.html",
            user_image=url_for("static", filename="images/" + filename),
        )
    else:
        return render_template("index.html", prediction="Method not allowed")


@app.route("/call_data", methods=["POST"])
def call_data():
    if request.method == "POST":
        print("POST!")
        data = request.form
        print(data["message"])
        return_object = call_gemini(data["message"])
        return json.dumps(return_object)


def call_gemini(user_input):
    text_input = (user_input,)
    gemini_input = [text_input]
    client = genai.Client(
        api_key=config["Gemini"]["API_KEY"],
    )
    global filename
    if filename != "images/question-man.jpg":
        image = PIL.Image.open("static/images/" + filename)
        gemini_input.append(image)

    try:
        response = client.models.generate_content(
            # model="gemini-2.0-flash-exp-image-generation",
            # model="gemini-2.5-flash-image-preview",
            model="gemini-3-pro-image-preview",
            contents=gemini_input,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"],
            ),
        )
    except Exception as e:
        print(e)
        return ["text", e["error"]["message"]]

    if response.candidates is None:
        return [["text", "我不想做這個～～"]]
    elif len(response.candidates) == 0:
        return [["text", "我並不想幫你做這個～～"]]
    elif response.candidates[0].content is None:
        return [["text", "我真的沒有想幫你做這個～～"]]
    elif response.candidates[0].content.parts is None:
        return [["text", "我真的真的不想這麼做～～"]]
    else:
        return_object = []
        for part in response.candidates[0].content.parts:
            print("part:", part)
            if part.text is not None:
                print(part.text)
                return_object.append(["text", part.text])
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save("static/images/gemini_output.jpg")
                filename = "gemini_output.jpg"
                return_object.append(["image", filename])
        return return_object


if __name__ == "__main__":
    app.run()
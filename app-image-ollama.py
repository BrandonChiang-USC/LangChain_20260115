import ollama
import base64
from IPython.display import Image, display

def image_to_base64(image_path):
    """Converts an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Example usage:
image_file_path = "cat.jpg"  # Replace with your image path
base64_image = image_to_base64(image_file_path)

question = "請列出這張圖片中的文字，使用繁體中文作答"

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", 
         "content": question, 
         "images": [base64_image]}
    ],
)
print(response["message"]["content"])
# Display the image in the notebook
display(Image(filename=image_file_path))
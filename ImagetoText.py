import cv2
import pytesseract
import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

# Set the API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # You can also use a .env file

# Put in the file location of the picture
image = cv2.imread('GPT/TTI/image.png')
# Convert the image to text
text = pytesseract.image_to_string(image)


with open('GPT/TTI/text.txt', 'w') as file:
    file.write(text)
# Creates a Document instance of your text
loader = TextLoader("GPT/TTI/text.txt")
# Stores your text within a Vector Index
index = VectorstoreIndexCreator().from_loaders(
    [loader])

x = str(index.query("Please write the following in comprehensive notes in bullet point form for a beginner to understand while keeping the important parts.", llm=ChatOpenAI()))
print(x)

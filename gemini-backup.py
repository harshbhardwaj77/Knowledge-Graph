import os
from dotenv import load_dotenv
# pip install python-dotenv

import google.generativeai as genai
# pip install -q -U google-generativeai

load_dotenv()

# https://ai.google.dev/tutorials/python_quickstart

model = genai.GenerativeModel("gemini-pro")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def prompt():
    input = "Tell me more about Borg of google"
    response = model.generate_content(input)
    print(response.text)
    return response


if __name__ == "__main__":
    prompt()




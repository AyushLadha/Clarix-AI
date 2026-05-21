import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# loading the .env file
load_dotenv()

# loading the api from the environment
api_key = os.getenv("Gemini_API_Key")

# create AI model object
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key = api_key)

# Send message and get response
message = [
    SystemMessage(content="You are a helpful data analyts assistant."),
    HumanMessage(content="In one sentence, what is the most important thing in automating data analysis?")
]

response = llm.invoke(message)

# print the response
print("AI says:", response.content[0]['text'])
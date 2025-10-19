from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


temp = 0.8
model = ChatOpenAI(model = "gpt-4", temperature = temp, max_completion_tokens = 50)
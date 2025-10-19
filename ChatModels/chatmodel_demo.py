from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4", temperature = 0.8, max_completion_tokens = 50)

result = model.invoke(
    "tell me a philosophical telugu shayari on heart"
)


print(result.content)
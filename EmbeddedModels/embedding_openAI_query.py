from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)
#dimensions represent the contextual maning captured in an embedding

result = embeddings.embed_query("Thsi is the information enquiry about embeddings")

print(str(result))
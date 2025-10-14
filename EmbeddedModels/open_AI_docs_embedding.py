from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)
#dimensions represent the contextual maning captured in an embedding

documents = ["The api_key client option must be set",
             "by passing api_key to the client",
             "by setting the OPENAI_API_KEY environment variable"
             "Ignore these example texts and focus on the code"]

result = embeddings.embed_documents(documents)

print(str(result))
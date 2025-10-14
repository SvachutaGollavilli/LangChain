from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions= 300)

documents = [
    "'Dancing With the Stars' Dedication Night: See the Full List of Songs and Who They're Honoring",
    "The upcoming installment of the popular Netflix series centers around the love story between Benedict Bridgerton and Sophie Baek.",
    "Trending Lay\'s Rebrands Because Customers Apparently Didn\’t Know Chips Were Made With \‘Real Potatoes",
    "A new report by McKinsey in conjunction with Telemundo has found that the young, digitally fluent Latino sports fan base",
    "The running back has fans feeling good about the New York Giants for the first time in a long time"
]

query = "tell me about chips and potatoes"

doc_embedddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embedddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(documents[index])
print("similarity score is ", score)
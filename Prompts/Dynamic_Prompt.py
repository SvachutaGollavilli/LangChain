from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from models import model
from langchain_core.prompts import PromptTemplate

load_dotenv()

paper_input = st.selectbox("select research papers", ["Attention is all you need", "Word2Vec", "GPT-3:Language models are few shot learners", "BERT:Pretraining of Deep Bidirectional Transformers", "Diffusion models beat GAN's on Image Synthesis" ])

style_input = st.selectbox("select style", ["Beginner Friendly", "Code-Oriented", "Technical", "Mathmatical"])

length_input = st.selectbox("Select Explanation length", ["Short(1-2 Paragraphs)", "Long(3-4 Paragraphs)", "Detail(5-6 Paragraphs)"])


template = PromptTemplate(
    template = "Please summarize the research paper titled \"{paper_input}\" with the following specifications:\nExplanation Style: {style_input}  \nExplanation Length: {length_input}  \n1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the çpaper.  \n   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  \n2. Analogies:  \n   - Use relatable analogies to simplify complex ideas.  \nIf certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.  \nEnsure the summary is clear, accurate, and aligned with the provided style and length.\n"
)

input_variables = ['paper_input', 'style_input', 'length_input']

validate_template = True

prompt = template.invoke({
    'paper_input': paper_input, 'style_input': style_input, 'length_input': length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
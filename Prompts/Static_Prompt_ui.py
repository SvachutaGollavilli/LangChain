from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from models import model


load_dotenv()

st.header("Research Tool")

user_input = st.text_input("enter the paper name")

if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
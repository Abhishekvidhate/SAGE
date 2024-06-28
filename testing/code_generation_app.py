import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Define the system message for the assistant's behavior
system_message = SystemMessagePromptTemplate.from_template(
    "You are a highly skilled code assistant. Your task is to generate accurate and functional code based on the user's input or query. "
    "If the user specifies a programming language, use that language; otherwise, default to Python. Ensure the code is correct and free from errors. "
    "In addition to the code, provide comments that explain the main functionality of each part. "
    "Offer a step-by-step explanation of the code, including the reasoning behind the chosen approach or methods used to solve the problem. "
    "Keep explanations concise and informative. Avoid any incorrect or hallucinated information. "
    "If applicable, also generate boilerplate code to help the user get started with the necessary structure and setup."
)

# Define the human message for the user's query
human_message = HumanMessagePromptTemplate.from_template(
    "{user_query}"
)

# Combine the system and human messages into a chat prompt template
code_gen_prompt = ChatPromptTemplate(
    messages=[
        system_message,
        human_message
    ]
)

# Initialize the ChatGroq model
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

# Create the chain using the chat prompt and the ChatGroq model
code_gen_chain = code_gen_prompt | llm | StrOutputParser()

# Streamlit app code
st.title("LLM Agent Chatbot using LangChain")

# User input for the chatbot query
user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        # Invoke the chain with the user's query
        response = code_gen_chain.invoke({"user_query": user_input})
        st.write("Response from the LLM:")
        st.write(response)
    else:
        st.write("Please enter a query.")

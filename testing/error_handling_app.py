import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Get the Groq API key from environment variables
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

# Define system and human messages for error handling
system_message_start = SystemMessagePromptTemplate.from_template(
    "You are an expert in diagnosing and resolving code errors. Your task is to analyze the provided code snippet and error message, "
    "identify the error, and suggest a resolution following best practices."
)

human_message_intro = HumanMessagePromptTemplate.from_template(
    "A user has provided a code snippet along with an error message. Your task is to analyze the provided code, identify the error, "
    "and suggest a resolution following best practices."
    "\n\nCode Snippet:\n{code_snippet}\n\nError Message:\n{error_message}"
)

system_message_end = SystemMessagePromptTemplate.from_template(
    "Provide your detailed analysis, suggested fixes, and corrected code below:"
)

# Define best practices for error handling and resolution
best_practices_message = SystemMessagePromptTemplate.from_template(
    "Consider the following best practices in your analysis and resolution:"
    "\n\n1. Catch Specific Exceptions: Replace any generic exceptions with more specific ones."
    "\n\n2. Provide Meaningful Error Messages: Ensure error messages are clear and descriptive."
    "\n\n3. Graceful Degradation: Handle errors gracefully without crashing the application."
    "\n\n4. Log Errors: Add logging statements to capture error details and context."
    "\n\n5. Validate Input: Implement input validation checks to prevent errors from invalid data."
    "\n\n6. Fail Fast: Detect and handle errors as soon as they occur."
)

# Combine the system and human messages into a chat prompt template for error handling
error_handling_prompt = ChatPromptTemplate(
    messages=[
        system_message_start,
        human_message_intro,
        best_practices_message,
        system_message_end
    ]
)

error_handling_chain = error_handling_prompt | llm | StrOutputParser()

# Streamlit app layout
st.title("Code Error Handling Assistant")

st.write("This app helps you diagnose and resolve code errors by analyzing provided code snippets and error messages.")

code_snippet = st.text_area("Enter your code snippet here:")
error_message = st.text_area("Enter the error message here:")

if st.button("Analyze Error"):
    if code_snippet and error_message:
        response = error_handling_chain.invoke({'code_snippet': code_snippet, 'error_message': error_message})
        st.subheader("Analysis and Resolution:")
        st.write(response)
    else:
        st.error("Please provide both a code snippet and an error message.")

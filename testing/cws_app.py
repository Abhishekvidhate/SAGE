import json
from typing import Any
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

chat_model = ChatGroq(model="llama3-8b-8192", temperature=0)


def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        return f"Failed to retrieve the webpage: {e}"


def web_search(query: str, num_results: int):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


get_links: Runnable[Any, Any] = (
        RunnablePassthrough()
        | RunnableLambda(
    lambda x: [
        {"url": url, "question": x["question"]}
        for url in web_search(query=x["question"], num_results=RESULTS_PER_QUESTION)
    ]
)
)

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{agent_prompt}"),
        (
            "user",
            "Write 3 search queries to find relevant code snippets or documentation for the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

AUTO_AGENT_INSTRUCTIONS = """
This task involves finding relevant code snippets and documentation for a given programming query. The search is conducted by a specific agent, determined by the programming topic.

examples:
task: "How to implement a binary search in Python?"
response: 
{
    "agent": "🖥️ Code Search Agent",
    "agent_role_prompt: "You are a seasoned software development assistant AI. Your primary goal is to compose comprehensive, insightful, and methodically arranged search queries to find relevant code snippets and documentation."
}
task: "Python requests library documentation"
response: 
{ 
    "agent":  "📘 Documentation Search Agent",
    "agent_role_prompt": "You are an experienced documentation search assistant AI. Your main objective is to produce comprehensive and insightful search queries to find relevant documentation for the specified programming topic."
}
"""

CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=AUTO_AGENT_INSTRUCTIONS), ("user", "task: {task}")]
)

SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
If the question cannot be answered using the text, simply summarize the text. Include all factual information, numbers, stats etc if available."""
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

scrape_and_summarize: Runnable[Any, Any] = (
        RunnableParallel(
            {
                "question": lambda x: x["question"],
                "text": lambda x: scrape_text(x["url"])[:10000],
                "url": lambda x: x["url"],
            }
        )
        | RunnableParallel(
    {
        "summary": SUMMARY_PROMPT | chat_model | StrOutputParser(),
        "url": lambda x: x["url"],
    }
)
        | RunnableLambda(lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")
)

multi_search = get_links | scrape_and_summarize.map() | (lambda x: "\n".join(x))

search_query = SEARCH_PROMPT | chat_model | StrOutputParser()
choose_agent = (
        CHOOSE_AGENT_PROMPT | chat_model | StrOutputParser()
)

get_search_queries = (
        RunnablePassthrough().assign(
            agent_prompt=RunnableParallel({"task": lambda x: x})
                         | choose_agent
                         | (lambda x: x.get("agent_role_prompt"))
        )
        | RunnableLambda(lambda x: {"agent_prompt": x["agent_prompt"], "question": x["task"]})
        | search_query
)

chain = (
        get_search_queries
        | (lambda x: [{"question": q} for q in x])
        | multi_search.map()
        | (lambda x: "\n\n".join(x))
)

# Streamlit app
st.title("Code and Documentation Search")

user_query = st.text_input("Enter your programming query:")

if st.button("Search"):
    if user_query:
        result = chain.invoke({"task": user_query})
        st.text_area("Search Results", result, height=400)
    else:
        st.error("Please enter a query.")

# # Streamlit app
# st.title("Code and Documentation Search")
#
# user_query = st.text_input("Enter your programming query:")
#
# if st.button("Search"):
#     if user_query:
#         # Debug input and intermediate steps
#         st.write("Running chain with input:", {"task": user_query})
#         intermediate_input = {"task": user_query}
#         st.write("Intermediate input to chain:", intermediate_input)
#
#         result = chain.invoke(intermediate_input)
#
#         # Debug result
#         st.write("Intermediate result:", result)
#         st.text_area("Search Results", result, height=400)
#     else:
#         st.error("Please enter a query.")

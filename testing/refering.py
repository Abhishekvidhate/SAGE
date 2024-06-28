from langchain_core.output_parsers.json import JsonOutputParser, OutputParserException
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
from typing import Any
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

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


def clean_json_response(response: str):
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        return json_str
    except Exception as e:
        raise OutputParserException("Failed to clean JSON response", llm_output=response) from e


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
    [("system", AUTO_AGENT_INSTRUCTIONS), ("user", "task: {task}")]
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
        "summary": SUMMARY_PROMPT | chat_model | JsonOutputParser(),
        "url": lambda x: x["url"],
    }
)
        | RunnableLambda(lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']['text']}")
)


def choose_agent(task: str):
    agent_response = chat_model.invoke(CHOOSE_AGENT_PROMPT.format_prompt(task=task))
    agent_response_cleaned = clean_json_response(agent_response.content)
    agent_data = JsonOutputParser().parse_result(agent_response_cleaned)
    return agent_data


def search_query(task: str, agent_prompt: str):
    search_prompt = SEARCH_PROMPT.format_prompt(agent_prompt=agent_prompt, question=task)
    search_response = chat_model.invoke(search_prompt)
    search_response_cleaned = clean_json_response(search_response.content)
    queries = JsonOutputParser().parse_result(search_response_cleaned)
    return queries


def main():
    st.title("AI-Powered Code Search")

    user_query = st.text_input("Enter your programming question:")
    if st.button("Search"):
        try:
            chain = (
                    RunnablePassthrough()
                    | RunnableLambda(lambda x: choose_agent(x["task"]))
                    | RunnableParallel(
                {
                    "task": lambda x: x["task"],
                    "queries": RunnableLambda(lambda x: search_query(x["task"], x["agent_role_prompt"])),
                }
            )
                    | get_links
                    | scrape_and_summarize
            )

            result = chain.invoke({"task": user_query})
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
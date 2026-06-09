import os
from typing import List
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
# import tavily
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """ Schema for Source used by the agent """

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """ Schema for Agent Response used by the agent """

    answer:str = Field(description="The Agent's answer to the user's question")
    sources: List[Source] = Field(default_factory=list,description="The list of sources used by the agent to answer the user's question")


# Tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def searchWeb(query: str) -> str:
#     """ Searches the web for information based on the query 

    
#     Args:
#         query: str
#     Returns:
#         The Search Results
#     """

#     print(f"Searching the web for {query}")
#     return Tavily.search(query=query)

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # llm = ChatOllama(model="gpt-oss:latest", temperature=0)
    llm = ChatOpenAI(model="gpt-5.5", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)
    result = agent.invoke({"messages": HumanMessage(content="List 3 AI Engineer Job Postings in India Remote with their approximate Salary in rupees from LinkedIn and list their details")})
    print(result)

if __name__ == "__main__":
    main()

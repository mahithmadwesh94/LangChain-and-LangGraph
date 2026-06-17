import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

index = os.environ.get("INDEX_NAME")

print("Initialising....")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-5.5", temperature=0)
vector_store = PineconeVectorStore(embedding=embeddings, index_name=index)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
Provide a detailed Answer:
""")

def formatDocs(docs):
    """Format the documents into a string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """Retrieval chain without LangChain Expressions Language
    Manual retrieves documents, formats them, and generates a response

    LIMITATIONS:
    - Cannot use LangChain Expressions Language
    - No streaming support
    - No async support
    - Harder to compose with other chains
    - More verbose and error prone
    """

    docs = retriever.invoke(query)
    context = formatDocs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

def retrieval_chain_with_lcel():
    """Retrieval chain with LangChain Expressions Language
    Automatic retrieves documents, formats them, and generates a response
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | formatDocs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )


    return retrieval_chain


if __name__ == "__main__":
    print("Running...")
    query = "What is pinecone in machine learning?"
    print("="*100)
    print("RAG pipeline without langchain expressions language")
    print("="*100)
    result = retrieval_chain_without_lcel(query)
    print("="*100)
    print()
    print()
    print(result)

    print("="*100)
    print("RAG pipeline with langchain expressions language")
    print("="*100)
    chain_with_lcel = retrieval_chain_with_lcel()
    result = chain_with_lcel.invoke({"question": query})
    print(result)
    print("="*100)

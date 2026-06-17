from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import os



import os
load_dotenv()

index = os.environ.get("INDEX_NAME")



if __name__ == "__main__":
    print("Ingestion...")
    loader = TextLoader("mediumBlog.txt",encoding="utf-8")
    documents = loader.load()
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Embedding...")
    # vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index)
    print("Finish")



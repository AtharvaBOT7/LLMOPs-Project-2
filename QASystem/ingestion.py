from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import os
import json
import boto3
import sys

# Bedrock Client Initialization

bedrock = boto3.client(service_name='bedrock_runtime')
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    bedrock_client=bedrock
)

def data_ingestion():
    loader = PyPDFDirectoryLoader(r"./data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    return docs

def get_vector_store(doc):
    vector_store_faiss = FAISS.from_documents(doc, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs)
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from QASystem.ingestion import get_vector_store, data_ingestion

import os
import json
import boto3
import sys

prompt_template = """ 
You are a highly intelligent question answering bot. Use the following context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer in as easy manner as you can. 

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables=["context", "question"]
)

def get_claude_llm():
    bedrock_llm = Bedrock(
        model_id="anthropic.claude-opus-4-5-20251101-v1:0",
        region_name="us-east-2",
        max_tokens = 512
    )
    return bedrock_llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm, 
        chain_type="stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]

if __name__ == "__main__":
    docs = data_ingestion()
    query = "What is Mamba?"

    vectorstore_faiss = get_vector_store(docs)

    llm = get_response_llm()

    get_response_llm(llm, vectorstore_faiss, query)
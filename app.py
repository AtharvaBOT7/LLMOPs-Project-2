import os
import json
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock  # Fix: langchain_community.llms

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from QASystem.ingestion import get_vector_store, data_ingestion
from QASystem.retrievalgeneration import get_llm, get_response_llm

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"  # Match your region from retrievalgeneration.py
)

def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and AWSBedrock")
    
    user_question = st.text_input("Ask a question from the pdf files")
    
    with st.sidebar:
        st.title("Update or Create the Vector Store")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done! Vector store created.")
                
        if st.button("Ask Question"):  # Better button name
            if user_question:  # Check if user entered a question
                with st.spinner("Generating answer..."):
                    faiss_index = FAISS.load_local(
                        "faiss_index",
                        bedrock_embeddings,
                        allow_dangerous_deserialization=True
                    )
                    llm = get_llm()
                    
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.write(response)
                    st.success("Done")
            else:
                st.warning("Please enter a question first!")
                
if __name__ == "__main__":
    main()
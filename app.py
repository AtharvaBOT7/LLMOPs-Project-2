import os
import json
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
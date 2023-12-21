import os
import pathlib
import textwrap

import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from langchain.vectorstores import DocArrayInMemorySearch
#from langchain.vectorstores.docarray.in_memory import DocArrayInMemorySearch

from langchain.schema.runnable import RunnableMap

import numpy as np

# Set up dotenv (.env holds environment variable, GOOGLE_API_KEY)
load_dotenv(override=True)

# Set up API auth
gemini_api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=gemini_api_key)

######### NO LANGCHAIN - SIMPLE API TEST #########
#Set up the model

# ## Configs
# generation_config = {
#   "temperature": 0.9,
#   "top_p": 1,
#   "top_k": 1,
#   "max_output_tokens": 2048,
# }
#
# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   }
# ]

## Declare model
#model = genai.GenerativeModel(model_name="gemini-pro")
# model = genai.GenerativeModel(model_name="gemini-pro",
#                               generation_config=generation_config,
#                               safety_settings=safety_settings)

# ## Output model info
# models = [m for m in genai.list_models()]
# print(models)

# ## Setup prompt

# Basic
# # Example 1
# prompt_parts = [
#   "What is the meaning of life?",
# ]
# response = model.generate_content(prompt_parts)
# print(response.text)

# # Example 2:
# # generate text
# prompt = 'Who are you and what can you do?'
# response = model.generate_content(prompt)
# # Markdown(response.text) # jupyter
# print(response.text)
######### END OF NO LANGCHAIN - SIMPLE API TEST #########



######### LANGCHAIN STUFF #########

# What is temperature? -> Randomness,
# ex: if 0.0, output is purely deterministic
#  aka output is exactly the same everytime

# Basic LLM Chain
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)
# Example 1
result = llm.invoke("What is a LLM?")
#Markdown(result.content)
print(result.content)

# Example 2
for chunk in llm.stream("Write a haiku about LLMs."):
    print(chunk.content)
    print("---")



# Basic Multi-chain
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

topic_str = "programming"
result = chain.invoke({"topic": topic_str})
print(result)



# # Basic RAG Search (ISSUES)
# docarray NOT INSTALLED PROPERLY EVEN THOUGH
# pip -q install "langchain[docarray]"
# AND
# pip install docarray (requirement already satisfied)
# DOES NOT WORK

# output_parser = StrOutputParser()
# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
# # Create embeddings model
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vector = embeddings.embed_query("hello, world!")
# print(vector[:5])
#
# vectors = embeddings.embed_documents(
#     [
#         "Today is Monday",
#         "Today is Tuesday",
#         "Today is April Fools day",
#     ]
# )
# print(f"({len(vectors)}, {len(vectors[0])})")

# #### broken 1
# documents = TextLoader("./minidocs-input.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# print(docs)
#
# db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# query = "What is Gemini"
# docs = db.similarity_search(query)
# print(docs[0].page_content)
# #### end of broken 1


# # Define your mini docs
# mini_docs = [
#     "Gemini Pro is a Large Language Model made by GoogleDeepMind",
#     "Gemini can refer to a star sign or a series of language models",
#     "Language models are trained by predicting the next token",
#     "LLMs can perform various NLP tasks and text generation"
# ]
#
# # Initialize the DocArrayInMemorySearch
# vectorstore = DocArrayInMemorySearch.from_texts(mini_docs, embedding=embeddings)
#
# # Create a retriever based on the vectorstore
# retriever = vectorstore.as_retriever()


# vectorstore = DocArrayInMemorySearch.from_texts(
#     # mini docs for embedding
#     ["Gemini Pro is a Large Language Model was made by GoogleDeepMind",
#      "Gemini can be either a star sign or a name of a series of language models",
#      "A Language model is trained by predicting the next token",
#      "LLMs can easily do a variety of NLP tasks as well as text generation"],
#
#     embedding=embeddings # passing in the embedder model
# )

#
# retriever = vectorstore.as_retriever()
#
# retriever.get_relevant_documents("what is Gemini?")
#
# retriever.get_relevant_documents("what is gemini pro?")
#
# template = """Answer the question a a full sentence, based only on the following context:
# {context}
#
# Return you answer in three back ticks
#
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
#
# retriever.get_relevant_documents("Who made Gemini Pro?")
#
# chain = RunnableMap({
#     "context": lambda x: retriever.get_relevant_documents(x["question"]),
#     "question": lambda x: x["question"]
# }) | prompt | model | output_parser
#
# chain.invoke({"question": "Who made Gemini Pro?"})



import os
import pathlib
import textwrap

import google.generativeai as genai
from IPython.display import display # for jupyter
from IPython.display import Markdown # for jupyter

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
#from langchain.vectorstores import DocArrayInMemorySearch
#from langchain.vectorstores.docarray.in_memory import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap
from langchain.vectorstores import Qdrant

from langchain_experimental.pal_chain import PALChain
from langchain.chains.llm import LLMChain

import requests
#from IPython.display import Image
from PIL import Image

# Set up dotenv (.env holds environment variable, GOOGLE_API_KEY)
load_dotenv(override=True)

# Set up API auth
gemini_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_api_key)

######### NO LANGCHAIN - SIMPLE API TEST #########
# # Set up the model

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

# # Basic LLM Chain
# llm = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.7)
# # Example 1
# result = llm.invoke("What is a LLM?")
# #Markdown(result.content)
# print(result.content)
#
# # Example 2
# for chunk in llm.stream("Write a haiku about LLMs."):
#     print(chunk.content)
#     print("---")



# # Basic Multi-chain
# model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.7)
# prompt = ChatPromptTemplate.from_template(
#     "tell me a short joke about {topic}"
# )
#
# output_parser = StrOutputParser()
#
# chain = prompt | model | output_parser
#
# topic_str = "programming"
# result = chain.invoke({"topic": topic_str})
# print(result)



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
#
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


# # Basic RAG Search (WORKING)
# output_parser = StrOutputParser()
# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
# # Create embeddings model
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# loader = TextLoader("./minidocs-input.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# # in-memory
# qdrant = Qdrant.from_documents(
#     docs,
#     embeddings,
#     location=":memory:",  # Local mode with in-memory storage only
#     collection_name="my_documents",
# )
# # query = "What is Gemini"
# # found_docs = qdrant.similarity_search(query)
# # print(found_docs[0].page_content)
#
# retriever = qdrant.as_retriever()
# result1 = retriever.get_relevant_documents("what is Gemini?")
# print(result1)
# result2 = retriever.get_relevant_documents("what is gemini pro?")
# print(result2)
# template = """Answer the question a a full sentence, based only on the following context:
# {context}
#
# Return you answer in three back ticks
#
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
#
# result3 = retriever.get_relevant_documents("Who made Gemini Pro?")
# print(result3)
# chain = RunnableMap({
#     "context": lambda x: retriever.get_relevant_documents(x["question"]),
#     "question": lambda x: x["question"]
# }) | prompt | model | output_parser
#
# result4 = chain.invoke({"question": "Who made Gemini Pro?"})
# print(result4)

# ## PAL Chain Example
# def main():
#     # PAL Chain
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
#     pal_chain = PALChain.from_math_prompt(model, verbose=True)
#
#     # question1 = "The cafeteria had 23 apples. \
#     # If they used 20 for lunch and bought 6 more,\
#     # how many apples do they have?"
#     # res1 = pal_chain.invoke(question1)
#     # print(res1)
#
#     question2 = "If you wake up at 7:00 a.m. and it takes you 1 hour and 30 minutes to get ready \
#      and walk to school, at what time will you get to school?"
#     res2 = pal_chain.invoke(question2)
#     print(res2)
#
# if __name__ == '__main__':
#     main()

## Multi Modal
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1200px-The_Earth_seen_from_Apollo_17.jpg"
content = requests.get(image_url).content
#Image(content,width=300) # IPython jupyter image viewer
# Save the image to a file
with open("image.jpg", "wb") as f:
    f.write(content)
# Open the image with the default image viewer
Image.open("image.jpg").show()

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image and who lives there?",
        },  # You can optionally provide text parts
        {
            "type": "image_url",
            "image_url": image_url
         },
    ]
)

print(llm.invoke([message]))

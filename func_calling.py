import os
import re
import langchain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Set up dotenv (.env holds environment variable, GOOGLE_API_KEY)
load_dotenv(override=True)

# Set up API auth
gemini_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_api_key)

def read_email(sender):
    print(f"Reading email from {sender}")


def play_song(song_name, artist):
    print(f"Playing {song_name} by {artist}")


def add_to_calendar(title, date, time):
    print(f"Adding event '{title}' on {date} at {time}")

def create_note(title, date):
    print(f"Creating note: '{title}' on {date}")


functions = {
    "read_email": read_email,
    "play_song": play_song,
    "add_calendar_event": add_to_calendar
}

prompt_template = "Execute the {func_name} function"
prompt = PromptTemplate(
    input_variables=["func_name"], template=prompt_template
)

model = ChatGoogleGenerativeAI(model="gemini-pro",
                               temperature=0)
llm = langchain.chains.LLMChain(llm=model, prompt=prompt)

def process_command(transcript):
    # Classify intent & extract entities
    parsed = llm.parse_obj(transcript)
    #                        , entity_mapping={
    #     "song_name": "str",
    #     "artist": "str",
    #     "title": "str",
    #     "date": "str",
    #     "time": "str"
    # })

    intent = parsed["intent"]
    entities = parsed["entities"]

    # Map to function
    func = functions[intent]

    # Execute function
    func(**entities)

    response = f"Successfully processed intent: {intent}"

    return response


command = "play blinded by the light song by bruce springsteen"
response = process_command(command)
print(response)

# # model-independent classify and QA chains
# # classify_chain = langchain.chains.ClassifierChain()
# # qa_chain = langchain.chains.QuestionAnsweringChain()
#
#
# # function wrapper
# @langchain.llm_method
# def execute(func_name, **kwargs):
#     function = functions[func_name]
#     return function(**kwargs)
#
#
# def process(input_text):
#     # Classify intent with classifier chain
#     intent = classify_chain.run(input_text)
#
#     if intent == "QA":
#         # QA use case
#         answer = qa_chain.run(input_text)
#         return answer
#
#     # Function execution use case
#     func_name, kwargs = extract_func_args(input_text)
#
#     return execute(func_name=func_name, **kwargs)


# def extract_func_args(text):
#     # Regex to extract function name
#     func_regex = r"(?:run|execute|call|start|begin) (?P<func_name>\w+)"
#
#     # Extract function name
#     match = re.search(func_regex, text)
#     if not match:
#         return None, None
#
#     func_name = match.group("func_name")
#
#     # Regex to extract key=value arguments
#     args_regex = r"--(?P<key>\w+) (?P<value>\w+|'.+?'|\d+)"
#
#     # Find all matches of arguments
#     args_matches = re.findall(args_regex, text)
#
#     # Construct kwargs dict
#     kwargs = {}
#     for match in args_matches:
#         kwargs[match[0]] = match[1]
#
#     return func_name, kwargs
#
# text = "execute create_note --title 'Meeting Notes' --date 2023-01-01"
#
# func_name, kwargs = extract_func_args(text)
#
# print(func_name) # 'create_note'
# print(kwargs) # {'title': 'Meeting Notes', 'date': '2023-01-01'}
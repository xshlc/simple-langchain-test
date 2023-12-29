import os
import google.generativeai as genai
from dotenv import load_dotenv
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part
from dummy_functions import stop_transcribing, pause_transcribing, resume_transcription


# Set up dotenv (.env holds environment variable, GOOGLE_API_KEY)
load_dotenv(override=True)

# Set up API auth
gemini_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(model_name="gemini-pro")

stop_transcribing_func = generative_models.FunctionDeclaration(
    name="stop_transcribing",
    description="Stops lecture transcription upon request",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)

transcription_tool = generative_models.Tool(
  function_declarations=[stop_transcribing_func]
)
def model_response(text):
    response = model.generate_content(
        text,
        generation_config={"temperature": 0},
        tools=[transcription_tool],
    )
    if response.candidates:
        function_args = response.candidates[0].content.parts[0].function_call
        return function_args
    else:
        return "No response generated."
    return response

# Test function
print(model_response("I want to stop transcribing the lecture."))


# stop_transcribing_func = generative_models.FunctionDeclaration(
#     name="stop_transcribing",
#     description="Stops lecture transcription upon request",
#     parameters={
#         "type": object,
#         "properties": {
#             "time": {
#                 "type": "string",
#                 "description": "Time to stop transcribing"
#             }
#         },
#         "required": [
#             "time"
#         ]
#     },
# )
#
# transcription_tool = generative_models.Tool(
#   function_declarations=[stop_transcribing_func]
# )
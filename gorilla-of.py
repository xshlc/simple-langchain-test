# Import Chat completion template and set-up variables
import openai
import urllib.parse
import json


openai.api_key = "EMPTY" # Key is ignored and does not matter
openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"
# Alternate mirrors
# openai.api_base = "http://34.132.127.197:8000/v1"

# Report issues
def raise_issue(e, model, prompt):
    issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
    issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
    issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
    print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")

# Query Gorilla server
def get_gorilla_response(prompt="I would like to translate from English to French.", model="gorilla-7b-hf-v1"):
  try:
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
  except Exception as e:
    raise_issue(e, model, prompt)


# Gorilla `gorilla-mpt-7b-hf-v1` with code snippets
# Translation
prompt = "I would like to translate 'I feel very good today.' from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))
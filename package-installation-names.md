
# Gemini package
pip install -q -U google-generativeai

# Jupyter (optional?)
pip install jupyter

# Environment variable for API key
pip install python-dotenv

# Langchain for Gemini
pip -q install langchain_experimental langchain_core
pip -q install google-generativeai==0.3.1
pip -q install google-ai-generativelanguage==0.4.0
pip -q install langchain-google-genai
pip -q install "langchain[docarray]"

# IPython
pip install ipython

# DocArrayInMemorySearch problem potential fixes
pip install pydantic==1.10.8 
pip install docarray==0.32.1 
* Sources for this suggestion:
 * https://github.com/langchain-ai/langchain/issues/14585 
 * https://github.com/langchain-ai/langchain/issues/12916
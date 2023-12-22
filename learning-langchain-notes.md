**pycharm** 

Check whether virtual environment (venv) is currently activated
On Windows: 
`echo $env:VIRTUAL_ENV  `
if this prints out the path of venv, then venv is active.
if this prints nothing, then deactivated state. 

Activate virtual environment (venv)
`.\venv\Scripts\activate`

**langchain** 

docs: https://python.langchain.com/docs/get_started/introduction 

langchain tutorial series on youtube
- video link: https://youtu.be/_v_fgW2SkkQ?si=IuHLL88XNG5P-pku 
- github: https://github.com/gkamradt/langchain-tutorials 
- quickstart: https://github.com/gkamradt/langchain-tutorials/blob/main/getting_started/Quickstart%20Guide.ipynb 
# Gemini Setup

Google's Google Colab
Jupyter Notebook: https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/python_quickstart.ipynb 

Google Gemini packages
`pip install jupyter`
`pip install -q -U google-generativeai`


In the `pip install` command you provided:

```bash
pip install -q -U google-generativeai
OR
pip install -q google-generativeai
```

The tags `-q` and `-U` have specific meanings:

1. **`-q` (or `--quiet`):**
   - **Meaning:** This flag is used to make the installation process less verbose by suppressing output except for errors and essential information.
   - **Usage:** If you include the `-q` flag, the installation process will be quieter, and you won't see as much output in the terminal. It's useful when you want to minimize the amount of information displayed during the installation.

2. **`-U` (or `--upgrade`):**
   - **Meaning:** This flag is used to upgrade an already installed package to the latest version. If the specified package is already installed, `pip install -U` ensures that it is upgraded to the newest available version.
   - **Usage:** If you include the `-U` flag, `pip` will check if the package is already installed and, if so, upgrade it to the latest version. If the package is not installed, it will be installed as usual.

So, in your specific example:

```bash
pip install -q -U google-generativeai
```

This command is installing the package `google-generativeai` with the following behaviors:

- `-q`: It's installing quietly, meaning it suppresses most of the output.
- `-U`: It's upgrading the package if it's already installed or installing it if it's not present.

This is a common practice to ensure you have the latest version of a package while keeping the installation process less verbose.


API Keys and .env file 
```bash
pip install python-dotenv
```

# Update pip

On Windows:
```powershell
python.exe -m pip install --upgrade pip
```

# Langchain Setup for Google Gemini

Tutorial video link: https://youtu.be/G3-YOEVg-xc?si=K90wo_EesdzXGe-a 

```powershell
pip -q install langchain_experimental langchain_core
pip -q install google-generativeai==0.3.1
pip -q install google-ai-generativelanguage==0.4.0
pip -q install langchain-google-genai
pip -q install "langchain[docarray]"
```


```powershell
pip show langchain langchain-core
```



# ERROR 1
## 1. Cannot install `pip -q install langchain_experimental langchain_core`

### Error message: 
```
pip -q install langchain_experimental langchain_core
  error: subprocess-exited-with-error

  × Building wheel for multidict (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [74 lines of output]
      *********************
      * Accelerated build *
      *********************
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-cpython-312
      creating build\lib.win-amd64-cpython-312\multidict
      copying multidict\_abc.py -> build\lib.win-amd64-cpython-312\multidict
      copying multidict\_compat.py -> build\lib.win-amd64-cpython-312\multidict
      copying multidict\_multidict_base.py -> build\lib.win-amd64-cpython-312\multidict
      copying multidict\_multidict_py.py -> build\lib.win-amd64-cpython-312\multidict
      copying multidict\__init__.py -> build\lib.win-amd64-cpython-312\multidict
      running egg_info
      writing multidict.egg-info\PKG-INFO
      writing dependency_links to multidict.egg-info\dependency_links.txt
      writing top-level names to multidict.egg-info\top_level.txt
      reading manifest file 'multidict.egg-info\SOURCES.txt'
      reading manifest template 'MANIFEST.in'
      warning: no previously-included files matching '*.pyc' found anywhere in distribution
      warning: no previously-included files found matching 'multidict\_multidict.html'
      warning: no previously-included files found matching 'multidict\*.so'
      warning: no previously-included files found matching 'multidict\*.pyd'
      warning: no previously-included files found matching 'multidict\*.pyd'
      no previously-included directories found matching 'docs\_build'
      adding license file 'LICENSE'
      writing manifest file 'multidict.egg-info\SOURCES.txt'
      C:\Users\Lena\AppData\Local\Temp\pip-build-env-lz60bzqm\overlay\Lib\site-packages\setuptools\command\build_py.py:207: _Warning: Package 'multidict._multilib' is absent fro
m the `packages` configuration.
      !!
     
              ********************************************************************************
              ############################
              # Package would be ignored #
              ############################
              Python recognizes 'multidict._multilib' as an importable package[^1],
              but it is absent from setuptools' `packages` configuration.
     
              This leads to an ambiguous overall configuration. If you want to distribute this
              package, please make sure that 'multidict._multilib' is explicitly added
              to the `packages` configuration field.
     
              Alternatively, you can also rely on setuptools' discovery methods
              (for example by using `find_namespace_packages(...)`/`find_namespace:`
              instead of `find_packages(...)`/`find:`).
     
              You can read more about "package discovery" on setuptools documentation page:
     
              - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
     
              If you don't want 'multidict._multilib' to be distributed and are
              already explicitly excluding 'multidict._multilib' via
              `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,
              you can try to use `exclude_package_data`, or `include-package-data=False` in
              combination with a more fine grained `package-data` configuration.
     
              You can read more about "package data files" on setuptools documentation page:
     
              - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
     
     
              [^1]: For Python, any directory (with suitable naming) can be imported,
                    even if it does not contain any `.py` files.
                    On the other hand, currently there is no concept of package data
                    directory, all directories are treated like packages.
              ********************************************************************************
     
      !!
        check.warn(importable)
      copying multidict\__init__.pyi -> build\lib.win-amd64-cpython-312\multidict
      copying multidict\py.typed -> build\lib.win-amd64-cpython-312\multidict
      running build_ext
      building 'multidict._multidict' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for multidict
ERROR: Could not build wheels for multidict, which is required to install pyproject.toml-based projects
```

### ERROR 1 FIX: 

https://github.com/imartinez/privateGPT/issues/1083 
- Fix highlight: 
  "Downloading [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) and installing C++ Build Tools and adding cl.exe to path fixed the issue."

#### Path for `cl.exe`: 
https://learn.microsoft.com/en-us/answers/questions/1398254/installed-visual-studio-build-tools-but-cannot-fin 

Finding the path:
https://github.com/carla-simulator/carla/issues/3235

### **Fix that actually worked:**
Install Visual Studio Build Tools: (!!!)
- C++ build tools
- .NET build tools
- Node.js build tools
https://youtu.be/hgNxAxyncdc?si=RcZIGbaeKwwcr7er

After fix:
`pip -q install langchain_experimental langchain_core` installs properly.

---


## Outputting Langchain Installation Info

### Command:
```powershell
pip show langchain langchain-core
```
#### Output:
```powershell
Name: langchain
Version: 0.0.352                                                                                                                                       
Summary: Building applications with LLMs through composability                                                                                         
Home-page: https://github.com/langchain-ai/langchain                                                                                                   
Author:                                                                                                                                                
Author-email:                                                                                                                                          
License: MIT                                                                                                                                           
Location: E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages                                                                        
Requires: aiohttp, dataclasses-json, jsonpatch, langchain-community, langchain-core, langsmith, numpy, pydantic, PyYAML, requests, SQLAlchemy, tenacity
Required-by: langchain-experimental                                                                                                                    
---
Name: langchain-core                                          
Version: 0.1.3                                                
Summary: Building applications with LLMs through composability
Home-page: https://github.com/langchain-ai/langchain          
Author:                                                       
Author-email:
License: MIT
Location: E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages
Requires: anyio, jsonpatch, langsmith, packaging, pydantic, PyYAML, requests, tenacity
Required-by: langchain, langchain-community, langchain-experimental, langchain-google-genai

```

## Setup API auth

Create a `.env` file
- Get API key from *Google AI Studio* (can also test prompts in playground)
```env
GOOGLE_API_KEY="##############"
```

Basic Python code setup
```python
import os  
import google.generativeai as genai  
from dotenv import load_dotenv

# Set up dotenv (.env holds environment variable, GOOGLE_API_KEY)  
load_dotenv(override=True)  
  
# Set up API auth  
gemini_api_key = os.getenv('GOOGLE_API_KEY')  
  
genai.configure(api_key=gemini_api_key)
```

## Create model
```python
#Set up the model  
  
## Configs  
generation_config = {  
  "temperature": 0.9,  
  "top_p": 1,  
  "top_k": 1,  
  "max_output_tokens": 2048,  
}  
  
safety_settings = [  
  {  
    "category": "HARM_CATEGORY_HARASSMENT",  
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  
  },  
  {  
    "category": "HARM_CATEGORY_HATE_SPEECH",  
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  
  },  
  {  
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  
  },  
  {  
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",  
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"  
  }  
]  
  
## Declare model  
#model = genai.GenerativeModel(model_name="gemini-pro")  
model = genai.GenerativeModel(model_name="gemini-pro",  
                              generation_config=generation_config,  
                              safety_settings=safety_settings)
```

## Outputting Gemini Model Info
```python
models = [m for m in genai.list_models()]  
print(models)
```

Run output:
```powershell
[Model(name='models/chat-bison-001',
      base_model_id='',
      version='001',
      display_name='PaLM 2 Chat (Legacy)',
      description='A legacy text-only model optimized for chat conversations',
      input_token_limit=4096,
      output_token_limit=1024,
      supported_generation_methods=['generateMessage', 'countMessageTokens'],
      temperature=0.25,
      top_p=0.95,
      top_k=40), Model(name='models/text-bison-001',
      base_model_id='',
      version='001',
      display_name='PaLM 2 (Legacy)',
      description='A legacy model that understands text and generates text as an output',
      input_token_limit=8196,
      output_token_limit=1024,
      supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'],
      temperature=0.7,
      top_p=0.95,
      top_k=40), Model(name='models/embedding-gecko-001',
      base_model_id='',
      version='001',
      display_name='Embedding Gecko',
      description='Obtain a distributed representation of a text.',
      input_token_limit=1024,
      output_token_limit=1,
      supported_generation_methods=['embedText', 'countTextTokens'],
      temperature=None,
      top_p=None,
      top_k=None), Model(name='models/gemini-pro',
      base_model_id='',
      version='001',
      display_name='Gemini Pro',
      description='The best model for scaling across a wide range of tasks',
      input_token_limit=30720,
      output_token_limit=2048,
      supported_generation_methods=['generateContent', 'countTokens'],
      temperature=0.9,
      top_p=1.0,
      top_k=1), Model(name='models/gemini-pro-vision',
      base_model_id='',
      version='001',
      display_name='Gemini Pro Vision',
      description='The best image understanding model to handle a broad range of applications',
      input_token_limit=12288,
      output_token_limit=4096,
      supported_generation_methods=['generateContent', 'countTokens'],
      temperature=0.4,
      top_p=1.0,
      top_k=32), Model(name='models/embedding-001',
      base_model_id='',
      version='001',
      display_name='Embedding 001',
      description='Obtain a distributed representation of a text.',
      input_token_limit=2048,
      output_token_limit=1,
      supported_generation_methods=['embedContent', 'countTextTokens'],
      temperature=None,
      top_p=None,
      top_k=None), Model(name='models/aqa',
      base_model_id='',
      version='001',
      display_name='Model that performs Attributed Question Answering.',
      description=('Model trained to return answers to questions that are grounded in provided '
                   'sources, along with estimating answerable probability.'),
      input_token_limit=7168,
      output_token_limit=1024,
      supported_generation_methods=['generateAnswer'],
      temperature=0.2,
      top_p=1.0,
      top_k=40)]
```


## If Jupyter, then also install IPython
```
pip install ipython
```

## Prompting stuff

### Basic examples

no langchain example 1:
```python
## Setup prompt  
prompt_parts = [  
  "What is the meaning of life?",  
]  
response = model.generate_content(prompt_parts)  
print(response.text)
```


no langchain example 2:
```python
from IPython.display import display  
from IPython.display import Markdown
## Setup prompt  
prompt = 'Who are you and what can you do?'  
  
response = model.generate_content(prompt)  

# Generate text if using jupyter
#Markdown(response.text)

print(response.text)
```

### **Langchain** examples

**Imports for langchain**
```python
from langchain_core.messages import HumanMessage  
from langchain_google_genai import ChatGoogleGenerativeAI
```


#### Basic LLM Chain
example 1:
```python
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)  
  
  
result = llm.invoke("What is a LLM?")  
# Generate text if using jupyter
#Markdown(result.content)  
print(result.content)
```

example 2:
```python
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
for chunk in llm.stream("Write a haiku about LLMs."):  
    print(chunk.content)  
    print("---")
```

#### Basic Multi-chain

**More imports for langchain**
```python
from langchain_google_genai import ChatGoogleGenerativeAI # This one should exist already
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
```

Basic python
```python
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
```

```powershell
Why did the programmer quit his job?

Because he didn't get arrays.
```


Jupyter version
```python
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "machine learning"})
```

#### Complex Chaining

##### Mini RAG (BROKEN)
Even more imports 
```python
from langchain_google_genai import ChatGoogleGenerativeAI # should already exist
from langchain_google_genai import GoogleGenerativeAIEmbeddings # should already exist
from langchain.vectorstores import DocArrayInMemorySearch
```

```python
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = DocArrayInMemorySearch.from_texts(
    # mini docs for embedding
    ["Gemini Pro is a Large Language Model was made by GoogleDeepMind",
     "Gemini can be either a star sign or a name of a series of language models",
     "A Language model is trained by predicting the next token",
     "LLMs can easily do a variety of NLP tasks as well as text generation"],

    embedding=embeddings # passing in the embedder model
)

retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("what is Gemini?")

retriever.get_relevant_documents("what is gemini pro?")

template = """Answer the question a a full sentence, based only on the following context:
{context}

Return you answer in three back ticks

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
```

More imports
```python
from langchain.schema.runnable import RunnableMap
```

```python
retriever.get_relevant_documents("Who made Gemini Pro?")

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "Who made Gemini Pro?"})
```

Broken code
```python
from langchain_google_genai import ChatGoogleGenerativeAI # should already exist
from langchain_google_genai import GoogleGenerativeAIEmbeddings # should already exist
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap

output_parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = DocArrayInMemorySearch.from_texts(
    # mini docs for embedding
    ["Gemini Pro is a Large Language Model was made by GoogleDeepMind",
     "Gemini can be either a star sign or a name of a series of language models",
     "A Language model is trained by predicting the next token",
     "LLMs can easily do a variety of NLP tasks as well as text generation"],

    embedding=embeddings # passing in the embedder model
)

retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("what is Gemini?")

retriever.get_relevant_documents("what is gemini pro?")

template = """Answer the question a a full sentence, based only on the following context:
{context}

Return you answer in three back ticks

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retriever.get_relevant_documents("Who made Gemini Pro?")

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "Who made Gemini Pro?"})
```

More RAG search examples: https://python.langchain.com/docs/expression_language/cookbook/retrieval 



##### Example: Google Generative AI Embeddings

https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai

Given examples in Langchain docs:
```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
  
vector = embeddings.embed_query("hello, world!")  
print(vector[:5])
```
```powershell
[0.05636945, 0.0048285457, -0.0762591, -0.023642512, 0.05329321]
```

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
  
vectors = embeddings.embed_documents(  
    [  
        "Today is Monday",  
        "Today is Tuesday",  
        "Today is April Fools day",  
    ]  
)  
print(f"({len(vectors)}, {len(vectors[0])})")
```
```
```powershell
(3, 768)
```

This works, meaning that `embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")` works.

The issue is `DocArrayInMemorySearch`. 

# ERROR 2

The following does not work:
```python
DocArrayInMemorySearch.from_texts(mini_docs, embedding=embeddings)
DocArrayInMemorySearch.from_documents(docs, embeddings)
```

Error message in console:
```
Traceback (most recent call last):
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\langchain_community\vectorstores\docarray\base.py", line 19, in _check_docarray_import
    import docarray
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\__init__.py", line 5, in <module>
    from docarray.array import DocList, DocVec
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\array\__init__.py", line 1, in <module>
    from docarray.array.any_array import AnyDocArray
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\array\any_array.py", line 22, in <module>
    from docarray.base_doc import BaseDoc
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\base_doc\__init__.py", line 1, in <module>
    from docarray.base_doc.any_doc import AnyDoc
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\base_doc\any_doc.py", line 3, in <module>
    from .doc import BaseDoc
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\docarray\base_doc\doc.py", line 22, in <module>
    from pydantic.main import ROOT_KEY
ImportError: cannot import name 'ROOT_KEY' from 'pydantic.main' (E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\pydantic\main.py)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\main.py", line 162, in <module>
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\langchain_core\vectorstores.py", line 510, in from_documents
    return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\langchain_community\vectorstores\docarray\in_memory.py", line 68, in from_texts
    store = cls.from_params(embedding, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\langchain_community\vectorstores\docarray\in_memory.py", line 39, in from_params
    _check_docarray_import()
  File "E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages\langchain_community\vectorstores\docarray\base.py", line 29, in _check_docarray_import
    raise ImportError(
ImportError: Could not import docarray python package. Please install it with `pip install "langchain[docarray]"`.
```

`\venv\Lib\site-packages\langchain_community\vectorstores\docarray\in_memory.py`
```python
@classmethod  
def from_texts(  
    cls,  
    texts: List[str],  
    embedding: Embeddings,  
    metadatas: Optional[List[Dict[Any, Any]]] = None,  
    **kwargs: Any,  
) -> DocArrayInMemorySearch:  
    """Create an DocArrayInMemorySearch store and insert data.  
  
    Args:        texts (List[str]): Text data.        embedding (Embeddings): Embedding function.        metadatas (Optional[List[Dict[Any, Any]]]): Metadata for each text            if it exists. Defaults to None.        metric (str): metric for exact nearest-neighbor search.            Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".            Defaults to "cosine_sim".  
    Returns:        DocArrayInMemorySearch Vector Store    """    store = cls.from_params(embedding, **kwargs)  
    store.add_texts(texts=texts, metadatas=metadatas)  # ERROR HERE !!!!!!!!!!
    return store 
```

`\venv\Lib\site-packages\langchain_community\vectorstores\docarray\in_memory.py`
```python
@classmethod  
def from_params(  
    cls,  
    embedding: Embeddings,  
    metric: Literal[  
        "cosine_sim", "euclidian_dist", "sgeuclidean_dist"  
    ] = "cosine_sim",  
    **kwargs: Any,  
) -> DocArrayInMemorySearch:  
    """Initialize DocArrayInMemorySearch store.  
  
    Args:        embedding (Embeddings): Embedding function.        metric (str): metric for exact nearest-neighbor search.            Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".            Defaults to "cosine_sim".        **kwargs: Other keyword arguments to be passed to the get_doc_cls method.    """    _check_docarray_import()  # ERROR HERE !!!!!!!!!!
    from docarray.index import InMemoryExactNNIndex  
  
    doc_cls = cls._get_doc_cls(space=metric, **kwargs)  
    doc_index = InMemoryExactNNIndex[doc_cls]()  # type: ignore  
    return cls(doc_index, embedding)
```

`\venv\Lib\site-packages\langchain_community\vectorstores\docarray\base.py`
```python
def _check_docarray_import() -> None:  
    try:  
        import docarray  # ERROR HERE !!!!!!!!!!
  
        da_version = docarray.__version__.split(".")  
        if int(da_version[0]) == 0 and int(da_version[1]) <= 31:  
            raise ImportError(  
                f"To use the DocArrayHnswSearch VectorStore the docarray "  
                f"version >=0.32.0 is expected, received: {docarray.__version__}."  
                f"To upgrade, please run: `pip install -U docarray`."            )  
    except ImportError:  
        raise ImportError(  
            "Could not import docarray python package. "  
            'Please install it with `pip install "langchain[docarray]"`.'        )
```

```
docarray NOT INSTALLED PROPERLY EVEN THOUGH  
pip -q install "langchain[docarray]"  
AND  
pip install docarray (requirement already satisfied)  

DOES NOT WORK...
import docarray  
print(docarray.__version__)

pip show docarray

Name: docarray
Version: 0.32.1
Summary: The data structure for multimodal data
Home-page: https://docarray.jina.ai/
Author: DocArray
Author-email:
License: Apache 2.0
Location: E:\CodeWorkstation\carl-ai\carl-langchain-test\venv\Lib\site-packages
Requires: numpy, orjson, pydantic, rich, types-requests, typing-inspect
Required-by:
```
Since `pip show docarray` outputs successfully, assuming that `docarray` is installed properly. 

similar error: 
https://github.com/langchain-ai/langchain/issues/14585 
https://github.com/langchain-ai/langchain/issues/12916
- suggested fix: 
	- `pip install pydantic==1.10.8 `
	- `pip install docarray==0.32.1 `

# Similarity search
https://www.pinecone.io/learn/what-is-similarity-search/

### Qdrant
https://python.langchain.com/docs/integrations/vectorstores/qdrant
#### Local Mode
##### In-memory
- quick experiments
- in-memory -> the data "gets lost when the client is destroyed - usually at the end of your script/notebook"
```python
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)
```

##### On-disk
- "without using the Qdrant server, may also store your vectors on disk so they’re persisted between runs"
```python
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path="/tmp/local_qdrant",
    collection_name="my_documents",
)
```

#### PAL Chaining
```python
from langchain_experimental.pal_chain import PALChain  
from langchain.chains.llm import LLMChain
```

```python
## Unsafe code
# Does not work
# Continue to next section of notes
model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)

pal_chain = PALChain.from_math_prompt(model, verbose=True)

question = "The cafeteria had 23 apples. \
If they used 20 for lunch and bought 6 more,\
how many apples do they have?"

pal_chain.invoke(question)

question = "If you wake up at 7:00 a.m. and it takes you 1 hour and 30 minutes to get ready \
 and walk to school, at what time will you get to school?"

pal_chain.invoke(question)
```

---
##### PAL Chain infinite looping error

See: https://docs.python.org/3/library/multiprocessing.html

Explanation: 
The error you are encountering is related to the multiprocessing module in Python. It's likely caused by the fact that the PALChain uses a Python REPL (Read-Eval-Print Loop) for its processing, and the multiprocessing module is not well-suited for use in environments where REPL is involved.

Here are a few suggestions to address the issue:

1. **Avoid using multiprocessing in REPL environments:**
   Since PALChain seems to utilize a Python REPL, it's generally a good practice to avoid using multiprocessing in REPL environments. The multiprocessing module is designed for use in standalone scripts and may not work well in interactive environments.

2. **Use if __name__ == '__main__': in your main script:**
   If you are executing the PALChain code directly in a script, make sure to wrap your code in an `if __name__ == '__main__':` block. This is a common practice when using multiprocessing in Python scripts.

   ```python
   if __name__ == '__main__':
       # Your PALChain code here
   ```

   This is necessary to prevent multiprocessing-related issues in Windows environments.

3. **Check PALChain Documentation or Community:**
   If the issue persists, check the documentation of PALChain or seek help from the community or developers. There may be specific guidelines or considerations for using PALChain in certain environments.

4. **Consider Using a Different Approach:**
   If the PALChain doesn't work well in your specific environment, consider using an alternative approach or library for your task. There might be other tools or libraries that better suit your needs.

Remember to adapt these suggestions based on the specifics of your use case and the PALChain library you are working with. If you have access to the PALChain library's documentation or community, it's recommended to check for any guidelines or recommendations regarding the use of multiprocessing or REPL environments.

---

### Working PAL Chain

Example 1:
```python
def main():  
    # PAL Chain  
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)  
    pal_chain = PALChain.from_math_prompt(model, verbose=True)  
    
    question2 = "If you wake up at 7:00 a.m. and it takes you 1 hour and 30 minutes to get ready \  
     and walk to school, at what time will you get to school?"    res2 = pal_chain.invoke(question2)  
    print(res2)
  
if __name__ == '__main__':  
    main()
```

Example 2:
```python
def main():  
    # PAL Chain  
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)  
    pal_chain = PALChain.from_math_prompt(model, verbose=True)  
    
    question1 = "The cafeteria had 23 apples. \
    If they used 20 for lunch and bought 6 more,\
    how many apples do they have?"  
    res1 = pal_chain.invoke(question1)
    print(res1)
  
if __name__ == '__main__':  
    main()
```

#### Multi Modal
```python
import requests
from IPython.display import Image

from langchain_core.messages import HumanMessage # should already exist
from langchain_google_genai import ChatGoogleGenerativeAI # should already exist
```

```python
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1200px-The_Earth_seen_from_Apollo_17.jpg"
content = requests.get(image_url).content
#Image(content,width=300) # jupyter IPython display image

# Save the image to a file  
with open("image.jpg", "wb") as f:  
    f.write(content)  
# Open the image with the default image viewer  
Image.open("image.jpg").show()
```

```python
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

# example
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

llm.invoke([message])
```

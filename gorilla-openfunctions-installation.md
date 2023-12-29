# Gorilla Local Installation

Using text file to bulk install
source: https://github.com/ShishirPatil/gorilla/blob/main/requirements.txt

gorilla-requirements.txt
contents:
```
openai  
anthropic  
tree_sitter  
tenacity==8.2.2  
pydantic==1.10.7  
rank-bm25==0.2.2
```

Error:
```
Collecting tokenizers>=0.13.0 (from anthropic->-r gorilla-requirements.txt (line 2))
  Downloading tokenizers-0.15.0.tar.gz (318 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 318.5/318.5 kB 4.0 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
     
      Cargo, the Rust package manager, is not installed or is not on PATH.
      This package requires Rust and Cargo to compile extensions. Install it through
      the system's package manager or via https://rustup.rs/
     
      Checking for Rust toolchain....
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

```
1. Fix by installing Rust at https://rustup.rs/
 * Windows: Add Rust as an environment variable to **both** User and System PATH 
 * Where to find? C:/Users/[username]/.cargo/bin
 * Rust installer should auto-add environment variable to User PATH
 * Add System path manually
 * Resolves pip install errors
2. Restart PyCharm
3. Try `pip install -r gorilla-requirements.txt` again

## Local OpenFunctions
pip install torch
pip install transformers
Note: torch does not work with Python 3.12. Downgrade Python to 3.11.

# AskCode

AskCode helps you get familiar with a codebase or project documents. Add Python,
PDF, Markdown, or text files, then ask questions in plain language. Answers are
based on matching project files and include file and line references when
available.

The app has a Streamlit interface, a FastAPI service, and a ChromaDB index. Use
Ollama to keep model processing local, or configure Gemini as a hosted option.

## Run locally with Ollama

Make sure the Ollama app is open, then install the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

In the project folder, set up the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export LLM_PROVIDER=ollama
```

Start the API in one terminal:

```bash
uvicorn api:app --reload --port 8000
```

Start the interface in a second terminal:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), add files from the
sidebar, and ask about your project. The API docs are at
[http://localhost:8000/docs](http://localhost:8000/docs).

## Use Gemini instead

Set these variables in the terminal before starting the API:

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-key"
```

Keep the key private and out of source control. Gemini sends the text needed
for search and answers to Google's hosted service.

## Run the tests

With the virtual environment active:

```bash
python -m unittest discover -s tests -v
```

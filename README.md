# RAG AI Project

This project is a local Retrieval-Augmented Generation (RAG) application for asking questions about documents stored in the project folder. It supports both a browser interface and a command-line interface, and it answers questions using only the text retrieved from the source documents.

## What the app does

- Reads PDF and TXT files from a local data folder
- Splits large documents into overlapping text chunks
- Creates embeddings using Google Gemini
- Stores vectors locally in ChromaDB
- Retrieves the most relevant passages for a question
- Uses Gemini to generate a grounded answer with source references

## Included project files

- `rag_bot.py` � core indexing and query logic
- `streamlit_app.py` � Streamlit web app
- `data/` � sample knowledge base documents
- `scripts/generate_sample_pdf.py` � script to generate the sample PDF
- `tests/test_rag_bot.py` � unit tests for chunking and citation behavior
- `requirements.txt` � Python dependencies

## Project structure

```text
RAG_AI_project/
+-- data/
�   +-- community_energy.pdf
�   +-- community_health.txt
�   +-- secure_software.txt
�   +-- urban_heat.txt
+-- scripts/
�   +-- generate_sample_pdf.py
+-- tests/
�   +-- test_rag_bot.py
+-- .chroma/
+-- .env
+-- .env.example
+-- rag_bot.py
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
+-- .gitignore
```

## Tech stack

- Python 3.11+
- Streamlit
- Google Gen AI SDK
- ChromaDB
- pypdf
- python-dotenv
- ReportLab

## Setup

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Create a `.env` file and add your Gemini API key:

```powershell
Copy-Item .env.example .env
```

Then set:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run the app

### Streamlit web app

```powershell
python -m streamlit run streamlit_app.py
```

Use the sidebar to index the documents in `data/`, then ask questions in the chat area.

### Command line

```powershell
python rag_bot.py ingest
python rag_bot.py ask "How can a city improve resilience during outages?"
```

You can also start a chat session:

```powershell
python rag_bot.py chat
```

## How the app works

1. It reads all supported files from the `data/` directory.
2. It splits the text into overlapping chunks.
3. It embeds each chunk with Gemini.
4. It stores those vectors in a local ChromaDB collection.
5. For each user question, it retrieves the most relevant chunks.
6. It sends the retrieved passages plus the question to Gemini and returns a grounded answer with source references.

## Notes

- The project is designed for local document search and Q&A.
- It requires an active Gemini API key to generate embeddings and answers.
- The vector database is stored in `.chroma/` and is refreshed when you re-index the documents.
- This app is intended for local use and is not a production multi-user system.

## Testing

Run the project tests with:

```powershell
python -m unittest discover -s tests -v
```

These tests validate chunking, citation formatting, and embedding behavior without requiring the Gemini API.

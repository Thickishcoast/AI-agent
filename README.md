# Beginner Gemma Chat

Beginner Gemma Chat is a command-line AI tutor that runs locally with
[Ollama](https://ollama.com/). It streams Gemma's response, keeps the chat
history between sessions, and stores useful facts about the user as semantic
long-term memories.

## Current setup

- **Interface:** interactive terminal chat
- **Chat model:** `gemma4:e4b`
- **Embedding model:** `embeddinggemma`
- **Python:** tested with Python 3.14.4
- **Python Ollama client:** 0.6.2
- **Local Ollama version:** 0.32.0

### Project files

| Path | Purpose |
| --- | --- |
| `main.py` | Runs the chat loop and coordinates history and memory. |
| `llm.py` | Streams thinking and response text from the Gemma model. |
| `semantic_memory.py` | Extracts, embeds, saves, and retrieves long-term memories. |
| `store_history.py` | Loads and saves conversation history. |
| `chat history/conversation_history.json` | Generated chat history. |
| `Memory/knowledge.json` | Generated memories and embeddings. |

The application sends relevant stored memories to the model before the latest
user message. After each successful response, it saves the conversation and
asks the model to extract any useful long-term facts from the user's message.

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- Enough disk space and memory for `gemma4:e4b`

## Setup

Clone the repository and enter the project directory:

```powershell
git clone "https://github.com/Thickishcoast/AI-agent"
cd "AI-agent"
```


Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure Ollama is running, then download the required models:

```powershell
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

You can confirm that they are available with:

```powershell
ollama list
```

## Run the chatbot

Run the application:

```powershell
python main.py
```

Type a message and press Enter. Enter `exit` or `quit` to stop the program.


## Data and privacy

The model runs through the local Ollama service. Conversation history and
long-term memory are stored locally in JSON files.


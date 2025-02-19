# Repository Agent

A tool for indexing and querying multiple Git repositories using vector search and large language models.

## Features

- Index multiple Git repositories
- Track changes and automatically reindex when needed
- Search across repositories with semantic understanding
- Support for both OpenAI and Ollama as AI providers
- Web interface for easy interaction
- Real-time progress tracking
- File-level granular search

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:

For OpenAI:
```env
OPENAI_API_KEY=your_api_key_here
REPOS_PATH=/path/to/your/repositories
AI_PROVIDER=openai
```

For Ollama:
```env
REPOS_PATH=/path/to/your/repositories
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Optional, defaults to this value
OLLAMA_MODEL=llama2  # Optional, defaults to llama2
```

## Usage

### Web Interface

Start the web server:
```bash
python web_app.py
```

Then open http://localhost:8000 in your browser.

### Command Line

Initialize and index repositories:

With OpenAI:
```bash
python repo_agent.py setup --repos-path /path/to/repos --provider openai --api-key your_api_key
```

With Ollama:
```bash
python repo_agent.py setup --repos-path /path/to/repos --provider ollama --ollama-model llama2
```

Ask questions:

With OpenAI:
```bash
python repo_agent.py ask "What does the login function do?" --repos-path /path/to/repos --provider openai --api-key your_api_key
```

With Ollama:
```bash
python repo_agent.py ask "What does the login function do?" --repos-path /path/to/repos --provider ollama --ollama-model llama2
```

## Supported AI Providers

### OpenAI
- Requires an API key
- Uses GPT-4 for chat and text-embedding-3-small for embeddings
- Generally provides higher quality results
- Costs money per API call

### Ollama
- Free and runs locally
- Supports various open-source models
- No API key required
- Performance depends on your hardware and the chosen model
- Requires Ollama to be installed and running locally

To use Ollama:
1. Install Ollama from https://ollama.ai
2. Start the Ollama service
3. Pull your desired model (e.g., `ollama pull llama2`)
4. Configure the application to use Ollama as the provider

## Configuration

All settings can be configured either through environment variables or command-line arguments:

| Setting | Environment Variable | Default Value | Description |
|---------|---------------------|---------------|-------------|
| AI Provider | `AI_PROVIDER` | `openai` | AI provider to use (`openai` or `ollama`) |
| OpenAI API Key | `OPENAI_API_KEY` | None | Required for OpenAI provider |
| Repositories Path | `REPOS_PATH` | `.` | Path to repositories directory |
| Ollama Base URL | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| Ollama Model | `OLLAMA_MODEL` | `llama2` | Model to use with Ollama |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
# Advanced AI Document Assistant

An AI-powered chatbot with RAG, web search, image generation, and code execution capabilities.

## Features
- 📄 Chat with PDF/TXT documents (RAG)
- 🔍 Real-time web search
- 🎨 AI image generation
- 💻 Python code execution
- 🤖 Powered by OpenRouter LLMs

## Setup

1. Clone this repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Create `.env` file with your API key:
```
   OPENROUTER_API_KEY=your-key-here
```
4. Run the app:
```bash
   streamlit run chatbot_advanced.py
```

## Usage

- Upload documents in the sidebar
- Enable/disable tools as needed
- Chat naturally - the AI will use appropriate tools

## Tools

- **Document Chat**: Ask questions about uploaded files
- **Web Search**: Get latest information from the web
- **Image Generation**: Create images from text descriptions
- **Code Execution**: Run Python code safely

## Deploy

Deploy for free on Streamlit Cloud - just connect your GitHub repo!
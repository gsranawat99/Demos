# Multi-Model RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot system that supports multiple language models (OpenAI, Claude, Groq) with a user-friendly Streamlit interface. This system provides detailed responses with source tracking and token usage monitoring.

## Features

- **Multi-Model Support**: 
  - OpenAI GPT models
  - Anthropic Claude
  - Groq models
- **Interactive Streamlit Interface**:
  - Model selection checkboxes
  - API key input fields
  - Real-time token usage display
  - Toggleable source information
  - Comparison mode for different models
- **Advanced RAG Capabilities**:
  - Document processing with dynamic chunk sizing
  - Source tracking and metadata
  - Token counting for multiple LLM types
  - Optimized retrieval using similarity threshold

## Prerequisites

- Python 3.9+
- API keys for desired models:
  - OpenAI API key
  - Anthropic API key (for Claude)
  - Groq API key

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements_streamlit.txt
```

3. Place your documents in the appropriate data folder

## Usage

1. Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

2. In the Streamlit interface:
   - Enter your API keys for the models you want to use
   - Select the models you want to compare
   - Enter your query
   - Toggle source visibility as needed
   - Monitor token usage in real-time

## Features in Detail

### Token Counting
- Accurate token counting for different model types
- Real-time usage monitoring
- Fallback counting methods when needed

### Source Tracking
- Track which parts of the source documents were used
- Toggle source visibility in the interface
- Detailed metadata for each response

### Model Comparison
- Compare responses from different models
- See how different models interpret the same query
- Evaluate token efficiency across models

## Project Structure

```
multi_rag/
├── dual_rag.py         # Core RAG implementation with multi-model support
├── streamlit_app.py    # Streamlit interface implementation
└── requirements_streamlit.txt  # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests. Some areas for potential contribution:
- Additional model support
- Enhanced token counting methods
- UI/UX improvements
- Documentation improvements

## Troubleshooting

Common issues and solutions:
- **Invalid API Keys**: Ensure you've entered valid API keys for each model you want to use
- **Token Counting Issues**: The system will fall back to approximate counting if exact counting isn't available
- **Model Availability**: If a model is unavailable, the system will indicate this in the interface

## License

MIT

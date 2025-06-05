import streamlit as st
import os
from typing import Dict, List
from dual_rag import DualRAGSystem, RAGConfig, LLMFactory
import tempfile
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Multi-Model RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = {
        'openai': [],
        'claude': [],
        'groq': []
    }
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'use_openai' not in st.session_state:
    st.session_state.use_openai = False
if 'use_claude' not in st.session_state:
    st.session_state.use_claude = False
if 'use_groq' not in st.session_state:
    st.session_state.use_groq = False
if 'enable_comparison' not in st.session_state:
    st.session_state.enable_comparison = False
if 'source_visibility' not in st.session_state:
    st.session_state.source_visibility = {}

def initialize_rag_system():
    """Initialize the RAG system with configuration."""
    config = RAGConfig()
    return DualRAGSystem(config)

def save_uploaded_file(uploaded_file):
    """Save uploaded PDF to a temporary file and return its path."""
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return temp_path
    return None

def toggle_source_visibility(key: str):
    """Toggle the visibility state of a source."""
    if key not in st.session_state.source_visibility:
        st.session_state.source_visibility[key] = False
    st.session_state.source_visibility[key] = not st.session_state.source_visibility[key]

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    
    # API Keys input with placeholders
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key", value="")
    claude_key = st.text_input("Anthropic (Claude) API Key", type="password", placeholder="Enter your Claude API key", value="")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key", value="")
    
    # Save API keys to environment temporarily for this session
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if claude_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    
    # Model selection
    st.subheader("Select Models")
    st.session_state.use_openai = st.checkbox("Use OpenAI", value=bool(openai_key), key="openai_checkbox")
    st.session_state.use_claude = st.checkbox("Use Claude", value=bool(claude_key), key="claude_checkbox")
    st.session_state.use_groq = st.checkbox("Use Groq", value=bool(groq_key), key="groq_checkbox")
    
    # Comparison mode
    st.subheader("Comparison Mode")
    st.session_state.enable_comparison = st.checkbox("Enable Answer Comparison", value=st.session_state.enable_comparison, key="comparison_checkbox")
    
    if st.session_state.enable_comparison:
        st.info("Comparison mode will show answers side by side in a table format")
    
    # PDF file upload
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        st.session_state.pdf_path = save_uploaded_file(uploaded_file)
        st.session_state.rag_system = initialize_rag_system()
        st.success(f"PDF uploaded: {uploaded_file.name}")

# Main chat interface
st.title("Multi-Model RAG Chat System")

def display_chat_messages(model_name: str):
    """Display chat messages for a specific model."""
    for idx, message in enumerate(st.session_state.messages[model_name]):
        with st.chat_message(message["role"]):
            # Display the message content
            st.write(message["content"])
            
            # Show tokens if available
            if "tokens" in message:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.caption(f"🔤 Input tokens: {message['tokens']['input']:,}")
                with col2:
                    st.caption(f"📤 Output tokens: {message['tokens']['output']:,}")
                st.caption(f"💫 Total tokens: {message['tokens']['input'] + message['tokens']['output']:,}")
            
            # Show sources button and information for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                source_key = f"source_btn_{model_name}_{idx}"
                button_label = "📚 Hide Sources" if st.session_state.source_visibility.get(source_key, False) else "📚 Show Sources"
                
                if st.button(button_label, key=source_key):
                    toggle_source_visibility(source_key)
                    st.rerun()
                
                # Display sources if visibility is true
                if st.session_state.source_visibility.get(source_key, False):
                    st.markdown("---")
                    st.markdown("#### Source Information:")
                    for source in message["sources"]:
                        st.markdown(f"📄 **{source['pdf_name']}** - Page {source['page_number']}")
                        with st.expander("View source content"):
                            st.markdown(source['content'])

def display_comparison(last_responses: Dict[str, Dict]):
    """Display comparison of model responses in a table format."""
    if not last_responses:
        return
    
    # Create comparison dataframe
    comparison_data = []
    for model, response in last_responses.items():
        total_tokens = response["input_tokens"] + response["output_tokens"]
        comparison_data.append({
            "Model": model.upper(),
            "Answer": response["content"],
            "Input Tokens": f"{response['input_tokens']:,}",
            "Output Tokens": f"{response['output_tokens']:,}",
            "Total Tokens": f"{total_tokens:,}",
        })
    
    df = pd.DataFrame(comparison_data)
    st.table(df)
    
    # Add toggleable source buttons for each model
    for model, response in last_responses.items():
        if "sources" in response:
            source_key = f"compare_source_btn_{model}"
            button_label = "📚 Hide Sources" if st.session_state.source_visibility.get(source_key, False) else "📚 Show Sources"
            
            if st.button(button_label, key=source_key):
                toggle_source_visibility(source_key)
                st.rerun()
            
            if st.session_state.source_visibility.get(source_key, False):
                st.markdown("---")
                st.markdown(f"#### Source Information for {model.upper()}:")
                for source in response["sources"]:
                    st.markdown(f"📄 **{source['pdf_name']}** - Page {source['page_number']}")
                    with st.expander("View source content"):
                        st.markdown(source['content'])

# Display format based on comparison mode
if st.session_state.get('enable_comparison', False):
    st.subheader("Model Comparison View")
    if st.session_state.messages['openai'] or st.session_state.messages['claude'] or st.session_state.messages['groq']:
        last_responses = {}
        for model in ['openai', 'claude', 'groq']:
            if st.session_state.messages[model]:
                last_msg = st.session_state.messages[model][-1]
                if last_msg["role"] == "assistant":
                    last_responses[model] = {
                        "content": last_msg["content"],
                        "input_tokens": last_msg["tokens"]["input"],
                        "output_tokens": last_msg["tokens"]["output"],
                        "sources": last_msg.get("sources", [])
                    }
        display_comparison(last_responses)
else:
    # Create columns based on selected models
    active_models = []
    if st.session_state.get('use_openai', False):
        active_models.append(('openai', "OpenAI Chat"))
    if st.session_state.get('use_claude', False):
        active_models.append(('claude', "Claude Chat"))
    if st.session_state.get('use_groq', False):
        active_models.append(('groq', "Groq Chat"))
    
    if active_models:
        cols = st.columns(len(active_models))
        for i, (model_key, model_name) in enumerate(active_models):
            with cols[i]:
                st.subheader(model_name)
                display_chat_messages(model_key)

# Query input
query = st.chat_input("Ask a question about the uploaded document")

if query:
    if not st.session_state.pdf_path:
        st.error("Please upload a PDF document first!")
    else:
        # Add user message to active chats
        active_models = []
        if st.session_state.use_openai and openai_key:
            active_models.append(('openai', LLMFactory.create_openai()))
        if st.session_state.use_claude and claude_key:
            active_models.append(('claude', LLMFactory.create_anthropic()))
        if st.session_state.use_groq and groq_key:
            active_models.append(('groq', LLMFactory.create_groq()))
        
        if not active_models:
            st.warning("Please select at least one model to use!")
            st.stop()
        
        # Add user message and get responses
        progress_text = st.empty()
        for model_key, _ in active_models:
            st.session_state.messages[model_key].append({"role": "user", "content": query})
        
        for i, (model_key, llm) in enumerate(active_models, 1):
            progress_text.text(f"Getting response from {model_key.upper()}... ({i}/{len(active_models)})")
            
            result = st.session_state.rag_system.process_query(st.session_state.pdf_path, query, llm)
            st.session_state.messages[model_key].append({
                "role": "assistant",
                "content": result["answer"],
                "tokens": {
                    "input": result["input_tokens"],
                    "output": result["output_tokens"]
                },
                "sources": result["sources"]
            })
        
        progress_text.empty()
        st.rerun()

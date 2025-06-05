from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
import tiktoken
import os

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    num_chunks: int = 5

@dataclass
class ChunkMetadata:
    """Metadata for each document chunk."""
    page_number: int
    pdf_name: str
    chunk_index: int

class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_openai() -> ChatOpenAI:
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    @staticmethod
    def create_anthropic() -> ChatAnthropic:
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.3,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    @staticmethod
    def create_groq() -> ChatGroq:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

class DualRAGSystem:
    """Main RAG system implementation."""
    
    def __init__(self, config: RAGConfig = RAGConfig()):
        """Initialize the RAG system with configuration."""
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    def process_document(self, pdf_path: str) -> Tuple[FAISS, int]:
        """Set up the RAG system with document processing and metadata."""
        # Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        document_chunks = splitter.split_documents(documents)
        
        # Add metadata to each chunk
        pdf_name = os.path.basename(pdf_path)
        for i, chunk in enumerate(document_chunks):
            chunk.metadata.update({
                'pdf_name': pdf_name,
                'chunk_index': i,
                'page_number': chunk.metadata.get('page', 0+1)  # PyPDFLoader already includes page numbers
            })
        
        # Create FAISS vector store from document chunks and embeddings
        vector_store = FAISS.from_documents(document_chunks, self.embeddings)
        
        return vector_store, self.config.chunk_size
    
    def retrieve(self, vector_store: FAISS, query: str) -> Dict[str, Any]:
        """Retrieve relevant chunks for the query with their metadata."""
        # Perform similarity search with metadata
        docs = vector_store.similarity_search(query, k=self.config.num_chunks)
        
        # Prepare the context and source information
        chunks_with_metadata = []
        for doc in docs:
            chunk_info = {
                'content': doc.page_content,
                'pdf_name': doc.metadata.get('pdf_name', 'Unknown'),
                'page_number': doc.metadata.get('page_number', 0),
                'chunk_index': doc.metadata.get('chunk_index', 0)
            }
            chunks_with_metadata.append(chunk_info)
        
        # Join the content with source information
        context_parts = []
        for chunk in chunks_with_metadata:
            context_parts.append(
                f"[Source: {chunk['pdf_name']}, Page: {chunk['page_number']}]\n{chunk['content']}"
            )
        
        return {
            'context': '\n\n'.join(context_parts),
            'sources': chunks_with_metadata
        }
    
    def generate_answer(self, context: str, query: str, llm: BaseLanguageModel) -> Dict[str, Any]:
        """Generate an answer using the specified LLM."""
        prompt = f"""Given the following context and query, provide a detailed answer. Use only the information from the context. Each context chunk includes its source information in square brackets.

Context:
{context}

Query:
{query}

Answer (please include relevant source page numbers in your response):"""
        
        try:
            response = llm.invoke(prompt)
            input_tokens = 0
            output_tokens = 0
            
            # Get usage statistics based on LLM type
            if isinstance(llm, ChatOpenAI):
                # For OpenAI, usage is directly available
                usage = getattr(response, 'usage', None)
                if usage:
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                else:
                    # Fallback to tiktoken counting for OpenAI models
                    encoding = tiktoken.encoding_for_model("gpt-4")
                    input_tokens = len(encoding.encode(prompt))
                    output_tokens = len(encoding.encode(response.content))
            
            elif isinstance(llm, (ChatGroq, ChatAnthropic)):
                # For Groq and Anthropic, check both dictionary and object formats
                usage = getattr(response, 'usage', {})
                if isinstance(usage, dict):
                    input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('output_tokens') or usage.get('completion_tokens', 0)
                else:
                    input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                
                # If no tokens found, estimate using character count (rough approximation)
                if input_tokens == 0:
                    input_tokens = len(prompt) // 4
                if output_tokens == 0:
                    output_tokens = len(response.content) // 4
            
            return {
                "content": response.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            return {
                "content": "Error generating answer",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
    
    def process_query(self, pdf_path: str, query: str, llm: BaseLanguageModel) -> Dict[str, Any]:
        """Complete RAG flow: document processing, retrieval, and answer generation."""
        # Step 1: Document processing
        vector_store, _ = self.process_document(pdf_path)
        
        # Step 2: Retrieval with metadata
        retrieval_results = self.retrieve(vector_store, query)
        
        # Step 3: Answer generation
        answer_data = self.generate_answer(retrieval_results['context'], query, llm)
        
        return {
            "answer": answer_data["content"],
            "context": retrieval_results['context'],
            "sources": retrieval_results['sources'],
            "input_tokens": answer_data["input_tokens"],
            "output_tokens": answer_data["output_tokens"]
        }

def main():
    """Main function to demonstrate the RAG system."""
    # Initialize configuration and RAG system
    config = RAGConfig()
    rag_system = DualRAGSystem(config)
    
    # Create LLM instances using the factory
    llm_openai = LLMFactory.create_openai()
    llm_groq = LLMFactory.create_groq()
    
    # Set up the query
    pdf_path = "data/attention1.pdf"
    query = "how many refrences are there in the document?"
    
    # Process query with OpenAI LLM
    result_openai = rag_system.process_query(pdf_path, query, llm_openai)
    print("\nOpenAI Result:")
    print("Answer:", result_openai["answer"])
    print("\nSources used:")
    for source in result_openai["sources"]:
        print(f"- {source['pdf_name']}, Page {source['page_number']}")
    
    # Process query with Groq LLM
    result_groq = rag_system.process_query(pdf_path, query, llm_groq)
    print("\nGroq Result:")
    print("Answer:", result_groq["answer"])
    print("\nSources used:")
    for source in result_groq["sources"]:
        print(f"- {source['pdf_name']}, Page {source['page_number']}")

if __name__ == "__main__":
    main()
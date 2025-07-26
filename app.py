import streamlit as st
import os
from typing import List, Dict, Any
import tempfile
from utils.pdf_processor import PDFProcessor
from utils.gemini_engine import GeminiEngine
from utils.simple_search import SimpleSearchEngine
from utils.simple_generator import SimpleResponseGenerator

# Configure page
st.set_page_config(
    page_title="Doc AI - RAG Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "gemini_engine" not in st.session_state:
    st.session_state.gemini_engine = None
if "search_engine" not in st.session_state:
    st.session_state.search_engine = SimpleSearchEngine()
if "response_generator" not in st.session_state:
    st.session_state.response_generator = SimpleResponseGenerator()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "use_gemini" not in st.session_state:
    st.session_state.use_gemini = True

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini engine (cached for performance)"""
    return GeminiEngine()

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files with enhanced accuracy"""
    pdf_processor = PDFProcessor(chunk_size=600, chunk_overlap=100)
    new_documents = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text from PDF
                text = pdf_processor.extract_text(tmp_file_path)
                if text.strip():
                    # Chunk the text
                    chunks = pdf_processor.chunk_text(text)
                    
                    # Add metadata to chunks
                    for i, chunk in enumerate(chunks):
                        new_documents.append({
                            "text": chunk,
                            "source": uploaded_file.name,
                            "chunk_id": i
                        })
                    
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No text found in {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
    
    return new_documents

def generate_response(query: str, search_engine, documents: List[Dict], use_gemini: bool = True):
    """Generate response using either Gemini or fallback generator"""
    
    # Search for relevant documents
    relevant_docs = search_engine.search(query, documents, k=3)
    
    if use_gemini and st.session_state.gemini_engine and st.session_state.gemini_engine.is_available():
        # Use Gemini for response generation
        response = st.session_state.gemini_engine.generate_response(query, relevant_docs)
    else:
        # Use fallback response generator
        response = st.session_state.response_generator.generate_response(query, relevant_docs)
    
    return response, relevant_docs

def main():
    # Initialize Gemini engine
    if st.session_state.gemini_engine is None:
        st.session_state.gemini_engine = initialize_gemini()
    
    # Professional Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">Doc AI</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">Advanced RAG Q&A Chatbot with Google Gemini</p>
        <p style="color: #888; font-size: 0.95rem;">Upload documents and get AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status bar at top
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    gemini_available = st.session_state.gemini_engine and st.session_state.gemini_engine.is_available()
    
    with status_col1:
        if gemini_available:
            st.success("AI Engine: Ready")
        else:
            st.warning("AI Engine: Offline")
    
    with status_col2:
        doc_count = len(st.session_state.processed_files)
        if doc_count > 0:
            st.info(f"Documents: {doc_count}")
        else:
            st.info("Documents: None")
    
    with status_col3:
        chat_count = len(st.session_state.chat_history)
        if chat_count > 0:
            st.info(f"Conversations: {chat_count}")
        else:
            st.info("Conversations: 0")
    
    with status_col4:
        engine_type = "Gemini AI" if (st.session_state.use_gemini and gemini_available) else "Local"
        st.info(f"Mode: {engine_type}")
    
    st.markdown("---")
    
    # Main content area
    if not st.session_state.documents:
        # Welcome screen - prominent document upload
        st.markdown("### Get Started")
        
        upload_col1, upload_col2 = st.columns([3, 2])
        
        with upload_col1:
            st.markdown("**Step 1: Upload Your Documents**")
            uploaded_files = st.file_uploader(
                "Select PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Drag and drop PDF files or click to browse"
            )
            
            if uploaded_files:
                st.markdown("**Step 2: Process Documents**")
                if st.button("Process Documents", type="primary", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        new_documents = process_uploaded_files(uploaded_files)
                        if new_documents:
                            st.session_state.documents.extend(new_documents)
                            st.session_state.search_engine.add_documents(new_documents)
                            st.rerun()
        
        with upload_col2:
            st.markdown("**System Status**")
            if gemini_available:
                st.success("Google Gemini: Available")
            else:
                st.error("Google Gemini: Not Available")
            st.success("PDF Processing: Ready")
            st.success("Search Engine: Ready")
            
            if not gemini_available:
                st.markdown("---")
                st.warning("Set up your Google API key for advanced AI features")
    
    else:
        # Main dashboard with documents loaded
        # Chat interface - more prominent
        st.markdown("### Ask Questions About Your Documents")
        
        # Quick AI engine toggle
        engine_col1, engine_col2 = st.columns([2, 1])
        with engine_col1:
            query = st.text_input(
                "Ask a question",
                placeholder="What would you like to know about your documents?",
                label_visibility="collapsed"
            )
        
        with engine_col2:
            if gemini_available:
                st.session_state.use_gemini = st.toggle("Use Gemini AI", value=True)
            else:
                st.session_state.use_gemini = False
                st.info("Using Local Engine")
        
        if query:
            if st.button("Ask Question", type="primary", use_container_width=True):
                with st.spinner("Analyzing documents..."):
                    try:
                        response, relevant_docs = generate_response(
                            query, 
                            st.session_state.search_engine, 
                            st.session_state.documents,
                            st.session_state.use_gemini
                        )
                        
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": response,
                            "sources": [doc.get("source", "Unknown") for doc in relevant_docs],
                            "engine": "Gemini AI" if st.session_state.use_gemini and gemini_available else "Local Engine"
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Chat history with better formatting
        if st.session_state.chat_history:
            st.markdown("### Recent Conversations")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"{chat['question'][:60]}{'...' if len(chat['question']) > 60 else ''}", expanded=(i==0)):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.caption(f"Sources: {', '.join(set(chat['sources']))} | Engine: {chat['engine']}")
            
            # Show more/clear buttons
            history_col1, history_col2 = st.columns(2)
            with history_col1:
                if len(st.session_state.chat_history) > 5:
                    if st.button("View All Conversations"):
                        for chat in st.session_state.chat_history:
                            with st.expander(f"{chat['question'][:60]}..."):
                                st.write("**Question:**", chat['question'])
                                st.write("**Answer:**", chat['answer'])
                                st.caption(f"Sources: {', '.join(set(chat['sources']))} | Engine: {chat['engine']}")
            
            with history_col2:
                if st.button("Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Bottom section with statistics and controls
        st.markdown("---")
        st.markdown("## Dashboard Statistics & Controls")
        
        # Create horizontal layout for bottom section with proper spacing
        bottom_col1, spacer1, bottom_col2, spacer2, bottom_col3 = st.columns([3, 0.3, 3, 0.3, 3])
        with bottom_col1:
            # Document management
            st.markdown("### Your Documents")
            if st.session_state.processed_files:
                for i, filename in enumerate(st.session_state.processed_files, 1):
                    st.write(f"📄 {filename}")
            else:
                st.write("No documents loaded")
            
            # Quick actions
            st.markdown("### Quick Actions")
            if st.button("Add More Documents", use_container_width=True):
                st.session_state.show_upload = True
            
            if st.button("Reset All", use_container_width=True, type="secondary"):
                if st.button("Confirm Reset", type="secondary"):
                    st.session_state.documents = []
                    st.session_state.processed_files = set()
                    st.session_state.chat_history = []
                    st.session_state.search_engine = SimpleSearchEngine()
                    st.rerun()
        
        with spacer1:
            st.empty()  # Creates visual spacing
        
        with bottom_col2:
            # AI Engine Details
            st.markdown("### AI Engine Status")
            if gemini_available:
                st.success("🤖 Google Gemini AI: Active")
                model_info = st.session_state.gemini_engine.get_model_info()
                st.write(f"**Model:** {model_info['model_name']}")
                st.write(f"**Provider:** {model_info['provider']}")
                st.write(f"**Status:** {model_info['status']}")
            else:
                st.warning("🤖 Google Gemini: Offline")
                st.info("Using Local Response Generator")
            
            st.success("📄 PDF Parser: Ready")
            st.success("🔍 Search Engine: Active")
            
            # Document Statistics
            st.markdown("### Document Statistics")
            doc_count = len(st.session_state.processed_files)
            chunk_count = len(st.session_state.documents)
            
            st.metric("Documents Processed", doc_count)
            st.metric("Text Chunks Created", chunk_count)
            
            if chunk_count > 0:
                avg_chunk_size = sum(len(doc['text']) for doc in st.session_state.documents) / chunk_count
                st.metric("Average Chunk Size", f"{avg_chunk_size:.0f} chars")
            
            # Search engine vocabulary size
            if hasattr(st.session_state.search_engine, 'vocab'):
                vocab_size = len(st.session_state.search_engine.vocab)
                st.metric("Search Terms Indexed", vocab_size)
        
        with spacer2:
            st.empty()  # Creates visual spacing
        
        with bottom_col3:
            # Conversation Statistics
            st.markdown("### Conversation Analytics")
            chat_count = len(st.session_state.chat_history)
            st.metric("Total Conversations", chat_count)
            
            if chat_count > 0:
                # Count by engine type
                gemini_count = sum(1 for chat in st.session_state.chat_history if "Gemini" in chat.get('engine', ''))
                local_count = chat_count - gemini_count
                
                st.metric("Gemini AI Responses", gemini_count)
                st.metric("Local AI Responses", local_count)
                
                # Average response length
                avg_response_length = sum(len(chat['answer']) for chat in st.session_state.chat_history) / chat_count
                st.metric("Avg Response Length", f"{avg_response_length:.0f} chars")
            
            # Performance Metrics
            st.markdown("### Performance Metrics")
            st.write("**Processing Configuration:**")
            st.write("• Chunk Size: 600 characters")
            st.write("• Chunk Overlap: 100 characters") 
            st.write("• Search Results: Top 3 chunks")
            st.write("• Text Extraction: Multi-method")
            st.write("• Search Algorithm: TF-IDF + Proximity")
            
            if st.session_state.documents:
                total_chars = sum(len(doc['text']) for doc in st.session_state.documents)
                st.write(f"**Total Content:** {total_chars:,} characters")
                st.write(f"**Memory Usage:** ~{total_chars * 4 / 1024 / 1024:.1f} MB")
        
        # Additional upload area (spans full width)
        if hasattr(st.session_state, 'show_upload') and st.session_state.show_upload:
            st.markdown("---")
            st.markdown("### Add More Documents")
            new_files = st.file_uploader(
                "Upload additional PDFs",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if new_files:
                if st.button("Process New Documents", use_container_width=True):
                    with st.spinner("Processing..."):
                        new_documents = process_uploaded_files(new_files)
                        if new_documents:
                            st.session_state.documents.extend(new_documents)
                            st.session_state.search_engine.add_documents(new_documents)
                            st.session_state.show_upload = False
                            st.rerun()

if __name__ == "__main__":
    main()
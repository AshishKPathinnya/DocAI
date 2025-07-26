# Doc AI - RAG Q&A Chatbot

## Overview

Doc AI is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to upload PDF documents and ask questions about their content. The system processes PDFs, creates embeddings for semantic search, and uses a language model to generate contextual responses based on the retrieved document content.

## System Architecture

The application follows a modular architecture with four main components:

1. **Frontend**: Streamlit web interface for user interaction
2. **Document Processing**: PDF text extraction and chunking
3. **Embedding System**: Vector embeddings and similarity search using FAISS
4. **Chat Engine**: Text generation using Hugging Face transformers

The system uses a RAG (Retrieval-Augmented Generation) pattern where user queries are first used to retrieve relevant document chunks, which are then provided as context to a language model for response generation.

## Screenshots and Live Link : https://bv2x42nzhokwdyz4zstjkq.streamlit.app/
<img width="1920" height="1080" alt="Screenshot 2025-07-27 001526" src="https://github.com/user-attachments/assets/56bb0f38-baa0-437d-9642-cff55c28541d" />

<img width="1920" height="1080" alt="Screenshot 2025-07-27 001616" src="https://github.com/user-attachments/assets/5dfcbecd-9da8-4768-8c16-869677bc4a13" />  
    
<img width="1920" height="1080" alt="Screenshot 2025-07-27 001645" src="https://github.com/user-attachments/assets/cc315041-c8a0-4419-a2de-4e59988848d9" />

<img width="1920" height="1080" alt="Screenshot 2025-07-27 001713" src="https://github.com/user-attachments/assets/358d7583-a349-4704-903a-56ab2a39aa7b" />

## Key Components

### PDF Processing (`utils/pdf_processor.py`)
- **Purpose**: Extracts and processes text from uploaded PDF files
- **Technology**: PyPDF2 for PDF text extraction
- **Features**: 
  - Text cleaning and normalization
  - Chunking with configurable size (500 chars) and overlap (50 chars)
  - Handles multi-page documents

### Embedding Management (`utils/embeddings.py`)
- **Purpose**: Creates and manages document embeddings for semantic search
- **Technology**: 
  - SentenceTransformers (all-MiniLM-L6-v2 model)
  - FAISS for vector similarity search
- **Features**:
  - 384-dimensional embeddings
  - Cosine similarity search
  - Cached model loading for performance

### Chat Engine (`utils/gemini_engine.py`)
- **Purpose**: Generates responses using language models
- **Technology**: 
  - Hugging Face Transformers
  - Google Gemini API-medium as default model
- **Features**:
  - CPU/GPU adaptive execution
  - Configurable generation parameters (temperature: 0.7, top_p: 0.9)
  - Maximum length limits for responses

### Main Application (`app.py`)
- **Purpose**: Streamlit frontend orchestrating all components
- **Features**:
  - File upload interface
  - Chat history management
  - Session state management
  - Component initialization and coordination

## Data Flow

1. **Document Upload**: Users upload PDF files through Streamlit interface
2. **Text Extraction**: PDFProcessor extracts and cleans text from PDFs
3. **Chunking**: Text is split into manageable chunks with overlap
4. **Embedding Creation**: SentenceTransformer creates vector embeddings for chunks
5. **Vector Storage**: FAISS index stores embeddings for fast similarity search
6. **Query Processing**: User questions are embedded and matched against document chunks
7. **Context Retrieval**: Most relevant chunks are retrieved based on similarity
8. **Response Generation**: ChatEngine generates responses using retrieved context

## External Dependencies

### Core ML Libraries
- **sentence-transformers**: For creating document embeddings
- **transformers**: For language model inference
- **torch**: PyTorch backend for model execution
- **faiss-cpu**: For efficient vector similarity search

### Document Processing
- **PyPDF2**: For PDF text extraction

### Web Framework
- **streamlit**: For web interface and user interaction

### Utilities
- **numpy**: For numerical operations on embeddings

## Deployment Strategy

The application is designed for single-user deployment with the following characteristics:

### Resource Management
- **Caching**: Uses Streamlit's `@st.cache_resource` for model loading
- **Session State**: Maintains chat history and processed documents in memory
- **Temporary Files**: Uses Python's tempfile for handling uploaded PDFs

### Performance Considerations
- **Model Loading**: One-time model initialization with caching
- **Memory Management**: In-memory storage for embeddings and documents
- **CPU/GPU Adaptive**: Automatically detects and uses available hardware

### Scalability Limitations
- **Single Session**: No persistence across browser sessions
- **Memory Bound**: Document storage limited by available RAM
- **No Database**: All data stored in session state

## Setup for Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install streamlit google-generativeai PyPDF2 numpy scikit-learn
   ```
3. Set up your Google Gemini API key:
   - Get an API key from [Google AI Studio](https://aistudio.google.com)
   - Set the environment variable: `GOOGLE_API_KEY=your_api_key_here`
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and deploy this repository
4. In the deployment settings, add your `GOOGLE_API_KEY` as a secret
5. The app will automatically use the default Streamlit port (8501) for cloud deployment

**Note**: The `.streamlit/config.toml` file is configured for both local development (with port 5000) and cloud deployment (default port). Streamlit Cloud will automatically override local port settings.


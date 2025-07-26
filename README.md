# Doc AI - Advanced RAG Q&A Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini AI that allows users to upload PDF documents and ask questions about their content. The system processes PDFs, creates searchable indexes, and uses Google's latest AI models to generate contextual responses.

## Features

- **Advanced RAG System**: Retrieval-Augmented Generation for accurate document-based Q&A
- **Google Gemini Integration**: Powered by Google's Gemini-1.5-Flash model with intelligent fallback
- **Enhanced PDF Processing**: Multi-method text extraction with improved accuracy
- **Smart Search Engine**: TF-IDF based similarity search with proximity scoring
- **Professional Dashboard**: Clean, responsive interface with real-time statistics
- **Multi-Document Support**: Process and query multiple documents simultaneously
- **Dual AI Modes**: Toggle between Google Gemini and local response generation
- **Real-time Analytics**: Live monitoring of AI engine status and performance metrics
  
## Screenshots and Live Link : https://bv2x42nzhokwdyz4zstjkq.streamlit.app/
<img width="1920" height="1080" alt="Screenshot 2025-07-27 001526" src="https://github.com/user-attachments/assets/56bb0f38-baa0-437d-9642-cff55c28541d" />

<img width="1920" height="1080" alt="Screenshot 2025-07-27 001616" src="https://github.com/user-attachments/assets/5dfcbecd-9da8-4768-8c16-869677bc4a13" />  
    
<img width="1920" height="1080" alt="Screenshot 2025-07-27 001645" src="https://github.com/user-attachments/assets/cc315041-c8a0-4419-a2de-4e59988848d9" />

<img width="1920" height="1080" alt="Screenshot 2025-07-27 001713" src="https://github.com/user-attachments/assets/358d7583-a349-4704-903a-56ab2a39aa7b" />

## Setup for Local Development

1. Clone this repository
2. Install dependencies:
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
4. In the deployment settings, add your `GOOGLE_API_KEY` as a secret:
   - Click on the "Advanced settings" section during deployment
   - Under "Secrets", add: `GOOGLE_API_KEY = your_actual_api_key_here`
   - Make sure to use your actual Google AI Studio API key
5. The app will automatically use the default Streamlit port (8501) for cloud deployment

## Architecture

The application follows a modular RAG architecture with four main components:

### Key Components

**PDF Processing (`utils/pdf_processor.py`)**
- Purpose: Extracts and processes text from uploaded PDF files
- Technology: PyPDF2 with enhanced multi-method extraction
- Features:
  - Advanced text cleaning and normalization
  - Paragraph-aware chunking (600 chars, 100 chars overlap)
  - Multi-page document handling with page context
  - Fallback extraction methods for complex PDFs

**Search Engine (`utils/simple_search.py`)**
- Purpose: Provides intelligent document retrieval
- Technology: Enhanced TF-IDF with proximity scoring
- Features:
  - Advanced tokenization with phrase matching
  - Question-answer pattern recognition
  - Multi-factor relevance scoring
  - Context-aware document ranking

**AI Engine (`utils/gemini_engine.py`)**
- Purpose: Generates intelligent responses using Google's latest AI
- Technology: Google Gemini-1.5-Flash API
- Features:
  - Context-aware response generation
  - Configurable response length and style
  - Error handling with graceful degradation
  - Secure API key management

**Local Fallback (`utils/simple_generator.py`)**
- Purpose: Provides responses when Gemini is unavailable
- Technology: Template-based response generation with advanced analysis
- Features:
  - Question classification and analysis
  - Context-aware response building
  - Multi-document synthesis
  - Intelligent content categorization

**Main Application (`app.py`)**
- Purpose: Streamlit frontend orchestrating all components
- Features:
  - Professional dashboard interface
  - Real-time system monitoring
  - Chat history management
  - Component coordination and state management

## Data Flow

1. **Document Upload**: Users upload PDF files through Streamlit interface
2. **Text Extraction**: Enhanced PDFProcessor extracts and cleans text from PDFs
3. **Intelligent Chunking**: Text is split into contextual chunks with smart overlap
4. **TF-IDF Indexing**: Search engine creates searchable indexes with relevance scoring
5. **Query Processing**: User questions are analyzed and matched against document content
6. **Context Retrieval**: Most relevant chunks are retrieved using advanced scoring
7. **AI Response Generation**: Gemini AI generates contextual responses with retrieved content
8. **Fallback Handling**: Local generator provides responses if Gemini is unavailable

## Dependencies

### Core Libraries
- **streamlit**: Modern web interface framework
- **google-generativeai**: Google Gemini AI integration
- **PyPDF2**: PDF document processing and text extraction
- **numpy**: Numerical computations for similarity calculations
- **scikit-learn**: Machine learning utilities for text processing

### System Requirements
- Python 3.11+
- Internet connection for Gemini API access
- Minimum 512MB RAM for document processing

## Performance Characteristics

### Resource Management
- **Caching**: Uses Streamlit's `@st.cache_resource` for model loading
- **Session State**: Maintains chat history and processed documents in memory
- **Temporary Files**: Secure handling of uploaded PDFs with automatic cleanup

### Scalability Considerations
- **Single Session**: Optimized for individual user sessions
- **Memory Efficient**: Smart chunking prevents memory overflow
- **API Rate Limiting**: Graceful handling of Gemini API limits

## Security Features

- **Secure API Key Management**: Environment-based key storage
- **File Validation**: PDF format verification before processing
- **Session Isolation**: User data isolated per browser session
- **Temporary File Cleanup**: Automatic removal of uploaded files

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set correctly
2. **PDF Processing Error**: Check if PDF is text-based (not scanned images)
3. **Slow Response**: Large documents may take time to process initially
4. **Memory Issues**: Restart session if processing many large documents

### Performance Tips
- Upload text-based PDFs for best results
- Process documents in smaller batches for better performance
- Use Local mode if experiencing API rate limits
- Clear chat history periodically to free memory

## Usage

1. Upload PDF documents using the file uploader
2. Click "Process Documents" to analyze your files
3. Ask questions about your documents in the chat interface
4. Toggle between Gemini AI mode and Local mode as needed
5. Monitor system performance through the real-time dashboard
---

**Note**: This application is designed for development and small-scale usage. For production deployment with multiple users, consider adding persistent storage, user authentication, and horizontal scaling capabilities.

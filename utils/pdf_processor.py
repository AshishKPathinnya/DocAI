import PyPDF2
import re
from typing import List

class PDFProcessor:
    """Handles PDF text extraction and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file with enhanced accuracy"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages with better handling
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    
                    # Try multiple extraction methods for better accuracy
                    page_text = ""
                    
                    # Primary extraction method
                    try:
                        page_text = page.extract_text()
                    except:
                        page_text = ""
                    
                    # If primary method fails or returns little text, try alternative
                    if not page_text or len(page_text.strip()) < 50:
                        try:
                            # Alternative extraction method using visitor pattern
                            visitor_text = ""
                            def visitor_body(text, cm, tm, fontDict, fontSize):
                                nonlocal visitor_text
                                visitor_text += text
                            
                            page.extract_text(visitor_text=visitor_body)
                            if len(visitor_text.strip()) > len(page_text.strip()):
                                page_text = visitor_text
                        except:
                            pass
                    
                    if page_text:
                        # Add page separator for better context
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Enhanced text cleaning
            text = self._enhanced_clean_text(text)
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _enhanced_clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better accuracy"""
        if not text:
            return ""
        
        # Remove page markers but preserve them for reference
        text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
        
        # Fix common PDF extraction issues
        
        # Fix broken words caused by line breaks
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Fix multiple line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix spacing issues
        text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)
        
        # Remove excessive whitespace but preserve paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Normalize internal spacing
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
                cleaned_lines.append('')
        
        # Join back and clean up
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive paragraph breaks
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2026', '...')  # Horizontal ellipsis
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with improved accuracy"""
        if not text:
            return []
        
        # First, split into paragraphs to preserve logical structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:].strip()
                    # Try to start overlap at word boundary
                    word_start = overlap_text.find(' ')
                    if word_start > 0:
                        overlap_text = overlap_text[word_start:].strip()
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Post-process chunks for very long paragraphs
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5:
                # Split long chunks at sentence boundaries
                sentences = self._split_into_sentences(chunk)
                sub_chunk = ""
                
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) + 1 <= self.chunk_size:
                        sub_chunk += sentence + " "
                    else:
                        if sub_chunk.strip():
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence + " "
                
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        # Filter out very short chunks
        return [chunk for chunk in final_chunks if len(chunk.strip()) > 50]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences more accurately"""
        # Simple sentence splitting with common abbreviations handling
        abbreviations = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'vs.', 'e.g.', 'i.e.'}
        
        # Replace abbreviations temporarily
        temp_text = text
        temp_replacements = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            temp_text = temp_text.replace(abbrev, placeholder)
            temp_replacements[placeholder] = abbrev
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', temp_text)
        
        # Restore abbreviations
        for i, sentence in enumerate(sentences):
            for placeholder, abbrev in temp_replacements.items():
                sentence = sentence.replace(placeholder, abbrev)
            sentences[i] = sentence.strip()
        
        return [s for s in sentences if s]
    
    def get_chunk_metadata(self, chunks: List[str], source_file: str) -> List[dict]:
        """Add metadata to chunks"""
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "text": chunk,
                "source": source_file,
                "chunk_id": i,
                "char_count": len(chunk),
                "word_count": len(chunk.split())
            })
        return metadata

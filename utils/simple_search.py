import re
from typing import List, Dict, Any
from collections import Counter
import math

class SimpleSearchEngine:
    """A lightweight text search engine using TF-IDF for document retrieval"""
    
    def __init__(self):
        self.documents = []
        self.word_freq = {}
        self.doc_freq = {}
        self.vocab = set()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search engine"""
        self.documents.extend(documents)
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index for documents"""
        # Clear existing index
        self.word_freq = {}
        self.doc_freq = {}
        self.vocab = set()
        
        # Process each document
        for i, doc in enumerate(self.documents):
            text = doc['text'].lower()
            words = self._tokenize(text)
            
            # Count word frequencies in this document
            word_count = Counter(words)
            self.word_freq[i] = word_count
            
            # Update vocabulary and document frequency
            for word in set(words):
                self.vocab.add(word)
                if word not in self.doc_freq:
                    self.doc_freq[word] = 0
                self.doc_freq[word] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with better accuracy"""
        # Preserve important punctuation and structure
        text = re.sub(r'[^\w\s\.\-]', ' ', text)
        
        # Handle hyphenated words and contractions
        text = re.sub(r"(\w+)[''](\w+)", r'\1\2', text)  # Handle contractions
        
        words = text.split()
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        # Filter and normalize words
        filtered_words = []
        for word in words:
            word = word.lower().strip('.')
            if len(word) > 2 and word not in stop_words and word.isalpha():
                filtered_words.append(word)
        
        return filtered_words
    
    def _calculate_tf_idf(self, query_words: List[str], doc_index: int) -> float:
        """Calculate TF-IDF score for a document given query words"""
        if doc_index not in self.word_freq:
            return 0.0
        
        score = 0.0
        doc_word_count = self.word_freq[doc_index]
        total_docs = len(self.documents)
        
        for word in query_words:
            if word in doc_word_count and word in self.doc_freq:
                # Term Frequency
                tf = doc_word_count[word] / sum(doc_word_count.values())
                
                # Inverse Document Frequency
                idf = math.log(total_docs / self.doc_freq[word])
                
                # TF-IDF Score
                score += tf * idf
        
        return score
    
    def search(self, query: str, documents: List[Dict[str, Any]] = None, k: int = 3) -> List[Dict[str, Any]]:
        """Enhanced search with improved accuracy"""
        # If documents are provided, use them temporarily, otherwise use stored documents
        search_documents = documents if documents is not None else self.documents
        
        if not search_documents:
            return []
        
        # If using provided documents, temporarily rebuild index
        if documents is not None:
            original_docs = self.documents
            self.documents = documents
            self._build_index()
        
        try:
            query_words = self._tokenize(query.lower())
            if not query_words:
                return []
            
            # Calculate scores for all documents
            scores = []
            for i in range(len(self.documents)):
                score = self._calculate_enhanced_score(query, query_words, i)
                scores.append((score, i))
            
            # Sort by score and return top k results
            scores.sort(reverse=True, key=lambda x: x[0])
            
            results = []
            for score, idx in scores[:k]:
                if score > 0:  # Only return documents with positive scores
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = score
                    doc['rank'] = len(results) + 1
                    results.append(doc)
            
            return results
        
        finally:
            # Restore original documents if they were temporarily changed
            if documents is not None:
                self.documents = original_docs
                self._build_index()
    
    def _calculate_enhanced_score(self, original_query: str, query_words: List[str], doc_index: int) -> float:
        """Calculate enhanced relevance score"""
        if doc_index not in self.word_freq:
            return 0.0
        
        doc_text = self.documents[doc_index]['text']
        doc_text_lower = doc_text.lower()
        
        # Base TF-IDF score
        base_score = self._calculate_tf_idf(query_words, doc_index)
        
        # Bonus scoring factors
        bonus_score = 0.0
        
        # 1. Exact phrase match (highest priority)
        if original_query.lower() in doc_text_lower:
            bonus_score += 2.0
        
        # 2. Partial phrase matches
        query_phrases = self._extract_phrases(original_query)
        for phrase in query_phrases:
            if len(phrase) > 1 and phrase.lower() in doc_text_lower:
                bonus_score += 1.5
        
        # 3. Sequential word matches
        text_words = self._tokenize(doc_text_lower)
        for i in range(len(text_words) - len(query_words) + 1):
            if text_words[i:i+len(query_words)] == query_words:
                bonus_score += 1.0
                break
        
        # 4. Proximity bonus (words appearing close together)
        proximity_bonus = self._calculate_proximity_bonus(query_words, text_words)
        bonus_score += proximity_bonus
        
        # 5. Question-answer pattern matching
        qa_bonus = self._calculate_qa_bonus(original_query, doc_text)
        bonus_score += qa_bonus
        
        # 6. Keyword density bonus
        keyword_density = sum(1 for word in text_words if word in query_words) / max(len(text_words), 1)
        bonus_score += keyword_density * 0.5
        
        return base_score + bonus_score
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query"""
        # Split by common delimiters but preserve important phrases
        phrases = []
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        phrases.extend([p.strip() for p in quoted_phrases if len(p.strip()) > 3])
        
        # Look for noun phrases (simple heuristic)
        words = query.split()
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if len(phrase) > 5:
                phrases.append(phrase)
        
        return phrases
    
    def _calculate_proximity_bonus(self, query_words: List[str], text_words: List[str]) -> float:
        """Calculate bonus for words appearing close together"""
        if len(query_words) < 2:
            return 0.0
        
        max_bonus = 0.0
        window_size = min(10, len(query_words) * 3)  # Flexible window size
        
        for i in range(len(text_words) - window_size + 1):
            window = text_words[i:i + window_size]
            matches_in_window = sum(1 for word in query_words if word in window)
            
            if matches_in_window >= 2:
                # Higher bonus for more matches in smaller window
                proximity_score = (matches_in_window / len(query_words)) * (1.0 / window_size) * 10
                max_bonus = max(max_bonus, proximity_score)
        
        return min(max_bonus, 1.0)  # Cap the bonus
    
    def _calculate_qa_bonus(self, query: str, doc_text: str) -> float:
        """Calculate bonus for question-answer patterns"""
        query_lower = query.lower()
        doc_lower = doc_text.lower()
        
        bonus = 0.0
        
        # Question word patterns
        question_patterns = {
            'what': ['definition', 'meaning', 'refers to', 'is defined as', 'means'],
            'how': ['process', 'method', 'procedure', 'steps', 'way to'],
            'when': ['time', 'date', 'during', 'period', 'year'],
            'where': ['location', 'place', 'situated', 'located', 'address'],
            'who': ['person', 'people', 'individual', 'author', 'responsible'],
            'why': ['reason', 'because', 'due to', 'cause', 'purpose']
        }
        
        for q_word, patterns in question_patterns.items():
            if q_word in query_lower:
                for pattern in patterns:
                    if pattern in doc_lower:
                        bonus += 0.3
                        break
        
        return min(bonus, 1.0)  # Cap the bonus
    
    def clear(self):
        """Clear all documents and reset the index"""
        self.documents = []
        self.word_freq = {}
        self.doc_freq = {}
        self.vocab = set()
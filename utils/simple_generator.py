from typing import List, Dict, Any
import re

class SimpleResponseGenerator:
    """A lightweight response generator using template-based responses"""
    
    def __init__(self):
        self.available = True
    
    def generate_response(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate an enhanced response based on the question and context documents"""
        if not context_docs:
            return "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure you have uploaded relevant PDF documents."
        
        # Advanced context preparation with ranking
        context_analysis = self._analyze_context_deeply(question, context_docs)
        
        # Generate comprehensive response
        response = self._generate_comprehensive_response(question, context_analysis)
        
        # Post-process and refine the response
        response = self._refine_response(response, question)
        
        return response
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents"""
        context_parts = []
        for doc in context_docs[:3]:  # Use top 3 documents
            context_parts.append(doc["text"])
        return " ".join(context_parts)
    
    def _analyze_context_deeply(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform deep analysis of context documents for better response generation"""
        analysis = {
            'primary_facts': [],
            'supporting_details': [],
            'definitions': [],
            'processes': [],
            'relationships': [],
            'question_type': self._classify_question_advanced(question),
            'key_concepts': self._extract_key_concepts(question),
            'context_segments': []
        }
        
        # Analyze each document
        for doc in context_docs[:5]:  # Analyze more documents
            doc_analysis = self._analyze_document_content(doc['text'], question)
            analysis['context_segments'].append({
                'source': doc['source'],
                'text': doc['text'],
                'relevance_score': doc.get('similarity_score', 0),
                'content_type': doc_analysis['content_type'],
                'key_sentences': doc_analysis['key_sentences'],
                'facts': doc_analysis['facts']
            })
        
        # Extract specific content types
        for segment in analysis['context_segments']:
            if segment['content_type'] == 'definition':
                analysis['definitions'].extend(segment['key_sentences'])
            elif segment['content_type'] == 'process':
                analysis['processes'].extend(segment['key_sentences'])
            elif segment['content_type'] == 'factual':
                analysis['primary_facts'].extend(segment['facts'])
            else:
                analysis['supporting_details'].extend(segment['key_sentences'])
        
        return analysis
    
    def _classify_question_advanced(self, question: str) -> Dict[str, Any]:
        """Advanced question classification"""
        question_lower = question.lower()
        
        classification = {
            'primary_type': 'general',
            'intent': 'information_seeking',
            'specificity': 'general',
            'complexity': 'simple'
        }
        
        # Primary question types
        if any(word in question_lower for word in ['what is', 'what are', 'what does', 'define']):
            classification['primary_type'] = 'definition'
        elif any(word in question_lower for word in ['how to', 'how do', 'how does', 'how can']):
            classification['primary_type'] = 'process'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            classification['primary_type'] = 'explanation'
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            classification['primary_type'] = 'temporal'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            classification['primary_type'] = 'location'
        elif any(word in question_lower for word in ['who', 'person', 'people']):
            classification['primary_type'] = 'entity'
        elif any(word in question_lower for word in ['which', 'what type', 'what kind']):
            classification['primary_type'] = 'classification'
        elif any(word in question_lower for word in ['how many', 'how much', 'number']):
            classification['primary_type'] = 'quantitative'
        
        # Intent classification
        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            classification['intent'] = 'comparison'
        elif any(word in question_lower for word in ['list', 'examples', 'types']):
            classification['intent'] = 'enumeration'
        elif '?' in question:
            classification['intent'] = 'direct_question'
        
        # Specificity
        if len(question.split()) > 10:
            classification['specificity'] = 'specific'
        elif any(word in question_lower for word in ['specifically', 'exactly', 'precisely']):
            classification['specificity'] = 'very_specific'
        
        # Complexity
        if any(word in question_lower for word in ['relationship', 'impact', 'effect', 'implications']):
            classification['complexity'] = 'complex'
        elif len(question.split()) > 15:
            classification['complexity'] = 'complex'
        
        return classification
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract key concepts from the question"""
        # Remove question words and common terms
        stop_words = {
            'what', 'how', 'when', 'where', 'who', 'why', 'which', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'may', 'might'
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        concepts = []
        
        # Extract multi-word concepts (noun phrases)
        words_clean = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Look for consecutive important words
        for i in range(len(words_clean)):
            concepts.append(words_clean[i])
            if i < len(words_clean) - 1:
                two_word = f"{words_clean[i]} {words_clean[i+1]}"
                concepts.append(two_word)
        
        return list(set(concepts))
    
    def _analyze_document_content(self, text: str, question: str) -> Dict[str, Any]:
        """Analyze document content to categorize and extract key information"""
        analysis = {
            'content_type': 'general',
            'key_sentences': [],
            'facts': [],
            'confidence': 0.0
        }
        
        sentences = self._split_sentences(text)
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Determine content type
        if any(pattern in text_lower for pattern in ['is defined as', 'means', 'refers to', 'is known as']):
            analysis['content_type'] = 'definition'
        elif any(pattern in text_lower for pattern in ['step', 'process', 'method', 'procedure', 'approach']):
            analysis['content_type'] = 'process'
        elif any(pattern in text_lower for pattern in ['because', 'due to', 'reason', 'cause', 'therefore']):
            analysis['content_type'] = 'explanation'
        elif re.search(r'\d+', text) and any(word in text_lower for word in ['percent', 'million', 'year', 'data']):
            analysis['content_type'] = 'factual'
        
        # Extract key sentences based on content type and question relevance
        question_words = [w.lower() for w in question.split() if len(w) > 2]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            relevance_score = 0
            
            # Score based on question word matches
            word_matches = sum(1 for word in question_words if word in sentence_lower)
            relevance_score += word_matches * 0.5
            
            # Score based on content indicators
            if analysis['content_type'] == 'definition' and any(word in sentence_lower for word in ['is', 'are', 'means']):
                relevance_score += 1.0
            elif analysis['content_type'] == 'process' and any(word in sentence_lower for word in ['step', 'then', 'next']):
                relevance_score += 1.0
            elif analysis['content_type'] == 'explanation' and any(word in sentence_lower for word in ['because', 'since', 'therefore']):
                relevance_score += 1.0
            
            # Extract factual information
            if re.search(r'\d+', sentence):
                analysis['facts'].append(sentence)
                relevance_score += 0.3
            
            if relevance_score > 0.5:
                analysis['key_sentences'].append((sentence, relevance_score))
        
        # Sort by relevance and keep top sentences
        analysis['key_sentences'].sort(key=lambda x: x[1], reverse=True)
        analysis['key_sentences'] = [s[0] for s in analysis['key_sentences'][:3]]
        
        return analysis
    
    def _generate_comprehensive_response(self, question: str, context_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive response using deep context analysis"""
        question_type = context_analysis['question_type']
        primary_type = question_type['primary_type']
        
        # Build response based on question type and available content
        if primary_type == 'definition' and context_analysis['definitions']:
            response = self._build_definition_response(question, context_analysis)
        elif primary_type == 'process' and context_analysis['processes']:
            response = self._build_process_response(question, context_analysis)
        elif primary_type == 'explanation':
            response = self._build_explanation_response(question, context_analysis)
        elif primary_type == 'quantitative' and context_analysis['primary_facts']:
            response = self._build_quantitative_response(question, context_analysis)
        elif primary_type == 'comparison':
            response = self._build_comparison_response(question, context_analysis)
        else:
            response = self._build_general_response(question, context_analysis)
        
        return response
    
    def _build_definition_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build a definition-focused response"""
        definitions = analysis['definitions'][:2]  # Top 2 definitions
        supporting = analysis['supporting_details'][:2]
        
        if not definitions:
            return self._build_general_response(question, analysis)
        
        # Start with the primary definition
        response = f"Based on the documents: {definitions[0]}"
        
        # Add additional definition if available
        if len(definitions) > 1:
            response += f" Additionally, {definitions[1]}"
        
        # Add supporting context if available
        if supporting:
            response += f" The documents also indicate that {supporting[0]}"
        
        return response
    
    def _build_process_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build a process-focused response"""
        processes = analysis['processes'][:3]  # Top 3 process descriptions
        
        if not processes:
            return self._build_general_response(question, analysis)
        
        response = f"According to the documents, the process involves: {processes[0]}"
        
        # Add additional steps or details
        if len(processes) > 1:
            response += f" The documentation further explains: {processes[1]}"
        
        if len(processes) > 2:
            response += f" Additionally, {processes[2]}"
        
        return response
    
    def _build_explanation_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build an explanation-focused response"""
        # Look for causal relationships and explanations
        all_content = (analysis['definitions'] + analysis['processes'] + 
                      analysis['supporting_details'] + analysis['primary_facts'])
        
        # Filter for explanatory content
        explanations = []
        for content in all_content:
            if any(word in content.lower() for word in ['because', 'due to', 'reason', 'cause', 'therefore', 'since']):
                explanations.append(content)
        
        if explanations:
            response = f"The documents explain that {explanations[0]}"
            if len(explanations) > 1:
                response += f" Furthermore, {explanations[1]}"
        else:
            # Fallback to most relevant content
            relevant_content = all_content[:2]
            if relevant_content:
                response = f"Based on the available information: {relevant_content[0]}"
                if len(relevant_content) > 1:
                    response += f" The documents also mention: {relevant_content[1]}"
            else:
                response = "I found limited explanatory information in the documents for this question."
        
        return response
    
    def _build_quantitative_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build a quantitative/factual response"""
        facts = analysis['primary_facts'][:3]
        
        if not facts:
            return self._build_general_response(question, analysis)
        
        response = f"According to the data in the documents: {facts[0]}"
        
        if len(facts) > 1:
            response += f" The documents also show: {facts[1]}"
        
        if len(facts) > 2:
            response += f" Additionally, {facts[2]}"
        
        return response
    
    def _build_comparison_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build a comparison-focused response"""
        all_content = (analysis['definitions'] + analysis['supporting_details'] + 
                      analysis['primary_facts'])
        
        # Look for comparative language
        comparative_content = []
        for content in all_content:
            if any(word in content.lower() for word in ['compared to', 'versus', 'different', 'similar', 'unlike', 'while']):
                comparative_content.append(content)
        
        if comparative_content:
            response = f"The documents provide this comparison: {comparative_content[0]}"
            if len(comparative_content) > 1:
                response += f" They also note: {comparative_content[1]}"
        else:
            # Provide separate information that can be compared
            if len(all_content) >= 2:
                response = f"Based on the documents: {all_content[0]} In contrast, {all_content[1]}"
            else:
                response = self._build_general_response(question, analysis)
        
        return response
    
    def _build_general_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Build a general response using best available content"""
        # Prioritize content by type and relevance
        content_priority = []
        
        # Add definitions first (highest priority)
        content_priority.extend([(item, 'definition') for item in analysis['definitions'][:2]])
        
        # Add processes
        content_priority.extend([(item, 'process') for item in analysis['processes'][:2]])
        
        # Add facts
        content_priority.extend([(item, 'fact') for item in analysis['primary_facts'][:2]])
        
        # Add supporting details
        content_priority.extend([(item, 'detail') for item in analysis['supporting_details'][:2]])
        
        if not content_priority:
            return "I found some relevant information in the documents, but couldn't extract specific details that directly answer your question. Please try rephrasing your question or check if the uploaded documents contain the information you're looking for."
        
        # Build response with the best available content
        primary_content = content_priority[0][0]
        response = f"Based on the documents: {primary_content}"
        
        # Add secondary content if available and different
        if len(content_priority) > 1:
            secondary_content = content_priority[1][0]
            if secondary_content != primary_content:
                response += f" The documents also indicate: {secondary_content}"
        
        return response
    
    def _refine_response(self, response: str, question: str) -> str:
        """Post-process and refine the response for better quality"""
        # Remove redundant phrases
        response = re.sub(r'\bthe documents?\b\s+', '', response, count=1)
        response = response.strip()
        
        # Ensure proper capitalization
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        # Remove excessive repetition
        sentences = response.split('. ')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                # Check for substantial overlap with existing sentences
                is_duplicate = False
                for existing in unique_sentences:
                    if len(set(sentence.lower().split()) & set(existing.lower().split())) > len(sentence.split()) * 0.7:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_sentences.append(sentence)
        
        refined_response = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if refined_response and not refined_response.endswith('.'):
            refined_response += '.'
        
        # Add context if response is too short
        if len(refined_response) < 50:
            refined_response += " For more detailed information, please refer to the source documents."
        
        return refined_response
    
    def _create_contextual_response(self, question: str, context: str, context_docs: List[Dict[str, Any]]) -> str:
        """Create a response using the context and question analysis"""
        
        # Identify question type
        question_lower = question.lower()
        question_words = question_lower.split()
        
        # Question type indicators
        what_questions = ['what', 'which']
        how_questions = ['how']
        when_questions = ['when']
        where_questions = ['where'] 
        who_questions = ['who']
        why_questions = ['why']
        
        question_type = 'general'
        if any(word in question_words for word in what_questions):
            question_type = 'what'
        elif any(word in question_words for word in how_questions):
            question_type = 'how'
        elif any(word in question_words for word in when_questions):
            question_type = 'when'
        elif any(word in question_words for word in where_questions):
            question_type = 'where'
        elif any(word in question_words for word in who_questions):
            question_type = 'who'
        elif any(word in question_words for word in why_questions):
            question_type = 'why'
        
        # Find key terms from the question
        key_terms = self._extract_key_terms(question)
        
        # Look for relevant sentences in context
        relevant_sentences = self._find_relevant_sentences(context, key_terms, question_lower)
        
        # Generate response based on question type
        if question_type == 'what':
            response = self._generate_what_response(relevant_sentences, key_terms)
        elif question_type == 'how':
            response = self._generate_how_response(relevant_sentences, key_terms)
        elif question_type == 'when':
            response = self._generate_when_response(relevant_sentences, key_terms)
        elif question_type == 'where':
            response = self._generate_where_response(relevant_sentences, key_terms)
        elif question_type == 'who':
            response = self._generate_who_response(relevant_sentences, key_terms)
        elif question_type == 'why':
            response = self._generate_why_response(relevant_sentences, key_terms)
        else:
            response = self._generate_general_response(relevant_sentences, key_terms)
        
        # If no specific response generated, provide relevant context
        if not response.strip():
            response = self._create_fallback_response(question, context_docs)
        
        return response
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from the question"""
        # Remove common question words and punctuation
        stop_words = {'what', 'how', 'when', 'where', 'who', 'why', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'do', 'does', 'did', 'can', 'could', 'would', 'should'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        return key_terms
    
    def _find_relevant_sentences(self, context: str, key_terms: List[str], question: str) -> List[str]:
        """Find sentences in context that contain key terms with enhanced accuracy"""
        # Split into sentences more accurately
        sentences = self._split_sentences(context)
        relevant_sentences = []
        
        question_lower = question.lower()
        question_words = [w.lower() for w in question.split() if len(w) > 2]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            relevance_score = 0.0
            
            # 1. Exact phrase match from question
            if len(question) > 10 and question_lower in sentence_lower:
                relevance_score += 3.0
            
            # 2. Key terms match
            term_matches = sum(1 for term in key_terms if term in sentence_lower)
            relevance_score += term_matches * 1.0
            
            # 3. Question words match
            question_matches = sum(1 for word in question_words if word in sentence_lower)
            relevance_score += question_matches * 0.5
            
            # 4. Sentence structure indicators (definition, explanation)
            structure_indicators = ['is', 'are', 'was', 'were', 'means', 'refers to', 'defined as', 'because', 'due to', 'therefore', 'thus', 'however', 'moreover']
            if any(indicator in sentence_lower for indicator in structure_indicators):
                relevance_score += 0.5
            
            # 5. Numerical or factual content bonus
            if re.search(r'\d+', sentence) or any(word in sentence_lower for word in ['percent', 'million', 'billion', 'year', 'date']):
                relevance_score += 0.3
            
            # 6. Proximity bonus for multiple key terms
            if term_matches > 1:
                relevance_score += 0.5
            
            if relevance_score > 0.5:  # Threshold for relevance
                relevant_sentences.append((sentence, relevance_score))
        
        # Sort by relevance score and return top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence[0] for sentence in relevant_sentences[:5]]  # Return more sentences
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences more accurately"""
        # Handle common abbreviations
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'vs.', 'e.g.', 'i.e.', 'cf.']
        
        # Temporarily replace abbreviations
        temp_text = text
        replacements = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            temp_text = temp_text.replace(abbrev, placeholder)
            replacements[placeholder] = abbrev
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', temp_text)
        
        # Restore abbreviations and clean up
        final_sentences = []
        for sentence in sentences:
            for placeholder, abbrev in replacements.items():
                sentence = sentence.replace(placeholder, abbrev)
            sentence = sentence.strip()
            if sentence:
                final_sentences.append(sentence)
        
        return final_sentences
    
    def _generate_what_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate enhanced response for 'what' questions"""
        if not sentences:
            return ""
        
        # Look for definitions or descriptions (prioritized)
        definition_sentences = []
        explanation_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # High priority: Definition patterns
            if any(pattern in sentence_lower for pattern in ['is defined as', 'means', 'refers to', 'is known as']):
                definition_sentences.append(sentence)
            # Medium priority: Description patterns  
            elif any(pattern in sentence_lower for pattern in ['is', 'are', 'was', 'were', 'describes', 'represents']):
                explanation_sentences.append(sentence)
        
        # Build comprehensive response
        if definition_sentences:
            response = f"Based on the document: {definition_sentences[0]}"
            if len(definition_sentences) > 1:
                response += f" Additionally, {definition_sentences[1]}"
        elif explanation_sentences:
            response = f"According to the document: {explanation_sentences[0]}"
            if len(explanation_sentences) > 1 and len(sentences) > 1:
                response += f" The document also mentions: {explanation_sentences[1]}"
        else:
            # Fallback to most relevant sentence
            response = f"The document states: {sentences[0]}"
            if len(sentences) > 1:
                response += f" It also indicates: {sentences[1]}"
        
        return response
    
    def _generate_how_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate response for 'how' questions"""
        if not sentences:
            return ""
        
        # Look for process descriptions
        process_words = ['step', 'process', 'method', 'way', 'procedure', 'approach']
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in process_words):
                return f"The document explains: {sentence}"
        
        return f"Here's what the document says: {sentences[0]}"
    
    def _generate_when_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate response for 'when' questions"""
        if not sentences:
            return ""
        
        # Look for time-related information
        time_patterns = [r'\d{4}', r'\d{1,2}/\d{1,2}', r'january|february|march|april|may|june|july|august|september|october|november|december']
        for sentence in sentences:
            if any(re.search(pattern, sentence.lower()) for pattern in time_patterns):
                return f"According to the document: {sentence}"
        
        return f"The document mentions: {sentences[0]}"
    
    def _generate_where_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate response for 'where' questions"""
        if not sentences:
            return ""
        
        # Look for location information
        location_words = ['located', 'place', 'address', 'city', 'country', 'region', 'area']
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in location_words):
                return f"The document indicates: {sentence}"
        
        return f"Based on the document: {sentences[0]}"
    
    def _generate_who_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate response for 'who' questions"""
        if not sentences:
            return ""
        
        # Look for people or organization mentions
        for sentence in sentences:
            # Check for capitalized words (potential names)
            if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence):
                return f"The document mentions: {sentence}"
        
        return f"According to the document: {sentences[0]}"
    
    def _generate_why_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate response for 'why' questions"""
        if not sentences:
            return ""
        
        # Look for explanations or reasons
        reason_words = ['because', 'due to', 'reason', 'cause', 'since', 'therefore', 'thus']
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in reason_words):
                return f"The document explains: {sentence}"
        
        return f"Based on the document: {sentences[0]}"
    
    def _generate_general_response(self, sentences: List[str], key_terms: List[str]) -> str:
        """Generate general response"""
        if not sentences:
            return ""
        
        return f"Here's what I found in the document: {sentences[0]}"
    
    def _create_fallback_response(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Create a fallback response when generation fails"""
        if context_docs:
            response = f"Based on the retrieved documents, here's relevant information:\n\n"
            for i, doc in enumerate(context_docs[:2], 1):
                response += f"{i}. From {doc['source']}: {doc['text'][:200]}...\n\n"
            response += "Please refer to the source context above for more detailed information."
            return response
        else:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the generator"""
        return {
            "model_name": "SimpleResponseGenerator",
            "type": "template-based",
            "available": True
        }
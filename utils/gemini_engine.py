import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
import streamlit as st

class GeminiEngine:
    """
    Chat engine using Google's Gemini AI for response generation
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini chat engine"""
        self.model_name = model_name
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Gemini model with API key"""
        try:
            # Get API key from environment or secrets
            api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            
            if not api_key:
                st.error("❌ Google API key not found. Please set GOOGLE_API_KEY in your environment.")
                return False
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing Gemini: {str(e)}")
            return False
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], max_length: int = 500) -> str:
        """Generate response using Gemini with retrieved context"""
        
        if not self.model:
            return "❌ Gemini model not initialized. Please check your API key."
        
        try:
            # Prepare context from retrieved documents
            context_text = ""
            for i, doc in enumerate(context_docs[:3]):  # Use top 3 most relevant documents
                context_text += f"Document {i+1} (from {doc.get('source', 'unknown')}):\n{doc['text']}\n\n"
            
            # Create prompt for Gemini
            prompt = f"""Based on the following documents, please answer the user's question accurately and comprehensively.

Context Documents:
{context_text}

User Question: {query}

Instructions:
- Use only the information provided in the context documents
- If the answer isn't in the documents, say so clearly
- Provide specific details and examples when available
- Keep the response focused and relevant to the question
- Maximum length: {max_length} characters

Answer:"""

            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                generated_text = response.text.strip()
                
                # Ensure response isn't too long
                if len(generated_text) > max_length:
                    generated_text = generated_text[:max_length-3] + "..."
                
                return generated_text
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Gemini is properly configured and available"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "status": "Available" if self.is_available() else "Not Available",
            "provider": "Google Gemini"
        }
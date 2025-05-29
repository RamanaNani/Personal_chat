import os
import openai
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self, api_key: str = None):
        """Initialize the QA engine."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        openai.api_key = self.api_key
        # Set timeout for API calls
        openai.timeout = 20  # Reduced timeout to 20 seconds

    def prepare_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare context from search results."""
        if not search_results['documents'] or not search_results['documents'][0]:
            return ""

        # Limit context to first 2 most relevant chunks to reduce token usage
        max_chunks = 2
        context_parts = []
        
        for i, chunk in enumerate(search_results['documents'][0][:max_chunks]):
            if len(chunk) > 1000:  # Limit chunk size
                chunk = chunk[:1000] + "..."
            context_parts.append(
                f"Document: {search_results['metadatas'][0][i]['filename']} "
                f"(part {search_results['metadatas'][0][i]['chunk_index'] + 1}/"
                f"{search_results['metadatas'][0][i]['total_chunks']})\n{chunk}"
            )
        
        return "\n\n".join(context_parts)

    def get_answer(self, question: str, context: str) -> str:
        """Get answer from OpenAI based on the context."""
        if not context:
            return "No relevant information found in the documents."

        try:
            logger.info(f"Making API call for question: {question[:50]}...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly and helpful assistant that answers questions based ONLY on the provided documents. Keep your answers concise and to the point. If the information is not in the documents, say so. If the question is about projects, mention that GitHub links are available for all projects."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.5,  # Reduced for more focused responses
                max_tokens=150,   # Reduced for faster responses
                timeout=20        # Reduced timeout
            )
            answer = response.choices[0].message.content
            logger.info("Successfully received API response")
            return answer
        except openai.error.Timeout:
            logger.error("OpenAI API request timed out")
            return "The request took too long to process. Please try again with a more specific question."
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return f"Error getting answer: {str(e)}" 
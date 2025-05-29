import os
import openai
from typing import Dict, Any

class QAEngine:
    def __init__(self, api_key: str = None):
        """Initialize the QA engine."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        openai.api_key = self.api_key

    def prepare_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare context from search results."""
        if not search_results['documents'] or not search_results['documents'][0]:
            return ""

        return "\n\n".join([
            f"Document: {search_results['metadatas'][0][i]['filename']} "
            f"(part {search_results['metadatas'][0][i]['chunk_index'] + 1}/"
            f"{search_results['metadatas'][0][i]['total_chunks']})\n{chunk}"
            for i, chunk in enumerate(search_results['documents'][0])
        ])

    def get_answer(self, question: str, context: str) -> str:
        """Get answer from OpenAI based on the context."""
        if not context:
            return "No relevant information found in the documents."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly and helpful assistant that answers questions based ONLY on the provided documents. If the information is not in the documents, say so. If the question is about projects, mention that GitHub links are available for all projects. Additionally, I am here to support you with job-related queries, especially for new graduates in AI or related fields."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting answer: {str(e)}" 
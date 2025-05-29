import json
from dotenv import load_dotenv
from vector_store import VectorStore
from document_processor import DocumentProcessor
from qa_engine import QAEngine

def load_projects():
    """Load project information from projects.json."""
    try:
        with open('projects.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading projects: {str(e)}")
        return {"projects": []}

def main():
    try:
        # Load environment variables
        load_dotenv()

        # Initialize components
        vector_store = VectorStore()
        doc_processor = DocumentProcessor()
        qa_engine = QAEngine()

        # Load project information
        projects_data = load_projects()

        # Only process and add documents if the collection is empty
        if vector_store.is_empty():
            print("No data in vector store. Processing and adding documents...")
            chunks, metadatas = doc_processor.process_documents()
            if not chunks:
                print("No documents found in the Documents folder.")
                return
            vector_store.add_documents(chunks, metadatas)
            print(f"Processed {len(chunks)} document chunks.")
        else:
            print("Vector store already contains data. Skipping document processing.")

        # Interactive Q&A loop
        while True:
            question = input("\nEnter your question (or 'quit' to exit, 'update' to reprocess documents): ")
            if question.lower() == 'quit':
                break
            if question.lower() == 'update':
                print("Reprocessing and updating documents...")
                vector_store.clear()
                chunks, metadatas = doc_processor.process_documents()
                vector_store.add_documents(chunks, metadatas)
                print(f"Updated with {len(chunks)} document chunks.")
                continue

            # Search for relevant chunks
            search_results = vector_store.search(question, n_results=5)
            
            # Prepare context and get answer
            context = qa_engine.prepare_context(search_results)
            print("\n--- Retrieved Context ---")
            print(context)
            print("--- End of Context ---\n")

            # Include project information if the question is about projects
            if "project" in question.lower():
                context += "\n\nProject Information:\n"
                for project in projects_data["projects"]:
                    context += f"{project['name']}: {project['description']} GitHub: {project['github_link']}\n"

            answer = qa_engine.get_answer(question, context)
            print("\nAnswer:", answer)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
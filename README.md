# Personal Chat

This project is a document-based Q&A system that uses OpenAI's API to answer questions based on provided documents. It processes both text and PDF documents, manages large document contexts, and utilizes a vector database (ChromaDB) for improved search capabilities.

## Features

- Process and query text and PDF documents.
- Use ChromaDB for efficient document retrieval.
- Interactive Q&A interface.
- Support for project-related queries with GitHub links.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RamanaNani/Personal_chat.git
   cd Personal_chat
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the main script to start the interactive Q&A session:
```bash
python main.py
```

- Enter your questions, and the system will provide answers based on the documents.
- Type `update` to reprocess and update the documents.
- Type `quit` to exit the session.

## Project Structure

- `main.py`: Entry point for the application.
- `document_processor.py`: Handles document reading and processing.
- `vector_store.py`: Manages vector database operations.
- `qa_engine.py`: Manages question-answering logic.
- `projects.json`: Contains information about projects and their GitHub links.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
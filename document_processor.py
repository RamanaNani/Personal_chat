from pathlib import Path
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, docs_folder: str = "Documents"):
        """Initialize the document processor."""
        self.docs_folder = Path(docs_folder)

    def is_text_file(self, file_path: Path) -> bool:
        """Check if the file is likely a text file."""
        if file_path.name.startswith('.') or file_path.name == 'DS_Store':
            return False
        
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml', '.yaml', '.yml'}
        return file_path.suffix.lower() in text_extensions

    def is_pdf_file(self, file_path: Path) -> bool:
        """Check if the file is a PDF file."""
        return file_path.suffix.lower() == '.pdf'

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            logger.info(f"Processing PDF file: {file_path.name}")
            reader = PdfReader(str(file_path))
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += f"\nPage {i+1}:\n{page_text}"
            logger.info(f"Successfully extracted text from {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def process_documents(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process all documents and return chunks with metadata."""
        if not self.docs_folder.exists():
            raise FileNotFoundError("Documents folder not found!")

        all_chunks = []
        all_metadatas = []

        for file_path in self.docs_folder.glob('**/*'):
            if file_path.is_file():
                content = ""
                if self.is_text_file(file_path):
                    try:
                        logger.info(f"Processing text file: {file_path.name}")
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        logger.info(f"Successfully read text file: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {str(e)}")
                elif self.is_pdf_file(file_path):
                    content = self.extract_text_from_pdf(file_path)

                if content:
                    chunks = self.split_text_into_chunks(content)
                    logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "filename": file_path.name,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        })

        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks, all_metadatas 
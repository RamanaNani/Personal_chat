from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qa_engine import QAEngine
from vector_store import VectorStore
from document_processor import DocumentProcessor
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Chat API",
    description="A document-based Q&A system using OpenAI's API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
doc_processor = DocumentProcessor()
qa_engine = QAEngine()

class Question(BaseModel):
    text: str

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Personal Chat API",
        "documentation": "/docs",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/update": "POST - Update documents",
            "/health": "GET - Health check"
        }
    }

@app.post("/ask")
async def ask_question(question: Question):
    """Ask a question and get an answer based on the documents"""
    try:
        start_time = time.time()
        logger.info(f"Processing question: {question.text}")

        # Process documents if the collection is empty
        if vector_store.is_empty():
            logger.info("Vector store is empty, processing documents...")
            chunks, metadatas = doc_processor.process_documents()
            if not chunks:
                raise HTTPException(status_code=404, detail="No documents found in the Documents folder.")
            vector_store.add_documents(chunks, metadatas)
            logger.info(f"Added {len(chunks)} document chunks")

        # Search for relevant chunks
        search_results = vector_store.search(question.text, n_results=3)
        
        # Prepare context and get answer
        context = qa_engine.prepare_context(search_results)
        answer = qa_engine.get_answer(question.text, context)
        
        process_time = time.time() - start_time
        logger.info(f"Question processed in {process_time:.2f} seconds")
        
        return {
            "answer": answer,
            "processing_time": f"{process_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def update_documents():
    """Update the document collection"""
    try:
        start_time = time.time()
        logger.info("Starting document update")
        
        vector_store.clear()
        chunks, metadatas = doc_processor.process_documents()
        vector_store.add_documents(chunks, metadatas)
        
        process_time = time.time() - start_time
        logger.info(f"Documents updated in {process_time:.2f} seconds")
        
        return {
            "message": f"Updated with {len(chunks)} document chunks.",
            "processing_time": f"{process_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Error updating documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
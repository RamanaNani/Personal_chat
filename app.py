from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qa_engine import QAEngine
from vector_store import VectorStore
from document_processor import DocumentProcessor
from dotenv import load_dotenv
import os
import time
import logging

# Load environment variables
load_dotenv()

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

# Initialize components lazily to avoid startup errors
vector_store = None
doc_processor = None
qa_engine = None

def get_components():
    """Initialize components if not already initialized"""
    global vector_store, doc_processor, qa_engine
    if vector_store is None:
        vector_store = VectorStore()
        doc_processor = DocumentProcessor()
        qa_engine = QAEngine()
    return vector_store, doc_processor, qa_engine

@app.on_event("startup")
async def startup_event():
    """Initialize components and process documents at startup"""
    logger.info("Initializing components at startup...")
    try:
        # Just initialize components, don't process documents automatically
        vector_store, doc_processor, qa_engine = get_components()
        logger.info("Components initialized, ready for requests!")
        
        # Uncomment the following lines if you want automatic document processing at startup
        # This is commented out to prevent container crashes on startup
        # if vector_store.is_empty():
        #     logger.info("Processing documents at startup...")
        #     chunks, metadatas = doc_processor.process_documents()
        #     if chunks:
        #         vector_store.add_documents(chunks, metadatas)
        #         logger.info(f"Startup: Processed {len(chunks)} document chunks")
        #     else:
        #         logger.warning("No documents found in the Documents folder")
        # else:
        #     logger.info("Vector store already contains data")
        
        logger.info("Startup initialization complete!")
    except Exception as e:
        logger.error(f"Error during startup initialization: {str(e)}")
        # Don't fail startup, let the app handle it later
        pass

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

@app.get("/warmup")
async def warmup():
    """Warmup endpoint to initialize components manually"""
    try:
        logger.info("Manual warmup requested...")
        vector_store, doc_processor, qa_engine = get_components()
        
        # Process documents if the collection is empty
        if vector_store.is_empty():
            logger.info("Processing documents during warmup...")
            chunks, metadatas = doc_processor.process_documents()
            if chunks:
                vector_store.add_documents(chunks, metadatas)
                logger.info(f"Warmup: Processed {len(chunks)} document chunks")
                return {
                    "status": "warmup_complete",
                    "message": f"Processed {len(chunks)} document chunks",
                    "ready": True
                }
            else:
                return {
                    "status": "no_documents",
                    "message": "No documents found in the Documents folder",
                    "ready": False
                }
        else:
            count = vector_store.collection.count()
            return {
                "status": "already_ready",
                "message": f"Vector store already contains {count} documents",
                "ready": True
            }
    except Exception as e:
        logger.error(f"Error during warmup: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "ready": False
        }

@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Personal Chat API",
        "documentation": "/docs",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/update": "POST - Update documents",
            "/health": "GET - Health check",
            "/warmup": "GET - Warmup"
        }
    }

@app.post("/ask")
async def ask_question(question: Question):
    """Ask a question and get an answer based on the documents"""
    try:
        start_time = time.time()
        logger.info(f"Processing question: {question.text}")

        # Get initialized components
        vector_store, doc_processor, qa_engine = get_components()

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
        
        # Get initialized components
        vector_store, doc_processor, qa_engine = get_components()
        
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
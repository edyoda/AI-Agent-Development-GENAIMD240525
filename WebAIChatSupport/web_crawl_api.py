from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import asyncio
import os
from pathlib import Path
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import aiofiles
import aiohttp
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from playwright.async_api import async_playwright
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class IndexRequest(BaseModel):
    urls: Optional[List[str]] = None
    force_reindex: bool = False

class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class IndexResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int

class SimpleDocumentIndexer:
    """Basic document indexer for AutoGen Memory."""

    def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
        self.memory = memory
        self.chunk_size = chunk_size

    async def _fetch_content(self, source: str) -> str:
        """Fetch fully rendered content from URL using Playwright."""
        if source.startswith(("http://", "https://")):
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(source, wait_until="networkidle")
                content = await page.content()
                await browser.close()
                return content
        else:
           async with aiofiles.open(source, "r", encoding="utf-8") as f:
               return await f.read()

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks: list[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk.strip())
        return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0

        for source in sources:
            try:
                content = await self._fetch_content(source)

                if "<" in content and ">" in content:
                    content = self._strip_html(content)

                chunks = self._split_text(content)

                for i, chunk in enumerate(chunks):
                    await self.memory.add(
                        MemoryContent(
                            content=chunk, 
                            mime_type=MemoryMimeType.TEXT, 
                            metadata={"source": source, "chunk_index": i}
                        )
                    )

                total_chunks += len(chunks)
                print(f"Indexed {len(chunks)} chunks from {source}")

            except Exception as e:
                print(f"Error indexing {source}: {str(e)}")

        return total_chunks

def extract_urls_from_domain(domain: str) -> List[str]:
    """Extracts all URLs from a given domain."""
    urls = set()
    try:
        response = requests.get(domain)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(domain, href)
            if full_url.startswith(domain):
                urls.add(full_url)

    except requests.exceptions.RequestException as e:
        print(f"Error while fetching {domain}: {e}")

    return list(urls)

# Initialize FastAPI app
app = FastAPI(
    title="EdYoda RAG Assistant API",
    description="A FastAPI application for querying EdYoda courses using RAG",
    version="1.0.0"
)

# Global variables
persistence_path = os.path.join(str(Path.home()), ".chromadb_autogen")
rag_memory = None
rag_assistant = None
oai_model_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_memory, rag_assistant, oai_model_client
    
    print(f"Using persistence path: {persistence_path}")
    
    # Initialize vector memory
    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="edyoda_courses",
            persistence_path=persistence_path,
            k=3,
            score_threshold=0.4,
        )
    )
    
    # Initialize OpenAI client
    oai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        temperature=0.85,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Create RAG assistant
    rag_assistant = AssistantAgent(
        name="rag_assistant", 
        model_client=oai_model_client, 
        memory=[rag_memory]
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global rag_memory
    if rag_memory:
        try:
            await rag_memory.close()
        except Exception as e:
            print(f"Error closing memory: {e}")

async def check_existing_documents() -> int:
    """Check if documents are already indexed."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persistence_path)
        
        try:
            collection = client.get_collection("edyoda_courses")
            count = collection.count()
            return count
        except Exception:
            return 0
            
    except Exception as e:
        print(f"Error checking existing documents: {e}")
        return 0

async def perform_indexing(urls: List[str]) -> int:
    """Perform document indexing."""
    indexer = SimpleDocumentIndexer(memory=rag_memory)
    chunks = await indexer.index_documents(urls)
    return chunks

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "EdYoda RAG Assistant API",
        "endpoints": {
            "ask": "/ask - Ask questions about EdYoda courses",
            "index": "/index - Index documents",
            "status": "/status - Check indexing status"
        }
    }

@app.get("/status")
async def get_status():
    """Get the current status of the indexed documents."""
    document_count = await check_existing_documents()
    return {
        "status": "ready" if document_count > 0 else "not_indexed",
        "indexed_documents": document_count,
        "persistence_path": persistence_path
    }

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index documents from URLs."""
    try:
        # Check existing documents
        existing_docs = await check_existing_documents()
        
        if existing_docs > 0 and not request.force_reindex:
            return IndexResponse(
                status="skipped",
                message=f"Documents already indexed ({existing_docs} documents). Use force_reindex=true to reindex.",
                chunks_indexed=0
            )
        
        # Get URLs to index
        if request.urls:
            urls_to_index = request.urls
        else:
            # Default: extract URLs from EdYoda domain
            all_urls = extract_urls_from_domain("https://www.edyoda.com")
            urls_to_index = [url for url in all_urls if url.rstrip('/').endswith('micro-degree')]
        
        if not urls_to_index:
            raise HTTPException(status_code=400, detail="No URLs to index")
        
        # Perform indexing in background
        chunks_indexed = await perform_indexing(urls_to_index)
        
        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(urls_to_index)} URLs",
            chunks_indexed=chunks_indexed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about EdYoda courses."""
    try:
        # Check if documents are indexed
        document_count = await check_existing_documents()
        if document_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No documents indexed. Please call /index endpoint first."
            )
        
        if not rag_assistant:
            raise HTTPException(status_code=500, detail="RAG assistant not initialized")
        
        # Get answer from RAG assistant
        response_text = ""
        stream = rag_assistant.run_stream(task=request.question)
        
        # Collect the streamed response
        async for chunk in stream:
            if hasattr(chunk, 'content') and chunk.content:
                response_text += chunk.content
            elif isinstance(chunk, str):
                response_text += chunk
        
        if not response_text:
            response_text = "I couldn't find relevant information to answer your question."
        
        return QuestionResponse(
            answer=response_text.strip(),
            sources=None  # You can enhance this to extract sources from metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/extract-urls")
async def extract_edyoda_urls():
    """Extract all micro-degree URLs from EdYoda domain."""
    try:
        all_urls = extract_urls_from_domain("https://www.edyoda.com")
        filtered_urls = [url for url in all_urls if url.rstrip('/').endswith('micro-degree')]
        
        return {
            "total_urls": len(all_urls),
            "micro_degree_urls": len(filtered_urls),
            "urls": filtered_urls
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting URLs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )

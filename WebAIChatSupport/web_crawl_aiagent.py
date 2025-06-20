import requests, json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from typing import List
import aiofiles
import aiohttp
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

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
        # Just split text into fixed-size chunks
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

                # Strip HTML if content appears to be HTML
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


def extract_urls_from_domain(domain):
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

all_urls = extract_urls_from_domain("https://www.edyoda.com")
filtered_urls = [url for url in all_urls if url.rstrip('/').endswith('micro-degree')]


import os
from pathlib import Path
import os, asyncio, requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Initialize vector memory with proper persistence
persistence_path = os.path.join(str(Path.home()), ".chromadb_autogen")
print(f"Using persistence path: {persistence_path}")

rag_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="edyoda_courses",  # Changed to match your use case
        persistence_path=persistence_path,
        k=3,  # Return top 3 results
        score_threshold=0.4,  # Minimum similarity score
    )
)

# Index documents only if needed
async def index_documents_if_needed() -> None:
    """Check if documents are already indexed, if not, index them."""
    
    # Check if collection exists and has documents
    try:
        # Try to access the collection to see if it exists and has content
        collection_info = await rag_memory._get_collection_info()
        document_count = collection_info.get('count', 0) if collection_info else 0
        
        print(f"Found {document_count} existing documents in memory")
        
        if document_count > 0:
            print("Using existing indexed documents")
            return
            
    except Exception as e:
        print(f"Collection doesn't exist or error checking: {e}")
    
    # If we reach here, we need to index
    print("No existing documents found. Starting indexing...")
    
    indexer = SimpleDocumentIndexer(memory=rag_memory)
    
    sources = filtered_urls
    
    chunks = await indexer.index_documents(sources)
    print(f"Successfully indexed {chunks} chunks from {len(sources)} documents")


# Alternative approach using ChromaDB client directly for checking
async def check_and_index_alternative() -> None:
    """Alternative method to check if indexing is needed."""
    try:
        # Try to get collection directly from ChromaDB
        import chromadb
        
        # Create client with same persistence path
        client = chromadb.PersistentClient(path=persistence_path)
        
        try:
            collection = client.get_collection("edyoda_courses")
            count = collection.count()
            print(f"Found existing collection with {count} documents")
            
            if count > 0:
                print("Using existing indexed documents")
                return
                
        except Exception:
            print("Collection doesn't exist, will create new one")
        
        # Index documents
        print("Starting document indexing...")
        indexer = SimpleDocumentIndexer(memory=rag_memory)
        
        sources = filtered_urls
        
        chunks = await indexer.index_documents(sources)
        print(f"Successfully indexed {chunks} chunks from {len(sources)} documents")
        
    except Exception as e:
        print(f"Error in check_and_index_alternative: {e}")
        # Fallback to always indexing
        await index_documents_if_needed()


oai_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    temperature=0.85,
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def main():
    try:
        # Check and index only if needed
        await check_and_index_alternative()
        
        # Create our RAG assistant agent
        rag_assistant = AssistantAgent(
            name="rag_assistant", 
            model_client=oai_model_client, 
            memory=[rag_memory]
        )

        # Ask questions
        print("\n" + "="*50)
        print("Starting RAG Assistant Query...")
        print("="*50)

        print("What do you want to know about courses of EdYoda? : ")
        asked_question = input()
        stream = rag_assistant.run_stream(
            #task="What will I learn in robotics course? Also, include the source URL from where the answer was found."
            task = asked_question
        )
        await Console(stream)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Always close the memory when done
        try:
            await rag_memory.close()
        except Exception as e:
            print(f"Error closing memory: {e}")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

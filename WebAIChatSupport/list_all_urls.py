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

def extract_urls_playwright(domain):
    urls = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(domain)
        page.wait_for_load_state("networkidle")

        links = page.query_selector_all("a[href]")
        for link in links:
            href = link.get_attribute("href")
            if href and href.startswith("/"):
                href = urljoin(domain, href)
            if href and href.startswith(domain):
                urls.add(href)

        browser.close()
    return list(urls)

# Usage
urls = extract_urls_playwright("https://www.edyoda.com")
print(urls)


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
    '''
    async def _fetch_content(self, source: str) -> str:
        """Fetch content from URL or file."""
        if source.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    return await response.text()
        else:
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                return await f.read()
    '''

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
                            content=chunk, mime_type=MemoryMimeType.TEXT, metadata={"source": source, "chunk_index": i}
                        )
                    )

                total_chunks += len(chunks)

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
print(filtered_urls)


import os
from pathlib import Path
import os, asyncio, requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Initialize vector memory

rag_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="autogen_docs",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
        k=3,  # Return top 3 results
        score_threshold=0.4,  # Minimum similarity score
    )
)

#rag_memory.clear()  # Clear existing memory


# Index AutoGen documentation
async def index_autogen_docs() -> None:
    indexer = SimpleDocumentIndexer(memory=rag_memory)
    
    '''
    sources = [
         "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/agents.html",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html",
        "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html",
    ]
    '''
    sources = [
            "https://www.edyoda.com/micro-degree/business-analyst-micro-degree",
            "https://www.edyoda.com/micro-degree/multi-cloud-devops-career-track-micro-degree",
            ]
    chunks: int = await indexer.index_documents(sources)
    print(f"Indexed {chunks} chunks from {len(sources)} AutoGen documents")

try:
    collection = rag_memory._client.get_collection(rag_memory.config.collection_name)
    if collection.count() > 0:
        print(f"Using existing memory with {collection.count()} items")
    else:
        print("Memory exists but is empty. Indexing documents...")
        asyncio.run(index_autogen_docs())
except Exception as e:
    print("Memory collection not found. Creating new one and indexing...")
    asyncio.run(index_autogen_docs())
    rag_memory.dump_component().model_dump_json()


#asyncio.run(index_autogen_docs())

oai_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    temperature=0.85,
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def main():
    # Create our RAG assistant agent
    rag_assistant = AssistantAgent(
        name="rag_assistant", model_client=oai_model_client, memory=[rag_memory]
    )

    # Ask questions about AutoGen
    stream = rag_assistant.run_stream(task="What all is the course content of multi-cloud career track?")
    await Console(stream)

    # Remember to close the memory when done
    await rag_memory.close()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

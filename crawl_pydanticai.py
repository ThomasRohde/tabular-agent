import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_title_and_summary(content: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation pages.
    Return a JSON object with 'title' and 'summary' keys.
    Create a clear title that represents the main topic of the page.
    For the summary: Create a concise summary of the main points in this page.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{content[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def process_page(url: str, content: str) -> Dict[str, Any]:
    """Process a single page."""
    # Get title and summary
    extracted = await get_title_and_summary(content, url)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "content_length": len(content),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return {
        "url": url,
        "title": extracted['title'],
        "summary": extracted['summary'],
        "content": content,
        "metadata": metadata
    }

def write_to_main_file(pages: List[Dict[str, Any]]):
    """Write all pages to a single markdown file."""
    try:
        filename = "pydantic_ai_docs.md"
        
        content = "# Pydantic AI Documentation\n\n"
        content += f"Generated on: {datetime.now(timezone.utc).isoformat()}\n\n"
        
        for page in pages:
            content += f"# {page['title']}\n\n"
            content += f"URL: {page['url']}\n\n"
            content += "## Summary\n"
            content += f"{page['summary']}\n\n"
            content += "## Content\n"
            content += f"{page['content']}\n\n"
            content += "## Metadata\n"
            content += "```json\n"
            content += f"{json.dumps(page['metadata'], indent=2)}\n"
            content += "```\n\n"
            content += "---\n\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Written all pages to {filename}")
    except Exception as e:
        print(f"Error writing main file: {e}")

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        processed_pages = []
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    page = await process_page(url, result.markdown_v2.raw_markdown)
                    processed_pages.append(page)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
        
        # Write all processed pages to a single file
        write_to_main_file(processed_pages)
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
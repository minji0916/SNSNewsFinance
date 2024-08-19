import asyncio
import aiohttp
from newspaper import Article
from fake_useragent import UserAgent
import pandas as pd
import psutil
import time
import chardet
from concurrent.futures import ProcessPoolExecutor
import os
from dotenv import load_dotenv
import logging
from config import setup_logging
import re

# 로그 설정
setup_logging()

# Load environment variables
load_dotenv()

# Load configuration from environment variables
CONCURRENCY = int(os.getenv('CONCURRENCY', 10))  
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 30))  
TIMEOUT = int(os.getenv('TIMEOUT', 30))  
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 2))  # 최대 재시도 횟수
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 1))  # 재시도 사이의 대기 시간(초)

ua = UserAgent()

async def fetch(session, url, timeout=TIMEOUT, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': ua.random}
            async with session.get(url, headers=headers, timeout=timeout, ssl=False) as response:
                content = await response.read()
                encoding = chardet.detect(content)['encoding'] or 'utf-8'
                return content.decode(encoding, errors='replace')
        except asyncio.TimeoutError:
            logging.warning(f"Timeout error for {url}, attempt {attempt + 1}/{max_retries}")
        except aiohttp.ClientError as e:
            logging.warning(f"Client error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
        except Exception as e:
            logging.warning(f"Failed to fetch {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(RETRY_DELAY)
    
    logging.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

def parse_article(html, url):
    if html:
        try:
            article = Article(url, language='ko')
            article.download(input_html=html)
            article.parse()
            return article.text
        except Exception as e:
            logging.error(f"Failed to parse {url}: {str(e)}")
    return None

async def extract_news_content(session, url):
    if not isinstance(url, str) or not url.startswith('http'):
        logging.warning(f"Invalid URL: {url}")
        return None

    html = await fetch(session, url)
    if html:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            return await loop.run_in_executor(pool, parse_article, html, url)
    return None

async def process_chunk(chunk, sem):
    conn = aiohttp.TCPConnector(ssl=False, limit=None)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.ensure_future(scrape_with_semaphore(url, session, sem)) for url in chunk]
        return await asyncio.gather(*tasks)

async def scrape_with_semaphore(url, session, sem):
    async with sem:
        return await extract_news_content(session, url)

async def scrape_urls(urls, chunk_size=CHUNK_SIZE, concurrency=CONCURRENCY):
    all_results = []
    sem = asyncio.Semaphore(concurrency)

    chunks = [urls[i:i+chunk_size] for i in range(0, len(urls), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        results = await process_chunk(chunk, sem)
        all_results.extend(results)
        
        logging.info(f"Processed {len(all_results)} out of {len(urls)} URLs")
        logging.info(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")
        
        if i < len(chunks) - 1:
            await asyncio.sleep(1)  # 각 청크 처리 후 1초 대기

    return all_results

def load_urls(csv_file_path):
    return pd.read_csv(csv_file_path)['Original Link'].tolist()

def clean_text(text):
    if text is None:
        return None
    
    # 광고 관련 텍스트 제거
    ad_patterns = [r'\b광고\b', r'\bAD\b', r'\b스폰서드\b', r'\bsponsored\b']
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 연속된 줄바꿈 제거 및 단락 구분
    text = re.sub(r'\n+', '\n\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def save_results(urls, contents, csv_file_path):
    cleaned_contents = [clean_text(content) for content in contents]
    pd.DataFrame({'url': urls, 'content': cleaned_contents}).to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    logging.info(f"Results saved to {csv_file_path}")

async def main():
    logging.info("Starting web scraping process...")
    start_time = time.time()

    urls = load_urls("data/raw_data.csv")
    logging.info(f"Loaded {len(urls)} URLs")

    scraped_contents = await scrape_urls(urls)

    save_results(urls, scraped_contents, "data/newspapker3k_scraped_results.csv")

    total_time = time.time() - start_time
    logging.info(f"Web scraping process completed. Total time taken: {total_time:.2f} seconds")
    logging.info(f"Average time per URL: {total_time/len(urls):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
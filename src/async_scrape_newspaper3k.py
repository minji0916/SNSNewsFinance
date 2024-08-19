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
import ssl

# 로그 설정
setup_logging()

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 설정 값 로드
CONCURRENCY = int(os.getenv('CONCURRENCY', 10))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 30))
TIMEOUT = int(os.getenv('TIMEOUT', 30))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 2))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 1))

# User-Agent 랜덤 생성기 설정
ua = UserAgent()

# SSL 컨텍스트 생성 함수
def create_ssl_context():
    context = ssl.create_default_context()
    context.set_ciphers('DEFAULT@SECLEVEL=1')  # 낮은 보안 수준의 암호도 허용
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1
    return context

async def fetch(session, url, timeout=TIMEOUT, max_retries=MAX_RETRIES):
    ssl_context = create_ssl_context()
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': ua.random}
            async with session.get(url, headers=headers, timeout=timeout, ssl=ssl_context) as response:
                content = await response.read()
                encoding = chardet.detect(content)['encoding'] or 'utf-8'
                return content.decode(encoding, errors='replace')
        except (asyncio.TimeoutError, aiohttp.ClientError, ssl.SSLError) as e:
            logging.warning(f"Error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
            if isinstance(e, ssl.SSLError) and attempt < max_retries - 1:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
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
            cleaned_text = clean_text(article.text)
            return cleaned_text
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
    conn = aiohttp.TCPConnector(ssl=create_ssl_context(), limit=None)
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
            await asyncio.sleep(1)

    return all_results

def clean_text(text):
    if text is None:
        return None
    
    ad_patterns = [r'\b광고\b', r'\bAD\b', r'\b스폰서드\b', r'\bsponsored\b']
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    
    return text

async def update_content(df):
    urls_to_scrape = df[df['Content'].isna()]['Original Link'].tolist()
    logging.info(f"Found {len(urls_to_scrape)} URLs to scrape")

    if urls_to_scrape:
        scraped_contents = await scrape_urls(urls_to_scrape)
        
        for url, content in zip(urls_to_scrape, scraped_contents):
            df.loc[df['Original Link'] == url, 'Content'] = content

    return df

async def main():
    logging.info("Starting web scraping process...")
    start_time = time.time()

    df = pd.read_csv("data/raw_data.csv")
    if 'Content' not in df.columns:
        df['Content'] = pd.NA

    updated_df = await update_content(df)
    updated_df.to_csv("data/raw_data_updated.csv", index=False, encoding='utf-8-sig')

    total_time = time.time() - start_time
    logging.info(f"Web scraping process completed. Total time taken: {total_time:.2f} seconds")
    logging.info(f"Results saved to data/raw_data_updated.csv")

if __name__ == "__main__":
    asyncio.run(main())
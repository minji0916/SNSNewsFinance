import logging
from dotenv import load_dotenv
import os
import json
import time

# .env 파일 로드
load_dotenv()

# 네이버 API의 클라이언트 ID와 시크릿 키 설정
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Naver news API 호출 시 설정
SEARCH_QUERIES = json.loads(os.getenv("SEARCH_QUERIES", '["주식", "채권"]'))
DISPLAY_COUNT = int(os.getenv("DISPLAY_COUNT", 10))
START_INDEX = int(os.getenv("START_INDEX", 1))
END_INDEX = int(os.getenv("END_INDEX", 300))
SORT_ORDER = os.getenv("SORT_ORDER", "sim")

# 로그 설정
LOG_FILE = os.getenv("LOG_FILE", 'news_project.log')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 로그 파일에 구분선 추가
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"New scraping session started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")

# DB 설정
MYSQL_HOST = os.getenv("DB_HOST", "localhost")
MYSQL_PORT = int(os.getenv("DB_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# 성능 관련 설정
CONCURRENCY = int(os.getenv("CONCURRENCY", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 30))
TIMEOUT = int(os.getenv("TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1))
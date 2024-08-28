from fetch_news import fetch_all_news
import asyncio
from async_scrape_newspaper3k import main as scrape_main
from summarize import summarize_news
from database import create_tables

# aidb 데이터베이스, 테이블 없는 경우 생성
create_tables()

async def main():
    """
    전체 파이프라인을 실행합니다.
    """
    fetch_all_news()  # 뉴스 데이터 가져오기

    # 뉴스 url에서 본문만 스크랩
    await scrape_main()

    # gemma2 사용해 뉴스 분석
    summarize_news()

if __name__ == "__main__":
    asyncio.run(main())

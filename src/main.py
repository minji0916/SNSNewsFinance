from fetch_news import fetch_all_news
from scrape_content import scrape_content_from_csv


def main():
    """
    전체 파이프라인을 실행합니다.
    """
    # fetch_all_news()  # 뉴스 데이터 가져오기
    scrape_content_from_csv() # 뉴스 url에서 본문만 스크랩

if __name__ == "__main__":
    main()

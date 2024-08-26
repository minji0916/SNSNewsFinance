import urllib.request
import json
import re
import html
import urllib.parse
import logging
from config import CLIENT_ID, CLIENT_SECRET, SEARCH_QUERIES, DISPLAY_COUNT, START_INDEX, END_INDEX, SORT_ORDER, setup_logging
from database import save_news_to_database

# 로그 설정
setup_logging()

def fetch_news(query):
    """
    주어진 검색어에 대해 뉴스 데이터를 가져와서 데이터베이스에 저장합니다.
    
    :param query: 검색어
    """
    encoded_query = urllib.parse.quote(query)  # 검색어를 URL 인코딩
    news_data = []
    
    for start_index in range(START_INDEX, END_INDEX, DISPLAY_COUNT):
        url = f"https://openapi.naver.com/v1/search/news?query={encoded_query}&display={DISPLAY_COUNT}&start={start_index}&sort={SORT_ORDER}"
        
        try:
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", CLIENT_ID)
            request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
            response = urllib.request.urlopen(request)
            response_code = response.getcode()
            
            if response_code == 200:
                response_body = response.read()
                response_dict = json.loads(response_body.decode('utf-8'))
                items = response_dict['items']
                
                for item in items:
                    # HTML 태그 제거
                    clean_title = re.sub(re.compile('<.*?>'), '', item['title'])
                    clean_description = re.sub(re.compile('<.*?>'), '', item['description'])
                    clean_pub_date = re.sub(re.compile('<.*?>'), '', item['pubDate'])
                    
                    # 뉴스 항목 추가
                    news_data.append((
                        query,  # 카테고리
                        html.unescape(item['originallink']),  # 원본 링크
                        html.unescape(clean_title),  # 제목
                        html.unescape(clean_description),  # 설명
                        clean_pub_date  # 발행일
                    ))
            else:
                logging.error(f"Error Code: {response_code}")
                error_details = response.read().decode('utf-8')
                logging.error(f"Error Details: {error_details}")

        except urllib.error.HTTPError as e:
            logging.error(f"HTTPError: {e.code} - {e.reason}")
            error_details = e.read().decode('utf-8')
            logging.error(f"Error Details: {error_details}")
    
    # 데이터베이스에 뉴스 데이터 저장
    save_news_to_database(news_data)

def fetch_all_news():
    """
    모든 검색어에 대해 뉴스 데이터를 가져와 데이터베이스에 저장합니다.
    """
    for query in SEARCH_QUERIES:
        fetch_news(query)
    logging.info("모든 뉴스 데이터가 데이터베이스에 저장되었습니다.")

if __name__ == "__main__":
    fetch_all_news()
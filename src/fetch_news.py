import urllib.request
import json
import re
import html
import urllib.parse
import pandas as pd
import logging
from config import CLIENT_ID, CLIENT_SECRET, SEARCH_QUERIES, DISPLAY_COUNT, START_INDEX, END_INDEX, SORT_ORDER, setup_logging
import os

# 로그 설정
setup_logging()

# 뉴스 데이터를 저장할 데이터프레임 초기화
news_df = pd.DataFrame(columns=["Category", "Title", "Original Link", "Description", "Publication Date", "Content"])

def fetch_news(query):
    """
    주어진 검색어에 대해 뉴스 데이터를 가져와서 데이터프레임에 추가합니다.
    
    :param query: 검색어
    """
    global news_df
    encoded_query = urllib.parse.quote(query)  # 검색어를 URL 인코딩
    current_index = len(news_df)  # 현재 데이터프레임의 인덱스
    
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
                    
                    # 데이터프레임에 뉴스 항목 추가
                    news_df.loc[current_index] = [
                        query,  # 카테고리 추가
                        html.unescape(clean_title),  # HTML 엔티티 제거
                        html.unescape(item['originallink']),  # HTML 엔티티 제거
                        html.unescape(clean_description),
                        clean_pub_date
                    ]
                    current_index += 1
            else:
                logging.error(f"Error Code: {response_code}")
                error_details = response.read().decode('utf-8')
                logging.error(f"Error Details: {error_details}")

        except urllib.error.HTTPError as e:
            logging.error(f"HTTPError: {e.code} - {e.reason}")
            error_details = e.read().decode('utf-8')
            logging.error(f"Error Details: {error_details}")

def save_to_csv(filename='data/raw_data.csv'):
    """
    데이터프레임을 CSV 파일로 저장합니다.
    파일이 존재하면 기존 데이터에 추가합니다.
    
    :param filename: 저장할 CSV 파일명
    """
    if os.path.exists(filename):
        # 파일이 존재하면 기존 데이터를 읽어온 후, 새로운 데이터를 추가합니다.
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, news_df], ignore_index=True)
    else:
        # 파일이 없으면 새로 생성합니다.
        updated_df = news_df

    try:
        updated_df.to_csv(filename, index=False)
        logging.info(f"CSV 파일이 '{filename}'로 저장되었습니다.")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")

def fetch_all_news():
    """
    모든 검색어에 대해 뉴스 데이터를 가져옵니다.
    """
    for query in SEARCH_QUERIES:
        fetch_news(query)
    save_to_csv()  # 뉴스 데이터를 CSV 파일로 저장하기

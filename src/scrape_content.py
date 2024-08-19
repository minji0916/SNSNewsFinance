import pandas as pd
import logging
from bs4 import BeautifulSoup
import requests
import certifi
import os
from config import setup_logging

# 로그 설정
setup_logging()

def extract_news_content(url):
    """
    주어진 URL에서 뉴스 콘텐츠를 추출합니다.
    
    :param url: 뉴스 기사 URL
    :return: 뉴스 콘텐츠 텍스트 또는 None
    """
    tags_to_check = ['br', 'p']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=certifi.where())
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
    except requests.RequestException as e:
        logging.error(f"요청 실패: {e}")
        return None
    except Exception as e:
        logging.error(f"파싱 실패: {e}")
        return None
    
    max_tag_count = 0
    news_content_div = None

    for div_limit in range(2, 10):
        for div in soup.find_all('div'):            
            num_nested_divs = len(div.find_all('div'))
            if num_nested_divs >= div_limit:
                continue
            
            tag_count = sum(len(div.find_all(tag)) for tag in tags_to_check)
            
            if tag_count > max_tag_count and len(div.get_text(strip=True)) >= 300:
                max_tag_count = tag_count
                news_content_div = div
        
        if news_content_div:
            content = news_content_div.get_text(strip=True)
            if len(content) >= 300:
                return content
    
    return None

def scrape_content_from_csv(input_csv='data/raw_data.csv', output_csv='data/soup_scraped_results.csv'):
    """
    CSV 파일에서 뉴스 콘텐츠를 스크랩하여 'Content' 컬럼에 저장합니다.
    
    :param input_csv: 입력 CSV 파일명
    :param output_csv: 출력 CSV 파일명
    """
    if not os.path.exists(input_csv):
        logging.error(f"파일이 존재하지 않습니다: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    if 'Content' not in df.columns:
        df['Content'] = None
    
    # 'content'가 비어있는 항목에 대해 콘텐츠를 스크랩합니다.
    for index, row in df[df['Content'].isna()].iterrows():
        url = row['Original Link']
        if pd.notna(url) and url:
            content = extract_news_content(url)
            if content:
                df.at[index, 'Content'] = content
                logging.info(f"스크랩 완료: {url}")
            else:
                df.at[index, 'Content'] = "FAIL"
                logging.warning(f"콘텐츠 추출 실패: {url}")

    # 업데이트된 DataFrame을 CSV 파일로 저장합니다.
    df.to_csv(output_csv, index=False)
    logging.info(f"업데이트된 CSV 파일이 '{output_csv}'로 저장되었습니다.")

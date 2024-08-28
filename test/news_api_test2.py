import urllib.request
import json
import re
import html
import urllib.parse
import logging
from SNSNewsFinance.src.config  import CLIENT_ID, CLIENT_SECRET, DISPLAY_COUNT, START_INDEX, END_INDEX, SORT_ORDER, setup_logging
from database import save_news_to_database

encText = urllib.parse.quote("왜안될까")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
news_data = []

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
        
        # 뉴스 항목 추가
        news_data.append((
            html.unescape(clean_title),  # 제목
            html.unescape(clean_description),  # 설명
        ))

print(news_data)
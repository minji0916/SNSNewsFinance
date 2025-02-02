# 네이버 검색 API 예제 - 블로그 검색
# 네이버 개발자 포럼에서 제공하는 코드를 그대로 가져온 것이기 때문에 안될 수가 없는 코드

import urllib.request
from config  import CLIENT_ID, CLIENT_SECRET

client_id = CLIENT_ID
client_secret = CLIENT_SECRET

encText = urllib.parse.quote("왜안될까")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
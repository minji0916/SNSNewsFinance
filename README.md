# 구현 중 : 최종 프로젝트
통합 SNS 기반 뉴스 및 금융 정보 서비스 : 뉴스 요약 및 분석, 금융 정보 제공 기능을 포함한 SNS 서비스 개발

이 미션에서는 사용자가 뉴스 요약 및 분석, 금융 정보를 제공받을 수 있는 SNS 서비스를 개발합니다. 
사용자에게 최신 뉴스를 요약하여 제공하고, 금융 정보를 시각화 및 분석하여 제공하는 기능을 포함합니다. 
서비스는 뉴스 및 금융 데이터 API와 통합되어야 하며, 사용자 맞춤형 추천 기능도 포함합니다. 
또한, 사용자가 뉴스와 금융 정보를 공유하고 토론할 수 있는 SNS 기능을 갖추어야 합니다.

## 구현 완료 : SNSNE News Scraper and Analyzer - AI
이 프로젝트는 뉴스 데이터를 수집, 저장, 분석하는 파이프라인을 구현합니다. 뉴스 API를 통해 데이터를 가져오고, 웹 스크래핑을 통해 콘텐츠를 수집하며, 데이터베이스에 저장하고 분석합니다.

## 기능
- 뉴스 API를 통한 데이터 수집
- 웹 스크래핑을 통한 뉴스 콘텐츠 수집
- 데이터베이스 저장 및 관리
- Ollma gemma2를 이용한 뉴스 내용 요약

# AI task 정리
1. 자연어 처리 모듈 개발 (뉴스 요약)
2. 머신러닝 모델 훈련 및 적용 (뉴스 및 금융 정보 추천 시스템)

## 파일 구조
```
SNSNE/
├── .vscode/                 # VS Code 설정 디렉토리
├── database/                # DBeaver 사용 : 데이터베이스 관련 파일 디렉토리
├── img/                     # git readme에 올릴 이미지
├── notebooks/               # 데이터 분석 및 실험을 위한 Jupyter 노트북 디렉토리
│   └── analysis.ipynb       # 분석용 Jupyter 노트북
├── src/                     # 소스 코드 디렉토리
│   ├── .env                 # Naver News API(클라이언트 ID, key), DB 관련 설정 등 환경 변수 파일 (gitignore 파일)
│   ├── async_scrape_ne...   # 뉴스 원문 스크래핑 파일 - 비동기로 속도 향상
│   ├── config.py            # 설정 파일 (.env 파일에 저장된 환경 변수를 설정)
│   ├── create_table.sql     # 테이블 생성 (News, UserNewsViews) - 현재, News 테이블만 사용됨
│   ├── database.py          # 데이터베이스 연결 및 데이터 처리 관련 파일
│   ├── fetch_news.py        # 뉴스 데이터를 가져오는 스크립트
│   ├── main.py              # 메인 실행 스크립트
│   ├── streamlit_stream...  # etc : vLLM 참고 코드 (사용 안함)
│   └── summarize.py         # 뉴스 요약 처리 스크립트
├── .gitignore               # Git 무시 파일 목록
├── 최후의수단.txt           # requirements.txt로 패키지 설치가 안되는 경우, 라이브러리 직접 설치
├── docker-compose.yml       # DB 관련 Docker Compose 설정 파일 (MariaDB 이미지)
├── requirements.txt         # 프로젝트 실행에 필요한 Python 라이브러리 목록이 저장된 파일 (pip freeze)
└── news_project.log         # 프로젝트 로그 파일 (gitignore 파일 - main 실행 시, 자동 생성됨)
```

## DB 엔티티 관계도
![alt text](/img/image.png)

## 실행 방법
1. python=3.10.11 가상환경 생성 후, requirements.txt 파일을 이용해 패키지 설치
- requirements.txt로 패키지 설치가 안될 경우, `최후의수단.txt`로 라이브러리 직접 설치
```
# 가상환경 생성 및 가상환경 들어가기
conda create --name myenv python=3.10.11
conda activate myenv

# 패키지 의존성 파일 설치
pip install -r requirements.txt

# 설치 확인
pip list
```

2. Docker를 사용하는 경우, docker-compose.yml 파일을 사용해 MariaDB 사용
    - Docker Desktop 실행
    - 프로젝트 파일에서 docker-compose.yml이 있는지 확인 후, 컨테이너 생성
   ```
       ls
       docker-compose up -d
   ```

2-1. Docker를 사용하지 않는 경우, MariaDB 서버를 로컬 시스템에 설치
    - (참고 : DBeaver를 사용해 MariaDB 연결하면 유지보수 용이)

3. `src/create_table.sql` 코드를 참고해서 MariaDB에 테이블 생성

4. 네이버 API의 클라이언트 ID와 시크릿 키와 DB 관련 설정을 `src/.env`에 입력합니다.

5. Ollma에서 gemma2 모델 다운 및 사용 잘 되는지 확인
```
   ollama run gemma2
```
7. 터미널에서 `src/main.py`를 실행하여 전체 파이프라인을 수행합니다.

## 테스트 방법
실행하는데 오래 걸릴 수 있으므로, `src/.env` 파일에서 해당 부분을 아래 설정으로 변경해줍니다.
```
# Naver news API 호출 시 설정
SEARCH_QUERIES=["주식"]  # 검색할 키워드를 좁힘
DISPLAY_COUNT=5         # 한 페이지에 5개의 뉴스만 반환
START_INDEX=1           # 첫 번째 뉴스부터 시작
END_INDEX=5             # 첫 5개의 뉴스만 반환
SORT_ORDER='date'       # 최신 뉴스 기준으로 정렬
```

 

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
from database import get_news_without_summary, update_news_summary
from config import setup_logging
import logging
import time

# 로깅 설정
setup_logging()

# Ollama를 사용하여 Gemma2 모델 초기화
model = Ollama(model="gemma2:latest")

# 템플릿 문자열 정의
template_string = """
작업: 다음 뉴스 기사를 분석하여 주요 포인트 3가지와 한 줄 인사이트를 추출하세요.

지시사항:
1. 기사의 내용을 면밀히 분석하세요.
2. 가장 중요하고 관련성 높은 정보를 바탕으로 3가지 주요 포인트를 추출하세요.
3. 3가지 포인트를 추출하기에 정보가 충분하지 않다면, 추출할 수 있는 포인트는 기입하고 그 외에는 '정보불충분'이라고 적어주십시오.
4. 각 포인트는 간결하게 한 문장으로 작성하되, 핵심 정보를 포함해야 합니다.
5. 기사 전체를 고려하여 독자에게 가장 유용할 수 있는 한 줄 인사이트를 도출하세요.
6. 인사이트는 기사의 함의나 잠재적 영향을 포함할 수 있습니다.

point_1 : 추출한 주요 포인트 중 첫번째를 작성하세요.
point_2 : 추출한 주요 포인트 중 두번째를 작성하세요.
point_3 : 추출한 주요 포인트 중 세번째를 작성하세요.
insight: 기사 전체를 고려하여 독자에게 가장 유용할 수 있는 한 줄 인사이트를 도출하세요. 인사이트는 기사의 함의나 잠재적 영향을 포함할 수 있습니다.

뉴스 기사: {text}

{format_instructions}
"""

# 출력 결과의 output format
response_schemas = [
    ResponseSchema(name="point_1", description="추출한 주요 포인트 중 첫번째를 작성하세요."),
    ResponseSchema(name="point_2", description="추출한 주요 포인트 중 두번째를 작성하세요."),
    ResponseSchema(name="point_3", description="추출한 주요 포인트 중 세번째를 작성하세요."),
    ResponseSchema(name="insight", description="기사 전체를 고려하여 독자에게 가장 유용할 수 있는 한 줄 인사이트를 도출하세요. 인사이트는 기사의 함의나 잠재적 영향을 포함할 수 있습니다.")
]

# output parser 지정
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# ChatPromptTemplate 정의
prompt_template = PromptTemplate.from_template(template_string)

def summarize_news():
    news_items = get_news_without_summary()
    
    for news in news_items:
        news_content = news['description'] if news['content'] == 'failed' else news['content']
        
        formatted_prompt = prompt_template.format(text=news_content, format_instructions=format_instructions)
        
        try:
            customer_response = model(formatted_prompt)
            output_dict = output_parser.parse(customer_response)
            summary_json = json.dumps(output_dict)
            update_news_summary(news['news_id'], summary_json)
            logging.info(f"뉴스 ID {news['news_id']}의 요약 처리 및 업데이트 완료")
        except Exception as e:
            logging.error(f"뉴스 ID {news['news_id']} 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    logging.info("뉴스 요약 프로세스 시작")
    start_time = time.time()
    summarize_news()
    total_time = time.time() - start_time
    logging.info("뉴스 요약 프로세스 완료")
    logging.info(f"총 처리 시간: {total_time:.2f} 초")
    
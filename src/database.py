import mysql.connector
from mysql.connector import Error
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
import logging

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            passwd=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )
        logging.info("MariaDB 데이터베이스 연결 성공")
    except Error as e:
        logging.error(f"데이터베이스 연결 오류: {e}")
    return connection

def save_news_to_database(news_data):
    connection = create_connection()
    if connection is None:
        return

    cursor = connection.cursor()

    # 중복 체크 및 삽입 SQL
    check_duplicate = "SELECT COUNT(*) FROM News WHERE news_url = %s"
    insert_news = """
    INSERT INTO News (category, news_url, title, description, publication_date)
    VALUES (%s, %s, %s, %s, %s)
    """

    try:
        for news_item in news_data:
            # 중복 체크
            cursor.execute(check_duplicate, (news_item[1],))  # news_url은 두 번째 항목
            result = cursor.fetchone()
            if result[0] == 0:  # 중복되지 않은 경우에만 삽입
                cursor.execute(insert_news, news_item)
                logging.info(f"새 뉴스 항목이 추가됨: {news_item[2]}")  # 제목 로깅
            else:
                logging.info(f"중복된 뉴스 항목 무시됨: {news_item[2]}")  # 제목 로깅

        connection.commit()
        logging.info("뉴스 데이터 저장 완료")
    except Error as e:
        logging.error(f"데이터 저장 오류: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            logging.info("MySQL 연결이 닫혔습니다.")

def get_news_without_content():
    connection = create_connection()
    if connection is None:
        return []

    cursor = connection.cursor()
    query = "SELECT news_url FROM News WHERE content IS NULL OR content = ''"

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return [result[0] for result in results]
    except Error as e:
        logging.error(f"데이터 조회 오류: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def update_news_content(news_url, content):
    connection = create_connection()
    if connection is None:
        return

    cursor = connection.cursor()
    query = "UPDATE News SET content = %s WHERE news_url = %s"

    try:
        cursor.execute(query, (content, news_url))
        connection.commit()
        logging.info(f"뉴스 내용 업데이트 완료: {news_url}")
    except Error as e:
        logging.error(f"데이터 업데이트 오류: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_news_without_summary():
    connection = create_connection()
    if connection is None:
        return []

    cursor = connection.cursor(dictionary=True)
    query = """
    SELECT news_id, content, description
    FROM News
    WHERE (summary IS NULL OR summary = '') AND content IS NOT NULL
    """
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        logging.error(f"데이터 조회 오류: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def update_news_summary(news_id, summary):
    connection = create_connection()
    if connection is None:
        return

    cursor = connection.cursor()
    query = "UPDATE News SET summary = %s WHERE news_id = %s"

    try:
        cursor.execute(query, (summary, news_id))
        connection.commit()
        logging.info(f"뉴스 요약 업데이트 완료: ID {news_id}")
    except Error as e:
        logging.error(f"데이터 업데이트 오류: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
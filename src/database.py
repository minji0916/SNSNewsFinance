# database.py
import mysql.connector
from mysql.connector import Error
import logging
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

def connect_to_database():
    """
    MySQL 데이터베이스에 연결을 시도하고, 연결 객체를 반환합니다.
    연결에 실패하면 None을 반환합니다.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            logging.info("MySQL database connection successful")
            return connection
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")
        return None

def save_to_database(news_df):
    """
    뉴스 데이터를 MySQL 데이터베이스에 저장합니다.

    :param news_df: 저장할 뉴스 데이터가 포함된 DataFrame
    """
    connection = connect_to_database()
    if connection is None:
        return

    cursor = connection.cursor()

    for _, row in news_df.iterrows():
        try:
            # Insert news data into the News table
            insert_news_query = """
            INSERT INTO News (category, news_url, title, description, publication_date)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_news_query, (
                row['Category'],
                row['Original Link'],
                row['Title'],
                row['Description'],
                row['Publication Date']
            ))

            # Commit the transaction
            connection.commit()

        except Error as e:
            logging.error(f"Failed to insert data into MySQL table: {e}")
            connection.rollback()

    # Close the cursor and connection
    cursor.close()
    connection.close()
    logging.info("MySQL connection is closed")

import mysql.connector
from mysql.connector import Error, errorcode
from datetime import datetime

def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return connection

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

def create_tables(connection):
    create_news_table = """
    CREATE TABLE IF NOT EXISTS News (
        news_id INT AUTO_INCREMENT PRIMARY KEY,
        Category TEXT,
        news_url VARCHAR(255),
        title VARCHAR(255),
        Description TEXT,
        summary TEXT,
        Publication_Date TIMESTAMP,
        embedding TEXT
    );
    """

    create_views_table = """
    CREATE TABLE IF NOT EXISTS UserNewsViews (
        view_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        news_id INT,
        view_date DATETIME,
        FOREIGN KEY (news_id) REFERENCES News(news_id)
    );
    """

    execute_query(connection, create_news_table)
    execute_query(connection, create_views_table)

def insert_sample_data(connection):
    cursor = connection.cursor()
    
    insert_news = """
    INSERT INTO News (Category, news_url, title, Description, summary, Publication_Date, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    news_data = (
        "정치",
        "http://example.com/news1",
        "샘플 뉴스 제목",
        "이것은 네이버 뉴스 API에서 제공하는 뉴스 요약입니다.",
        "LLM을 이용한 뉴스 요약 내용",
        datetime.now(),
        "뉴스 임베딩 값"
    )
    cursor.execute(insert_news, news_data)
    
    news_id = cursor.lastrowid
    
    insert_view = """
    INSERT INTO UserNewsViews (user_id, news_id, view_date)
    VALUES (%s, %s, %s)
    """
    view_data = (1, news_id, datetime.now())
    cursor.execute(insert_view, view_data)
    
    connection.commit()
    print("Sample data inserted successfully")

def main():
    host = "localhost"
    user = "root"
    password = "1234"
    db_name = "AI_DB"

    # 서버에 연결
    connection = create_server_connection(host, user, password)
    if connection is None:
        print("Failed to connect to the database. Exiting.")
        return
    
    # 데이터베이스 생성
    create_database_query = f"CREATE DATABASE IF NOT EXISTS {db_name}"
    create_database(connection, create_database_query)

    # 생성된 데이터베이스에 연결
    connection = create_db_connection(host, user, password, db_name)

    # 테이블 생성
    create_tables(connection)

    # 샘플 데이터 삽입
    insert_sample_data(connection)

    # 연결 종료
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

if __name__ == "__main__":
    main()
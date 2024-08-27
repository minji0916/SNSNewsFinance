CREATE TABLE IF NOT EXISTS News (
        news_id INT AUTO_INCREMENT PRIMARY KEY,
        category TEXT,
        news_url VARCHAR(255),
        title VARCHAR(255),
        description TEXT,
        content TEXT,
        summary TEXT,
        publication_date VARCHAR(255),
        embedding TEXT
);

CREATE TABLE IF NOT EXISTS UserNewsViews (
        view_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        news_id INT,
        view_date VARCHAR(255),
        FOREIGN KEY (news_id) REFERENCES News(news_id)
);
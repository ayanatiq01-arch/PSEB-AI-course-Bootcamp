CREATE DATABASE library;
USE library;

CREATE TABLE books (
    id INT PRIMARY KEY,
    title VARCHAR(100),
    author VARCHAR(100),
    year INT,
    price DECIMAL(10,2)
);

INSERT INTO books (id, title, author, year, price)
VALUES
(1, 'Harry Potter', 'J.K. Rowling', 1997, 800),
(2, 'The Alchemist', 'Paulo Coelho', 1988, 450),
(3, 'Naruto', ' Masashi Kishimoto', 2018, 950),
(4, 'Bleach', 'Tite Kubo', 1997, 600),
(5, 'Hunger Games', 'Suzanne Collins', 1937, 400),
(6, 'The Hobbit', 'J.R.R. Tolkien', 1937, 750),
(7, 'Pride and Prejudice', 'Jane Austen', 2016, 550),
(8, 'Deep Work', 'Cal Newport', 2016, 650);

SELECT Price FROM books
WHERE price > 500;

SELECT * FROM books
ORDER BY price DESC;

SELECT COUNT(*) AS total_books
FROM books;

UPDATE books
SET price = 1000
WHERE id = 3;

SELECT * FROM books
WHERE id = 3;

CREATE TABLE loan_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_data TEXT,
    prediction VARCHAR(20),
    reasons TEXT
);
SHOW TABLES;
DESCRIBE loan_predictions;
INSERT INTO loan_predictions (input_data, prediction, reasons)
VALUES ('[1, 0, 50000]', 'Approved', 'AMT_CREDIT increased; NAME_TYPE decreased');
SELECT * FROM loan_predictions;

Machine Learning-Based Phishing Email Detection System
1. Project Objective

This project is an individual B207 Cyber Security assignment. It builds a machine learning system that analyses email text and classifies each email as Safe Email or Phishing Email, using TF-IDF feature extraction and a Logistic Regression classifier. The system includes a command-line interface, SQLite storage for analysis history, and basic reporting.

2. Cybersecurity Problem

Phishing emails remain one of the most common initial attack vectors used to steal credentials, deliver malware, and commit fraud. Manual review of every incoming email does not scale, so this project explores whether a supervised machine learning model can reliably flag suspicious emails from their text content alone, while accounting for the operational cost of both missed phishing (false negatives) and wrongly flagged legitimate emails (false positives).

3. Dataset
File: Phishing_Email.csv
Source URL: (https://www.kaggle.com/datasets/subhajournal/phishingemails)
Records: 18,650 (before cleaning)
Columns:
Unnamed: 0 — row index (dropped, not predictive)
Email Text — raw email content (input feature)
Email Type — Safe Email / Phishing Email (target)
Known data quality issues: 19 empty/whitespace text records, 1,111 exact duplicates — both are detected programmatically and removed during cleaning (see Stage 4 output).
4. Project Structure
phishing-email-detector/
├── data/                  Phishing_Email.csv goes here
├── models/                saved TF-IDF vectorizer + classifier
├── reports/figures/       generated charts (Figures 1-3, 7-8)
├── database/              SQLite database
├── screenshots/           evidence screenshots
├── tests/                 functional test suite
├── src/                   source modules
├── main.py                pipeline entry point
├── setup_project.py       environment/database setup script
└── requirements.txt
5. Installation
bash
pip install -r requirements.txt
6. Setup

Run once, before anything else:

bash
python setup_project.py

This installs dependencies, creates all project folders, and initialises the SQLite database.

Place the dataset at data/Phishing_Email.csv before continuing.

7. Model Training
bash
python main.py train

This runs data loading, cleaning, EDA, text preprocessing, TF-IDF extraction, model training, evaluation, and error analysis, then saves the trained vectorizer and classifier to models/.

8. Running the CLI
bash
python main.py cli

Menu options let you analyse a new email, view analysis history, view flagged phishing attempts, view model information, and view a reporting summary.

9. Database

SQLite database at database/phishing_detector.db, table analysis, storing prediction, prediction score, text length, top contributing terms, and timestamp for each analysis. Only the first 500 characters of any submitted email are stored. All queries are parameterised.

10. Testing
bash
python main.py test

Runs the 12-test functional test plan (startup, valid/empty/long/special-character input, database storage and retrieval, invalid CLI input, model reload).

11. Model Evaluation Results
Metric	    Result
Accuracy	98.23%
Precision	96.64%
Recall	    98.70%
F1-score	97.66%

Confusion matrix and full classification report are printed by python main.py train and saved as figures in reports/figures/.

12. Limitations
Trained on a single public dataset; performance on emails from other sources or languages is untested.
TF-IDF captures lexical patterns only, not sender metadata, headers, or attachments.
High accuracy does not mean the system is production-ready — false negatives still carry direct security risk and require human-in-the-loop review.
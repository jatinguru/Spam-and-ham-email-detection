Spam Email Detection using Enron Email Dataset
Overview
This project uses machine learning to classify emails as spam or ham (non-spam). It combines the Enron Email Dataset (for ham emails) with an online SMS spam dataset (for spam examples). 
A Naive Bayes classifier is used with TF-IDF vectorization for feature extraction.

Dataset Used
Ham Emails: Extracted from the Enron email dataset (maildir/)

Spam Emails: SMS spam dataset from UCI Repository
(fetched from GitHub for convenience)

Technologies Used
1.Python
2.Pandas
3.Scikit-learn
4.Regular Expressions
5.Jupyter Notebook

Model Used
1.TF-IDF Vectorizer: Converts text into numeric form.
2.Multinomial Naive Bayes: A popular classifier for text data.
3.Train-Test Split: 80% training and 20% testing.

Results
The trained model achieved high accuracy in distinguishing spam from ham emails, showing reliable performance based on the classification report.

Future Work
Implement a web-based GUI using Flask or Streamlit

Use deep learning models (e.g., LSTM) for better accuracy

Detect phishing or scam content more specifically


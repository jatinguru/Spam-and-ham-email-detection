import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Enron Ham Emails
def load_ham_emails(maildir_path):
    ham = []
    for user in os.listdir(maildir_path):
        user_path = os.path.join(maildir_path, user)
        for folder in ['inbox', 'sent']:
            folder_path = os.path.join(user_path, folder)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            email = f.read()
                            ham.append(email)
                    except:
                        continue
    return ham

ham_emails = load_ham_emails(r"D:\enron_mail_20150507\maildir")

# 2. Create Ham DataFrame
enron_df = pd.DataFrame({'message': ham_emails})
enron_df['label'] = 0  # Ham = 0

# 3. Load Spam Data (Online SMS Dataset)
spam_data = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
    sep='\t', header=None, names=['label', 'message']
)
spam_data['label'] = spam_data['label'].map({'ham': 0, 'spam': 1})

# 4. Combine Both Datasets
combined_df = pd.concat([enron_df, spam_data], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# 5. Preprocess Text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

combined_df['cleaned'] = combined_df['message'].apply(preprocess)

# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    combined_df['cleaned'], combined_df['label'], test_size=0.2, random_state=42
)

# 7. Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 9. Predict and Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

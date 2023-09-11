import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from flask import Flask, redirect, url_for, render_template, request, abort

app = Flask(__name__)

df = pd.read_csv('spam.csv', encoding='latin-1')

df_cln = df[['v1', 'v2']]

df_cln = df_cln.drop_duplicates()

X_train, X_test, y_train, y_test = train_test_split(df_cln['v2'], df_cln['v1'], test_size=0.3, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_tfidf = feature_extraction.fit_transform(X_train)
X_test_tfidf = feature_extraction.transform(X_test)

smote = SMOTE(random_state=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

@app.route('/')
def index():
    return render_template('MailPaste.html')

@app.route('/check', methods=['POST', 'GET'])
def check():
    if request.method == 'POST':
        mailString = request.form['mail']
        result = MailType(mailString)
        return render_template('MailPaste.html', result=result)

def MailType(mailStr):
    res = model.predict(feature_extraction.transform([mailStr]))
    if res[0] == 'ham':
        return "Ham Mail/SMS"
    else:
        return "Spam Mail/SMS"

if __name__ == '__main__':
    app.run(debug=True)

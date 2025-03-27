import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from flask import Flask, request, jsonify, render_template
import os

try:
    # Load dataset
    dataset_path = "spam.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Please download the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset and place it in the project folder as spam.csv")

    df = pd.read_csv(dataset_path, encoding='latin-1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Data preprocessing
    def clean_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub("\d+", "", text)
        return text

    df['message'] = df['message'].apply(clean_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Model training
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()
    model = make_pipeline(vectorizer, classifier)
    model.fit(X_train, y_train)

    # Save the model
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Flask app
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template("index.html")

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            message = request.form['message']
            with open("spam_model.pkl", "rb") as f:
                model = pickle.load(f)
            prediction = model.predict([message])[0]
            return render_template("index.html", message=message, prediction='Spam' if prediction else 'Ham')
        except Exception as e:
            return render_template("index.html", message="An error occurred: " + str(e))

    if __name__ == '__main__':
        try:
            app.run(debug=True)
        except Exception as e:
            print("An error occurred: " + str(e))
except Exception as e:
    print("An error occurred: " + str(e))
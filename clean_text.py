import pickle
import re

import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()


with open("model/modelo_salvo", "rb") as file:
    load_data = pickle.load(file)
    load_model = load_data['model']
    tfidf_vectorizer = load_data['tfidf_vectorizer']


def clean_text(text: str) -> str:
    """
    Removes URLs, punctuation, and numbers from the given text.
    Converts the text to lowercase, splits it into words, and removes stop words.
    Applies stemming to each word.
    
    Parameters:
    - text (str): The input text to be cleaned.
    
    Returns:
    - str: The cleaned text.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    text = text.lower().split()

    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]

    return ' '.join(text)


def classify_msg(msg: str) -> int:
    """
    Classifies a given message using a trained model.

    Parameters:
        msg (str): The message to be classified.

    Returns:
        int: The predicted class label for the message.
    """
    cleaned_msg = clean_text(msg)
    transformed_msg = tfidf_vectorizer.transform([cleaned_msg]).toarray()

    prediction = load_model.predict(transformed_msg)

    return prediction[0]

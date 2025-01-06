import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import re
from urllib.parse import urlparse
import pickle

# Feature extraction functions
def get_url_length(url):
    return len(url)

def has_ip_address(url):
    return int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))

def count_special_chars(url):
    special_chars = ['-', '@', '_']
    return sum([url.count(char) for char in special_chars])

def has_https(url):
    return int(urlparse(url).scheme == 'https')

def extract_features(url):
    return pd.DataFrame([{
        'url_length': get_url_length(url),
        'has_ip': has_ip_address(url),
        'special_chars': count_special_chars(url),
        'https': has_https(url)
    }])

# Load dataset and train model if not already done
def train_model():
    df = pd.read_csv("urls.csv")

    # Check if 'url' and 'label' columns exist
    if 'url' not in df.columns or 'label' not in df.columns:
        raise KeyError("The CSV file must contain 'url' and 'label' columns.")

    df['url_length'] = df['url'].apply(get_url_length)
    df['has_ip'] = df['url'].apply(has_ip_address)
    df['special_chars'] = df['url'].apply(count_special_chars)
    df['https'] = df['url'].apply(has_https)

    X = df[['url_length', 'has_ip', 'special_chars', 'https']]
    y = df['label']

    model = RandomForestClassifier()
    model.fit(X, y)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

# Load or train the model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    train_model()
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

def predict_url(url):
    features = extract_features(url)
    return model.predict(features)[0]

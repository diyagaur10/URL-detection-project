import sys
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tldextract
from urllib.parse import urlparse
from scipy.sparse import hstack
import warnings

# Suppress the specific UserWarning from TfidfVectorizer
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")


# --- 1. Lexical Feature Tokenizer ---
def makeTokens(url_str):
    """
    Tokenizes a URL into its constituent parts for TF-IDF vectorization.
    """
    try:
        if not isinstance(url_str, str):
            return []

        if not url_str.startswith("http"):
            url_str = "http://" + url_str

        tokens = []
        for delim in ['/', '-', '.']:
            for part in url_str.split(delim):
                if part:
                    tokens.append(part.lower())
        
        ext = tldextract.extract(url_str)
        if ext.subdomain: tokens.append(ext.subdomain.lower())
        if ext.domain: tokens.append(ext.domain.lower())
        if ext.suffix: tokens.append(ext.suffix.lower())
        
        common_to_remove = {'http', 'https', 'www', 'com'}
        final_tokens = [t for t in list(set(tokens)) if t not in common_to_remove]
        
        return final_tokens
    except:
        return []

# --- 2. Statistical Feature Extraction ---
def compute_statistical_features(url_str):
    """
    Computes statistical and structural features from a URL string.
    """
    # ### FIXED ###: The number of features must match the actual count.
    NUM_STATISTICAL_FEATURES = 17
    
    try:
        if not isinstance(url_str, str) or len(url_str) == 0:
            return [0] * NUM_STATISTICAL_FEATURES
            
        if not url_str.startswith("http"):
            url_str = "http://" + url_str
            
        parsed_url = urlparse(url_str)
        ext = tldextract.extract(url_str)

        features = []
        
        # Length-based features (4 features)
        features.append(len(url_str))
        features.append(len(parsed_url.netloc))
        features.append(len(parsed_url.path))
        features.append(len(ext.domain))
        
        # Count of special characters (11 features)
        features.append(url_str.count('.'))
        features.append(url_str.count('-'))
        features.append(url_str.count('@'))
        features.append(url_str.count('?'))
        features.append(url_str.count('&'))
        features.append(url_str.count('|'))
        features.append(url_str.count('='))
        features.append(url_str.count('_'))
        features.append(url_str.count('~'))
        features.append(url_str.count('%'))
        features.append(url_str.count('/'))
        
        # Presence of IP address in domain (1 feature)
        domain_is_ip = 0
        if ext.domain:
            if all(c.isdigit() or c == '.' for c in ext.domain):
                is_ip = all(part.isdigit() for part in ext.domain.split('.'))
                domain_is_ip = 1 if is_ip else 0
        features.append(domain_is_ip)
            
        # Presence of sensitive words (1 feature)
        sensitive_words = ['secure', 'login', 'signin', 'ebayisapi', 'account', 'confirm', 'webscr']
        features.append(1 if any(word in url_str.lower() for word in sensitive_words) else 0)
        
        return features
    except Exception:
        return [0] * NUM_STATISTICAL_FEATURES

# --- Global model variables ---
vectorizer = None
logit_model = None

# --- Main execution block ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "urldata.csv")

    try:
        urls_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"FATAL: urldata.csv not found at {csv_file_path}. Exiting.", file=sys.stderr)
        sys.exit(1)

    if "url" not in urls_data.columns or "label" not in urls_data.columns:
        print("FATAL: urldata.csv must contain 'url' and 'label' columns. Exiting.", file=sys.stderr)
        sys.exit(1)

    
    urls_data.dropna(subset=['url', 'label'], inplace=True)
    urls_data['url'] = urls_data['url'].astype(str).str.strip()
    urls_data = urls_data[urls_data['url'].str.len() > 0]
    urls_data.reset_index(drop=True, inplace=True)
    

   
    
    vectorizer = TfidfVectorizer(tokenizer=makeTokens, max_features=5000)
    lexical_features = vectorizer.fit_transform(urls_data['url'])
    
    statistical_features = np.array([compute_statistical_features(url) for url in urls_data['url']])
    
    X = hstack([lexical_features, statistical_features])
    y = urls_data['label']
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    logit_model = LogisticRegression(solver='liblinear', C=0.5, class_weight='balanced', max_iter=1000)
    logit_model.fit(X_train, y_train)
    
    
    
    y_pred = logit_model.predict(X_test)
    
    
    
    
    cm = confusion_matrix(y_test, y_pred)
    # Ensure classes are sorted for consistent column/index naming
    class_labels = sorted(list(y.unique()))
   
    if len(sys.argv) > 1:
        input_url = sys.argv[1]
        
        input_lexical = vectorizer.transform([input_url])
        input_statistical = np.array([compute_statistical_features(input_url)]).reshape(1, -1)
        input_combined = hstack([input_lexical, input_statistical])
        
        prediction_array = logit_model.predict(input_combined)
        predicted_label = prediction_array[0]
        
        if predicted_label == "good":
            print("It is a secure website")
        elif predicted_label == "bad":
            print("It is a phishing website")
        else:
            # Fallback for any unexpected labels
            print(f"Predicted: {predicted_label}")
    else:
        print("\nINFO: To check a new URL, run the script with the URL as an argument:")
        print("python ml_model_detector.py \"some-suspicious-url.com/login\"", file=sys.stderr)
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tldextract
import os

# --- Tokenizer (refined from original script) ---
def makeTokens(url_str):
    # Assuming url_str is a plain string like "http://example.com/path"
    all_tokens = []

    # 1. Split by '/'
    # Normalize to lowercase early and filter empty strings
    tokens_from_slash = [t.lower() for t in url_str.split('/') if t]
    all_tokens.extend(tokens_from_slash)

    # 2. For each part from slash, split by '-' and then by '.'
    for part_slashed in tokens_from_slash:
        tokens_from_hyphen = [t.lower() for t in part_slashed.split('-') if t]
        all_tokens.extend(tokens_from_hyphen)
        for part_hyphened in tokens_from_hyphen:
            tokens_from_dot = [t.lower() for t in part_hyphened.split('.') if t]
            all_tokens.extend(tokens_from_dot)
    
    # Add tldextract features (all lowercase)
    ext = tldextract.extract(url_str)
    if ext.subdomain: all_tokens.append(ext.subdomain.lower())
    if ext.domain: all_tokens.append(ext.domain.lower())
    if ext.suffix: all_tokens.append(ext.suffix.lower())
    
    # Unique tokens
    unique_tokens = list(set(all_tokens))
    
    # Remove specific common tokens (original removed 'com'; http/https are good to remove if tokenized)
    common_to_remove = {'com', 'http', 'https'} 
    final_tokens = [token for token in unique_tokens if token not in common_to_remove and token] # Ensure no empty strings
    
    return final_tokens

# --- Global model variables (will be loaded by main) ---
vectorizer_model = None
logit_classifier_model = None

# --- Model Loading (adapted from original script) ---
def load_and_train_model_on_the_fly():
    global vectorizer_model, logit_classifier_model
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "urldata.csv")

    try:
        urls_data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"FATAL: urldata.csv not found at {csv_file_path}. Please ensure it is in the same directory as the script.", file=sys.stderr)
        sys.exit(1) # Exit with error code

    y = urls_data["label"]  # Expect "bad" or "good" (or whatever your labels are)
    url_list = urls_data["url"]

    vectorizer_model = TfidfVectorizer(tokenizer=makeTokens)
    X = vectorizer_model.fit_transform(url_list)
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Added max_iter for potential convergence issues with default.
    logit_classifier_model = LogisticRegression(max_iter=1000) 
    logit_classifier_model.fit(X_train, y_train)
    # IMPORTANT: Training on every call is very inefficient. Pre-train and save/load model for production.

# --- Prediction (adapted from original script) ---
def predict_url_status(url_to_check):
    if vectorizer_model is None or logit_classifier_model is None:
        print("FATAL: Model components not loaded. Call load_and_train_model_on_the_fly first.", file=sys.stderr)
        sys.exit(1)

    if not url_to_check.startswith("http://") and not url_to_check.startswith("https://"):
        url_to_check = "http://" + url_to_check 

    X_predict = vectorizer_model.transform([url_to_check])
    prediction = logit_classifier_model.predict(X_predict)
    return prediction[0] 

# --- Main execution block for command-line usage ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml_model_detector.py <URL_to_check>", file=sys.stderr)
        sys.exit(1)

    input_url = sys.argv[1]
    
    load_and_train_model_on_the_fly() # This happens on every script call due to spawn behavior
    
    result = predict_url_status(input_url)
    print(result) # Output "bad" or "good" (or your specific labels) to stdout
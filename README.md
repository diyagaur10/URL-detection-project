**Fake Website Detection System**
This project is a Fake Website Detection System designed to identify potentially malicious websites using Machine Learning and a user-friendly GUI. The system provides real-time URL validation and SSL certification checks to ensure online safety.

**Key Features**

**Phishing Detection:**

Uses a Logistic Regression model trained on a labeled dataset of URLs to classify websites as secure or malicious.
Employs TF-IDF Vectorization with a custom tokenizer for effective feature extraction from URLs.

**SSL Certification Check:**

Validates the presence of a valid SSL certificate for a given URL to assess its security for encrypted communication.

**User-Friendly GUI:**

Built with Tkinter for easy interaction.
Allows users to input URLs, check their safety, and view results in an intuitive interface.
Customizable background for a personalized look.

**Custom URL Feature Extraction:**

Extracts key components like subdomains, domains, suffixes, and tokenized elements of the URL to enhance model accuracy.

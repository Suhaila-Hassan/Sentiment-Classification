# Emotion Classification
This project demonstrates an end-to-end machine learning pipeline for emotion classification using text comments.

## Dataset
The dataset used is **Emotion_classify_Data** csv dataset containing two columns:
- Comment: A string of text input.
- Emotion: The emotion label associated with the comment (e.g., Anger, Fear, Joy).

## Project Workflow

### 1. Importing Libraries
Data handling: Pandas, Numpy
NLP: NLTK
Machine Learning: Scikit-Learn
Visualization: Matplotlib, Seaborn

### 2. Exploratory Data Analysis (EDA)
- Load dataset.
- Checking for missing values.
- Displaying emotion class distribution.

### 3. Data Preprocessing
- Text Cleaning: Lowercasing, removing stopwords.
- Tokenization: Using NLTK’s word_tokenize.
- Stemming & Lemmatization: Leveraging PorterStemmer and WordNetLemmatizer.
- Split the dataset into 80% training and 20% testing using train_test_split.
- Perform vectorization techniques on dataset:
  A. CountVectorizer
  B. TF-IDF Vectorizer

### 4. Machine Learning Models
#### Models Used:
Two classification algorithms are trained and tested using both CountVectorizer and TF-IDF features:
- Logistic Regression
- Multinomial Naive Bayes

#### Evaluation Metrics:
- Accuracy Score
- Classification Report
- Confusion Matrix

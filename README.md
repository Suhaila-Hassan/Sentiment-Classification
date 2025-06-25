# 🎭 Emotion Classification System

A comprehensive machine learning-powered emotion classification system that analyzes text and predicts emotions with confidence scores. Built with Python, scikit-learn, and Streamlit for an interactive web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## 🌟 Features

- **Multi-emotion Classification**: Detects various emotions including joy, sadness, anger, fear, disgust, surprise, and more
- **Interactive Web Interface**: Beautiful Streamlit-powered web app with real-time analysis
- **High Accuracy**: Achieved best performance with Logistic Regression + Count Vectorizer (LogReg_Count)
- **Comprehensive Analysis**: Provides confidence scores for all emotion categories
- **Text Processing**: Advanced NLP preprocessing with stemming, lemmatization, and stop word removal
- **Visualization**: Dynamic charts and word clouds for emotion distribution
- **File Upload Support**: Analyze text from uploaded files
- **Text Statistics**: Detailed analysis of input text characteristics

## 🚀 Demo

### Web Interface
The Streamlit app provides an intuitive interface for emotion analysis:

- **Text Input**: Type or upload text files for analysis
- **Real-time Results**: Instant emotion prediction with confidence scores
- **Visual Analytics**: Interactive charts showing confidence distributions
- **Text Statistics**: Comprehensive text analysis metrics

### Sample Predictions
```
Input: "I'm so excited about this new opportunity!"
Output: JOY (87.3% confidence)

Input: "This situation makes me really angry and frustrated."
Output: ANGER (92.1% confidence)
```

## 📊 Model Performance

Our emotion classification system was trained and evaluated on multiple algorithms:

| Model | Vectorizer | Accuracy |
|-------|------------|----------|
| **Logistic Regression** | **Count Vectorizer** | **Best Performance** ⭐ |
| Logistic Regression | TF-IDF | High Performance |
| Naive Bayes | Count Vectorizer | Good Performance |
| Naive Bayes | TF-IDF | Good Performance |

### Key Metrics
- **Best Model**: Logistic Regression with Count Vectorizer
- **Text Preprocessing**: Stemming, Lemmatization, Stop word removal
- **Feature Engineering**: Count-based vectorization
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score

## 🎯 Usage

### Running the Web App

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically load the trained models

## 🔬 Technical Details

### Data Preprocessing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove punctuation and special characters
   - Filter out non-alphabetic tokens

2. **Tokenization**
   - Split text into individual words
   - Remove stop words (common words like 'the', 'and', etc.)

3. **Normalization**
   - **Lemmatization**: Reduce words to their root form
   - **Stemming**: Remove suffixes to get word stems

4. **Vectorization**
   - Convert text to numerical features
   - Count Vectorizer for best performance

### Model Architecture

```python
# Best performing pipeline
Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(random_state=0))
])
```

### Supported Emotions

The system can classify the following emotions:
- 😊 **Joy**
- 😠 **Anger**
- 😨 **Fear**

## 📈 Visualization Features

### Training Visualizations
- Emotion distribution charts
- Word clouds for each emotion category
- Model performance comparison
- Confusion matrices for each model

### Web App Visualizations
- Real-time confidence score charts
- Interactive emotion distribution
- Text statistics dashboard
- Dynamic progress indicators

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

## 📖 API Documentation

### Core Functions

#### `preprocess_text(text)`
Preprocesses input text for emotion classification.

**Parameters:**
- `text` (str): Raw input text

**Returns:**
- `str`: Cleaned and processed text

#### `predict_emotion(text, model, vectorizer)`
Predicts emotion for given text.

**Parameters:**
- `text` (str): Input text to classify
- `model`: Trained classifier model
- `vectorizer`: Fitted text vectorizer

**Returns:**
- `tuple`: (predicted_emotion, confidence_scores)


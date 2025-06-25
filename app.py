import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .emotion-result {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .sample-text {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .sample-text:hover {
        background: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}' if 'punkt' in item else f'corpora/{item}')
        except LookupError:
            try:
                nltk.download(item, quiet=True)
            except:
                pass

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("🚫 Model files not found! Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    try:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"❌ Error in text preprocessing: {str(e)}")
        st.info("💡 Please make sure NLTK data is downloaded properly.")
        return text.lower()

# Predict emotion
def predict_emotion(text, model, vectorizer):
    if not text.strip():
        return None, None
    
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    prediction_proba = model.predict_proba(text_vector)[0]
    
    return prediction, prediction_proba

def create_confidence_chart(emotions, probabilities):
    """Create an enhanced confidence chart"""
    prob_df = pd.DataFrame({
        'Emotion': emotions,
        'Confidence': probabilities * 100
    }).sort_values('Confidence', ascending=False)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    
    # Create horizontal bar chart
    bars = ax.barh(prob_df['Emotion'], prob_df['Confidence'], color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, prob_df['Confidence']):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f'{value:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)
    
    # Styling
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotions', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Classification Confidence Scores', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(probabilities * 100) + 10)
    ax.grid(axis='x', alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, prob_df

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Emotion Classification App</h1>', unsafe_allow_html=True)
    
    # Load model and vectorizer
    with st.spinner("🔄 Loading models..."):
        model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 About This App")
        st.markdown("""
        <div class="info-box">
            <h4>🤖 AI-Powered Emotion Detection</h4>
            <p>This app uses machine learning to analyze text and predict emotions with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Available Emotions")
        try:
            emotions = model.classes_
            emotion_icons = {
                'joy': '😊', 'happiness': '😊', 'happy': '😊',
                'sadness': '😢', 'sad': '😢',
                'anger': '😠', 'angry': '😠',
                'fear': '😨', 'scared': '😨',
                'disgust': '🤢', 'disgusted': '🤢',
                'surprise': '😲', 'surprised': '😲',
                'love': '❤️', 'trust': '🤝', 'anticipation': '🤔'
            }
            
            for emotion in emotions:
                icon = emotion_icons.get(emotion.lower(), '🎭')
                st.markdown(f"**{icon} {emotion.title()}**")
        except:
            st.markdown("🎭 Various emotions available")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **📝 Enter text** in the input area
        2. **🎯 Click 'Analyze'** to classify
        3. **📊 View results** and confidence scores
        4. **🧪 Try samples** for quick testing
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown("### 📝 Text Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose your input method:",
            ["✍️ Type text", "📁 Upload file"],
            horizontal=True,
            help="Select how you'd like to provide text for analysis"
        )
        
        user_text = ""
        
        if input_method == "✍️ Type text":
            user_text = st.text_area(
                "Enter your text here:",
                height=200,
                placeholder="Type or paste your message here to analyze its emotional content...",
                help="Enter any text you'd like to analyze for emotional content"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt'],
                help="Upload a .txt file containing the text you want to analyze"
            )
            
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("📄 File content preview:", value=user_text[:500] + "..." if len(user_text) > 500 else user_text, 
                           height=150, disabled=True)
        
        # Analysis button
        analyze_btn = st.button(
            "🎯 Analyze Emotion", 
            type="primary", 
            use_container_width=True,
            help="Click to analyze the emotional content of your text"
        )
        
        if analyze_btn:
            if user_text.strip():
                with st.spinner("🔍 Analyzing emotional content..."):
                    prediction, probabilities = predict_emotion(user_text, model, vectorizer)
                    
                    if prediction is not None:
                        st.success("✅ Analysis completed successfully!")
                        st.session_state.prediction = prediction
                        st.session_state.probabilities = probabilities
                        st.session_state.analyzed_text = user_text
                        st.session_state.emotions = model.classes_
                    else:
                        st.error("❌ Unable to analyze the text. Please try again.")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            probabilities = st.session_state.probabilities
            emotions = st.session_state.emotions
            
            # Predicted emotion display
            confidence = max(probabilities) * 100
            
            # Get emoji for emotion
            emotion_icons = {
                'joy': '😊', 'happiness': '😊', 'happy': '😊',
                'sadness': '😢', 'sad': '😢',
                'anger': '😠', 'angry': '😠',
                'fear': '😨', 'scared': '😨',
                'disgust': '🤢', 'disgusted': '🤢',
                'surprise': '😲', 'surprised': '😲',
                'love': '❤️', 'trust': '🤝', 'anticipation': '🤔'
            }
            
            emotion_icon = emotion_icons.get(prediction.lower(), '🎭')
            
            st.markdown(f"""
            <div class="emotion-result">
                {emotion_icon} {prediction.upper()}<br>
                <small style="font-size: 1rem; opacity: 0.9;">{confidence:.1f}% confidence</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart
            st.markdown("#### 📈 Detailed Confidence Scores")
            fig, prob_df = create_confidence_chart(emotions, probabilities)
            st.pyplot(fig, use_container_width=True)
            
            # Scores table
            with st.expander("📋 View Detailed Scores Table"):
                prob_df['Confidence'] = prob_df['Confidence'].round(2)
                prob_df['Confidence'] = prob_df['Confidence'].astype(str) + '%'
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👆 Enter text and click 'Analyze Emotion' to see results here.")
    
    # Text Statistics Section
    if hasattr(st.session_state, 'analyzed_text'):
        st.markdown("---")
        st.markdown("### 📊 Text Statistics")
        
        text = st.session_state.analyzed_text
        cleaned = preprocess_text(text)
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("📝 Characters", len(text)),
            ("🔤 Words", len(text.split())),
            ("📖 Sentences", text.count('.') + text.count('!') + text.count('?')),
            ("🔧 Processed Words", len(cleaned.split()) if cleaned else 0)
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #4ecdc4;">{label}</h4>
                    <h2 style="margin: 0; color: #333;">{value}</h2>
                </div>
                """, unsafe_allow_html=True)
    


if __name__ == "__main__":
    main()
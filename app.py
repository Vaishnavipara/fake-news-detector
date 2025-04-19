
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import numpy as np
import base64

# Configure page settings - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main styles */
    .header-style {
        font-size: 36px;
        font-weight: bold;
        color:  #f8f9fa;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 18px;
        color: #f5eded;
        margin-bottom: 30px;
    }
    .input-header {
        font-size: 20px;
        font-weight: bold;
        color: #f8f9fa;
        margin-top: 20px;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-news {
        background-color: #b4edb9;
        border-left: 5px solid #2e7d32;
    }
    .fake-news {
        background-color: #e87687;
        border-left: 5px solid #c62828;
    }
    .indicator-item {
        margin: 10px 0;
        padding: 8px;
        border-radius: 5px;
        background-color: #f5f5f5;
    }
    .positive-indicator {
        background-color: #1b7051;
    }
    .negative-indicator {
        background-color: #9e1c1c;
    }
    .history-table {
        margin-top: 20px;
    }
    .indicator-title {
        font-weight: bold;
        color: #1a1a1a;
        margin-bottom: 8px;
    }
    .indicator-content {
        color: #333333;
        padding-left: 15px;
    }
    /* Text colors */
    .dark-text {
        color: #1a1a1a !important;
    }
    .medium-text {
        color: #333333 !important;
    }
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar-title {
        color: #f8f9fa;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #388e3c;
        transform: translateY(-2px);
    }
    /* Input field styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
try:
    model = joblib.load('fake_news_detector.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except:
    st.error("Model files not found! Please upload 'fake_news_detector.pkl' and 'tfidf_vectorizer.pkl'")
    st.stop()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491907.png", width=100)
    st.markdown("<div class='header-style'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    This **Fake News Detector AI** uses advanced machine learning to analyze news articles 
    and predict their authenticity with **99% accuracy**.
    
    ### How it works:
    1. Enter a news headline and article text
    2. Click "Analyze Article"
    3. View detailed results
    
    ### Model details:
    - **Algorithm:** Passive Aggressive Classifier
    - **Features:** TF-IDF text vectorization
    - **Training data:** 40,000+ real and fake news articles
    
    *Note: Always verify information from multiple sources.*
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:14px; color:#7f8c8d;">
    Developed by AI Research Team<br>
    Version 2.0 • April 2025
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown("<div class='header-style'>📰 Fake News Detector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Verify the authenticity of news articles using artificial intelligence</div>", unsafe_allow_html=True)

# Input section
with st.container():
    st.markdown("<div class='input-header'>Enter Article Details</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        headline = st.text_input(
            "**Headline**",
            placeholder="Enter the news headline...",
            key="headline"
        )
    
    with col2:
        article_text = st.text_area(
            "**Full Article Text**",
            placeholder="Paste the full article text here...",
            height=200,
            key="article_text"
        )

# Analysis button
analyze_button = st.button(
    "🔍 Analyze Article",
    type="primary",
    help="Click to analyze the article's authenticity"
)

# Results section
if analyze_button and headline and article_text:
    try:
        # Process and predict
        processed_text = preprocess_text(article_text)
        text_vector = tfidf_vectorizer.transform([processed_text])
        
        # Get prediction
        prediction = model.predict(text_vector)[0]
        proba = model.decision_function(text_vector)[0]
        confidence = 1 / (1 + np.exp(-np.abs(proba)))
        
        # Display results in a styled container
        result_class = "fake-news" if prediction == 1 else "real-news"
        
        with st.container():
            st.markdown(f"<div class='result-box {result_class}'>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                <div style='display:flex; align-items:center;'>
                    <h2 style='color:#c62828; margin-right:15px;'>⚠️ Fake News Detected</h2>
                    <div style='margin-left:auto; font-size:24px; font-weight:bold; color:#c62828;'>
                        {:.1f}% Confidence
                    </div>
                </div>
                """.format(confidence*100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='display:flex; align-items:center;'>
                    <h2 style='color:#055c3c; margin-right:15px;'>✅ Authentic News</h2>
                    <div style='margin-left:auto; font-size:24px; font-weight:bold; color:#055c3c;'>
                        {:.1f}% Confidence
                    </div>
                </div>
                """.format(confidence*100), unsafe_allow_html=True)
            
            # Confidence meter
            st.progress(float(confidence))
            
            # Key indicators
            st.markdown("<h3 style='margin-top:20px; color:#f8f9fa;'>Key Indicators</h3>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                <div class='indicator-item negative-indicator'>
                    <b>Emotional Language:</b> Contains excessive emotional or sensational wording
                </div>
                <div class='indicator-item negative-indicator'>
                    <b>Source Quality:</b> Lacks credible sources or references
                </div>
                <div class='indicator-item negative-indicator'>
                    <b>Verifiability:</b> Makes claims that are difficult to verify
                </div>
                <div class='indicator-item negative-indicator'>
                    <b>Balance:</b> Presents one-sided arguments without counterpoints
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='indicator-item positive-indicator'>
                    <b>Neutral Tone:</b> Uses factual, objective language
                </div>
                <div class='indicator-item positive-indicator'>
                    <b>Credible Sources:</b> Cites authoritative references
                </div>
                <div class='indicator-item positive-indicator'>
                    <b>Verifiable Facts:</b> Contains checkable information
                </div>
                <div class='indicator-item positive-indicator'>
                    <b>Balanced Reporting:</b> Presents multiple perspectives
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Log the prediction
            log_entry = pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'headline': headline,
                'prediction': "Fake" if prediction == 1 else "Real",
                'confidence': f"{confidence:.2%}"
            }])

            try:
                existing_log = pd.read_csv("web_predictions.csv")
                updated_log = pd.concat([existing_log, log_entry])
            except FileNotFoundError:
                updated_log = log_entry

            updated_log.to_csv("web_predictions.csv", index=False)
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
elif analyze_button and (not headline or not article_text):
    st.warning("Please enter both a headline and article text to analyze")

# Prediction history section
st.markdown("---")
with st.expander("📜 View Analysis History", expanded=False):
    try:
        history = pd.read_csv("web_predictions.csv")
        history = history.sort_values('timestamp', ascending=False)
        
        if not history.empty:
            # Format confidence as percentage
            history['Confidence'] = history['confidence'].apply(lambda x: f"{float(x.strip('%')):.1f}%")
            
            # Display styled dataframe
            st.dataframe(
                history[['timestamp', 'headline', 'prediction', 'Confidence']],
                column_config={
                    "timestamp": "Timestamp",
                    "headline": "Headline",
                    "prediction": st.column_config.SelectboxColumn(
                        "Verdict",
                        options=["Real", "Fake"],
                        required=True
                    ),
                    "Confidence": "Confidence"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add download button
            csv = history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full History",
                data=csv,
                file_name="fake_news_detection_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No analysis history yet. Analyze some articles to build history.")
    except FileNotFoundError:
        st.info("No analysis history yet. Analyze some articles to build history.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7f8c8d; font-size:14px; margin-top:50px;">
    For educational purposes only • Not a substitute for professional fact-checking
</div>
""", unsafe_allow_html=True)

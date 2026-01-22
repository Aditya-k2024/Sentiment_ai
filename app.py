import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import plotly.graph_objects as go

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --- 1. SETUP & CACHING ---
# We use @st.cache_resource so we only load the heavy model/NLTK data ONCE.

@st.cache_resource
def load_resources():
    # A. Download NLTK data quietly
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
            
    # B. Load Model & Vectorizer
    try:
        model = joblib.load('Notebook & models/lr_model.pkl')
        vectorizer = joblib.load('Notebook & models/tf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Load them now
model, tfidf = load_resources()

# --- 2. PREPROCESSING FUNCTIONS ---
# This must match EXACTLY what you used during training

# Define helper for POS tagging
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Initialize tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove negatives from stop_words (Critical for sentiment)
    for negative in ['no', 'not', 'nor', 'never', "don't", "won't", "didn't", "isn't"]:
        stop_words.discard(negative)
        
    # A. Clean HTML
    text = re.sub(r'<<.*?>>', ' ', text)
    
    # B. Keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # C. Tokenize
    tokens = word_tokenize(text)
    
    # D. POS Tag & Lemmatize
    pos_tags = nltk.pos_tag(tokens)
    clean_tokens = []
    
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        
        if lemma not in stop_words:
            clean_tokens.append(lemma)
            
    return " ".join(clean_tokens)

# --- UI LAYOUT: CYBER DASHBOARD (BIG INPUT VERSION) ---
def render_hacker_dashboard():
    # 1. CYBER CSS
    st.markdown("""
    <style>
        /* Import Hacker Font */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

        /* Main App Background */
        .stApp {
            background-color: #050505;
            color: #00ff41;
            font-family: 'Share Tech Mono', monospace;
        }

        /* Input Area - NOW BIGGER */
        .stTextArea textarea {
            background-color: #0d0d0d;
            color: #00ff41;
            border: 1px solid #333;
            font-family: 'Share Tech Mono', monospace;
            font-size: 1.1rem; /* Slightly larger text for readability */
        }
        .stTextArea textarea:focus {
            border: 1px solid #00ff41;
            box-shadow: 0 0 10px #00ff41;
        }

        /* The Button */
        .stButton > button {
            background-color: black;
            color: #00ff41;
            border: 1px solid #00ff41;
            text-transform: uppercase;
            letter-spacing: 2px;
            width: 100%;
            height: 60px; /* Taller button to match big input */
            font-size: 1.2rem;
            transition: 0.3s;
            margin-top: 10px;
        }
        .stButton > button:hover {
            background-color: #00ff41;
            color: black;
            box-shadow: 0 0 15px #00ff41;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Share Tech Mono', monospace;
            text-transform: uppercase;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. HEADER
    st.markdown("<h1 style='text-align: center; border-bottom: 1px solid #333; padding-bottom: 10px;'>/// SENTIMENT_ANALYSIS ///</h1>", unsafe_allow_html=True)
    
    # 3. INPUT (Changed height from 100 to 250)
    st.write(">> PASTE_DATA_STREAM_BELOW:")
    user_input = st.text_area("Label Hidden", label_visibility="collapsed", height=100)
    
    # 4. BUTTON
    analyze = st.button("INITIATE_ANALYSIS_PROTOCOL")

    # 5. LOGIC & VISUALIZATION
    if analyze and user_input:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        
        # Get Prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Get Probability
        try:
            proba = model.predict_proba(vectorized_text)[0]
            confidence = max(proba) 
        except:
            pass
            
        # --- BUILD THE CHART ---
        if prediction == 1 or prediction == "positive":
            chart_color = "#00ff41" # Matrix Green
            result_text = "POSITIVE"
        else:
            chart_color = "#ff003c" # Cyber Red
            result_text = "NEGATIVE"

        fig = go.Figure(data=[go.Pie(
            labels=['Confidence', 'Uncertainty'],
            values=[confidence, 1-confidence],
            hole=.7, 
            marker_colors=[chart_color, '#1a1a1a'],
            textinfo='none', 
            hoverinfo='label+percent',
            sort=False
        )])

        fig.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            annotations=[dict(text=f"{int(confidence*100)}%", x=0.5, y=0.5, font_size=40, showarrow=False, font_color=chart_color, font_family="Share Tech Mono")]
        )

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            st.markdown(f"<h3 style='color: {chart_color}'> >> ANALYSIS_COMPLETE</h3>", unsafe_allow_html=True)
            st.markdown(f"**DETECTED CLASS:** {result_text}")
            st.markdown(f"**SYSTEM CONFIDENCE:** {confidence*100:.2f}%")
            st.code(f"RAW_TEXT_SAMPLE: {user_input[:50]}...", language="bash")

        with r_col2:
            st.markdown(f"<div style='text-align: center; color: {chart_color}'>NEURAL_CERTAINTY_MATRIX</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

# CALL THE FUNCTION
if model:
    render_hacker_dashboard()

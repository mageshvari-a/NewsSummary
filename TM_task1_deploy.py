import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import re
import string
import os

# ✅ Ensure NLTK knows where to look
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

# ✅ Debugging: Check if NLTK path is set correctly
print("🔍 NLTK Data Path:", NLTK_DATA_PATH)
print("📂 Files in NLTK Directory:", os.listdir(NLTK_DATA_PATH) if os.path.exists(NLTK_DATA_PATH) else "❌ Not Found!")

# ✅ Check and load tokenizer
try:
    tokenizer_path = "tokenizers/punkt/english.pickle"
    tokenizer = nltk.data.load(tokenizer_path)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

# ✅ Check if VADER lexicon exists
vader_path = os.path.join(NLTK_DATA_PATH, "sentiment/vader_lexicon.txt")
if not os.path.exists(vader_path):
    raise FileNotFoundError(f"❌ Vader lexicon not found! Expected path: {vader_path}")

# Ensure the analyzer loads from the correct path
sia = SentimentIntensityAnalyzer()

# ✅ Function to perform sentiment analysis
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# ✅ Function to generate a word cloud
def generate_wordcloud(text, title):
    if not text.strip():
        st.warning(f"⚠️ No valid words found for {title}. Skipping word cloud.")
        return
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=15)
    st.pyplot(plt)

# ✅ Function to generate unigram & bigram word clouds
def generate_ngram_wordcloud(text, n, title):
    if not text.strip():
        st.warning(f"⚠️ No valid words found for {title}. Skipping word cloud.")
        return

    tokens = word_tokenize(text.lower())  # Tokenize & convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove numbers & special characters
    if not tokens:
        st.warning(f"⚠️ No valid n-grams found for {title}. Skipping word cloud.")
        return

    ngram_list = list(ngrams(tokens, n))
    ngram_freq = Counter(ngram_list)

    # Convert to string format for WordCloud
    ngram_words = {' '.join(gram): freq for gram, freq in ngram_freq.items()}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=15)
    st.pyplot(plt)

# ✅ Streamlit UI
st.title("📰 News Sentiment Analysis & Word Clouds")
st.write("Upload a `.txt` file with news summaries for analysis.")

# ✅ Upload file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    # Read file content
    data = uploaded_file.read().decode("utf-8").strip()
    
    if not data:
        st.error("❌ The uploaded file is empty. Please upload a valid text file.")
    else:
        # ✅ Extract summaries using regex
        summaries = re.findall(r"Summary:\s*(.*?)\nKeywords:", data, re.DOTALL)

        if not summaries:
            st.error("❌ No summaries found in the file. Please check the file format.")
        else:
            # ✅ Debugging: Show extracted summaries
            st.subheader("🔎 Extracted Summaries (First 3)")
            st.write(summaries[:3])

            # ✅ Create DataFrame
            df = pd.DataFrame({"summary": summaries})

            # ✅ Perform sentiment analysis
            df['sentiment_score'] = df['summary'].apply(analyze_sentiment)
            df['sentiment_label'] = df['sentiment_score'].apply(
                lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
            )

            # ✅ Show sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df['sentiment_label'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["green", "red", "gray"], ax=ax)
            ax.set_title("Sentiment Distribution")
            ax.set_ylabel("Number of Articles")
            st.pyplot(fig)

            # ✅ Show DataFrame
            st.subheader("📋 Processed Data")
            st.dataframe(df)

            # ✅ Concatenate all summaries for word clouds
            all_text = " ".join(df['summary']).strip()

            # ✅ Check if text is valid
            if all_text:
                st.write("✅ Processed Text for Word Cloud (First 300 chars)")
                st.write(all_text[:300])  # Show first 300 characters

                # ✅ Generate Unigram Word Cloud
                st.subheader("☁️ Unigram Word Cloud")
                generate_wordcloud(all_text, "Unigram Word Cloud")

                # ✅ Generate Bigram Word Cloud
                st.subheader("☁️ Bigram Word Cloud")
                generate_ngram_wordcloud(all_text, 2, "Bigram Word Cloud")
            else:
                st.error("❌ No valid text found for word cloud. Check file content.")

            # ✅ Download processed data
            processed_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sentiment Data", processed_csv, "sentiment_results.csv", "text/csv")

st.write("🚀 **Built with Streamlit & NLTK**")

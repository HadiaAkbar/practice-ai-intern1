import nltk
import heapq
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import PyPDF2
import os

# Setup NLTK (very reliable on Streamlit)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class DocumentSummarizer:
    def __init__(self):
        # We use NLTK instead of spaCy for better stability on Streamlit Cloud
        self.stop_words = set(stopwords.words('english'))
        
        # Load AI model
        try:
            # Using a very stable, small model
            self.ai_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except:
            self.ai_model = None

    def get_sentences_and_words(self, text):
        # Use NLTK's sent_tokenize instead of spaCy
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum() and w not in self.stop_words]
        return sentences, clean_words

    def frequency_based_summary(self, text, count=3):
        sents, words = self.get_sentences_and_words(text)
        if not sents: return ""
        
        # Count words
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        
        if not freq: return ""
        
        # Score sentences
        scores = {}
        for s in sents:
            for w in word_tokenize(s.lower()):
                if w in freq:
                    scores[s] = scores.get(s, 0) + freq[w]
        
        best_sents = heapq.nlargest(count, scores, key=scores.get)
        return " ".join(best_sents)

    def tfidf_based_summary(self, text, count=3):
        sents, _ = self.get_sentences_and_words(text)
        if len(sents) <= count: return text
        
        vec = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform(sents)
        scores = matrix.toarray().sum(axis=1)
        
        top_idx = scores.argsort()[-count:][::-1]
        # Keep original order
        return " ".join([sents[i] for i in sorted(top_idx)])

    def abstractive_summary(self, text):
        if not self.ai_model: return "AI model not loaded."
        try:
            # Simple chunking for long text
            res = self.ai_model(text[:1024], max_length=130, min_length=30, do_sample=False)
            return res[0]['summary_text']
        except Exception as e:
            return f"Error: {str(e)}"

    def get_analytics(self, text):
        sents, words = self.get_sentences_and_words(text)
        word_counts = pd.Series(words).value_counts().head(10)
        
        return {
            "word_freq": word_counts.to_dict(),
            "keywords": word_counts.index.tolist()[:5],
            "num_sentences": len(sents),
            "num_words": len(words)
        }

def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text += p.extract_text()
        return text
    return ""

def save_summary(text, path, type='txt'):
    if type == 'txt':
        with open(path, 'w') as f:
            f.write(text)
    elif type == 'pdf':
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(path, pagesize=letter)
        t = c.beginText(40, 750)
        t.setFont("Helvetica", 10)
        # Simple wrapping
        for line in text.split('.'):
            if line.strip():
                t.textLine(line.strip() + '.')
        c.drawText(t)
        c.save()

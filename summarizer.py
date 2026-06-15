import nltk
import spacy
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import PyPDF2
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class DocumentSummarizer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python3 -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.stop_words = set(stopwords.words('english'))
        # Initialize abstractive summarizer as a bonus feature
        try:
            # Using a smaller model for stability in the environment
            self.abstractive_model = pipeline("text2text-generation", model="t5-small")
        except Exception as e:
            print(f"Error loading abstractive model: {e}")
            self.abstractive_model = None

    def preprocess_text(self, text):
        # Lowercasing
        text = text.lower()
        # Tokenization and sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        words = word_tokenize(text)
        # Stopword removal
        filtered_words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return sentences, filtered_words

    def frequency_based_summary(self, text, num_sentences=3):
        sentences, filtered_words = self.preprocess_text(text)
        
        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
        if not word_frequencies:
            return ""

        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    if len(sent.split(' ')) < 30:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

    def tfidf_based_summary(self, text, num_sentences=3):
        sentences, _ = self.preprocess_text(text)
        if len(sentences) <= num_sentences:
            return text

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Sum TF-IDF scores for each sentence
        sentence_scores = tfidf_matrix.toarray().sum(axis=1)
        
        # Rank sentences
        ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[::-1]]
        summary = ' '.join(ranked_sentences[:num_sentences])
        return summary

    def abstractive_summary(self, text):
        if not self.abstractive_model:
            return "Abstractive summarization model is currently unavailable."
        # Handle long text by chunking if necessary
        max_chunk = 512
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        try:
            summaries = self.abstractive_model(chunks, max_length=100, min_length=30, do_sample=False)
            return " ".join([s.get('generated_text', s.get('summary_text', '')) for s in summaries])
        except Exception as e:
            return f"Error during abstractive summarization: {str(e)}"

    def get_analytics(self, text):
        sentences, filtered_words = self.preprocess_text(text)
        
        # Word frequency analysis
        word_freq = pd.Series(filtered_words).value_counts().head(10)
        
        # Keyword extraction (top 5)
        keywords = word_freq.index.tolist()[:5]
        
        # Sentence importance scoring (using frequency)
        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        sentence_scores = []
        for sent in sentences:
            score = sum(word_frequencies.get(word, 0) for word in word_tokenize(sent.lower()) if word in word_frequencies)
            sentence_scores.append(score)
            
        return {
            "word_freq": word_freq.to_dict(),
            "keywords": keywords,
            "sentence_scores": sentence_scores,
            "num_sentences": len(sentences),
            "num_words": len(filtered_words)
        }

def read_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif extension.lower() == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    return None

def save_summary(summary, output_path, format='txt'):
    if format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    elif format == 'pdf':
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        text_object = c.beginText(40, height - 40)
        text_object.setFont("Helvetica", 12)
        
        # Simple line wrapping
        lines = summary.split('\n')
        for line in lines:
            # Very basic wrap
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + word) < 90:
                    current_line += word + " "
                else:
                    text_object.textLine(current_line)
                    current_line = word + " "
            text_object.textLine(current_line)
            
        c.drawText(text_object)
        c.save()

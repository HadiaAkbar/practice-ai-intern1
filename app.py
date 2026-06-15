import streamlit as st
import os
from summarizer import DocumentSummarizer, read_file, save_summary
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(page_title="AI-Powered Document Summarizer", layout="wide")

# Initialize Summarizer
@st.cache_resource
def load_summarizer():
    return DocumentSummarizer()

summarizer = load_summarizer()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Summarizer", "Analytics", "About"])

st.title("TEYZIX CORE Internship - AI Summarizer")

if page == "Summarizer":
    st.header("Document Summarization")
    
    # Input Method
    input_method = st.radio("Select Input Method", ["Direct Text", "Upload File"])
    
    input_text = ""
    if input_method == "Direct Text":
        input_text = st.text_area("Enter text to summarize", height=300)
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
        if uploaded_file is not None:
            # Save temporary file
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            input_text = read_file(uploaded_file.name)
            os.remove(uploaded_file.name)

    if input_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summarization Settings")
            method = st.selectbox("Select Approach", ["Frequency-based", "TF-IDF based", "Abstractive (Transformers)"])
            summary_length = st.slider("Summary Length (Number of sentences)", 1, 10, 3)
            
        if st.button("Generate Summary"):
            with st.spinner("Processing..."):
                if method == "Frequency-based":
                    summary = summarizer.frequency_based_summary(input_text, summary_length)
                elif method == "TF-IDF based":
                    summary = summarizer.tfidf_based_summary(input_text, summary_length)
                else:
                    summary = summarizer.abstractive_summary(input_text)
                
                st.session_state['last_summary'] = summary
                st.session_state['last_input'] = input_text

            st.subheader("Summary Result")
            st.write(summary)
            
            # Export options
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("Export as TXT"):
                    save_summary(summary, "outputs/summary.txt", "txt")
                    st.success("Saved to outputs/summary.txt")
            with export_col2:
                if st.button("Export as PDF"):
                    save_summary(summary, "outputs/summary.pdf", "pdf")
                    st.success("Saved to outputs/summary.pdf")

            with st.expander("Compare Original vs Summarized"):
                c1, c2 = st.columns(2)
                c1.markdown("**Original Text**")
                c1.write(input_text)
                c2.markdown("**Summarized Text**")
                c2.write(summary)

elif page == "Analytics":
    st.header("Document Analytics")
    
    if 'last_input' in st.session_state:
        text = st.session_state['last_input']
        analytics = summarizer.get_analytics(text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentences", analytics['num_sentences'])
        col2.metric("Filtered Words", analytics['num_words'])
        col3.write("**Top Keywords**")
        col3.write(", ".join(analytics['keywords']))
        
        st.subheader("Word Frequency Analysis")
        df_freq = pd.DataFrame(list(analytics['word_freq'].items()), columns=['Word', 'Frequency'])
        st.bar_chart(df_freq.set_index('Word'))
        
        st.subheader("Sentence Importance Scoring")
        st.line_chart(analytics['sentence_scores'])
    else:
        st.info("Please run the summarizer first to see analytics.")

elif page == "About":
    st.header("About the Project")
    st.write("""
    This AI-Powered Document Summarization System was developed as part of the TEYZIX CORE Internship (June Batch).
    
    **Features:**
    - Extractive Summarization (Frequency & TF-IDF)
    - Abstractive Summarization (BART Transformer)
    - PDF & TXT File Support
    - Interactive Analytics Dashboard
    - Export functionality (TXT/PDF)
    
    **Developed by:** Intern
    """)

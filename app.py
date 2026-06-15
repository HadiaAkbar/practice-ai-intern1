import streamlit as st
import os
from summarizer import DocumentSummarizer, read_file, save_summary
import pandas as pd

# Simple Page Config
st.set_page_config(page_title="Intern Project: AI Summarizer")

# Load the summarizer (cached so it doesn't reload every time)
@st.cache_resource
def get_model():
    return DocumentSummarizer()

model = get_model()

# Simple Sidebar
st.sidebar.title("Menu")
choice = st.sidebar.selectbox("Select Page", ["Home", "Analytics", "About"])

st.title("AI Document Summarizer")
st.markdown("---")

if choice == "Home":
    st.subheader("Summarize Your Document")
    
    # Input options
    option = st.selectbox("How to input text?", ["Type it here", "Upload a file"])
    
    text_data = ""
    if option == "Type it here":
        text_data = st.text_area("Paste your text here:", height=200)
    else:
        file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
        if file:
            # Simple way to handle uploaded file
            with open("temp_file", "wb") as f:
                f.write(file.getbuffer())
            text_data = read_file("temp_file")
            os.remove("temp_file")

    if text_data:
        # Settings
        st.write("### Settings")
        method = st.radio("Choose Method:", ["Simple Frequency", "TF-IDF", "AI Abstractive"])
        num_sent = st.slider("Number of sentences:", 1, 10, 3)
        
        if st.button("Start Summarizing"):
            with st.spinner("Wait a moment..."):
                if method == "Simple Frequency":
                    result = model.frequency_based_summary(text_data, num_sent)
                elif method == "TF-IDF":
                    result = model.tfidf_based_summary(text_data, num_sent)
                else:
                    result = model.abstractive_summary(text_data)
                
                st.session_state['summary'] = result
                st.session_state['original'] = text_data
            
            st.success("Done!")
            st.write("### Summary:")
            st.write(result)
            
            # Export buttons
            c1, c2 = st.columns(2)
            if c1.button("Save as Text"):
                save_summary(result, "outputs/summary.txt", "txt")
                st.info("Saved to outputs/summary.txt")
            if c2.button("Save as PDF"):
                save_summary(result, "outputs/summary.pdf", "pdf")
                st.info("Saved to outputs/summary.pdf")

elif choice == "Analytics":
    st.subheader("Data Analysis")
    if 'original' in st.session_state:
        data = model.get_analytics(st.session_state['original'])
        
        st.write(f"**Total Sentences:** {data['num_sentences']}")
        st.write(f"**Total Words (clean):** {data['num_words']}")
        st.write(f"**Top Keywords:** {', '.join(data['keywords'])}")
        
        # Simple Charts
        st.write("#### Word Frequency Chart")
        df = pd.DataFrame(list(data['word_freq'].items()), columns=['Word', 'Count'])
        st.bar_chart(df.set_index('Word'))
    else:
        st.warning("Please summarize something first!")

else:
    st.subheader("Project Info")
    st.write("Developed by: AI Intern")
    st.write("Project: AI-Powered Document Summarizer")
    st.write("Company: TEYZIX CORE")

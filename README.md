# AI-Powered Document Summarization System

## Project Overview
This project is an AI-driven system designed to automatically summarize long text documents. It was developed as part of the **TEYZIX CORE Internship (June Batch)**. The system provides both extractive and abstractive summarization techniques, a user-friendly Streamlit interface, and detailed document analytics.

## Features
- **Data Input System**: Supports direct text input, .txt files, and .pdf files.
- **Text Preprocessing**: Includes lowercasing, stopword removal, tokenization, and sentence segmentation using NLTK and spaCy.
- **Summarization Logic**:
  - **Extractive**: Frequency-based scoring and TF-IDF based ranking.
  - **Abstractive (Bonus)**: Uses Hugging Face Transformers (BART model).
- **Output System**: Adjustable summary length and comparison view.
- **Analytics Module**: Word frequency analysis, keyword extraction, and sentence importance scoring.
- **File Handling**: Export summaries as .txt or .pdf files.
- **Streamlit UI**: A live web-based interface for easy interaction.

## Installation & Setup

1. **Clone the project folder**
2. **Install dependencies**:
   ```bash
   pip install nltk spacy streamlit transformers PyPDF2 reportlab pandas matplotlib seaborn torch
   ```
3. **Download NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main Streamlit application and UI logic.
- `summarizer.py`: Core logic for text processing, summarization, and analytics.
- `sample_docs/`: Directory containing test documents.
- `outputs/`: Directory where exported summaries are saved.
- `requirements.txt`: List of required Python packages.

## Technical Details
- **Language**: Python
- **Libraries**: NLTK, spaCy, Streamlit, Transformers, Scikit-learn, Pandas, Matplotlib.
- **Design**: Modular function-based architecture for clean and readable code.

---
**Developed by:** Intern (TEYZIX CORE June Batch)

# CX Sentiment Analyzer PT-BR

Customer Service sentiment analyzer in Brazilian Portuguese using Python and Streamlit.

## 🎯 Business Problem

Support teams receive dozens of tickets per day in Portuguese and struggle to quickly identify which ones are **frustrated customers** vs **satisfied customers**.  
Goal: detect sentiment (Ruim/Bom) from free-text tickets to help CX and Product prioritize actions.

## 🧠 What this project does

- Creates a small CX dataset in PT-BR (100 synthetic tickets) based on real support situations.
- Cleans and normalizes the text (`text_clean`) for analysis.
- Builds a simple PT-BR sentiment model using a custom lexicon (words like *demora, erro, ótimo, excelente*).
- Achieves around **83% accuracy** against logical labels (regra de negócio).
- Exposes the model via a **Streamlit app** where you type a ticket and get:
  - A sentiment score.
  - A prediction: **Bom** or **Ruim**.

## 🧪 Tech stack

- Python (pandas, seaborn, nltk)
- Jupyter Notebook for EDA and modeling: `cx_ticket_sentiment_modeling.ipynb`
- Streamlit app: `app.py`

## 📚 Learning & Pitfalls

- External CSV URLs can break (HTTP 404) → I created my own PT-BR CX dataset.
- VADER (English lexicon) performed poorly in PT-BR → I built a custom PT-BR lexicon.
- Random labels produced low accuracy → I defined a logical labelling rule based on domain knowledge.
- Mixing `.ipynb` and `.py` content broke the app → I separated analysis (notebook) and product (`app.py`).

## 🚀 Next steps

- Replace the lexicon-based model with a TF-IDF + Logistic Regression classifier.
- Evaluate on a larger real-world CX dataset.
- Deploy the Streamlit app publicly.

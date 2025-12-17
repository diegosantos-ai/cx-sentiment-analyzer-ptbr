# CX Sentiment Analyzer PT-BR

Customer Service sentiment analyzer in Brazilian Portuguese using Python and Streamlit.

##  Business Problem

Support teams receive dozens of tickets per day in Portuguese and struggle to quickly identify which ones are **frustrated customers** vs **satisfied customers**.  
Goal: detect sentiment (Ruim/Bom) from free-text tickets to help CX and Product prioritize actions.

##  What this project does

- Creates a small CX dataset in PT-BR (100 synthetic tickets) based on real support situations.
- Cleans and normalizes the text (`text_clean`) for analysis.
- Builds a simple PT-BR sentiment model using a custom lexicon (words like *demora, erro, ótimo, excelente*).
- Achieves around **83% accuracy** against logical labels (regra de negócio).
- Exposes the model via a **Streamlit app** where you type a ticket and get:
  - A sentiment score.
  - A prediction: **Bom** or **Ruim**.

##  Tech stack

- Python (pandas, seaborn, nltk)
- Jupyter Notebook for EDA and modeling: `cx_ticket_sentiment_modeling.ipynb`
- Streamlit app: `app.py`

##  Learning & Pitfalls

- External CSV URLs can break (HTTP 404) → I created my own PT-BR CX dataset.
- VADER (English lexicon) performed poorly in PT-BR → I built a custom PT-BR lexicon.
- Random labels produced low accuracy → I defined a logical labelling rule based on domain knowledge.
- Mixing `.ipynb` and `.py` content broke the app → I separated analysis (notebook) and product (`app.py`).
- In tests with external sentences (20 examples outside the original dataset), the model performs well on clear complaints but tends to classify praises and neutral sentences as **Negative**, with 60–70% confidence.
- This reveals a bias in the training data (focused mainly on customer pain points) and highlights the importance of having a manually labeled validation set and more diverse positive/neutral examples.
- Next iteration: create a small "golden set" with human labels (Positive / Negative / Neutral) and enrich the training dataset with short positive sentences and good-recovery cases (problems that were solved well).


##  Next steps

- Evolve from a lexical model to a supervised classifier using TF-IDF + Logistic Regression (implemented in the notebook, achieving 100% accuracy on the current test set).
- Increase the dataset with more real/synthetic Customer Experience (CX) tickets in PT-BR to validate the model's generalization.
- Apply stratified cross-validation (k-fold) to obtain a more robust performance estimate.
- Compare with other algorithms (Naive Bayes, Random Forest) and choose the best trade-off between performance and simplicity.
- Publish the Streamlit app with the supervised model in production.


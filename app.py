import streamlit as st

# ----- léxico PT-BR -----
ptbr_positive = [
    "ótimo", "excelente", "perfeita", "rápido", "eficiente",
    "obrigado", "educado", "atencioso", "resolvido", "funciona",
    "recomendo", "nota 10", "perfeito"
]

ptbr_negative = [
    "grosseiro", "demora", "travando", "erro", "ruim",
    "inaceitável", "urgente", "sem resposta", "reembolso",
    "atraso", "atrasada", "atrasado", "demorou", "espera",
    "aguardando", "não funciona"
]

def sentiment_ptbr(text):
    text = text.lower()
    pos_score = sum(1 for w in ptbr_positive if w in text)
    neg_score = sum(1 for w in ptbr_negative if w in text)
    return (pos_score - neg_score) / max(1, len(text.split()))

st.title("CX Sentiment Analyzer PT-BR")
st.write("Modelo baseado em tickets de atendimento em português.")

ticket = st.text_area(
    "Digite o texto do atendimento:",
    "Atendente demorou para responder no chat"
)

if st.button("Analisar sentimento"):
    score = sentiment_ptbr(ticket)
    label = "Bom" if score >= 0 else "Ruim"
    st.write(f"Score: {score:.2f}")
    st.write(f"Predição: {label}")

import streamlit as st
import pandas as pd
import joblib
import os
import re
from datetime import datetime

# ----- Configuração da página -----
st.set_page_config(
    page_title="CX Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Estilização Customizada -----
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2C3E50;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ----- Funções Auxiliares -----
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../data/sentiment_model.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("⚠️ Modelo analítico não encontrado. Por favor, contate o suporte técnico.")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\u00C0-\u00FF\s]', '', text)
    text = text.lower()
    return text.strip()

def save_feedback(text, predicted, corrected):
    feedback_file = os.path.join(os.path.dirname(__file__), '../data/feedback.csv')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    df_new = pd.DataFrame([{
        "timestamp": timestamp,
        "text": text,
        "predicted_sentiment": predicted,
        "corrected_sentiment": corrected
    }])
    
    if not os.path.exists(feedback_file):
        df_new.to_csv(feedback_file, index=False)
    else:
        df_new.to_csv(feedback_file, mode='a', header=False, index=False)

# ----- Inicializar Estado -----
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Texto", "Sentimento", "Confiança", "Feedback"])
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ----- Lógica do Modelo -----
model = load_model()

def classify_sentiment(text: str, model) -> tuple:
    if not model:
        return None, 0.0, {}
    
    cleaned_text = clean_text(text)
    
    # Predict probabilities
    try:
        probas = model.predict_proba([cleaned_text])[0]
        classes = model.classes_
        probs_dict = dict(zip(classes, probas))
        
        predicted_label = model.predict([cleaned_text])[0]
        confidence = probs_dict[predicted_label]
        
        return predicted_label, confidence, probs_dict
    except Exception as e:
        st.error(f"Erro na inferência: {str(e)}")
        return None, 0.0, {}

def get_cx_insight(label: str) -> str:
    insights = {
        "Bom": "Sinal POSITIVO. O cliente expressa satisfação ou aprovação. Recomendação: Monitorar fidelização, identificar promotores e reforçar o comportamento positivo.",
        "Ruim": "Sinal DE ALERTA. O cliente expressa insatisfação, frustração ou problema crítico. Recomendação: Prioridade ATA (Alta), acionamento de suporte preventivo e análise de causa raiz.",
        "Neutro": "Sinal INFORMATIVO. O texto é imparcial ou não contém carga emocional clara. Recomendação: Acompanhar evolução e contexto para evitar detração silenciosa."
    }
    return insights.get(label, "Sem recomendação disponível.")

# ----- UI Principal -----
st.title("🧠 CX Sentiment Intelligence")
st.markdown("### Análise Preditiva de Sentimento & Feedback Loop")
st.markdown("---")

# ----- Sidebar -----
with st.sidebar:
    st.header("Painel de Controle")
    st.info("Modelo v2.0 (Treinado em 900k amostras)")
    
    st.markdown("---")
    st.subheader("Métricas da Sessão")
    if len(st.session_state.history) > 0:
        total = len(st.session_state.history)
        sentiment_counts = st.session_state.history['Sentimento'].value_counts()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total", total)
        with col_m2:
            neg_pct = (sentiment_counts.get('Ruim', 0) / total) * 100
            st.metric("Risco (%)", f"{neg_pct:.1f}%")
            
        st.markdown("#### Distribuição")
        st.bar_chart(sentiment_counts)
    else:
        st.markdown("*Aguardando dados...*")
    
    if st.button("Limpar Sessão", help="Reiniciar histórico local"):
        st.session_state.history = st.session_state.history.iloc[0:0]
        st.rerun()

# ----- Área de Análise -----
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Entrada de Dados")
    input_text = st.text_area(
        "Insira o comentário do cliente:",
        height=150,
        placeholder="Ex: O atendimento foi excelente, mas o prazo de entrega deixou a desejar..."
    )
    
    analyze_btn = st.button("Executar Análise", type="primary", use_container_width=True)

with col_result:
    st.subheader("Diagnóstico")
    
    if analyze_btn and input_text:
        with st.spinner("Processando linguagem natural..."):
            label, conf, probs = classify_sentiment(input_text, model)
            
            if label:
                # Salvar estado para feedback
                st.session_state.current_prediction = {
                    "text": input_text,
                    "label": label,
                    "confidence": conf,
                    "probs": probs
                }
                st.session_state.feedback_submitted = False
                
    # Exibir Resultado (Persistente até nova análise)
    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        
        # Header do Resultado
        color_map = {"Bom": "green", "Ruim": "red", "Neutro": "orange"}
        color = color_map.get(pred['label'], "gray")
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(240,242,246,0.5); border-left: 6px solid {color};">
            <h2 style="margin:0; color: {color};">{pred['label'].upper()}</h2>
            <p style="margin:0; font-size: 1.1em;">Confiabilidade do Modelo: <strong>{pred['confidence']:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Insight Estratégico")
        st.info(get_cx_insight(pred['label']))
        
        with st.expander("Ver Detalhes Probabilísticos"):
            st.json(pred['probs'])
        
        # ----- Feedback Loop -----
        st.markdown("---")
        st.markdown("#### Validação Humana")
        
        if not st.session_state.feedback_submitted:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("👍 Correto", use_container_width=True):
                    save_feedback(pred['text'], pred['label'], pred['label'])
                    st.session_state.feedback_submitted = True
                    st.toast("Feedback registrado! O modelo agradece.", icon="🤖")
                    
                    # Add to history
                    new_row = pd.DataFrame([{
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Texto": pred['text'], 
                        "Sentimento": pred['label'], 
                        "Confiança": f"{pred['confidence']:.2f}",
                        "Feedback": "Correto"
                    }])
                    st.session_state.history = pd.concat([new_row, st.session_state.history], ignore_index=True)
                    st.rerun()

            with col_f2:
                if st.button("👎 Incorreto", use_container_width=True):
                    st.session_state.show_correction = True

            if st.session_state.get("show_correction", False):
                with st.form("correction_form"):
                    correct_label = st.selectbox("Qual seria a classificação correta?", ["Bom", "Neutro", "Ruim"])
                    submit_correction = st.form_submit_button("Enviar Correção")
                    
                    if submit_correction:
                        save_feedback(pred['text'], pred['label'], correct_label)
                        st.session_state.feedback_submitted = True
                        st.session_state.show_correction = False
                        st.success("Correção enviada para retreino futuro.")
                        
                        # Add to history
                        new_row = pd.DataFrame([{
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Texto": pred['text'], 
                            "Sentimento": pred['label'], 
                            "Confiança": f"{pred['confidence']:.2f}",
                            "Feedback": f"Corrigido para {correct_label}"
                        }])
                        st.session_state.history = pd.concat([new_row, st.session_state.history], ignore_index=True)
                        st.rerun()
        else:
            st.success("✅ Feedback registrado com sucesso.")

# ----- Histórico Recente -----
st.markdown("---")
st.subheader("Histórico da Sessão")
if len(st.session_state.history) > 0:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

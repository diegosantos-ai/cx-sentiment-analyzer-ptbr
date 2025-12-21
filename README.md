# CX Sentiment Intelligence (PT-BR) 🧠

Sistema de inteligência de sentimento e Customer Experience (CX) focado em português brasileiro. Este projeto utiliza Machine Learning para classificar tickets e comentários de clientes, fornecendo insights estratégicos e recomendações de ação.

---

## 🚀 Funcionalidades

### 1. Modelagem Preditiva Avançada
- **Machine Learning**: Modelo de Regressão Logística (TF-IDF) treinado em **+900.000 amostras** de comentários reais (Twitter/Reviews).
- **Classificação**: Identifica sentimentos **Bom**, **Ruim** e **Neutro** com alta precisão para o contexto brasileiro.
- **Probabilístico**: Fornece nível de confiança (%) para cada predição.

### 2. Dashboard Estratégico
- **Interface Profissional**: UI moderna e intuitiva focada em analistas de CX.
- **Insights de Negócio**: Traduz a classificação técnica em recomendações acionais (ex: "Risco de Churn", "Oportunidade de Fidelização").
- **Métricas em Tempo Real**: Monitoramento de risco e KPIs da sessão atual.

### 3. Feedback Loop (Aprendizado Contínuo)
- **Validação Humana**: Botões de `👍 Correto` e `👎 Incorreto` para validar as análises.
- **Coleta de Dados**: Correções são salvas automaticamente em `data/feedback.csv` para re-treinamento futuro do modelo, garantindo que o sistema aprenda com os erros.

---

## 📂 Estrutura do Projeto

```
cx-sentiment-analyzer-ptbr/
├── dashboard/
│   └── app.py              # Aplicação Streamlit (Frontend & Lógica)
├── data/
│   ├── TrainingDataSets/   # Datasets brutos (Twitter, Reviews)
│   ├── sentiment_model.pkl # Modelo treinado serializado
│   └── feedback.csv        # Log de feedback humano
├── notebooks/
│   └── cx_ticket_sentiment_modeling.ipynb # Notebooks de exploração
├── reports/                # Relatórios gerados
├── train_model.py          # Script de treinamento do pipeline
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

---

## 🛠️ Como Executar

### Pré-requisitos
- Python 3.9+
- Pip / Virtualenv

### Instalação

1. Clone o repositório:
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd cx-sentiment-analyzer-ptbr
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate
   # Linux/Mac:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

### Rodando a Aplicação
Execute o comando abaixo na raiz do projeto:

```bash
streamlit run dashboard/app.py
```

Acesse o dashboard em: `http://localhost:8501`

### Retreinando o Modelo
Caso adicione novos dados em `data/TrainingDataSets` ou queira incorporar o feedback:

```bash
python train_model.py
```
O script consolidará os dados, treinará um novo pipeline e atualizará o arquivo `sentiment_model.pkl`.

---

## 📊 Especificações Técnicas

- **Modelo**: Logistic Regression com class_weight='balanced'.
- **Vetorização**: TF-IDF (n-grams 1-2, max_features=100k).
- **Dados de Treino**: Twitter Sentiment Analysis Datasets (PT-BR).
- **Métricas (Test Set)**:
    - Acurácia Global: ~81%
    - F1-Score (Macro): 0.82

---

## 🤝 Contribuição & Feedback Loop

O diferencial deste projeto é a capacidade de evolução. Utilize a interface do dashboard para corrigir classificações incorretas. Essas correções são vitais para ajustar o modelo a vocabulários específicos de diferentes nichos de negócio.

---
*Desenvolvido para times de Customer Experience.*

## Autor
**Diego Santos**  
[diegosantos-ai](https://github.com/diegosantos-ai)

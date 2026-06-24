"""
app.py — Debulhe o Código Florestal (Streamlit)

Interface de pergunta e resposta sobre a Lei 12.651/2012, com tradução para
linguagem simples. O motor de IA por trás não é exposto na interface
(prática comum em produtos -- o usuário não precisa saber qual modelo
responde, só que a resposta é confiável e cita a base legal).

Para rodar:
    streamlit run app.py

Pré-requisitos:
    pip install streamlit numpy sentence-transformers langchain langchain-core \
        langchain-anthropic beautifulsoup4 requests

Nota: a busca por similaridade usa um índice vetorial em NumPy puro (ver
pipeline.IndiceVetorial), não ChromaDB -- evita uma cadeia de dependências
(opentelemetry/protobuf) que causou falhas de import recorrentes no Streamlit
Community Cloud. Com apenas 91 artigos, um banco vetorial completo seria
desnecessário de qualquer forma.
"""

import streamlit as st

import pipeline

st.set_page_config(
    page_title="Debulhe o Código Florestal",
    page_icon="🌾",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Identidade visual (mesma paleta do protótipo HTML e do pitch)
# ---------------------------------------------------------------------------

MATA = "#1F3D2B"
MATA_ESCURA = "#14271C"
TERRA = "#B5703B"
SOL = "#E8A33D"
PERGAMINHO = "#F7F3EA"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {PERGAMINHO};
    }}
    h1, h2, h3 {{
        color: {MATA_ESCURA};
        font-family: Georgia, serif;
        font-weight: 800;
    }}
    .base-legal {{
        border: 1.5px dashed {MATA};
        border-radius: 8px;
        padding: 10px 14px;
        background-color: {PERGAMINHO};
        font-family: monospace;
        font-size: 0.85rem;
        margin-top: 12px;
    }}
    .aviso-externo {{
        background-color: #FBF1DE;
        border-left: 4px solid {SOL};
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 12px;
        font-size: 0.92rem;
        color: #6b4d1f;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inicialização do pipeline (cacheada -- roda só uma vez por sessão)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo_embedding():
    from sentence_transformers import SentenceTransformer
    import torch

    # Modelo mais leve que BAAI/bge-m3 (568M parâmetros) -- necessário porque
    # o Streamlit Community Cloud gratuito tem RAM limitada (~1GB), e o
    # bge-m3 causava o processo ser encerrado silenciosamente por excesso de
    # memória (sem traceback Python, só a tela genérica "Oh no" do Streamlit).
    # paraphrase-multilingual-MiniLM-L12-v2 (118M parâmetros, ~5x menor) é
    # suficiente para um corpus de 91 artigos em português.
    #
    # Ressalva conhecida: este mesmo modelo já apresentou falha de
    # recuperação em outro projeto (capstone IRPJ), motivando a troca para
    # bge-m3 naquele caso -- mas o contexto era diferente (pares
    # pergunta+resposta longos, com seções "Notas:" que diluíam o embedding).
    # Aqui o corpus é mais uniforme (artigos de lei). Se a qualidade da busca
    # piorar perceptivelmente, considerar voltar ao bge-m3 e usar uma conta
    # paga do Streamlit Cloud (ou outro serviço de deploy com mais RAM).
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=dispositivo
    )


@st.cache_resource(show_spinner="Baixando e processando o Código Florestal...")
def montar_indice():
    """Pipeline completo de ingestão: download -> parsing -> remissões ->
    embeddings -> indexação no índice vetorial. Roda uma única vez
    (cache_resource)."""
    artigos = pipeline.parsear_lei_completa()
    chunks = pipeline.construir_chunks(artigos)

    modelo_embedding = carregar_modelo_embedding()

    collection = pipeline.criar_indice()
    pipeline.indexar_chunks(collection, chunks, modelo_embedding)

    return collection, modelo_embedding, len(chunks)


def obter_llm():
    """Retorna a instância de LLM usada pela aplicação.

    O motor não é exposto na interface (o usuário só vê 'consultando...',
    sem nome de provedor) -- decisão de produto, não de ocultar informação:
    o código-fonte deste arquivo é público e mostra exatamente qual motor
    está em uso, para quem quiser conferir.

    A chave é lida de st.secrets (Streamlit Community Cloud) com fallback
    para variável de ambiente (uso local).
    """
    import os
    from langchain_anthropic import ChatAnthropic

    chave = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    if not chave:
        raise ValueError(
            "Chave de API não encontrada. Configure ANTHROPIC_API_KEY como "
            "Secret no Streamlit Cloud, ou como variável de ambiente localmente."
        )
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1, api_key=chave)


def gerar_resposta(pergunta: str, collection, modelo_embedding, llm) -> tuple[str, list[dict]]:
    """Roda o retriever expandido + chain de tradução, retornando a resposta
    e os documentos recuperados (para exibir a procedência, se desejado)."""
    docs = pipeline.retriever_expandido(
        pergunta, collection, modelo_embedding, k=5, profundidade_expansao=1
    )
    contexto = pipeline.formatar_contexto(docs)
    leis_externas = pipeline.formatar_leis_externas(docs)
    prompt_usuario = pipeline.montar_prompt_usuario(pergunta, contexto, leis_externas)

    mensagens = [
        ("system", pipeline.SYSTEM_PROMPT_TRADUCAO),
        ("human", prompt_usuario),
    ]
    resposta = llm.invoke(mensagens)
    return resposta.content, docs


# ---------------------------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------------------------

st.title("🌾 Debulhe o Código Florestal")
st.markdown(
    "Pergunte sobre sua propriedade. Receba a resposta em português simples, "
    "com a lei certinha citada para quem quiser confirmar."
)

try:
    collection, modelo_embedding, n_artigos = montar_indice()
    st.success(f"Base carregada: {n_artigos} artigos do Código Florestal indexados.", icon="✅")
except Exception as e:
    st.error(
        f"Não foi possível carregar a base da lei. Detalhe técnico: {e}",
        icon="⚠️",
    )
    st.stop()

pergunta = st.text_input(
    "Sua pergunta",
    placeholder="Ex: tenho 100 hectares de floresta na Amazônia, quanto preciso preservar?",
)

col1, col2 = st.columns([1, 4])
with col1:
    perguntar = st.button("Perguntar", type="primary", use_container_width=True)

if perguntar and pergunta.strip():
    try:
        llm = obter_llm()
    except ModuleNotFoundError as e:
        st.error(
            f"Biblioteca não instalada: {e}. "
            "Veja o cabeçalho deste arquivo para a lista de dependências.",
            icon="⚠️",
        )
        st.stop()

    with st.spinner("Consultando o Código Florestal..."):
        try:
            resposta, docs = gerar_resposta(pergunta, collection, modelo_embedding, llm)
        except Exception as e:
            st.error(f"Erro ao gerar a resposta: {e}", icon="⚠️")
            st.stop()

    st.markdown("### Resposta")
    st.markdown(resposta)

    with st.expander("Ver artigos recuperados (procedência)"):
        for d in docs:
            origem = "🔍 busca direta" if d["origem"] == "semantica" else "🔗 via remissão"
            st.markdown(f"**Art. {d['numero']}** — {origem}")

elif perguntar:
    st.warning("Digite uma pergunta antes de continuar.", icon="✏️")

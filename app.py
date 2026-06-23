"""
app.py — Entenda o Código Florestal (Streamlit)

Interface de pergunta e resposta sobre a Lei 12.651/2012, com tradução para
linguagem simples. Suporta dois backends de LLM, selecionáveis na barra
lateral: Claude (Anthropic) e Ollama (modelo aberto, executado localmente).

Para rodar:
    streamlit run app.py

Pré-requisitos:
    pip install streamlit numpy sentence-transformers langchain langchain-core \
        langchain-anthropic langchain-ollama beautifulsoup4 requests

Nota: a busca por similaridade usa um índice vetorial em NumPy puro (ver
pipeline.IndiceVetorial), não ChromaDB -- evita uma cadeia de dependências
(opentelemetry/protobuf) que causou falhas de import recorrentes no Streamlit
Community Cloud. Com apenas 91 artigos, um banco vetorial completo seria
desnecessário de qualquer forma.
"""

import streamlit as st

import pipeline

st.set_page_config(
    page_title="Entenda o Código Florestal",
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

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("BAAI/bge-m3", device=dispositivo)


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


def obter_llm(backend: str, modelo_ollama: str = None):
    """Retorna uma instância de LLM (LangChain) conforme o backend escolhido.

    A chave da Anthropic é lida de st.secrets (Streamlit Community Cloud) se
    disponível, com fallback para a variável de ambiente ANTHROPIC_API_KEY
    (uso local). Isso permite o mesmo código funcionar nos dois ambientes
    sem alteração.
    """
    if backend == "Claude (Anthropic)":
        import os
        from langchain_anthropic import ChatAnthropic

        chave = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
        if not chave:
            raise ValueError(
                "ANTHROPIC_API_KEY não encontrada. Configure como Secret no "
                "Streamlit Cloud, ou como variável de ambiente localmente."
            )
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.1, api_key=chave)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=modelo_ollama or "llama3.1", temperature=0.1)


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
# Barra lateral: seleção de backend
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuração do modelo")
    backend = st.radio(
        "Motor de tradução",
        ["Claude (Anthropic)", "Ollama (ferramenta aberta)"],
        index=1,
        help="A versão entregue ao haCARthon usa Ollama (ferramenta aberta, "
             "exigida pelo edital). A opção Claude fica disponível para "
             "comparação e foi usada durante o desenvolvimento.",
    )

    modelo_ollama = None
    if backend == "Ollama (ferramenta aberta)":
        modelo_ollama = st.text_input(
            "Nome do modelo Ollama",
            value="llama3.1",
            help="Use o nome exato que aparece em 'ollama list' na sua máquina.",
        )
        st.caption(
            "Requer o Ollama instalado e rodando localmente "
            "([ollama.com/download](https://ollama.com/download)). "
            "⚠️ Não funciona quando o app está publicado no Streamlit "
            "Community Cloud -- o servidor deles não tem acesso a um "
            "servidor Ollama. Use esta opção só ao rodar o app na sua "
            "própria máquina."
        )
    else:
        st.caption("Requer a variável de ambiente ANTHROPIC_API_KEY configurada.")

    st.markdown("---")
    st.caption(
        "Fonte: Lei nº 12.651/2012 (texto compilado, Casa Civil). "
        "91 artigos, grafo de remissões internas e externas."
    )

# ---------------------------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------------------------

st.title("🌾 Entenda o Código Florestal")
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
        llm = obter_llm(backend, modelo_ollama)
    except ModuleNotFoundError as e:
        st.error(
            f"Biblioteca não instalada para o backend selecionado: {e}. "
            "Veja o cabeçalho deste arquivo para a lista de dependências.",
            icon="⚠️",
        )
        st.stop()

    with st.spinner(f"Consultando o Código Florestal (via {backend})..."):
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

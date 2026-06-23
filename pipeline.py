"""
pipeline.py — lógica compartilhada entre as duas versões do app (Claude e Ollama).

Todo o código de parsing, extração de remissões, indexação e retriever
expandido já foi validado no notebook Pipeline_Completo_CodigoFlorestal.ipynb
(91 artigos, 72 remissões internas, 28 externas). Este módulo reaproveita
essas mesmas funções, sem alterações de lógica -- só organizado para ser
importado pelo Streamlit em vez de rodado célula a célula.
"""

import re
import json
import requests
from bs4 import BeautifulSoup

URL_LEI_COMPILADA = "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2012/lei/L12651compilado.htm"

# ---------------------------------------------------------------------------
# Parsing em artigos (3 correções aplicadas durante a sessão de depuração)
# ---------------------------------------------------------------------------

PADRAO_ARTIGO = re.compile(r"(?=Art\.\s*\d+[ºo°]?(?:-[A-Z])?\b)")
PADRAO_NUMERO = re.compile(r"Art\.\s*(\d+)([ºo°]?(?:-[A-Z])?)")

PADRAO_CITACAO_ENTRE_ASPAS = re.compile(r"[\x93\u201c\"][^\x94\u201d\"]*[\x94\u201d\"]", re.DOTALL)

PADRAO_CABECALHO_ESTRUTURAL = re.compile(
    r"\n\s*(CAPÍTULO\s+[IVXLCDM]+(?:-[A-Z])?\s*\n.*|Seção\s+[IVXLCDM]+\s*\n.*)",
    re.IGNORECASE | re.DOTALL
)


def baixar_texto_lei() -> str:
    """Download do texto compilado da Lei 12.651/2012 (Casa Civil)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(URL_LEI_COMPILADA, headers=headers, timeout=30)
    resp.raise_for_status()
    resp.encoding = "ISO-8859-1"

    soup = BeautifulSoup(resp.text, "html.parser")
    texto_completo = soup.get_text(separator="\n")

    indice_inicio = texto_completo.find("CAPÍTULO I")
    if indice_inicio == -1:
        raise ValueError(
            "Não encontrei 'CAPÍTULO I' no texto extraído -- o HTML pode ter mudado "
            "de estrutura, ou o encoding pode estar incorreto."
        )
    return texto_completo[indice_inicio:]


def mascarar_citacoes(texto: str) -> tuple[str, dict[str, str]]:
    mapa_restauracao = {}
    contador = 0

    def substituir(match):
        nonlocal contador
        chave = f"__CITACAO_{contador}__"
        mapa_restauracao[chave] = match.group(0)
        contador += 1
        return chave

    texto_mascarado = PADRAO_CITACAO_ENTRE_ASPAS.sub(substituir, texto)
    return texto_mascarado, mapa_restauracao


def restaurar_citacoes(texto: str, mapa_restauracao: dict[str, str]) -> str:
    for chave, original in mapa_restauracao.items():
        texto = texto.replace(chave, original)
    return texto


def remover_cabecalho_estrutural_residual(texto_artigo: str) -> str:
    match = PADRAO_CABECALHO_ESTRUTURAL.search(texto_artigo)
    if match:
        return texto_artigo[:match.start()].rstrip()
    return texto_artigo


def dividir_em_artigos(texto: str) -> list[str]:
    texto_mascarado, mapa_restauracao = mascarar_citacoes(texto)
    partes = PADRAO_ARTIGO.split(texto_mascarado)
    blocos_mascarados = [p.strip() for p in partes[1:] if p.strip()]
    return [restaurar_citacoes(b, mapa_restauracao) for b in blocos_mascarados]


def resolver_redacao_vigente(blocos: list[str]) -> dict[str, str]:
    artigos_vigentes = {}
    for b in blocos:
        m = PADRAO_NUMERO.match(b)
        if not m:
            continue
        chave = m.group(1) + m.group(2)
        artigos_vigentes[chave] = b
    return artigos_vigentes


def parsear_lei_completa() -> dict[str, str]:
    """Pipeline completo de parsing: download -> divisão em artigos ->
    remoção de cabeçalho residual. Retorna {numero_artigo: texto}."""
    texto_lei = baixar_texto_lei()
    blocos = dividir_em_artigos(texto_lei)
    artigos = resolver_redacao_vigente(blocos)
    artigos = {n: remover_cabecalho_estrutural_residual(t) for n, t in artigos.items()}
    return artigos


# ---------------------------------------------------------------------------
# Extração de remissões internas e externas
# ---------------------------------------------------------------------------

PADRAO_REMISSAO = re.compile(
    r"art(?:igo)?s?\.?\s*(\d+)[ºo°]?(?:-([A-Z]))?",
    re.IGNORECASE
)

PADRAO_NOTA_EDITORIAL = re.compile(
    r"\((?:Reda[çc][ãa]o|Inclu[íi]d[ao]|Acrescid[oa]|Vide)[^)]*\)",
    re.IGNORECASE
)

PADRAO_LEI_EXTERNA = re.compile(
    r"Lei\s+n[ºo°]?\.?\s*([\d.]+)(?:,\s*de\s*\d{1,2}\s*de\s*\w+\s*de\s*(\d{4}))?",
    re.IGNORECASE
)
NUMERO_DESTA_LEI = "12.651"


def normalizar_numero(numero: str) -> str:
    return re.sub(r"[ºo°]", "", numero)


def extrair_remissoes(texto_artigo: str, proprio_numero: str) -> list[str]:
    matches = PADRAO_REMISSAO.findall(texto_artigo)
    citados = set()
    for numero, sufixo in matches:
        chave = numero + (f"-{sufixo}" if sufixo else "")
        citados.add(chave)

    proprio_normalizado = normalizar_numero(proprio_numero)
    citados = {c for c in citados if normalizar_numero(c) != proprio_normalizado}

    return sorted(citados, key=lambda x: (int(re.match(r"\d+", x).group()), x))


def extrair_remissoes_externas(texto_artigo: str) -> list[str]:
    texto_sem_notas = PADRAO_NOTA_EDITORIAL.sub("", texto_artigo)
    matches = PADRAO_LEI_EXTERNA.findall(texto_sem_notas)
    leis_citadas = set()
    for numero, ano in matches:
        if numero == NUMERO_DESTA_LEI:
            continue
        rotulo = f"Lei {numero}" + (f"/{ano}" if ano else "")
        leis_citadas.add(rotulo)
    return sorted(leis_citadas)


def construir_chunks(artigos: dict[str, str]) -> list[dict]:
    """Monta a lista de chunks com metadados de remissão, pronta para indexar."""
    chunks = []
    for n, texto in artigos.items():
        chunks.append({
            "numero": n,
            "texto": texto,
            "remissoes_internas": extrair_remissoes(texto, n),
            "remissoes_externas": extrair_remissoes_externas(texto),
            "lei": "12.651/2012",
        })
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB: indexação e leitura
# ---------------------------------------------------------------------------

def desserializar_metadado(metadado: dict) -> dict:
    resultado = dict(metadado)
    for campo in ("remissoes_internas", "remissoes_externas"):
        valor = resultado.get(campo, "")
        resultado[campo] = valor.split(",") if valor else []
    return resultado


def indexar_chunks(collection, chunks: list[dict], modelo_embedding) -> None:
    """Gera embeddings e indexa os chunks no ChromaDB, em batch."""
    textos = [c["texto"] for c in chunks]
    embeddings = modelo_embedding.encode(
        textos, batch_size=32, normalize_embeddings=True
    )

    ids = [f"art_{c['numero']}" for c in chunks]
    metadados = [
        {
            "numero": c["numero"],
            "lei": c["lei"],
            "remissoes_internas": ",".join(c["remissoes_internas"]),
            "remissoes_externas": ",".join(c["remissoes_externas"]),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=textos,
        metadatas=metadados,
    )


def buscar_por_numero_artigo(collection, numeros: list[str]) -> list[dict]:
    if not numeros:
        return []
    resultado = collection.get(
        where={"numero": {"$in": numeros}},
        include=["documents", "metadatas"],
    )
    return [
        {"numero": m["numero"], "texto": doc, "metadata": desserializar_metadado(m)}
        for doc, m in zip(resultado["documents"], resultado["metadatas"])
    ]


def retriever_expandido(
    pergunta: str,
    collection,
    modelo_embedding,
    k: int = 5,
    profundidade_expansao: int = 1,
) -> list[dict]:
    """Busca semântica inicial + expansão por remissões internas."""
    embedding_pergunta = modelo_embedding.encode([pergunta], normalize_embeddings=True)
    resultado_semantico = collection.query(
        query_embeddings=embedding_pergunta.tolist(),
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs_recuperados = {}
    for doc, m in zip(resultado_semantico["documents"][0], resultado_semantico["metadatas"][0]):
        meta = desserializar_metadado(m)
        docs_recuperados[meta["numero"]] = {
            "numero": meta["numero"],
            "texto": doc,
            "metadata": meta,
            "origem": "semantica",
        }

    numeros_recuperados = set(docs_recuperados.keys())
    fronteira = set(numeros_recuperados)

    for _ in range(profundidade_expansao):
        candidatos_expansao = set()
        for numero in fronteira:
            remissoes = docs_recuperados[numero]["metadata"]["remissoes_internas"]
            for r in remissoes:
                if r and r not in numeros_recuperados:
                    candidatos_expansao.add(r)

        if not candidatos_expansao:
            break

        novos_docs = buscar_por_numero_artigo(collection, list(candidatos_expansao))
        nova_fronteira = set()
        for d in novos_docs:
            d["origem"] = "remissao"
            docs_recuperados[d["numero"]] = d
            numeros_recuperados.add(d["numero"])
            nova_fronteira.add(d["numero"])
        fronteira = nova_fronteira

    return list(docs_recuperados.values())


# ---------------------------------------------------------------------------
# Formatação de contexto para o prompt de tradução
# ---------------------------------------------------------------------------

def formatar_contexto(docs: list[dict]) -> str:
    partes = []
    for d in docs:
        origem_label = "(via remissão)" if d["origem"] == "remissao" else ""
        partes.append(f"--- Art. {d['numero']} {origem_label} ---\n{d['texto']}")
    return "\n\n".join(partes)


def formatar_leis_externas(docs: list[dict]) -> str:
    todas = set()
    for d in docs:
        todas.update(d["metadata"]["remissoes_externas"])
    return ", ".join(sorted(todas)) if todas else "nenhuma"


SYSTEM_PROMPT_TRADUCAO = """Você traduz texto jurídico do Código Florestal brasileiro (Lei 12.651/2012)
para linguagem simples.

DIRETRIZES DE LINGUAGEM SIMPLES:
- Frases curtas, uma ideia por frase
- Voz ativa (evite "foi estabelecido que" -> prefira "a lei estabelece")
- Evite nominalizações e gerundismo
- Explique siglas e termos técnicos na primeira menção (ex: "APP (Área de Preservação Permanente)")
- Vá direto: condição -> consequência, na ordem em que acontecem
- Não use jargão como "nos termos do", "para os fins do disposto em"
- Quando houver números/percentuais, aplique ao caso concreto da pergunta do usuário, se possível

REGRA CRÍTICA - NUNCA VIOLAR:
- Nunca omita prazos, exceções ou condições do texto original, mesmo que isso deixe a explicação menos "limpa"
- Se a simplificação reduzir uma exceção a algo ambíguo, prefira manter a frase um pouco mais longa mas precisa
- Sempre cite o(s) artigo(s) exato(s) ao final da resposta

AVISO DE REMISSÃO EXTERNA:
- Se a lista de "leis externas mencionadas" não estiver vazia, inclua uma frase
  ao final avisando que esses pontos não foram detalhados nesta consulta
- Não invente o conteúdo dessas leis externas -- apenas avise que existem

FORMATO DE SAÍDA:
1. Resposta direta à pergunta em linguagem simples
2. Linha em branco
3. "Base legal: " seguido da citação do(s) artigo(s) usados
4. Se houver remissão externa não resolvida, uma linha final de aviso
"""


def montar_prompt_usuario(pergunta: str, contexto: str, leis_externas: str) -> str:
    return f"""Trechos recuperados do Código Florestal:
{contexto}

Leis externas mencionadas nestes trechos (não detalhadas nesta consulta): {leis_externas}

Pergunta do usuário: {pergunta}"""

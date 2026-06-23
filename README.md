# Entenda o Código Florestal

Ferramenta de perguntas e respostas sobre a Lei nº 12.651/2012 (Código
Florestal brasileiro), com tradução para linguagem simples. Projeto
desenvolvido para o haCARthon — Maratona de Soluções para o CAR (2026),
Desafio 3.

## Como funciona

1. O texto compilado da lei (fonte: Casa Civil) é dividido em artigos.
2. Um grafo de remissões internas (entre artigos da própria lei) e externas
   (a outras 14 leis citadas) é construído automaticamente.
3. A pergunta do usuário é comparada por similaridade semântica
   (embeddings BAAI/bge-m3) contra os artigos indexados no ChromaDB.
4. Quando um artigo recuperado cita outro, o sistema busca também o artigo
   citado — capturando exceções que uma busca puramente semântica poderia
   não recuperar.
5. Um modelo de linguagem traduz o conteúdo recuperado para português
   simples, sem omitir exceções, condições ou prazos do texto original.

## Executando localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para usar o backend Claude (Anthropic), configure a variável de ambiente:

```bash
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

Para usar o backend Ollama (ferramenta aberta), instale o
[Ollama](https://ollama.com/download) e baixe um modelo antes de rodar:

```bash
ollama pull llama3.1
```

## Publicando no Streamlit Community Cloud

Ao publicar, configure o secret `ANTHROPIC_API_KEY` no painel do app
(Settings → Secrets) se quiser usar o backend Claude. O backend Ollama
**não funciona** no Streamlit Community Cloud, pois o servidor não tem
acesso a um servidor Ollama local — funciona apenas ao rodar o app na
própria máquina.

## Estrutura do projeto

- `app.py` — interface Streamlit, com seletor de backend (Claude/Ollama)
- `pipeline.py` — lógica de ingestão da lei, grafo de remissões, indexação
  no ChromaDB e retriever expandido
- `requirements.txt` — dependências Python

## Licença e fonte de dados

Texto da Lei nº 12.651/2012 obtido do site da Presidência da República
(Casa Civil), fonte pública oficial. Este projeto é disponibilizado como
código aberto, alinhado ao espírito de Bem Público Digital do projeto
CAR DPG.

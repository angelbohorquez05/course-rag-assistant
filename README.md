# course-rag-assistant

Final project for the **Machine Learning Advanced** course (ITS Angelo Rizzoli, Milan).

An assistant that answers questions about a folder of course materials using **RAG** (Retrieval-Augmented Generation), and cites where each answer comes from (document and page).

## How it works

1. **Read** the documents (PDF, page by page)
2. **Split** them into chunks, each labelled with its document and page
3. **Embed** each chunk with Gemini
4. **Retrieve** the chunks closest to the question (cosine similarity) and let the LLM answer using only those

## Status

- [x] Stage 1, blocks 1 and 2: read PDF and split into chunks
- [ ] Stage 1, blocks 3 and 4: embeddings, retrieval and answer
- [ ] Stage 2: whole course + Weaviate vector database
- [ ] Stage 3: generic folder reader
- [ ] Stage 4: lecture filter and query rewriting
- [ ] Stage 5: evaluation
- [ ] Stage 6: user interface

## Setup

```bash
uv sync
cp .env.example .env   # then add your GEMINI_API_KEY
```

## Note

Course documents go in `data/`, which is **not** pushed to GitHub.

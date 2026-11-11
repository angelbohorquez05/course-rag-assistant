"""Reusable building blocks of the course RAG assistant.

Every function was first written and explained step by step in stage1_minimal_rag.ipynb.
Comments say where each piece comes from:
    SAME     identical to the course code
    ADAPTED  based on the course code, with changes explained in the comment
    NEW      written for this project
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pypdf
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai.types import Content, Part

load_dotenv()            # reads GEMINI_API_KEY from the .env file
client = genai.Client()

# Which model turns text into vectors:
#   "gemini": gemini-embedding-2 (better quality, free tier limited to 1000 texts per day)
#   "ollama": nomic-embed-text running on this laptop (no limits, works offline)
EMBEDDING_BACKEND = "ollama"
EMBEDDING_MODEL = "gemini-embedding-2"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
# First choice, then fallbacks (in class these models were sometimes "under high demand")
# Free-tier requests per day (from AI Studio): 3.6/3.8 Flash 20 each, 3.5/3.1 Flash Lite 500 each,
# Gemma 4 31B 14,400. Best quality first, then the models with more daily quota.
GENERATION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-4-31b-it",
]
OLLAMA_MODEL = "qwen3.5:0.8b"   # local last resort (see `ollama list` for the models you have)

BATCH_SIZE = 20
RETRY_DELAY_SECONDS = 30
MAX_RETRIES = 5


# ---------------------------------------------------------------- reading and cleaning

# ADAPTED from the course (first_rag.ipynb, read_pdf): same pypdf reading,
# but we keep each page separately instead of joining all the text.
def read_pdf(path):
    """Return a list of dicts, one per page, with its page number and text."""
    reader = pypdf.PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": page_number, "text": text})
    return pages


# NEW (not in the course): automatic removal of headers/footers.
def clean_pages(pages, min_share=0.5):
    """Remove lines repeated on many pages (headers/footers) and lines that are only a page number."""
    line_counts = Counter()
    for p in pages:
        unique_lines = {line.strip() for line in p["text"].splitlines() if line.strip()}
        line_counts.update(unique_lines)

    threshold = max(2, int(len(pages) * min_share))
    repeated = {line for line, count in line_counts.items() if count >= threshold}

    cleaned = []
    for p in pages:
        kept = [
            line for line in p["text"].splitlines()
            if line.strip() and line.strip() not in repeated and not line.strip().isdigit()
        ]
        cleaned.append({"page": p["page"], "text": "\n".join(kept)})
    return cleaned, repeated


# ---------------------------------------------------------------- chunking

# ADAPTED from the course (first_rag.ipynb, naive_chunks): same sliding window and same
# overlap check. NEW: we split page by page, attach a label (document, pages, chunk_num)
# and skip exact duplicates (e.g. an agenda slide repeated in every section).
def split_into_chunks(pages, document_name, chunk_size=1000, overlap=200, dedupe=True):
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    seen = {}                                    # chunk text -> its position in `chunks`
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size]
            if len(piece.strip()) < 50:          # skip almost empty leftovers at the end of a page
                continue
            key = piece.strip()
            if dedupe and key in seen:           # same text already seen: keep one copy, add the page
                pages_of_copy = chunks[seen[key]]["pages"]
                if p["page"] not in pages_of_copy:
                    pages_of_copy.append(p["page"])
            else:
                seen[key] = len(chunks)
                chunks.append({
                    "document": document_name,
                    "pages": [p["page"]],
                    "chunk_num": len(chunks),
                    "text": piece,
                })
            if start + chunk_size >= len(text):  # reached the end of the page
                break
    return chunks


# NEW: merge "animation builds". When a slide deck is exported to PDF, every click of an
# animation becomes a new page, and each page contains the lines of the previous one plus
# a few new ones (not always at the end: a footer can stay at the bottom).
# If ALL the lines of a chunk also appear in a longer chunk of the same document, we drop it
# and add its pages to the longer one, which already holds all of its information.
def merge_contained_chunks(chunks):
    def lines_of(chunk):
        return {" ".join(line.split()) for line in chunk["text"].splitlines() if line.strip()}

    by_length = sorted(chunks, key=lambda c: len(c["text"]), reverse=True)
    kept = []
    for chunk in by_length:                      # longest first, so "containers" are kept first
        lines = lines_of(chunk)
        container = next(
            (k for k in kept if k["document"] == chunk["document"] and lines <= lines_of(k)),
            None,
        )
        if container is None:
            kept.append(chunk)
        else:
            container["pages"] = sorted(set(container["pages"]) | set(chunk["pages"]))

    kept.sort(key=lambda c: (c["document"], c["pages"][0]))    # back to reading order
    for i, chunk in enumerate(kept):
        chunk["chunk_num"] = i
    return kept


def pages_label(chunk):
    """'p. 4', or 'p. 2, 3, 11' for a chunk found on several pages."""
    return "p. " + ", ".join(str(n) for n in chunk["pages"])


# NEW: the same read -> clean -> split pipeline, repeated for every PDF in a folder.
# Cleaning is done per document, so each lecture finds its own header
# (the copyright line is written in slightly different ways in different lectures).
# The three switches let us turn our improvements off, to measure them (see stage3_evaluation).
def build_chunks_from_folder(folder, clean=True, dedupe=True, merge=True):
    """Read every PDF in `folder` and return all their chunks in one list."""
    all_chunks = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        pages = read_pdf(pdf_path)
        if clean:
            pages, _ = clean_pages(pages)
        chunks = split_into_chunks(pages, document_name=pdf_path.stem, dedupe=dedupe)
        if merge:
            chunks = merge_contained_chunks(chunks)
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------- embeddings

# ADAPTED from the course (weaviate_tutorial.ipynb, movie embeddings): same batches of 20,
# same Content/Part objects, same check on the number of embeddings, same wait-and-retry.
# NEW: a maximum number of retries, and no retry when the error is ours (ValueError).
def embed_texts(texts, verbose=True, kind="document"):
    """Return a (n_texts, dim) float32 array with one embedding per text.

    kind: "document" for chunks, "query" for questions (only used by the Ollama backend).
    """
    if EMBEDDING_BACKEND == "ollama":
        return embed_texts_ollama(texts, verbose, kind)

    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # One Content per text: if we pass a plain list of strings, Gemini treats it
                # as ONE content made of many parts and returns a single embedding.
                contents = [Content(parts=[Part.from_text(text=t)]) for t in batch]
                response = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
                vectors = [e.values for e in response.embeddings]
                if len(vectors) != len(batch):
                    raise ValueError(f"Expected {len(batch)} embeddings, got {len(vectors)}")
                all_vectors.extend(vectors)
                break
            except ValueError:
                raise                      # our own check failed: retrying would give the same result
            except Exception as error:
                if attempt == MAX_RETRIES:
                    raise
                print(f"Batch {start}: {error}. Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY_SECONDS}s")
                time.sleep(RETRY_DELAY_SECONDS)
        if verbose:
            print(f"Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.array(all_vectors, dtype=np.float32)


# ADAPTED from the course (ollama_api_tutorial.ipynb / litellm_api_tutorial.ipynb used
# nomic-embed-text with Ollama). NEW: the task prefixes. nomic-embed-text was trained to
# receive "search_document: " before texts to store and "search_query: " before questions.
def embed_texts_ollama(texts, verbose=True, kind="document"):
    import ollama                                # imported here: only needed with this backend
    prefix = "search_query: " if kind == "query" else "search_document: "
    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [prefix + t for t in texts[start:start + BATCH_SIZE]]
        response = ollama.embed(model=OLLAMA_EMBEDDING_MODEL, input=batch)
        all_vectors.extend(response.embeddings)
        if verbose:
            print(f"Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)} (local)")
    return np.array(all_vectors, dtype=np.float32)


# NEW (the course uses np.save/np.load by hand): automatic cache that detects outdated files.
def load_or_compute_embeddings(chunks, name, folder="data"):
    """Load cached embeddings if they match the current chunks, otherwise compute and save them."""
    if EMBEDDING_BACKEND != "gemini":
        name = f"{name}_{EMBEDDING_BACKEND}"      # vectors from different models must never be mixed
    vectors_path = Path(folder) / f"{name}_embeddings.npy"
    chunks_path = Path(folder) / f"{name}_chunks.json"

    if vectors_path.exists() and chunks_path.exists():
        cached_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if cached_chunks == chunks:
            print("Loaded embeddings from cache")
            return np.load(vectors_path)
        print("Chunks changed: recomputing embeddings")

    vectors = embed_texts([c["text"] for c in chunks])
    np.save(vectors_path, vectors)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return vectors


# ---------------------------------------------------------------- retrieval

# SAME formula as the course (image search in class, 23 Sept): cosine similarity + argsort.
# NEW: optional filter on the lecture, like Quri does when you ask inside a lecture
# (same idea as the Filter used in weaviate_tutorial.ipynb).
def search(question, chunks, embeddings, top_k=5, lecture=None, question_vector=None):
    """Return the top_k chunks most similar to the question, optionally only from one lecture."""
    if lecture is None:
        candidates = np.arange(len(chunks))                     # every chunk
    else:
        candidates = np.array([i for i, c in enumerate(chunks) if c["document"] == lecture])
        if len(candidates) == 0:
            raise ValueError(f"No chunks for lecture {lecture!r}")

    if question_vector is None:                 # can be passed in to avoid embedding the same question twice
        question_vector = embed_texts([question], verbose=False, kind="query")[0]
    vectors = embeddings[candidates]
    similarities = (vectors @ question_vector) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vector)
    )
    best = np.argsort(-similarities)[:top_k]       # positions sorted from most to least similar
    return [(float(similarities[i]), chunks[candidates[i]]) for i in best]


# ---------------------------------------------------------------- generation

# ADAPTED from the course (first_rag.ipynb, TEMPLATE_PROMPT): same rules (only the sources,
# say when the answer is missing, cite the source). NEW: cite the page, answer in the question's language.
PROMPT_TEMPLATE = """You are a teaching assistant for a course.
Answer the question using ONLY the information in the sources below.
If the sources do not contain the answer, do not invent anything: reply ONLY with the code
NOT_FOUND_IT (question in Italian), NOT_FOUND_ES (Spanish) or NOT_FOUND_EN (English).
After each statement, cite its source in the form [document, p. N].
Answer in the same language as the question.

SOURCES:
{sources}

QUESTION: {question}
"""


# NEW: fallback between models (same idea as the LiteLLM Router seen in class).
def generate_with_fallback(prompt, models=GENERATION_MODELS, attempts_per_model=3, wait_seconds=10):
    """Try each model in order; on a server error (5xx, e.g. 'high demand') wait and retry."""
    for model in models:
        for attempt in range(1, attempts_per_model + 1):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text, model
            except errors.ServerError as error:
                print(f"{model}: {error.code} (attempt {attempt}/{attempts_per_model})")
                time.sleep(wait_seconds * attempt)   # wait a bit longer each time
            except errors.ClientError as error:
                # 404 = model not available for this key, 429 = free-tier quota used up for this model
                if error.code not in (404, 429):
                    raise                            # other 4xx (e.g. wrong key) are our problem: stop
                print(f"{model}: {error.code}, skipping it")
                break
        print(f"{model} unavailable, trying the next model")

    # Last resort: a local model with Ollama (same call as the course's ollama_api_tutorial.ipynb).
    print(f"All Gemini models busy: answering locally with Ollama ({OLLAMA_MODEL})")
    import ollama                                    # imported here: only needed when Gemini fails
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
    )
    return response.message.content, f"ollama/{OLLAMA_MODEL}"


# ADAPTED from the course (first_rag.ipynb, answer): same structure.
# NEW: numpy search instead of Weaviate, optional lecture filter, model fallback.
# NEW: fixed, a bit dramatic, messages for when the slides do not contain the answer.
# The model only chooses the language; the text is always exactly this.
NOT_FOUND_MESSAGES = {
    "NOT_FOUND_IT": "Ho cercato in ogni slide, pagina dopo pagina, lezione dopo lezione. Ci ho messo l'anima. "
                    "Ma su questo, il corso tace.\n\nQuesto argomento non è nei materiali del corso.",
    "NOT_FOUND_ES": "Busqué en cada slide, página tras página, lección tras lección. Le puse el alma. "
                    "Pero sobre esto, el curso guarda silencio.\n\nEste tema no está en los materiales del curso.",
    "NOT_FOUND_EN": "I searched every slide, page after page, lesson after lesson. I put my soul into it. "
                    "But on this, the course is silent.\n\nThis topic is not in the course materials.",
}


def finalize_answer(text):
    """Replace a NOT_FOUND code with its fixed message; leave normal answers unchanged."""
    for code, message in NOT_FOUND_MESSAGES.items():
        if code in text:
            return message
    return text


def build_prompt(question, results):
    """Put the retrieved chunks, labelled with document and pages, into the prompt."""
    sources = "\n\n".join(
        f"[{chunk['document']}, {pages_label(chunk)}]\n{chunk['text']}" for _, chunk in results
    )
    return PROMPT_TEMPLATE.format(sources=sources, question=question)


def answer(question, chunks, embeddings, top_k=5, lecture=None):
    results = search(question, chunks, embeddings, top_k, lecture)
    prompt = build_prompt(question, results)
    text, model_used = generate_with_fallback(prompt)
    text = finalize_answer(text)
    print(f"(answered by {model_used})")
    return text, results

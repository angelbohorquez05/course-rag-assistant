"""'How it works' mode: the project explained like slides, using the real data of the app.

Each slide shows one step of the pipeline with live numbers and examples, without showing code.
The slide texts are in Italian (the language of the presentation) and use the terms seen in class.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rag_core import (
    EMBEDDING_BACKEND,
    GENERATION_MODELS,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    build_chunks_from_folder,
    build_prompt,
    clean_pages,
    pages_label,
    read_pdf,
    search,
)


# ---------------------------------------------------------------- data for the slides (computed once)

@st.cache_resource(show_spinner="Preparing the slides...")
def pipeline_stats(folder):
    """Numbers for each step of the pipeline, per lecture."""
    rows, removed = [], {}
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        pages = read_pdf(pdf_path)
        clean, repeated = clean_pages(pages)
        rows.append({
            "lezione": pdf_path.stem,
            "pagine": len(pages),
            "caratteri prima della pulizia": sum(len(p["text"]) for p in pages),
            "caratteri dopo la pulizia": sum(len(p["text"]) for p in clean),
        })
        removed[pdf_path.stem] = sorted(repeated)
    counts = {
        "1. chunking semplice, senza pulizia": len(build_chunks_from_folder(folder, clean=False, dedupe=False, merge=False)),
        "2. + pulizia (intestazioni, numeri di pagina)": len(build_chunks_from_folder(folder, clean=True, dedupe=False, merge=False)),
        "3. + duplicati esatti rimossi": len(build_chunks_from_folder(folder, clean=True, dedupe=True, merge=False)),
        "4. + animazioni unite": len(build_chunks_from_folder(folder, clean=True, dedupe=True, merge=True)),
    }
    return pd.DataFrame(rows), removed, counts


@st.cache_resource
def map_2d(_embeddings):
    """Squeeze the vectors to 2 dimensions with PCA, only to draw them (the search uses all of them)."""
    mean = _embeddings.mean(axis=0)
    _, _, vt = np.linalg.svd(_embeddings - mean, full_matrices=False)
    components = vt[:2]
    return mean, components, (_embeddings - mean) @ components.T


# ---------------------------------------------------------------- design (minimal, consistent)
# One accent colour, one muted grey, thin lines. Every step slide has the same layout:
# a small uppercase label, a title, the pipeline strip with the current step lit up,
# then one idea, one visual, and the key code.

ACCENT = "#4A90E2"
MUTED = "#6f6f6f"
TEXT = "#e8e8e8"
LINE = "#3a3a3a"

PIPELINE = ["PDF", "Pulizia", "Chunking", "Embedding", "Retrieval", "Prompt", "LLM"]

CSS = f"""
<style>
.kicker {{ color: {ACCENT}; font-size: 0.78rem; letter-spacing: 0.18em; text-transform: uppercase;
          margin-bottom: 0.1rem; }}
.slide-title {{ font-size: 2.1rem; font-weight: 600; margin: 0 0 0.6rem 0; line-height: 1.15; }}
.lead {{ font-size: 1.15rem; color: {TEXT}; margin: 0.4rem 0 1rem 0; }}
.lead b {{ color: {ACCENT}; font-weight: 600; }}
.card {{ border: 1px solid {LINE}; border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%; }}
.card h4 {{ margin: 0 0 0.4rem 0; font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; }}
.card p {{ margin: 0; font-size: 1.05rem; line-height: 1.5; }}
.big-number {{ font-size: 2.6rem; font-weight: 600; line-height: 1; }}
.big-label {{ color: {MUTED}; font-size: 0.85rem; margin-top: 0.3rem; }}
div[data-testid="stMetric"] {{ border: 1px solid {LINE}; border-radius: 14px; padding: 0.8rem 1rem; }}
</style>
"""


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def header(kicker, title, step=None):
    html(CSS)
    html(f"<div class='kicker'>{kicker}</div><div class='slide-title'>{title}</div>")
    if step is not None:
        pipeline_strip(step)


def pipeline_strip(active):
    """The 7 steps of the system as a thin line of nodes; the current one is lit up."""
    n = len(PIPELINE)
    gap = 640 / (n - 1)
    parts = []
    for k in range(n - 1):
        x1, x2 = 20 + k * gap + 7, 20 + (k + 1) * gap - 7
        colour = ACCENT if k < active else LINE
        parts.append(f"<line x1='{x1}' y1='18' x2='{x2}' y2='18' stroke='{colour}' stroke-width='1.5'/>")
    for k, name in enumerate(PIPELINE):
        x = 20 + k * gap
        if k == active:
            parts.append(f"<circle cx='{x}' cy='18' r='7' fill='{ACCENT}'/>")
        elif k < active:
            parts.append(f"<circle cx='{x}' cy='18' r='5' fill='none' stroke='{ACCENT}' stroke-width='1.5'/>")
        else:
            parts.append(f"<circle cx='{x}' cy='18' r='5' fill='none' stroke='{LINE}' stroke-width='1.5'/>")
        colour = TEXT if k == active else MUTED
        weight = 600 if k == active else 400
        parts.append(f"<text x='{x}' y='44' fill='{colour}' font-size='12' font-weight='{weight}' "
                     f"text-anchor='middle' font-family='sans-serif'>{name}</text>")
    html(f"<svg viewBox='0 0 680 56' width='100%' style='margin:0.2rem 0 1.2rem 0'>{''.join(parts)}</svg>")


def illustration(svg_body, height=150):
    html(f"<svg viewBox='0 0 680 {height}' width='100%' style='margin:0.4rem 0 0.8rem 0'>{svg_body}</svg>")


# ---------------------------------------------------------------- the slides (text in Italian)
# Short on purpose: keywords, numbers, drawings and key code. The explanation is said aloud.

def slide_title(course_name, author):
    html(CSS)
    stack = "".join(
        f"<rect x='{250 + k * 14}' y='{40 - k * 12}' width='120' height='84' rx='8' fill='#0e0f12' "
        f"stroke='{MUTED if k < 2 else ACCENT}' stroke-width='1.5'/>"
        for k in range(3))
    lines = "".join(f"<line x1='292' y1='{30 + k * 12}' x2='{360 - k * 10}' y2='{30 + k * 12}' "
                    f"stroke='{ACCENT}' stroke-width='1.5' stroke-linecap='round'/>" for k in range(4))
    bubble = (f"<line x1='400' y1='70' x2='446' y2='70' stroke='{LINE}' stroke-width='1.5' stroke-dasharray='4 4'/>"
              f"<rect x='452' y='46' width='92' height='48' rx='24' fill='none' stroke='{TEXT}' stroke-width='1.5'/>"
              f"<circle cx='480' cy='70' r='3' fill='{TEXT}'/><circle cx='498' cy='70' r='3' fill='{TEXT}'/>"
              f"<circle cx='516' cy='70' r='3' fill='{TEXT}'/>")
    html(f"<div style='text-align:center;margin-top:9vh'>"
         f"<svg viewBox='0 0 680 140' width='70%'>{stack}{lines}{bubble}</svg>"
         f"<div class='kicker' style='margin-top:1.4rem'>Machine Learning Advanced · Progetto finale</div>"
         f"<div style='position:relative;display:inline-block;margin-top:0.4rem'>"
         f"<div style='position:relative;font-size:5.4rem;font-weight:700;letter-spacing:0.04em;line-height:1.1;"
         f"background:linear-gradient(120deg,#ffffff 15%,{ACCENT} 85%);-webkit-background-clip:text;background-clip:text;"
         f"color:transparent'>Lumi</div></div>"
         f"<div style='color:{MUTED};font-size:1.15rem;margin:-0.3rem 0 0.8rem'>Assistente RAG per lo studio</div>"
         f"<div class='lead'>Studia con l'AI, <b>senza uscire dal tema</b>.</div>"
         f"<div style='color:{MUTED};margin-top:2rem'>{author}</div></div>")


def slide_problem():
    header("Il problema e la soluzione", "Studiare con l'AI, ma restando sul tema")
    left, right = st.columns(2, gap="medium")
    with left:
        html(f"<div class='card'><h4 style='color:{MUTED}'>Il problema</h4>"
             f"<svg viewBox='0 0 300 110' width='100%'>"
             f"<rect x='20' y='35' width='70' height='40' rx='20' fill='none' stroke='{TEXT}' stroke-width='1.5'/>"
             f"<path d='M95 55 C 140 10, 170 100, 215 40 S 260 80, 285 30' fill='none' stroke='{MUTED}' "
             f"stroke-width='1.5' stroke-dasharray='5 5'/>"
             f"<text x='245' y='95' fill='{MUTED}' font-size='26' font-family='sans-serif'>?</text></svg>"
             f"<p>I chatbot generici <b>escono dal tema</b>: altre definizioni, altri esempi, a volte inventano.</p>"
             f"</div>")
    with right:
        pages = "".join(f"<rect x='{200 + k * 10}' y='{25 + k * 8}' width='60' height='70' rx='6' fill='#0e0f12' "
                        f"stroke='{ACCENT}' stroke-width='1.5'/>" for k in range(3))
        html(f"<div class='card' style='border-color:{ACCENT}'><h4 style='color:{ACCENT}'>La soluzione</h4>"
             f"<svg viewBox='0 0 300 110' width='100%'>"
             f"<rect x='20' y='35' width='70' height='40' rx='20' fill='none' stroke='{TEXT}' stroke-width='1.5'/>"
             f"<line x1='95' y1='55' x2='195' y2='55' stroke='{ACCENT}' stroke-width='1.5'/>{pages}"
             f"<path d='M232 70 l8 8 l14 -18' fill='none' stroke='{ACCENT}' stroke-width='2'/></svg>"
             f"<p>Risposte <b>solo</b> dal materiale del corso, con <b>lezione e pagina</b>.</p></div>")
    html("<div style='height:1.4rem'></div>")
    a, b, c = st.columns(3, gap="medium")
    for col, (word, text) in zip((a, b, c), [("Retrieval", "Trovare le slide giuste"),
                                              ("Augmented", "Metterle nel prompt"),
                                              ("Generation", "L'LLM risponde solo con quelle")]):
        col.markdown(f"<div class='card'><h4 style='color:{ACCENT}'>{word}</h4><p>{text}</p></div>",
                     unsafe_allow_html=True)


def slide_data(stats):
    df, _, _ = stats
    header("Passo 01", "I dati", step=0)
    html("<div class='lead'>Il sistema legge le slide del corso <b>pagina per pagina</b>: "
         "così ogni risposta può citare la pagina esatta.</div>")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lezioni", len(df))
    c2.metric("Pagine", int(df["pagine"].sum()))
    c3.metric("Caratteri", f"{int(df['caratteri prima della pulizia'].sum()):,}".replace(",", "."))
    st.bar_chart(df.set_index("lezione")["pagine"], horizontal=True, color=ACCENT, height=240)
    st.code("""reader = pypdf.PdfReader(path)
for page_number, page in enumerate(reader.pages, start=1):
    pages.append({"page": page_number, "text": page.extract_text()})""", language="python")


def slide_cleaning(stats):
    df, _, _ = stats
    header("Passo 02", "Pulizia del testo", step=1)
    html("<div class='lead'>Una riga presente in <b>metà delle pagine</b> è rumore: si toglie.</div>")
    page = []
    for k, x in enumerate((60, 260, 460)):
        page.append(f"<rect x='{x}' y='10' width='160' height='120' rx='8' fill='none' stroke='{LINE}' stroke-width='1.5'/>")
        page.append(f"<line x1='{x + 16}' y1='28' x2='{x + 144}' y2='28' stroke='#E24B4A' stroke-width='1.5'/>")
        page.append(f"<text x='{x + 80}' y='24' fill='#E24B4A' font-size='10' text-anchor='middle' "
                    f"font-family='sans-serif'>© copyright</text>")
        for j in range(4):
            page.append(f"<line x1='{x + 16}' y1='{52 + j * 14}' x2='{x + 144 - (j * 13 + k * 9) % 50}' "
                        f"y2='{52 + j * 14}' stroke='{ACCENT}' stroke-width='1.5' stroke-linecap='round'/>")
        page.append(f"<text x='{x + 144}' y='122' fill='#E24B4A' font-size='10' text-anchor='end' "
                    f"font-family='sans-serif' text-decoration='line-through'>{k + 1}</text>")
    illustration("".join(page), height=140)
    st.caption("In rosso: il copyright e il numero di pagina, presenti su ogni slide. In blu: il contenuto che resta. "
               "La regola funziona con qualsiasi documento, senza scrivere a mano cosa togliere.")
    before = int(df["caratteri prima della pulizia"].sum())
    after = int(df["caratteri dopo la pulizia"].sum())
    c1, c2 = st.columns([1, 2])
    c1.metric("Caratteri", f"{after:,}".replace(",", "."), f"{after - before:,}".replace(",", "."))
    c2.code("""threshold = max(2, int(len(pages) * 0.5))
repeated = {line for line, count in line_counts.items()
            if count >= threshold}""", language="python")


def slide_chunks(stats, chunks):
    _, _, counts = stats
    header("Passo 03", "Chunking", step=2)
    html("<div class='lead'>Una lezione intera mescola tanti argomenti: si divide in pezzi piccoli, "
         "i <b>chunk</b>. Ognuno tiene la sua etichetta: <b>lezione e pagine</b>.</div>")
    blocks = []
    for k in range(4):
        x = 40 + k * 150
        blocks.append(f"<rect x='{x}' y='{14 + (k % 2) * 26}' width='190' height='34' rx='6' fill='none' "
                      f"stroke='{ACCENT}' stroke-width='1.5'/>")
        blocks.append(f"<text x='{x + 12}' y='{36 + (k % 2) * 26}' fill='{MUTED}' font-size='11' "
                      f"font-family='sans-serif'>chunk {k + 1}</text>")
    for k in range(3):
        x = 40 + (k + 1) * 150
        blocks.append(f"<rect x='{x}' y='14' width='40' height='60' fill='{ACCENT}' opacity='0.18'/>")
    illustration("".join(blocks), height=84)
    st.caption("1000 caratteri per chunk, con 200 caratteri in comune tra un chunk e il successivo (overlap), "
               "così un'idea tagliata sul bordo resta intera in almeno un chunk.")

    html("<div class='kicker' style='margin-top:1.2rem'>Due problemi trovati guardando i risultati</div>")
    left, right = st.columns(2, gap="medium")
    agenda = "".join(f"<rect x='{30 + k * 62}' y='20' width='50' height='62' rx='5' fill='none' "
                     f"stroke='{MUTED if k else ACCENT}' stroke-width='1.5' {'stroke-dasharray=\'4 4\'' if k else ''}/>"
                     f"<line x1='{40 + k * 62}' y1='36' x2='{70 + k * 62}' y2='36' stroke='{MUTED if k else ACCENT}' "
                     f"stroke-width='1.5'/>" for k in range(4))
    left.markdown(f"<div class='card'><h4 style='color:{ACCENT}'>Duplicati</h4>"
                  f"<svg viewBox='0 0 300 100' width='100%'>{agenda}</svg>"
                  f"<p>La stessa slide (es. l'agenda) si ripete in più punti: si tiene <b>una sola copia</b>, "
                  f"con tutte le sue pagine.</p></div>", unsafe_allow_html=True)
    build = "".join(
        f"<rect x='{30 + k * 62}' y='20' width='50' height='62' rx='5' fill='none' "
        f"stroke='{ACCENT if k == 3 else MUTED}' stroke-width='1.5'/>"
        + "".join(f"<line x1='{38 + k * 62}' y1='{32 + j * 11}' x2='{72 + k * 62}' y2='{32 + j * 11}' "
                  f"stroke='{ACCENT if k == 3 else MUTED}' stroke-width='1.5'/>" for j in range(k + 1))
        for k in range(4))
    right.markdown(f"<div class='card'><h4 style='color:{ACCENT}'>Animazioni</h4>"
                   f"<svg viewBox='0 0 300 100' width='100%'>{build}</svg>"
                   f"<p>Ogni clic dell'animazione diventa una pagina che contiene la precedente: si tiene "
                   f"<b>la versione completa</b>, senza perdere informazione.</p></div>", unsafe_allow_html=True)
    html("<div style='height:1rem'></div>")
    a, b = st.columns([1, 2], gap="medium")
    a.markdown(f"<div class='card'><div class='big-number'>{list(counts.values())[1]} → "
               f"{list(counts.values())[-1]}</div><div class='big-label'>Chunk dopo il taglio → "
               f"senza duplicati e animazioni</div></div>", unsafe_allow_html=True)
    with b:
        example = max(chunks, key=lambda c: len(c["pages"]))
        with st.expander(f"Esempio: un'animazione di {len(example['pages'])} pagine diventata un solo chunk"):
            st.caption(f"{example['document']} · {pages_label(example)}")
            st.text(example["text"])


def slide_embeddings(chunks, embeddings, lecture_title):
    header("Passo 04", "Embedding", step=3)
    model = OLLAMA_EMBEDDING_MODEL if EMBEDDING_BACKEND == "ollama" else EMBEDDING_MODEL
    html(f"<div class='lead'>Ogni chunk diventa un vettore di <b>{embeddings.shape[1]} numeri</b>: "
         f"un indirizzo sulla mappa del significato.</div>")
    _, _, points = map_2d(embeddings)
    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1],
                       "Lezione": [lecture_title(c["document"]) for c in chunks]})
    st.scatter_chart(df, x="x", y="y", color="Lezione", height=380)
    st.caption("Ogni punto è un chunk, ogni colore una lezione. I chunk della stessa lezione stanno vicini: "
               "il modello ha capito di cosa parla ogni slide, senza che nessuno glielo dicesse.")
    st.code(f"""response = ollama.embed(model="{model}", input=batch)   # in locale, con cache .npy""",
            language="python")


def slide_search(chunks, embeddings, lecture_title):
    header("Passo 05", "Retrieval", step=4)
    html("<div class='lead'>La domanda diventa un vettore con lo stesso modello, e si confronta con "
         "<b>tutti i chunk</b>: si prendono i <b>5 più simili</b>.</div>")
    left, right = st.columns([2, 3], gap="medium")
    with left:
        html(f"<svg viewBox='0 0 240 170' width='100%'>"
             f"<line x1='30' y1='150' x2='210' y2='60' stroke='{ACCENT}' stroke-width='2'/>"
             f"<line x1='30' y1='150' x2='200' y2='20' stroke='{TEXT}' stroke-width='2'/>"
             f"<path d='M95 118 A 70 70 0 0 0 86 106' fill='none' stroke='{MUTED}' stroke-width='1.5'/>"
             f"<text x='100' y='104' fill='{MUTED}' font-size='13' font-family='sans-serif'>θ</text>"
             f"<text x='214' y='64' fill='{ACCENT}' font-size='12' font-family='sans-serif'>slide</text>"
             f"<text x='190' y='14' fill='{TEXT}' font-size='12' font-family='sans-serif'>domanda</text>"
             f"<circle cx='30' cy='150' r='3' fill='{TEXT}'/></svg>")
        html(f"<div style='text-align:center;color:{MUTED}'>Angolo piccolo → significato simile</div>")
    with right:
        st.code("""similarities = (vectors @ question_vector) / (
    np.linalg.norm(vectors, axis=1)
    * np.linalg.norm(question_vector))
best = np.argsort(-similarities)[:top_k]""", language="python")
    question = st.text_input("Prova una domanda",
                             value=st.session_state.get("demo_question", "Che cos'è l'overfitting e come si evita?"),
                             key="demo_question_input")
    st.session_state.demo_question = question
    results = search(question, chunks, embeddings, top_k=5)
    st.session_state.demo_results = results
    df = pd.DataFrame({
        "Slide": [f"{lecture_title(c['document'])} · {pages_label(c)}" for _, c in results],
        "Similarità coseno": [s for s, _ in results],
    }).set_index("Slide")
    st.bar_chart(df, horizontal=True, color=ACCENT, height=220)
    st.caption("Le 5 slide trovate per questa domanda: più lunga la barra, più la slide è vicina alla domanda.")


def slide_prompt():
    header("Passo 06", "Il prompt", step=5)
    html("<div class='lead'>Le 5 slide trovate, con la loro etichetta, e <b>regole chiare</b>.</div>")
    question = st.session_state.get("demo_question", "Che cos'è l'overfitting e come si evita?")
    results = st.session_state.get("demo_results")
    if results is None:
        st.warning("Prima fai una ricerca nel passo precedente.")
        return
    st.caption("Regole: usare solo le slide, testo facile da leggere, fonti (lezione e pagina) solo alla fine, dire quando la risposta non c'è, "
               "rispondere nella lingua della domanda.")
    st.code(build_prompt(question, results), language=None, wrap_lines=True, height=400)


def slide_evaluation():
    header("Passo 07", "Valutazione")
    html("<div class='lead'>Un esame di <b>15 domande</b>, con la pagina giusta come soluzione.</div>")
    cards = [("47%", "La slide giusta è la prima"),
             ("67%", "La slide giusta è tra le 5 date all'LLM"),
             ("257", "Chunk, un terzo in meno senza perdere informazione")]
    cols = st.columns(3, gap="medium")
    for col, (number, meaning) in zip(cols, cards):
        col.markdown(f"<div class='card'><div class='big-number'>{number}</div>"
                     f"<div class='big-label'>{meaning}</div></div>", unsafe_allow_html=True)
    st.caption("Le domande in spagnolo sono le più difficili: il modello di embedding locale conosce soprattutto l'inglese.")

def slide_future():
    header("Prossimi passi", "Dove può arrivare")
    items = [("Memoria", "Ricorda la conversazione, per fare domande di seguito."),
             ("Tool calling", "L'LLM riscrive la domanda prima di cercare."),
             ("Multilingue", "Embedding forti anche in spagnolo e italiano."),
             ("Schemi ricreati", "Ogni schema delle slide ridisegnato in automatico."),
             ("Weaviate", "Database vettoriale per migliaia di documenti.")]
    rows = "".join(
        f"<div style='display:grid;grid-template-columns:3.2rem 13rem 1fr;align-items:baseline;"
        f"padding:1.05rem 0;border-bottom:1px solid {LINE}'>"
        f"<span style='color:{ACCENT};font-size:0.85rem;letter-spacing:0.1em'>{i:02d}</span>"
        f"<span style='font-size:1.15rem;font-weight:600'>{title}</span>"
        f"<span style='color:#b5b5b5;font-size:1.05rem'>{text}</span></div>"
        for i, (title, text) in enumerate(items, start=1)
    )
    html(f"<div style='max-width:820px;margin:0.6rem auto 0;border-top:1px solid {LINE}'>{rows}</div>")
    html(f"<div class='lead' style='text-align:center;margin-top:2rem'>Funziona con qualsiasi corso: "
         f"<b>basta una cartella di PDF</b>.</div>")


# ---------------------------------------------------------------- stepper

def render(chunks, embeddings, lecture_title, data_folder, course_name, author=""):
    stats = pipeline_stats(data_folder)
    slides = [
        ("Titolo", lambda: slide_title(course_name, author)),
        ("Problema e soluzione", slide_problem),
        ("Dati", lambda: slide_data(stats)),
        ("Pulizia", lambda: slide_cleaning(stats)),
        ("Chunking", lambda: slide_chunks(stats, chunks)),
        ("Embeddings", lambda: slide_embeddings(chunks, embeddings, lecture_title)),
        ("Retrieval", lambda: slide_search(chunks, embeddings, lecture_title)),
        ("Prompt", slide_prompt),
        ("Valutazione", slide_evaluation),
        ("Prossimi passi", slide_future),
    ]
    if "slide" not in st.session_state:
        st.session_state.slide = 0
    i = st.session_state.slide

    slides[i][1]()

    st.divider()
    left, middle, right = st.columns([1, 3, 1])
    left.button("← Indietro", disabled=i == 0, width="stretch",
                on_click=lambda: st.session_state.update(slide=i - 1))
    middle.markdown(
        f"<div style='text-align:center;color:{MUTED};letter-spacing:0.3em'>"
        + " ".join("●" if k == i else "○" for k in range(len(slides)))
        + f"</div><div style='text-align:center;color:{MUTED};font-size:0.8rem'>{i + 1} / {len(slides)}</div>",
        unsafe_allow_html=True)
    right.button("Avanti →", disabled=i == len(slides) - 1, width="stretch",
                 on_click=lambda: st.session_state.update(slide=i + 1))

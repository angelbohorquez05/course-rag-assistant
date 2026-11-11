"""Web interface for the course RAG assistant.

Run it from the project folder with:
    uv run streamlit run app.py
"""
from pathlib import Path

import streamlit as st

import walkthrough
from diagrams import find_diagram
from rag_core import (
    build_chunks_from_folder,
    build_prompt,
    generate_with_fallback,
    load_or_compute_embeddings,
    pages_label,
    search,
)

DATA_FOLDER = "data/rag_papers"
COURSE_NAME = "Course materials"
AUTHOR = "Angel Bohorquez · ITS Angelo Rizzoli"
TOP_K = 5

# Readable names for the lectures. Any PDF not listed here is shown with its file name,
# so the app also works with other courses without changing the code.
LECTURE_TITLES = {
    "lec1_slides": "01 Introduction to Data Science",
    "lec2_slides": "02 Manipulating Data Using Numpy",
    "lec3_slides": "03 Cleaning Data Using Pandas",
    "lec4_slides": "04 Visualizing Data Using Matplotlib",
    "lec5_slides": "05 Supervised Learning",
    "lec6_slides": "06 Model Evaluation",
    "lec8_slides": "08 Feature Engineering",
}

st.set_page_config(page_title="Course Assistant", layout="centered")

# A few style touches: centered welcome title, left-aligned sidebar buttons.
st.markdown(
    """
    <style>
    .welcome { text-align: center; margin-top: 18vh; }
    .welcome .kicker { color: #9a9a9a; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .welcome h1 { font-size: 1.9rem; font-weight: 600; margin-top: 0; }
    section[data-testid="stSidebar"] button { justify-content: flex-start; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit runs this whole file again at every interaction. @st.cache_resource keeps the
# result in memory, so the PDFs are read and the embeddings loaded only once.
@st.cache_resource(show_spinner="Loading the course materials...")
def load_course():
    chunks = build_chunks_from_folder(DATA_FOLDER)
    embeddings = load_or_compute_embeddings(chunks, name="course")
    return chunks, embeddings


def title_of(lecture):
    return LECTURE_TITLES.get(lecture, lecture)


# Show the original slide next to the answer, so schemas and charts are visible too.
# The page is rendered from the PDF as an image (pypdfium2); nothing is sent to any model.
@st.cache_data(show_spinner=False)
def slide_image(document, page):
    try:
        import pypdfium2 as pdfium
    except ImportError:                         # optional: without the library the app still works
        return None
    pdf = pdfium.PdfDocument(str(Path(DATA_FOLDER) / f"{document}.pdf"))
    image = pdf[page - 1].render(scale=1.6).to_pil()
    pdf.close()
    return image


def show_best_slide(results):
    """The slide closest to the question: its diagram redrawn if we have one, otherwise a photo of the
    page. For an animation, its last (complete) page."""
    _, chunk = results[0]
    diagram = find_diagram(chunk["document"], chunk["pages"])
    if diagram is not None:
        st.markdown(f"<div style='margin:0.6rem 0'>{diagram['svg']()}</div>", unsafe_allow_html=True)
        st.caption(f"{diagram['title']} · ricreato da {title_of(chunk['document'])} · {pages_label(chunk)}")
        return
    page = max(chunk["pages"])
    image = slide_image(chunk["document"], page)
    if image is not None:
        st.image(image, caption=f"{title_of(chunk['document'])} · p. {page}", width="stretch")


chunks, embeddings = load_course()
lectures = sorted({c["document"] for c in chunks})

# session_state survives the reruns: it is the "memory" of the page.
if "messages" not in st.session_state:
    st.session_state.messages = []          # the chat: questions, answers and their sources
if "lecture" not in st.session_state:
    st.session_state.lecture = None         # None = search the whole course


def open_scope(lecture):
    """Clicking a lecture changes where we search and starts a new chat."""
    st.session_state.lecture = lecture
    st.session_state.messages = []


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Course Assistant")
    mode = st.radio("Mode", ["Assistant", "How it works"], horizontal=True, label_visibility="collapsed")
    if mode == "Assistant":                 # the course list is only useful in the chat
        st.button("New chat", width="stretch", on_click=open_scope,
                  args=(st.session_state.lecture,))
        st.caption("ENROLLED COURSES")
        st.button(COURSE_NAME, width="stretch", on_click=open_scope, args=(None,),
                  type="primary" if st.session_state.lecture is None else "secondary")
        for lecture in lectures:
            st.button(f"　{title_of(lecture)}", key=f"lecture_{lecture}", width="stretch",
                      on_click=open_scope, args=(lecture,),
                      type="primary" if st.session_state.lecture == lecture else "secondary")

# ---------------------------------------------------------------- main page
if mode == "How it works":
    # The same app, explained step by step like slides, with its real data.
    walkthrough.render(chunks, embeddings, title_of, DATA_FOLDER, COURSE_NAME, AUTHOR)
    st.stop()

scope = st.session_state.lecture
if scope is None:
    kicker, heading, placeholder = "Course", COURSE_NAME, "Ask about this course..."
else:
    kicker, heading, placeholder = "Lecture", title_of(scope), "Ask about this lecture..."

if not st.session_state.messages:
    # Empty chat: big centered title with the current scope (whole course or one lecture).
    st.markdown(
        f'<div class="welcome"><div class="kicker">{kicker}</div><h1>{heading}</h1></div>',
        unsafe_allow_html=True,
    )
else:
    st.caption(heading)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            with st.expander("✓ Searched the slides", expanded=False):
                for score, chunk in message["sources"]:
                    st.markdown(f"**{title_of(chunk['document'])}** · {pages_label(chunk)} · "
                                f"similarity {score:.3f}")
                    st.text(chunk["text"])
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_best_slide(message["sources"])
            st.caption(f"Answered by {message['model']}")

question = st.chat_input(placeholder)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching the slides...", expanded=False) as status:
            results = search(question, chunks, embeddings, top_k=TOP_K, lecture=scope)
            for score, chunk in results:
                st.markdown(f"**{title_of(chunk['document'])}** · {pages_label(chunk)} · "
                            f"similarity {score:.3f}")
            status.update(label="Writing the answer...")
            text, model_used = generate_with_fallback(build_prompt(question, results))
            status.update(label="✓ Searched the slides", state="complete")
        st.markdown(text)
        show_best_slide(results)
        st.caption(f"Answered by {model_used}")

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": results, "model": model_used}
    )

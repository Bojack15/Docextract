import os
import tempfile
import base64
import io
import threading
from pathlib import Path
import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
from streamlit.runtime.scriptrunner import get_script_run_ctx

from chunker import ChunkConfig
from pipeline import process_file, export_json
from vector_store import VectorStore


def get_session_id():
    ctx = get_script_run_ctx()
    return ctx.session_id if ctx else "default"


@st.cache_data
def get_pdf_images(file_bytes):
    # Render first 5 pages at 90 DPI for fast loading & client presentation
    return convert_from_bytes(file_bytes, dpi=90, last_page=5)


@st.cache_resource
def get_bg_tasks():
    return {}

BG_TASKS = get_bg_tasks()


def background_processing_task(session_id, file_bytes, filename, config, ocr_lang, dpi, do_omr, do_store, db_path, do_export):
    BG_TASKS[(session_id, filename)] = {"status": "processing", "doc": None, "error": None}
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = process_file(tmp_path, config, ocr_lang, dpi, is_omr=do_omr)
        doc.filename = filename
        doc.filepath = filename

        if do_store:
            vs = VectorStore(path=db_path)
            vs.add(doc)

        if do_export:
            out = f"./{Path(filename).stem}_extracted.json"
            export_json(doc, out)

        BG_TASKS[(session_id, filename)]["doc"] = doc
        BG_TASKS[(session_id, filename)]["status"] = "completed"
    except Exception as e:
        BG_TASKS[(session_id, filename)]["error"] = str(e)
        BG_TASKS[(session_id, filename)]["status"] = "failed"
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@st.fragment(run_every=1)
def check_status_fragment(session_id, selected_filename):
    if "processed_docs" not in st.session_state:
        st.session_state["processed_docs"] = {}
        
    any_running = False
    any_finished = False
    
    for (s_id, fname), task in list(BG_TASKS.items()):
        if s_id == session_id:
            if task["status"] == "processing":
                if fname == selected_filename:
                    any_running = True
            elif task["status"] == "completed":
                st.session_state["processed_docs"][fname] = task["doc"]
                BG_TASKS.pop((s_id, fname), None)
                any_finished = True
            elif task["status"] == "failed":
                st.error(f"Failed to process {fname}: {task['error']}")
                BG_TASKS.pop((s_id, fname), None)
                any_finished = True
                
    if any_running:
        st.info("Processing...")
        st.spinner("Parsing layouts...")
        
    if any_finished:
        st.rerun()

# ─────────────────── Page Config ───────────────────
st.set_page_config(
    page_title="DocExtract",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── Premium CSS Styling ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

/* Global app overrides */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
    color: #f8fafc;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Outfit', sans-serif;
    color: #cbd5e1;
    font-size: 1.25rem;
    font-weight: 600;
}

/* Custom premium hero section */
.hero {
    background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.05) 90.2%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.hero h1 {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    font-weight: 800;
    font-size: 2.5rem;
    letter-spacing: -0.5px;
}

.hero p {
    color: #94a3b8;
    margin: 0.6rem 0 0 0;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Glassmorphic grid boxes */
.stat-box {
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    padding: 1rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.2);
}

.stat-box .val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.stat-box .lbl {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.75px;
    margin-top: 0.4rem;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Semantic Search and Preview result card */
.result {
    background: rgba(30, 41, 59, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 16px;
    border-left: 5px solid #6366f1;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.1);
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

.result:hover {
    background-color: rgba(30, 41, 59, 0.35);
    border-left-color: #8b5cf6;
}

.result .meta {
    font-size: 0.825rem;
    color: #94a3b8;
    margin-bottom: 0.75rem;
    font-weight: 500;
}

.result .body {
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* Streamlit Tabs Styling */
div[data-testid="stTabBar"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 20px !important;
    backdrop-filter: blur(8px) !important;
}

div[data-testid="stTabBar"] button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #64748b !important;
    background-color: transparent !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    padding: 8px 16px !important;
    border: none !important;
}

div[data-testid="stTabBar"] button[aria-selected="true"] {
    color: #c084fc !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

/* Style file uploader dropzone */
div[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(139, 92, 246, 0.3) !important;
    background: rgba(30, 41, 59, 0.2) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease, background-color 0.3s ease !important;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.05) !important;
}

/* Slider and Selectbox typography and colors */
.stSlider label, .stSelectbox label, .stTextInput label, .stSlider div, .stSelectbox div, .stTextInput div {
    font-family: 'Outfit', sans-serif !important;
}

/* Premium Buttons Styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.5) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

button[kind="secondary"]:hover {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: rgba(139, 92, 246, 0.4) !important;
    color: #f8fafc !important;
}

/* Prevent image overflow in column containers and ensure centering */
img, .stImage, div[data-testid="stImage"] {
    max-width: 100% !important;
    height: auto !important;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
</style>

""", unsafe_allow_html=True)

# ─────────────────── Header ────────────────────────
st.markdown("""
<div class="hero">
    <h1>DocExtract</h1>
    <p>Pipeline engine for multi-structured documents, granular chunk splitting, and SQLite semantic storage.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────── Sidebar Settings ───────────────────────
with st.sidebar:
    st.markdown("### Settings")
    chunk_size = st.slider("Chunk size (words)", 128, 2048, 512, step=64)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 256, 50, step=10)
    ocr_lang = st.text_input("OCR language", value="eng")
    dpi = st.selectbox("OCR DPI", [150, 200, 300, 400], index=2)
    db_path = st.text_input("Vector DB path", value="./vector_db")

# ─────────────────── Application Tabs ──────────────────────────
tab_process, tab_search, tab_manage = st.tabs(["Process Documents", "Semantic Search", "Manage Store"])

# ─────────────────── Process Tab ───────────────────
with tab_process:
    files = st.file_uploader(
        "Upload PDF or image files",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        accept_multiple_files=True,
    )

    if files:
        selected_file_name = st.selectbox("Select file to manage", [f.name for f in files])
        selected_file = next(f for f in files if f.name == selected_file_name)
        show_preview = st.toggle("Preview File", value=False)

        st.write("---")

        if show_preview:
            st.markdown("### Document Preview")
            if selected_file.name.lower().endswith(".pdf"):
                try:
                    with st.spinner("Rendering document pages..."):
                        images = get_pdf_images(selected_file.getvalue())

                    col_p1, col_p2, col_p3 = st.columns([1, 8, 1])
                    with col_p2:
                        with st.container(height=600):
                            for i, img in enumerate(images, 1):
                                st.image(img, caption=f"Page {i}", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load PDF preview: {e}")
            else:
                col_p1, col_p2, col_p3 = st.columns([1, 8, 1])
                with col_p2:
                    st.image(selected_file, use_container_width=True)
            st.write("---")

        action_col = st.container()

        with action_col:
            st.markdown("### Processing Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                do_store = st.checkbox("Index in vector DB", value=True)
            with col2:
                do_export = st.checkbox("Export JSON", value=False)
            with col3:
                do_omr = st.checkbox("OMR Mode", value=False)

            if "processed_docs" not in st.session_state:
                st.session_state["processed_docs"] = {}

            # Clear cached documents that are no longer in the uploaded files list
            uploaded_names = {f.name for f in files}
            st.session_state["processed_docs"] = {
                name: doc for name, doc in st.session_state["processed_docs"].items()
                if name in uploaded_names
            }

            if st.button("Process Document", type="primary", use_container_width=True):
                session_id = get_session_id()
                if BG_TASKS.get((session_id, selected_file.name), {}).get("status") == "processing":
                    st.warning("Already processing this file!")
                else:
                    config = ChunkConfig(size=chunk_size, overlap=chunk_overlap)
                    file_bytes = selected_file.getvalue()
                    t = threading.Thread(
                        target=background_processing_task,
                        args=(
                            session_id,
                            file_bytes,
                            selected_file.name,
                            config,
                            ocr_lang,
                            dpi,
                            do_omr,
                            do_store,
                            db_path,
                            do_export
                        )
                    )
                    t.start()
                    st.rerun()

            check_status_fragment(get_session_id(), selected_file.name)

            # Render stats and previews for any processed documents in session state
            if st.session_state["processed_docs"]:
                for name, doc in st.session_state["processed_docs"].items():
                    st.write(f"### Processed Results for {name}")

                    # Grid statistics display
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f'<div class="stat-box"><div class="val">{doc.total_pages}</div><div class="lbl">Pages</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-box"><div class="val">{doc.total_words:,}</div><div class="lbl">Words</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-box"><div class="val">{len(doc.chunks)}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="stat-box"><div class="val">{doc.method.value.upper()}</div><div class="lbl">Extraction</div></div>', unsafe_allow_html=True)

                    # Extracted layout previews
                    with st.expander("Structured Page Outlines"):
                        for page in doc.pages:
                            icon = "[Text]" if page.method.value == "text" else "[OCR]"
                            conf = f" — {page.confidence:.1f}% OCR Confidence" if page.confidence else ""
                            st.markdown(f"**{icon} Page {page.number}** — {page.word_count} words{conf}")
                            if page.text:
                                st.text(page.text[:300] + ("..." if len(page.text) > 300 else ""))

                    with st.expander("Text Chunk Samples"):
                        for chunk in doc.chunks[:5]:
                            st.markdown(
                                f'<div class="result">'
                                f'<div class="meta">Chunk #{chunk.index} · Pages {chunk.start_page}–{chunk.end_page} · {chunk.word_count} words</div>'
                                f'<div class="body">{chunk.text[:220]}...</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        if len(doc.chunks) > 5:
                            st.info(f"Showing 5 of {len(doc.chunks)} chunks")

# ─────────────────── Search Tab ────────────────────
with tab_search:
    query = st.text_input("Semantic search query", placeholder="Query database semantic space...")
    col_n, col_f = st.columns([1, 2])
    with col_n:
        n_results = st.number_input("Results", 1, 50, 5)
    with col_f:
        filter_file = st.text_input("Filter by filename")

    if query and st.button("Search database", type="primary", use_container_width=True):
        vs = VectorStore(path=db_path)
        where = {"filename": filter_file} if filter_file else None
        results = vs.search(query, n=n_results, where=where)

        if not results:
            st.warning("No matches found.")
        else:
            st.markdown(f"**Found {len(results)} matches**")
            for i, r in enumerate(results, 1):
                meta = r["metadata"]
                dist = r["distance"]
                score = max(0, 1 - dist) * 100
                text = r["text"][:500] + ("..." if len(r["text"]) > 500 else "")
                st.markdown(
                    f'<div class="result">'
                    f'<div class="meta">Result #{i} · <b>{meta.get("filename", "?")}</b> · '
                    f'Pages {meta.get("start_page", "?")}-{meta.get("end_page", "?")} · '
                    f'Relevance: {score:.1f}%</div>'
                    f'<div class="body">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────── Manage Tab ────────────────────
with tab_manage:
    try:
        vs = VectorStore(path=db_path)
        info = vs.stats()

        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="stat-box"><div class="val">{info["total_documents"]}</div><div class="lbl">Indexed Documents</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="val">{info["total_chunks"]}</div><div class="lbl">Total Chunks</div></div>', unsafe_allow_html=True)

        docs = info.get("documents", [])
        if docs:
            st.markdown("### Database Catalogue")
            for d in docs:
                cols = st.columns([3, 1, 1, 1, 1])
                cols[0].write(f"**{d['filename']}**")
                cols[1].write(f"{d['total_pages']} pages")
                cols[2].write(f"{d['chunks']} chunks")
                cols[3].write(d["method"].upper())
                if cols[4].button("Delete", key=f"del_{d['document_id']}"):
                    vs.delete(d["document_id"])
                    st.rerun()

        st.markdown("---")
        if st.button("Erase SQLite Vector Store", type="secondary"):
            vs.reset()
            st.success("Vector store dropped and re-initialized.")
            st.rerun()
    except Exception:
        st.info("Vector database unit empty. Ingest a document to initialize storage structures.")

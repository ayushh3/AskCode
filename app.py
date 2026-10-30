import os

import requests
import streamlit as st


API_URL = os.getenv("ASKCODE_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 180


def show_api_error(error: requests.RequestException) -> None:
    if error.response is not None:
        try:
            message = error.response.json().get("detail", error.response.text)
        except ValueError:
            message = error.response.text
        st.error(f"AskCode couldn't complete that request: {message}")
    else:
        st.error("AskCode can't reach its local service. Check that the API is running.")


st.set_page_config(page_title="AskCode", page_icon="⌘", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2.5rem; }
    .askcode-eyebrow {
        color: #64748b; font-size: 0.78rem; font-weight: 650;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .askcode-title { font-size: 2.35rem; font-weight: 700; letter-spacing: -0.05em; }
    .askcode-subtitle { color: #64748b; font-size: 1.05rem; margin-top: 0.25rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(120, 130, 150, 0.18); }
    [data-testid="stFileUploader"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## AskCode")
    st.caption("Your project, in context.")

    st.markdown("### Project files")
    uploaded_files = st.file_uploader(
        "Add files to your workspace",
        type=["pdf", "md", "txt", "py"],
        accept_multiple_files=True,
        help="Supported files: Python, PDF, Markdown, and plain text.",
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Index selected files", use_container_width=True):
        files = [
            ("files", (file.name, file.getvalue(), "application/octet-stream"))
            for file in uploaded_files
        ]
        try:
            with st.spinner("Adding your files to the workspace..."):
                response = requests.post(
                    f"{API_URL}/index",
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
            indexed = response.json()["indexed"]
            st.success(
                f"Added {len(indexed)} file{'s' if len(indexed) != 1 else ''} "
                "to your workspace."
            )
        except requests.RequestException as error:
            show_api_error(error)

    st.divider()
    st.markdown("### In this workspace")
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        response.raise_for_status()
        sources = response.json()["documents"]
        if not sources:
            st.caption("Files you add will appear here.")
        for source in sources:
            file_col, remove_col = st.columns([5, 1])
            file_col.markdown(f"`{source}`")
            if remove_col.button("×", key=f"remove-{source}", help=f"Remove {source}"):
                delete_response = requests.delete(
                    f"{API_URL}/documents/{requests.utils.quote(source, safe='')}",
                    timeout=10,
                )
                delete_response.raise_for_status()
                st.rerun()
    except requests.RequestException as error:
        show_api_error(error)

st.markdown('<div class="askcode-eyebrow">Developer workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="askcode-title">AskCode</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="askcode-subtitle">Get oriented in your code and project documents.</div>',
    unsafe_allow_html=True,
)
st.write("")

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    st.caption("Workspace service is ready")
except requests.RequestException:
    st.warning("The workspace service is offline. Start the API to ask about your files.")

st.markdown("### What are you working on?")
question = st.text_area(
    "Ask about your project",
    placeholder="Ask about code or a document in your workspace...",
    label_visibility="collapsed",
    height=110,
)

if st.button("Search workspace", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Looking through your files..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        result = response.json()
        st.markdown("### Answer")
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Files and lines"):
                for source in result["sources"]:
                    st.markdown(
                        f"- `{source['source']}` · lines "
                        f"{source['start_line']}–{source['end_line']}"
                    )
    except requests.RequestException as error:
        show_api_error(error)

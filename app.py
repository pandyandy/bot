import streamlit as st

from ui import (
    wrap_doc_in_html,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from core.caching import bootstrap_caching
from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

MAIN_LOGO_URL = 'https://assets-global.website-files.com/5e21dc6f4c5acf29c35bb32c/5e21e66410e34945f7f25add_Keboola_logo.svg'
MINI_LOGO_URL = 'https://components.keboola.com/images/default-app-icon.png'

st.markdown(
    f'''
    <div style="text-align: right;">
        <img src="{MAIN_LOGO_URL}" alt="Logo" width="150">
    </div>
    ''',
    unsafe_allow_html=True
)

st.header("AskMyPDF")

# Enable caching for expensive functions
bootstrap_caching()

openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the secrets as OPENAI_API_KEY. You can get your key at"
        " https://platform.openai.com/api-keys."
    )

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file.",
        type=["pdf", "docx", "txt"],
        help="Scanned documents are not supported yet!",
        accept_multiple_files=True
)

    model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

    with st.expander("Advanced Options"):
        return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
        show_full_doc = st.checkbox("Show parsed contents of the document")

if not uploaded_files:
    st.info("Upload a file to get started.")
    st.stop()

files = []
for uploaded_file in uploaded_files:
    try:
        file = read_file(uploaded_file)
        files.append(file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)

chunked_files = [chunk_file(file, chunk_size=300, chunk_overlap=0) for file in files]

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

#with st.spinner("Indexing document... This may take a while ⏳"):
folder_index = embed_files(
        files=chunked_files,
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )

if show_full_doc:
    with st.sidebar:
        with st.expander("Document"):
            for file in files:
                st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": "Hello! I'm here to assist with any questions you have about the uploaded documents. How can I help you today?",
    },
]

# Add a reset button
if st.sidebar.button("Reset Chat", use_container_width=True):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["messages"] = INITIAL_MESSAGE
    st.session_state["history"] = []

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = INITIAL_MESSAGE

for message in st.session_state['messages']:
    if message["role"] == "user":
        st.chat_message("user", avatar='🧑‍💻').write(message["content"])
    else:
        st.chat_message("assistant", avatar=MINI_LOGO_URL).write(message["content"], unsafe_allow_html=True)

def extract_answer_without_sources(result_answer: str) -> str:
    normalized_answer = result_answer.replace("\nSOURCES:", " SOURCES:")
    normalized_answer = normalized_answer.replace("SOURCES:\n", "SOURCES: ")
    answer = normalized_answer.split("SOURCES:")[0].strip()
    return answer

if query := st.chat_input("Ask a question about the document"):
    st.session_state['messages'].append({"role": "user", "content": query})
    with st.chat_message("user", avatar='🧑‍💻'):
        st.write(query)

    with st.spinner("Processing your request, please wait..."):
        llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0.2)
        result = query_folder(
            folder_index=folder_index,
            query=query,
            return_all=return_all_chunks,
            llm=llm,
        )
    #st.write(result)
    pages = sorted(set(int(source.metadata["page"]) for source in result.sources))
    sources = ", ".join(map(str, pages))
    if sources:
        answer = f"""
        {result.answer}
        
        <span style="color:grey; font-size: small; font-style: italic;">The information was found on the following pages: {sources}.</span>
        """
    else:
        answer = extract_answer_without_sources(result.answer)
        
    st.session_state['messages'].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar=MINI_LOGO_URL):
        st.write(answer, unsafe_allow_html=True)

    
    with st.sidebar:
        with st.expander("Sources"):
            for source in result.sources:
                try:
                    clean_content = source.page_content.encode('utf-8', 'ignore').decode('utf-8')
                    st.markdown(clean_content)
                    st.markdown(source.metadata["source"])
                except UnicodeEncodeError as e:
                    st.error(f"Error encoding source content: {e}")
                st.markdown("---")
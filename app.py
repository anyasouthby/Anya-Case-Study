
import sys

import streamlit as st

sys.path.append("src")

from retrieval import Retriever
from llm import LocalLLM
from agent import run_agent


st.set_page_config(
    page_title="AI Document Analysis Assistant",
    page_icon="📄",
    layout="wide",
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


@st.cache_resource
def load_components():
    """Load the retrieval system and language model once."""
    retriever = Retriever("data/dataset")
    llm = LocalLLM()
    return retriever, llm

def get_source_path(filename: str):
    """Return the path to a source document in the dataset."""
    return f"data/dataset/{filename}"

def get_source_file_info(filename: str):
    """Return the source path, actual file type and download filename."""

    source_path = get_source_path(filename)

    with open(source_path, "rb") as file:
        header = file.read(5)

    if header == b"%PDF-":
        return source_path, "pdf", filename

    return source_path, "html", f"{filename.rsplit('.', 1)[0]}.html"

st.title("AI Document Analysis Assistant")

st.write(
    "Ask questions about the supplied UK government documents. "
    "The assistant will select the most appropriate analysis capability "
    "for your request."
)


retriever, llm = load_components()


question = st.text_area(
    "What would you like to know?",
    placeholder=(
        "Try asking a factual question, requesting an executive summary, "
        "or asking about risks, opportunities, actions or stakeholders."
    ),
    height=100,
)

if st.button("Analyse", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Analysing the documents..."):

            result = run_agent(
                request=question,
                retriever=retriever,
                llm=llm,
            )

        st.session_state.analysis_result = result


result = st.session_state.analysis_result


if result is not None:

    tool = result["tool"]

    st.subheader("Capability selected")

    capability_names = {
        "qa": "Question & Answer",
        "insights": "Insight Extraction",
        "summary": "Executive Summary",
    }

    st.info(capability_names.get(tool, tool))


    # ---------------------------------------------------------
    # Question & Answer
    # ---------------------------------------------------------

    if tool == "qa":

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources & evidence")

        sources_by_document = {}

        for source in result["sources"]:

            filename = source["filename"]

            if filename not in sources_by_document:
                sources_by_document[filename] = []

            sources_by_document[filename].append(source)


        for filename, document_sources in sources_by_document.items():

            st.markdown(f"### {filename}")

            st.write(
                f"{len(document_sources)} retrieved "
                f"evidence chunk(s) from this document."
            )

            for source in document_sources:

                st.markdown(
                    f"**Retrieved evidence — chunk "
                    f"{source['chunk_id']}**"
                )

                st.markdown(
                    "> " + source["text"].replace("\n", "\n> ")
                )


            source_path, file_type, download_filename = (
                get_source_file_info(filename)
            )

            with open(source_path, "rb") as file:

                st.download_button(
                    label=f"Open source document ({file_type.upper()})",
                    data=file,
                    file_name=download_filename,
                    key=f"source_{filename}",
                )

            st.divider()


        with st.expander("How this answer was generated"):

            st.write("Capability: Question & Answer")
            st.write("Retrieval: semantic similarity search")
            st.write(
                f"Retrieved chunks: {len(result['sources'])}"
            )
            st.write(
                "Language model: Qwen2.5-3B-Instruct-4bit"
            )


    # ---------------------------------------------------------
    # Insight extraction
    # ---------------------------------------------------------

    elif tool == "insights":

        insights = result["insights"]

        st.subheader("Risks")

        for item in insights.get("risks", []):
            st.write(f"- {item}")


        st.subheader("Opportunities")

        for item in insights.get("opportunities", []):
            st.write(f"- {item}")


        st.subheader("Actions")

        for item in insights.get("actions", []):

            if isinstance(item, dict):

                owner = item.get("owner", "")
                action = item.get("action", "")

                if owner:
                    st.write(f"- **{owner}:** {action}")
                else:
                    st.write(f"- {action}")

            else:
                st.write(f"- {item}")


        st.subheader("Deadlines")

        for item in insights.get("deadlines", []):

            if isinstance(item, dict):

                date = item.get("date", "")
                description = item.get("description", "")

                if date:
                    st.write(
                        f"- **{date}:** {description}"
                    )
                else:
                    st.write(f"- {description}")

            else:
                st.write(f"- {item}")


        st.subheader("Stakeholders")

        for item in insights.get("stakeholders", []):

            if isinstance(item, dict):

                organisation = item.get("organisation", "")
                role = item.get("role", "")

                if role:
                    st.write(
                        f"- **{organisation}:** {role}"
                    )
                else:
                    st.write(f"- {organisation}")

            else:
                st.write(f"- {item}")


        st.subheader("Sources")

        unique_sources = []

        for source in result["retrieved_sources"]:

            filename = source["filename"]

            if filename not in unique_sources:
                unique_sources.append(filename)

        for filename in unique_sources:
            st.write(f"- {filename}")


        with st.expander("How this analysis was generated"):

            st.write("Capability: Insight Extraction")
            st.write(
                "Retrieval: semantic similarity search"
            )
            st.write(
                f"Retrieved chunks: "
                f"{len(result['retrieved_sources'])}"
            )
            st.write(
                "Language model: Qwen2.5-3B-Instruct-4bit"
            )


    # ---------------------------------------------------------
    # Executive summary
    # ---------------------------------------------------------

    elif tool == "summary":

        st.subheader("Executive Summary")
        st.write(result["summary"])


        st.subheader("Sources")

        unique_sources = []

        for source in result["sources"]:

            filename = source["filename"]

            if filename not in unique_sources:
                unique_sources.append(filename)

        for filename in unique_sources:
            st.write(f"- {filename}")


        with st.expander("How this summary was generated"):

            st.write("Capability: Executive Summary")
            st.write(
                "Retrieval: semantic similarity search"
            )
            st.write(
                f"Retrieved chunks: {len(result['sources'])}"
            )
            st.write(
                "Language model: Qwen2.5-3B-Instruct-4bit"
            )


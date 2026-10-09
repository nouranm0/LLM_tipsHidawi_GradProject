import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📄 AI Document Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze documents, ask questions, and compare PDFs using AI.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a feature:",
    [
        "📊 Document Analysis",
        "💬 Ask Questions",
        "🔄 Compare Documents"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "PePared by: NOURAN MUHAMMAD -- " \
    "ID: 524"
)


# ============================================================
# DOCUMENT ANALYSIS
# ============================================================

if page == "📊 Document Analysis":

    st.header("📊 Document Analysis")

    st.write(
        "Upload a PDF to generate an AI-powered analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key="analysis_pdf"
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "🔍 Analyze Document",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing document..."
            ):

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/analyze",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        },
                        timeout=300
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state[
                            "analysis_result"
                        ] = data

                    else:

                        st.error(
                            response.text
                        )

                except Exception as e:

                    st.error(
                        f"Connection error: {e}"
                    )

        # ----------------------------------------------------
        # SHOW RESULT
        # ----------------------------------------------------

        if "analysis_result" in st.session_state:

            data = st.session_state[
                "analysis_result"
            ]

            st.divider()

            st.subheader("📄 Document")

            st.write(
                data["document"]
            )

            st.subheader("📝 Summary")

            st.write(
                data["summary"]
            )

            st.subheader("🔑 Key Points")

            for point in data["key_points"]:

                st.markdown(
                    f"- {point}"
                )

            st.subheader("⚠️ Risks")

            if data["risks"]:

                for risk in data["risks"]:

                    st.markdown(
                        f"- {risk}"
                    )

            else:

                st.write(
                    "No risks were identified."
                )

            st.subheader("📚 References")

            references = data["references"]

            if references:

                st.write(
                    "Source pages: "
                    + ", ".join(
                        str(page)
                        for page in references
                    )
                )

            else:

                st.write(
                    "No references available."
                )


# ============================================================
# ASK QUESTIONS
# ============================================================

elif page == "💬 Ask Questions":

    st.header("💬 Ask Questions")

    st.write(
        "Upload a document and ask questions about its content."
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key="question_pdf"
    )

    question = st.text_area(
        "Your question:",
        placeholder=(
            "Example: What steps are involved "
            "in component-level estimation?"
        ),
        height=100
    )

    if st.button(
        "💬 Ask Question",
        use_container_width=True
    ):

        if uploaded_file is None:

            st.warning(
                "Please upload a PDF first."
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/ask",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        },
                       
                        data={
    "question": question.strip()
},
                        timeout=300
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.divider()

                        st.subheader(
                            "🤖 Answer"
                        )

                        st.write(
                            data["answer"]
                        )

                        st.subheader(
                            "📚 Sources"
                        )

                        references = data[
                            "references"
                        ]

                        if references:

                            st.write(
                                "Source pages: "
                                + ", ".join(
                                    str(page)
                                    for page in references
                                )
                            )

                        else:

                            st.write(
                                "No source pages found."
                            )

                    else:

                        st.error(
                            response.text
                        )

                except Exception as e:

                    st.error(
                        f"Connection error: {e}"
                    )


# ============================================================
# COMPARE DOCUMENTS
# ============================================================

elif page == "🔄 Compare Documents":

    st.header("🔄 Compare Documents")

    st.write(
        "Upload two PDF documents to identify similarities, "
        "differences and potential risks."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📄 Document A")

        file_a = st.file_uploader(
            "Upload Document A",
            type=["pdf"],
            key="file_a"
        )

    with col2:

        st.subheader("📄 Document B")

        file_b = st.file_uploader(
            "Upload Document B",
            type=["pdf"],
            key="file_b"
        )

    if st.button(
        "🔍 Compare Documents",
        use_container_width=True
    ):

        if file_a is None or file_b is None:

            st.warning(
                "Please upload both documents."
            )

        else:

            with st.spinner(
                "Comparing documents..."
            ):

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/compare",
                        files={
                            "file_a": (
                                file_a.name,
                                file_a.getvalue(),
                                "application/pdf"
                            ),
                            "file_b": (
                                file_b.name,
                                file_b.getvalue(),
                                "application/pdf"
                            )
                        },
                        timeout=300
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.divider()

                        st.subheader(
                            "📊 Comparison Result"
                        )

                        st.markdown(
                            data["comparison"]
                        )

                    else:

                        st.error(
                            response.text
                        )

                except Exception as e:

                    st.error(
                        f"Connection error: {e}"
                    )
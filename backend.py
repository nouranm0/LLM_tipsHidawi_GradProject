import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


# ============================================================
# CONFIGURATION
# ============================================================
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found. "
        "Set it as an environment variable before running the backend."
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Document Intelligence API",
    description="RAG-based document analysis, question answering and comparison.",
    version="1.0.0"
)


# ============================================================
# LLMs
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

json_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)


# ============================================================
# EMBEDDINGS
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class DocumentAnalysis(BaseModel):
    document: str = Field(description="Document name")
    summary: str = Field(description="Document summary")
    key_points: list[str] = Field(
        description="Main points from the document"
    )
    risks: list[str] = Field(
        description="Potential risks or concerns supported by the document"
    )
    references: list[int] = Field(
        description="Relevant source page numbers"
    )


parser = PydanticOutputParser(
    pydantic_object=DocumentAnalysis
)


# ============================================================
# PROMPTS
# ============================================================

rag_prompt = ChatPromptTemplate.from_template("""
You are an AI Document Intelligence Assistant.

Answer the question using ONLY the provided document context.

If the answer is not available in the context, clearly say so.

Do not invent facts.

Context:
{context}

Question:
{question}

Provide a clear and concise answer.
""")


analysis_prompt = ChatPromptTemplate.from_template("""
Analyze the document context below.

Extract:

- A concise summary
- Important key points
- Potential risks or concerns explicitly supported by the text
- Relevant source page numbers

Do not invent information.

If no risks are supported by the text, return an empty list.

Use only the supplied context.

Document name:
{document_name}

Context:
{context}

{format_instructions}
""")


summary_prompt = ChatPromptTemplate.from_template("""
Summarize the document context below.

Include:

1. Main purpose
2. Main topics
3. Important conclusions

Do not add facts that are not present in the document.

Context:
{context}
""")


comparison_prompt = ChatPromptTemplate.from_template("""
Compare the following two documents.

Identify:

- Similarities
- Differences
- Important changes
- Potential risks or concerns

Do not invent facts.

If a detail is absent, state that it is not specified.

DOCUMENT A:
{document_a}

DOCUMENT B:
{document_b}
""")


# ============================================================
# CHAINS
# ============================================================

rag_chain = rag_prompt | llm | StrOutputParser()

analysis_chain = (
    analysis_prompt
    | json_llm
    | parser
)

summary_chain = (
    summary_prompt
    | llm
    | StrOutputParser()
)

comparison_chain = (
    comparison_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# PDF FUNCTIONS
# ============================================================

def extract_pdf_pages(path):
    """
    Extract text from every PDF page while preserving page numbers.
    """

    reader = PdfReader(path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            documents.append({
                "page": page_number,
                "text": text.strip()
            })

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    return documents


def extract_pdf_text(path):
    """
    Extract the complete PDF text with page markers.
    """

    documents = extract_pdf_pages(path)

    return "\n\n".join(
        f"[Page {document['page']}]\n{document['text']}"
        for document in documents
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        page_chunks = text_splitter.split_text(
            document["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return chunks


# ============================================================
# VECTOR STORE
# ============================================================

def create_vector_store(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "page": chunk["page"],
            "chunk_id": index
        }
        for index, chunk in enumerate(chunks)
    ]

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vector_store


# ============================================================
# DOCUMENT CONTEXT
# ============================================================

def create_full_context(chunks):

    return "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk in chunks
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(path):

    documents = extract_pdf_pages(path)

    chunks = create_chunks(documents)

    vector_store = create_vector_store(chunks)

    full_context = create_full_context(chunks)

    return {
        "documents": documents,
        "chunks": chunks,
        "vector_store": vector_store,
        "full_context": full_context
    }


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

def analyze_document(path):

    processed = process_document(path)

    result = analysis_chain.invoke({
        "document_name": Path(path).name,
        "context": processed["full_context"],
        "format_instructions": parser.get_format_instructions()
    })

    return result, processed


# ============================================================
# ASK QUESTION
# ============================================================

def ask_document(
    vector_store,
    question,
    k=4
):

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join(
        f"[Page {doc.metadata['page']}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    references = sorted({
        doc.metadata["page"]
        for doc in docs
    })

    return {
        "answer": answer,
        "references": references
    }


# ============================================================
# SUMMARIZE DOCUMENT
# ============================================================

def summarize_document(
    full_context
):

    return summary_chain.invoke({
        "context": full_context
    })


# ============================================================
# COMPARE DOCUMENTS
# ============================================================

def compare_documents(
    pdf_path_a,
    pdf_path_b
):

    text_a = extract_pdf_text(
        pdf_path_a
    )

    text_b = extract_pdf_text(
        pdf_path_b
    )

    result = comparison_chain.invoke({
        "document_a": text_a,
        "document_b": text_b
    })

    return result


# ============================================================
# API MODELS
# ============================================================

class QuestionRequest(BaseModel):

    question: str

    vector_store: object = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Document Intelligence API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            content = await file.read()

            temp.write(content)

            temp_path = temp.name

        analysis, processed = analyze_document(
            temp_path
        )

        return {
            "document": analysis.document,
            "summary": analysis.summary,
            "key_points": analysis.key_points,
            "risks": analysis.risks,
            "references": analysis.references
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ============================================================
# ASK ENDPOINT
# ============================================================

# NOTE:
# The Streamlit app will communicate with this endpoint
# using the document again. This keeps the project simple.
#
# For the first version, we process the uploaded PDF,
# create FAISS, then answer the question.

@app.post("/ask")
async def ask_endpoint(
    file: UploadFile = File(...),
    question: str = Form("")
):

    if not question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            content = await file.read()

            temp.write(content)

            temp_path = temp.name

        processed = process_document(
            temp_path
        )

        result = ask_document(
            processed["vector_store"],
            question
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ============================================================
# COMPARE ENDPOINT
# ============================================================

@app.post("/compare")
async def compare_endpoint(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):

    if not file_a.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Document A must be a PDF."
        )

    if not file_b.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Document B must be a PDF."
        )

    path_a = None
    path_b = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_a:

            temp_a.write(
                await file_a.read()
            )

            path_a = temp_a.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_b:

            temp_b.write(
                await file_b.read()
            )

            path_b = temp_b.name

        result = compare_documents(
            path_a,
            path_b
        )

        return {
            "comparison": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if path_a and os.path.exists(path_a):
            os.remove(path_a)

        if path_b and os.path.exists(path_b):
            os.remove(path_b)

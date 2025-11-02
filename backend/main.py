from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import os

# --- Latest LangChain Imports (No langchain.chains) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories and Global Objects ---
VECTOR_DIR = "backend/vector_stores"
DATA_DIR = "backend/org_data"
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")
retrieval_chains = {}  # Cache for retrieval chains per org

# --- Utility Functions ---


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def combine_texts_from_org(org_path: str) -> str:
    """Combine text from all PDFs in the org's folder."""
    all_texts = []
    for filename in os.listdir(org_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(org_path, filename)
            pdf_text = extract_text_from_pdf(pdf_path)
            if pdf_text.strip():
                all_texts.append(pdf_text)
    return "\n".join(all_texts)


def get_vectorstore(org_id: str):
    """Load the FAISS vector store for the org."""
    index_path = os.path.join(VECTOR_DIR, f"{org_id}_index.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    return None

# --- Endpoints ---


@app.post("/upload")
async def upload_pdf(orgId: str = Form(...), file: UploadFile = None):
    """Upload a PDF and rebuild the org's vector index."""
    if not file:
        return {"error": "No file uploaded"}

    org_path = os.path.join(DATA_DIR, orgId)
    os.makedirs(org_path, exist_ok=True)
    pdf_dest = os.path.join(org_path, file.filename)
    with open(pdf_dest, "wb") as f:
        f.write(await file.read())

    combined_text = combine_texts_from_org(org_path)
    if not combined_text.strip():
        return {"error": "No readable text found in PDFs."}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_text(combined_text)
    docs = [Document(page_content=chunk) for chunk in docs]

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(os.path.join(VECTOR_DIR, f"{orgId}_index.faiss"))

    if orgId in retrieval_chains:
        del retrieval_chains[orgId]  # Invalidate cache after upload

    return {"status": "success", "pdfs_processed": len(os.listdir(org_path))}


@app.post("/chat")
async def chat(orgId: str = Form(...), message: str = Form(...)):
    """Chat endpoint using org-specific data with pure LCEL chain."""
    if orgId not in retrieval_chains:
        store = get_vectorstore(orgId)
        if not store:
            return {"response": "This organization has not uploaded any PDFs yet."}

        retriever = store.as_retriever(search_kwargs={"k": 3})

        # Modern prompt template
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant for the organization {org_id}. "
            "Answer based only on the following context:\n{context}\n\n"
            "Question: {input}\n\nAnswer:"
        )

        # Pure LCEL chain: retriever -> format docs -> prompt -> LLM -> parser
        retrieval_chains[orgId] = (
            {
                "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "input": RunnablePassthrough(),
                "org_id": lambda _: orgId,  # Inject org_id
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    chain = retrieval_chains[orgId]
    response = await chain.ainvoke(message)  # Async invoke
    return {"response": response}


@app.get("/")
def root():
    return {"message": "Chatbot backend running with latest LangChain (pure LCEL)"}

# Scenario: AI Research Assistant for a Corporate Innovation Team
# A corporate innovation lab reviews AI research papers to stay ahead of trends.
# This RAG chatbot lets the team ask natural questions about uploaded papers.

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ── 1. Load PDF ──────────────────────────────────────────────────────────────
def load_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# ── 2. Chunk the document ────────────────────────────────────────────────────
def chunk_document(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

# ── 3. Build vector store ────────────────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_ai_research")

# ── 4. Build RAG chain ───────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ── 5. Interactive chat loop ─────────────────────────────────────────────────
def chat_loop(qa_chain):
    print("\n=== AI Research Assistant ===")
    print("Ask questions about the research paper. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue
        answer = qa_chain.run(query)
        print(f"Assistant: {answer}\n")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = "ai_research.pdf"  # Replace with your PDF path
    print("Loading document...")
    pages = load_document(pdf_path)
    print(f"Loaded {len(pages)} pages. Chunking...")
    chunks = chunk_document(pages)
    print(f"Created {len(chunks)} chunks. Building vector store...")
    vectorstore = build_vectorstore(chunks)
    print("Vector store ready. Building RAG chain...")
    qa_chain = build_rag_chain(vectorstore)
    chat_loop(qa_chain)

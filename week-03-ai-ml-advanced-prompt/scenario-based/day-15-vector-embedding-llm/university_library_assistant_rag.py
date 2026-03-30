# Scenario: University Library Assistant
# A university library deploys a RAG chatbot so students can ask questions
# about textbooks and course notes instead of manually searching through PDFs.

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ── 1. Load textbook PDF ─────────────────────────────────────────────────────
def load_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# ── 2. Chunk into concept-level sections ────────────────────────────────────
def chunk_document(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(pages)

# ── 3. Build vector store ────────────────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_university")

# ── 4. Build RAG chain ───────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ── 5. Interactive chat loop ─────────────────────────────────────────────────
def chat_loop(qa_chain):
    print("\n=== University Library Assistant ===")
    print("Ask questions about your textbook. Type 'exit' to quit.\n")
    print("Example: 'Explain the difference between supervised and unsupervised learning'")
    print("Example: 'What is gradient descent?'\n")
    while True:
        query = input("Student: ").strip()
        if query.lower() == "exit":
            print("Good luck with your studies!")
            break
        if not query:
            continue
        answer = qa_chain.run(query)
        print(f"Assistant: {answer}\n")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = "Introduction_to_Data_Science.pdf"  # Replace with your PDF path
    print("Loading textbook...")
    pages = load_document(pdf_path)
    print(f"Loaded {len(pages)} pages. Chunking...")
    chunks = chunk_document(pages)
    print(f"Created {len(chunks)} chunks. Building vector store...")
    vectorstore = build_vectorstore(chunks)
    print("Vector store ready. Building RAG chain...")
    qa_chain = build_rag_chain(vectorstore)
    chat_loop(qa_chain)

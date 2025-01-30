import streamlit as st
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import textwrap
import docx

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.full_text = ""
        
    def chunk_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    def process_document(self, text):
        self.full_text = text
        self.chunks = self.chunk_text(text)
        embeddings = self.embedding_model.encode(self.chunks)
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(np.array(embeddings).astype('float32'))
        
    def semantic_search(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.chunks[i] for i in I[0]]

def read_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.type == "text/plain":
        return file.getvalue().decode('utf-8')
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")

def get_answer(query, context, use_semantic_search=False):
    model = genai.GenerativeModel('gemini-pro')
    
    if use_semantic_search:
        context_prompt = "\n".join(context)
    else:
        context_prompt = context
        
    prompt = f"""Based on the following document content, please answer the question.
    If you cannot find the answer in the provided content, say so clearly.
    
    Document content: {context_prompt}
    
    Question: {query}
    
    Answer: """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Document parsing & RAG System")

    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.full_text = ""

    st.sidebar.header("Configuration")
    search_method = st.sidebar.radio(
        "Search Method",
        ["Full Document", "Semantic Search"],
        help="Choose how to process the document for answering questions"
    )
    
    if search_method == "Semantic Search":
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
        k_chunks = st.sidebar.slider("Number of relevant chunks", 1, 5, 3)

    
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, TXT, or DOCX)",
        type=['pdf', 'txt', 'docx']
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                content = read_file(uploaded_file)
                st.session_state.full_text = content
                if search_method == "Semantic Search":
                    st.session_state.processor.process_document(content)
                st.success(f"Successfully parsed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Query section
    st.header("🔍 Ask Questions")
    query = st.text_input("Enter your question about the document:")

    if query and st.button("Get Answer"):
        if not st.session_state.full_text:
            st.error("Please upload a document first!")
            return

        with st.spinner("Generating answer..."):
            try:
                if search_method == "Semantic Search":
                    relevant_chunks = st.session_state.processor.semantic_search(query, k=k_chunks)
                    answer = get_answer(query, relevant_chunks, use_semantic_search=True)
                    
                    st.subheader("🤖 Answer:")
                    st.write(answer)
                    
                    st.subheader("📑 Relevant Passages:")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        with st.expander(f"Passage {i}"):
                            st.write(chunk)
                else:
                    answer = get_answer(query, st.session_state.full_text)
                    st.subheader("🤖 Answer:")
                    st.write(answer)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
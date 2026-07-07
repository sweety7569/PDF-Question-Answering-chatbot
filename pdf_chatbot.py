import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Gemini API Key
import google.generativeai as genai

genai.configure(api_key="MY_KEY")

# Gemini Model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Streamlit Title
st.title("📄 PDF Question Answering Chatbot")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    st.success("PDF uploaded successfully!")

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)

    # Store in FAISS
    embedding_matrix = np.array(
        embeddings
    ).astype("float32")

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)

    # Retrieval Function
    def retrieve_chunks(query, k=3):

        query_embedding = embedding_model.encode(
            [query]
        ).astype("float32")

        distances, indices = index.search(
            query_embedding,
            k
        )

        retrieved_text = "\n\n".join(
            [chunks[i] for i in indices[0]]
        )

        return retrieved_text

    # RAG Function
    def ask_pdf(question):

        context = retrieve_chunks(question)

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = gemini_model.generate_content(
            prompt
        )

        return response.text

    # Question Input
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        with st.spinner(
            "Searching PDF..."
        ):

            answer = ask_pdf(question)

        st.subheader("Answer")

        st.write(answer)
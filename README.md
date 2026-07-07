🚀 Overview

The AI-Powered PDF Question Answering Chatbot is a Retrieval-Augmented Generation (RAG) application that allows users to upload any PDF document and ask questions in natural language. The chatbot retrieves the most relevant content from the document using semantic search and generates accurate, context-aware answers with Google's Gemini AI.

✨ Features:
📄 Upload any PDF document
💬 Ask questions in natural language
🤖 AI-powered answers using Google Gemini
🔍 Semantic search using FAISS
🧠 Sentence embeddings with Sentence Transformers
⚡ Fast document retrieval
🌐 Interactive web interface built with Streamlit
☁️ Deployed on Streamlit Community Cloud

🛠️ Technologies Used:
Python
Streamlit
Google Gemini API
FAISS
Sentence Transformers
LangChain Text Splitters
PyPDF
NumPy

🏗️ How It Works:
Upload a PDF document.
Extract text from the PDF.
Split the text into smaller chunks.
Generate embeddings using Sentence Transformers.
Store embeddings in a FAISS vector database.
Retrieve the most relevant chunks based on the user's query.
Send the retrieved context and question to Google Gemini.
Display the AI-generated answer.

pip install transformers faiss-cpu sentence-transformers streamlit langchain
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from transformers import pipeline

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

# Corpus of documents (knowledge base)
documents = [
    "You can reset your password by going to the account settings page.",
    "Our return policy allows returns within 30 days with a receipt.",
    "Account settings can be accessed via the top-right dropdown menu."
]

# Generate embeddings for the documents
document_embeddings = embedding_model.encode(documents)

# Create FAISS index for efficient retrieval
dimension = document_embeddings.shape[1]  # Embedding size
faiss_index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean distance) index
faiss_index.add(np.array(document_embeddings))  # Add embeddings to the index
def retrieve_documents(query, k=2):
    """
    Retrieve top-k relevant documents for a given query.
    """
    query_embedding = embedding_model.encode([query])  # Encode the query
    distances, indices = faiss_index.search(query_embedding, k)  # Search FAISS index
    return [documents[i] for i in indices[0]]  # Retrieve top-k documents


# Load a lightweight text-generation model (you can switch to GPT-based models)
generation_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_response(query, retrieved_docs):
    """
    Generate a response using the retrieved documents and query.
    """
    # Combine context and query
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nUser Query: {query}\nAnswer:"
    
    # Generate the response
    response = generation_pipeline(prompt, max_length=100, do_sample=True)
    return response[0]['generated_text']
def rag_chatbot(query):
    """
    RAG chatbot pipeline: retrieval + generation.
    """
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)
    
    # Generate response using retrieved documents
    response = generate_response(query, retrieved_docs)
    return response


st.title("RAG Chatbot")
st.write("Ask a question, and I'll find the most relevant information for you!")

# Input from the user
user_query = st.text_input("Your Query:")

if user_query:
    # Get chatbot response
    bot_response = rag_chatbot(user_query)
    st.write("*Chatbot:*", bot_response)
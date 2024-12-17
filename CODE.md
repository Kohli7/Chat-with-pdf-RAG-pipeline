 pip install pdfplumber
 pip install PyMuPDF sentence-transformers faiss-cpu openai textwrap
pip install sentence-transformers
import sentence_transformers
print(sentence_transformers.__version__)
pip install faiss-cpu
pip install faiss-gpu
pip install --upgrade pip
import faiss
print(faiss.__version__)
pip install openai
pip install --upgrade pip
import openai

openai.api_key = "your-openai-api-key"

   *** THOSE ARE THE REQURIMENTS ***

-------------------------------------------------------------
   import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
----------------------------------------------------------------
import textwrap

def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)
----------------------------------------------------------------
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(chunks):
    return [model.encode(chunk) for chunk in chunks]
---------------------------------------------------------------
import faiss
import numpy as np

def store_embeddings_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    return index
---------------------------------------------------------------
def query_faiss(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]).astype(np.float32), k)
    return I[0]
------------------------------------------------------------------
import openai

openai.api_key = "your-openai-api-key"

def generate_response(query, context):
    prompt = f"Answer the following question using the context: {context}\nQuestion: {query}"
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=150
    )
    return response.choices[0].text.strip()
---------------------------------------------------------------------
def generate_comparison_response(query, chunks):
    # Extract relevant terms for comparison (customize this based on your data).
    comparison_data = []
    for chunk in chunks:
        comparison_data.append(extract_comparison_data(chunk))  # Custom function to extract comparison data.

    # Format comparison data as a table or bullet points.
    return format_comparison_as_table(comparison_data)  # Custom function to format the response.
-----------------------------------------------------------------------------
ALL TOGETHER 
def main(pdf_path, query):
    # Step 1: Extract and process PDF text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    # Step 2: Embed the chunks and store in FAISS
    embeddings = embed_text(chunks)
    index = store_embeddings_in_faiss(embeddings)
    # Step 3: Embed the query and search for relevant chunks
    query_embedding = model.encode(query)
    relevant_chunk_indices = query_faiss(index, query_embedding)
     # Step 4: Retrieve the context and generate a response
    relevant_chunks = [chunks[i] for i in relevant_chunk_indices]
    context = " ".join(relevant_chunks)  # Combine relevant chunks for context
    response = generate_response(query, context)
    return response
    ---------------------------------------------------------------------------------
    import fitz  # PyMuPDF

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
     for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to process the PDF and generate a response
def main(pdf_path, query):
    # Step 1: Extract and process PDF text
    text = extract_text_from_pdf(pdf_path)  # Extract text from PDF
    print("Extracted Text:")
    print(text[:500])  # Print the first 500 characters of extracted text
     # Example processing steps (these need to be defined as per your code):
    # chunks = chunk_text(text)
    # embeddings = embed_text(chunks)
    # index = store_embeddings_in_faiss(embeddings)
     # Example of generating a response (replace with actual logic)
    response = "This is a sample response based on the extracted text."
    return response

# Define the PDF file path and user query
pdf_path = "/Users/neeleshsuragani/Desktop/task 1/UNIT IV 2024-2025.pdf"  # Corrected path
query = "What is the main topic of the document?"

# Call the main function to process the PDF and generate a response
response = main(pdf_path, query)

# Print the response
print("Generated Response:")
print(response)

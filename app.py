from flask import Flask, render_template, request, redirect, url_for
import fitz
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import faiss
import numpy as np
from transformers import pipeline
import os

app = Flask(__name__)

# Global variables to hold model, index, and text data
model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline('text-generation', model="gpt2")

# Initialize the FAISS index
index = None
text_split = None

# Function to extract text from PDF
def extract_pdf_text(filename):
    doc = fitz.open(filename=filename)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


@app.route("/")
def index_page():

    return render_template("index.html")

# Home route to display the PDF upload form
@app.route("/create_indexer", methods=["GET", "POST"])
def create_indexer():
    global index, text_split
    
    if request.method == "POST":
        # Get the PDF file from the form
        pdf_file = request.files["pdfFile"]

        if pdf_file:
            # Ensure the uploads directory exists
            if not os.path.exists("uploads"):
                os.mkdir("uploads")

            # Save the PDF file temporarily
            filename = os.path.join("uploads", "user_pdf.pdf")
            pdf_file.save(filename)

            # Extract text from the PDF
            extracted_text = extract_pdf_text(filename)

            # Split the extracted text into sentences
            text_split = sent_tokenize(extracted_text)

            # Generate embeddings for the sentences
            embeddings = model.encode(text_split)
            embeddings_np = np.array(embeddings)

            # Initialize FAISS index and add embeddings
            global index
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)

            return redirect(url_for('rag_pdf'))  # Redirect to the RAG page after file upload
        
    return "Not Post request"


# Route to handle the RAG query after PDF upload
@app.route("/rag_pdf", methods=["GET", "POST"])
def rag_pdf():
    global index, text_split

    generated_text = ""  # Initialize generated_text to an empty string

    if request.method == "POST":
        # Get the user query from the form
        user_query = request.form.get("query")

        if user_query and index is not None and text_split is not None:
            # Embed the user query
            user_query_embedding = model.encode([user_query])

            # Perform search on the FAISS index
            distances, indices = index.search(np.array(user_query_embedding), 3)

            # Get relevant text based on indices
            relevant_text = "\n".join([text_split[i] for i in indices[0]])

            # Generate the response using the GPT-2 model
            response = generator(f"Context: {relevant_text} Question: {user_query}", max_length=200)
            generated_text = response[0]['generated_text']

            # Optionally print the result for debugging
            print("Generated Text:", generated_text)

    return render_template("rag_pdf.html", generated_text=generated_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)

from fastapi import FastAPI, File, UploadFile
import os
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import json
import os

app = FastAPI()
LEARNING_DIR = "learning_models"
@app.post("/learn")
async def learn(pdf: UploadFile = File(...)):
    # Save uploaded file
    pdf_path = os.path.join(LEARNING_DIR, pdf.filename)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Extract text and generate embeddings
    text = extract_text_from_pdf(pdf_path)
    embeddings = generate_embeddings(text)
    save_learned_model(pdf.filename, embeddings, text)
    return {"message": f"Learned from {pdf.filename}"}

@app.post("/ask")
def ask_question(pdf_name: str, question: str):
    try:
        answer = query_model(question, pdf_name)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


# Load embeddings from saved model
def load_learned_model(pdf_name):
    with open(os.path.join(LEARNING_DIR, f"{pdf_name}.json")) as f:
        data = json.load(f)
        return data["embeddings"], data["text"]

# Build a Faiss index
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# Query the model
def query_model(query, pdf_name):
    embeddings, text = load_learned_model(pdf_name)
    index = build_faiss_index(embeddings)

    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_embedding, k=1)  # Retrieve the top result
    return text[I[0][0]]

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=True)


os.makedirs(LEARNING_DIR, exist_ok=True)

def save_learned_model(pdf_name, embeddings, text):
    data = {
        "pdf_name": pdf_name,
        "text": text,
        "embeddings": embeddings.tolist()  # Convert tensor to list
    }
    save_path = os.path.join(LEARNING_DIR, f"{pdf_name}.json")
    with open(save_path, "w") as f:
        json.dump(data, f)

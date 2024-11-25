
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
import json
import os

LEARNING_DIR = "learning_models"
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

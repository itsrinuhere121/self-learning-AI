from sentence_transformers import SentenceTransformer
import pdfplumber
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer

text_model = SentenceTransformer('all-MiniLM-L6-v2')
def text_to_embedding(text):
    return text_model.encode(text, convert_to_tensor=True)
def extract_from_pdf(pdf_path):
    text = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text += page.extract_text() or ""

            # Extract images
            for img in page.images:
                x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                cropped_image = page.crop((x0, y0, x1, y1)).to_image()
                images.append(cropped_image)

    return text, images

image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_to_embedding(image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = image_model.get_image_features(**inputs)
    return outputs

LEARNING_DIR = "learning_models"
os.makedirs(LEARNING_DIR, exist_ok=True)

def save_to_knowledge_base(pdf_name, text, text_embedding, image_embeddings):
    data = {
        "text": text,
        "text_embedding": text_embedding.tolist(),
        "image_embeddings": [embedding.tolist() for embedding in image_embeddings],
    }
    with open(os.path.join(LEARNING_DIR, f"{pdf_name}.json"), "w") as f:
        json.dump(data, f)

def load_knowledge_base():
    knowledge = []
    for file in os.listdir(LEARNING_DIR):
        with open(os.path.join(LEARNING_DIR, file), "r") as f:
            knowledge.append(json.load(f))
    return knowledge

qa_model = T5ForConditionalGeneration.from_pretrained("t5-small")
qa_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = qa_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

def combined_context(knowledge):
    return " ".join([item["text"] for item in knowledge])
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/learn")
async def learn(pdf: UploadFile = File(...)):
    pdf_path = os.path.join(LEARNING_DIR, pdf.filename)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())
    
    # Process PDF
    text, images = extract_from_pdf(pdf_path)
    text_embedding = text_to_embedding(text)
    image_embeddings = [image_to_embedding(img) for img in images]
    save_to_knowledge_base(pdf.filename, text, text_embedding, image_embeddings)
    
    return {"message": f"Learned from {pdf.filename}"}

@app.post("/ask")
def ask_question(question: str):
    # Load all knowledge
    knowledge = load_knowledge_base()
    context = combined_context(knowledge)
    
    # Generate answer
    answer = generate_answer(question, context)
    return {"answer": answer}

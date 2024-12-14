from fastapi import FastAPI, File, UploadFile
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import pdfplumber
import os
import json
from io import BytesIO
from sentence_transformers import SentenceTransformer
# Initialize FastAPI
app = FastAPI()

# Initialize Models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = T5ForConditionalGeneration.from_pretrained("t5-small")
qa_tokenizer = T5Tokenizer.from_pretrained("t5-small")
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Knowledge Base Directory
LEARNING_DIR = "knowledge_base"
os.makedirs(LEARNING_DIR, exist_ok=True)

# Helper Functions
def extract_text_and_images_from_pdf(pdf_path):
    """
    Extract text and images from a PDF file.
    """
    text = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text += page.extract_text() or ""

            # Extract images from the page
            for img in page.images:
                x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
                pil_image = convert_pageimage_to_pil(cropped_image)
                if pil_image:
                    images.append(pil_image)
    return text, images

def convert_pageimage_to_pil(page_image):
    """
    Convert a pdfplumber PageImage to a PIL.Image.Image.
    """
    buffer = BytesIO()
    page_image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)

def process_text(text):
    """
    Convert text to embeddings using SentenceTransformer.
    """
    return text_model.encode(text, convert_to_tensor=True)

def process_image(image):
    """
    Convert an image (PIL.Image.Image) to embeddings using CLIP.
    """
    inputs = image_processor(images=image, return_tensors="pt")
    features = image_model.get_image_features(**inputs)
    return features

def save_knowledge(pdf_name, text, text_embedding, image_embeddings):
    """
    Save the extracted knowledge to the knowledge base directory.
    """
    knowledge = {
        "text": text,
        "text_embedding": text_embedding.tolist(),
        "image_embeddings": [embedding.tolist() for embedding in image_embeddings]
    }
    with open(os.path.join(LEARNING_DIR, f"{pdf_name}.json"), "w") as f:
        json.dump(knowledge, f)

def load_knowledge():
    """
    Load all knowledge from the knowledge base directory.
    """
    all_knowledge = []
    for file in os.listdir(LEARNING_DIR):
        with open(os.path.join(LEARNING_DIR, file), "r") as f:
            all_knowledge.append(json.load(f))
    return all_knowledge

def generate_markdown_response(question, context):
    """
    Generate a markdown response to a question based on the provided context.
    """
    input_text = f"question: {question} context: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = qa_model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate markdown with placeholder for images (if applicable)
    markdown_response = f"### Question: {question}\n\n**Answer:**\n\n{answer}\n"
    return markdown_response

# FastAPI Endpoints
@app.post("/learn")
async def learn(pdf: UploadFile = File(...)):
    """
    Endpoint to learn from a PDF.
    """
    # Save the uploaded PDF
    pdf_path = os.path.join(LEARNING_DIR, pdf.filename)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Extract and process the PDF
    text, images = extract_text_and_images_from_pdf(pdf_path)
    text_embedding = process_text(text)
    image_embeddings = [process_image(img) for img in images]

    # Save processed knowledge
    save_knowledge(pdf.filename, text, text_embedding, image_embeddings)

    return {"message": f"Successfully learned from {pdf.filename}"}

@app.post("/ask")
async def ask_question(question: str):
    """
    Endpoint to answer a question based on learned knowledge.
    """
    # Load all knowledge
    knowledge = load_knowledge()
    combined_text = " ".join([item["text"] for item in knowledge])

    # Generate a response in markdown
    markdown_response = generate_markdown_response(question, combined_text)

    return {"response_markdown": markdown_response}

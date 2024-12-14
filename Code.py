from fastapi import FastAPI, File, UploadFile
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import pdfplumber
from fastapi.responses import FileResponse
import os
import json
from io import BytesIO
from reportlab.pdfgen import canvas
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
def create_valid_pdf(pdf_path):
    """
    Creates a valid PDF file with sample content using reportlab.
    """
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "This is a valid PDF with sample text content.")
    c.drawString(100, 730, "You can add more lines or images for testing purposes.")
    c.save()

def extract_text_and_images_from_pdf(pdf_path):
    """
    Extracts text and images from a PDF file using pdfplumber.
    Ensures images are converted to PIL.Image.Image format.
    """
    text = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text += page.extract_text() or ""

            # Extract and process images
            for img in page.images:
                x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
                pil_image = convert_to_pil_image(cropped_image)
                if pil_image:
                    images.append(pil_image)
    return text, images

def convert_to_pil_image(page_image):
    """
    Converts pdfplumber's PageImage to a PIL.Image.Image object.
    """
    buffer = BytesIO()
    page_image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)

def process_text(text):
    """
    Encodes the text using SentenceTransformer to generate embeddings.
    """
    return text_model.encode(text, convert_to_tensor=True)

def process_image(image):
    """
    Encodes an image (PIL.Image.Image) using CLIP to generate embeddings.
    """
    inputs = image_processor(images=image, return_tensors="pt")
    features = image_model.get_image_features(**inputs)
    return features

def save_knowledge(pdf_name, text, text_embedding, image_embeddings):
    """
    Saves the extracted knowledge to the knowledge base directory.
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
    Loads all knowledge from the knowledge base directory.
    Skips invalid or non-JSON files.
    """
    all_knowledge = []
    for file in os.listdir(LEARNING_DIR):
        file_path = os.path.join(LEARNING_DIR, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_knowledge.append(json.load(f))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Skipping invalid file: {file_path}. Error: {e}")
    return all_knowledge

# def load_knowledge():
#     """
#     Loads all knowledge from the knowledge base directory.
#     """
#     all_knowledge = []
#     for file in os.listdir(LEARNING_DIR):
#         with open(os.path.join(LEARNING_DIR, file), "r") as f:
#             all_knowledge.append(json.load(f))
#     return all_knowledge

def generate_markdown_response(question, context):
    """
    Generates a markdown response to a question based on the provided context.
    """
    input_text = f"question: {question} context: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = qa_model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate markdown
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
from pydantic import BaseModel
class QuestionRequest(BaseModel):
    question: str
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     """
#     Endpoint to answer a question based on learned knowledge.
#     """
#     # Extract question from the request payload
#     question = request.question

#     # Load all knowledge
#     knowledge = load_knowledge()
#     combined_text = " ".join([item["text"] for item in knowledge])

#     # Generate a response in markdown
#     markdown_response = generate_markdown_response(question, combined_text)
#     return {"response_markdown": markdown_response}
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to answer a question based on learned knowledge.
    """
    # Extract question from the request payload
    question = request.question
    knowledge = load_knowledge()
    combined_text = " ".join([item["text"] for item in knowledge])

    # Generate Markdown response
    markdown_response = generate_markdown_response(question, combined_text)

    # Save Markdown response to file
    response_file = os.path.join(LEARNING_DIR, "response.md")
    with open(response_file, "w", encoding="utf-8") as f:
        f.write(markdown_response)

    # Return the Markdown file as a downloadable response
    return FileResponse(response_file, media_type="text/markdown", filename="response.md")

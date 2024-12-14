import os
import json
import base64
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
from io import BytesIO
import pdfplumber
from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer
# Initialize FastAPI
app = FastAPI()

# Initialize Models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = T5ForConditionalGeneration.from_pretrained("t5-small")
qa_tokenizer = T5Tokenizer.from_pretrained("t5-small")
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to store the LLM data (acting as the knowledge base)
LLM_PATH = "llm_data.json"

# Helper Functions

def load_llm_data():
    """
    Load the existing LLM (knowledge) data from the file.
    If the file does not exist or is empty, return default data.
    """
    if os.path.exists(LLM_PATH):
        try:
            with open(LLM_PATH, 'r') as f:
                llm_data = json.load(f)
                if llm_data:  # Ensure the file is not empty
                    return llm_data
                else:
                    print("Warning: The LLM file is empty, initializing with default values.")
                    return {"text": [], "images": []}
        except json.JSONDecodeError:
            print("Error: The LLM file is corrupted, initializing with default values.")
            return {"text": [], "images": []}
    else:
        print("LLM file does not exist, initializing with default values.")
        return {"text": [], "images": []}

def update_llm_data(new_text, new_images):
    """
    Update the LLM data by appending new text and images in base64 format.
    """
    llm_data = load_llm_data()
    llm_data["text"].append(new_text)
    # Convert images to base64 before saving
    base64_images = [convert_image_to_base64(img) for img in new_images]
    llm_data["images"].extend(base64_images)
    
    # Save updated data to the LLM file
    with open(LLM_PATH, 'w') as f:
        json.dump(llm_data, f)

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

def convert_image_to_base64(image):
    """
    Converts an image (PIL.Image.Image) to a base64-encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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
    Endpoint to learn from a PDF and update the LLM.
    """
    # Save the uploaded PDF
    pdf_path = os.path.join("uploads", pdf.filename)
    os.makedirs("uploads", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Extract and process the PDF
    text, images = extract_text_and_images_from_pdf(pdf_path)
    text_embedding = process_text(text)
    image_embeddings = [process_image(img) for img in images]

    # Update the LLM with new data
    update_llm_data(text, images)

    return {"message": f"Successfully learned from {pdf.filename}"}
from pydantic import BaseModel
class QuestionRequest(BaseModel):
    question: str
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to answer a question based on learned knowledge.
    """
    # Extract question from the request payload
    question = request.question
    # Load all knowledge
    llm_data = load_llm_data()
    combined_text = " ".join(llm_data["text"])

    # Generate a response in markdown
    markdown_response = generate_markdown_response(question, combined_text)

    # Return the markdown response
    return {"response_markdown": markdown_response}

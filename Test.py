# Importing required libraries for testing and debugging
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import pdfplumber
from io import BytesIO
import torch
import os
import json
from sentence_transformers import SentenceTransformer

# Initialize models for text and image processing
text_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = T5ForConditionalGeneration.from_pretrained("t5-small")
qa_tokenizer = T5Tokenizer.from_pretrained("t5-small")
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Setting up test directory for debugging
TEST_DIR = "/mnt/data/test_knowledge_base"
os.makedirs(TEST_DIR, exist_ok=True)

# Test PDF setup for debugging
test_pdf_path = "./Chapter-12.pdf"
with open(test_pdf_path, "wb") as f:
    f.write(b"%PDF-1.4\n%Test PDF\n1 0 obj\n<<>>\nstream\nBT\n/F1 12 Tf\n10 0 Td\n(Test PDF Content)Tj\nET\nendstream\nendobj\n")

# Helper Functions
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

# Test Function to Debug Image Handling and Processing
def test_pdf_processing(pdf_path):
    """
    Processes the test PDF for debugging text and image extraction and embedding.
    """
    # Extract content from PDF
    text, images = extract_text_and_images_from_pdf(pdf_path)

    # Debug output for text and image processing
    print("Extracted Text:", text[:100])  # Show first 100 characters of text

    # Process text into embeddings
    text_embedding = process_text(text)
    print("Text Embedding Shape:", text_embedding.shape)

    # Process images into embeddings
    image_embeddings = []
    for img in images:
        try:
            embedding = process_image(img)
            image_embeddings.append(embedding)
            print("Image Embedding Shape:", embedding.shape)
        except Exception as e:
            print("Error processing image:", e)

    # Save processed results to JSON
    result = {
        "text": text,
        "text_embedding": text_embedding.tolist(),
        "image_embeddings": [emb.tolist() for emb in image_embeddings]
    }
    result_path = os.path.join(TEST_DIR, "test_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f)

    return result_path

# Run the test to debug and validate
test_result_path = test_pdf_processing(test_pdf_path)
print(test_result_path)

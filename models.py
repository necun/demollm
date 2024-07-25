# models.py

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import logging
import io
import pytesseract
from prompts import *
import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify more languages if needed


class AIModel:
    def __init__(self, model_name, tokenizer_name, access_token):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_auth_token=access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_auth_token=access_token)
        self.model.config.use_cache = False
        self.model.eval()
        logging.basicConfig(level=logging.INFO)
    
    def chat(self, image, msgs, sampling=True, temperature=0.1, stream=False):
        return self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature,
            stream=stream
        )



class ImageProcessor:
    @staticmethod
    def load_image(image_bytes):
        print("Loading image")
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    @staticmethod
    def extract_text(image):
        width, height = image.size
        split_point = int(height * 0.12)
        top_section = image.crop((0, 0, width, split_point))
                # Convert the top section to bytes
        top_section_bytes = io.BytesIO()
        top_section.save(top_section_bytes, format='JPEG')
        top_section_bytes = top_section_bytes.getvalue()

        # Perform OCR on the cropped section
        results = reader.readtext(top_section_bytes)
        res = ""
        # Display the results
        for result in results:
            res+=result[1] +" " # result[1] contains the detected text
        logging.info("detected text: " + res)

        # # pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
        # extracted_text = pytesseract.image_to_string(top_section)
        # logging.info(f"extracted text: {extracted_text}")
        
        
        return res
# controllers.py

from models import AIModel, ImageProcessor
from fastapi.responses import  JSONResponse
import logging
import asyncio
from prompts import *
from exceptions_handlers import *
import torch
# Disable gradient computation
torch.set_grad_enabled(False)


class TemplateController:
    def __init__(self, model):
        self.model = model

    async def handle_template_request(self, image_bytes, template_id):
        image = ImageProcessor.load_image(image_bytes)
        template = templates[template_id]
        
        if not template:
            return JSONResponse(content={"error": "Invalid template ID"}, status_code=400)
        
        template_msgs = [{'role': 'user', 'content': template}]
        
        try:
            with torch.no_grad():
                res = self.model.chat(
                    image=image,
                    msgs=template_msgs,
                
                    sampling=True,
                    temperature=0.1,
                )
            return JSONResponse(content=res, status_code=200)
        except TypeError as e:
            logging.error(f"Error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    async def handle_template_stream(self, image_bytes):
        image = ImageProcessor.load_image(image_bytes)
        text = ImageProcessor.extract_text(image)

        template_id = ""
        if "VISITORS FORM" in text:
            template_id = "VISITORS FORM"
        elif "To Do" in text:
            template_id = "To Do"
        elif "Schedule Meeting" in text:
            template_id = "Schedule Meeting"
        elif "ENQUIRY FORM" in text:
            template_id = "ENQUIRY FORM"
        elif "MOM" in text:
            template_id = "MOM"
        else:
            return JSONResponse(content={"error": "Kindly scan the template properly"}, status_code=400)

        response =await self._template_stream_handler(image, template_id)
        return response

    async def _template_stream_handler(self, image, template_id):
        template = templates[template_id]

        if not template:
            return '{"error":"Invalid Template ID"}'
            

        template_msgs = [{'role': 'user', 'content': template}]
        logging.info(f"Template messages: {template_msgs}")

        with torch.no_grad():
            res = self.model.chat(
                image=image,
                msgs=template_msgs,
             
                sampling=True,
                temperature=0.1,
            )
        print(res)

        return res

        # end_of_stream_token = "<|eot_id|>"
        # empty_chunks_count = 0

        # for chunk in template_stream_response:
        #     logging.info(f"Chunk received: {chunk}")

        #     if isinstance(chunk, str) and chunk != "":
        #         empty_chunks_count = 0

        #         if end_of_stream_token in chunk:
        #             chunk = chunk.replace(end_of_stream_token, "")
        #             yield chunk
        #             break

        #         yield chunk
        #     elif chunk == "":
        #         empty_chunks_count += 1
        #         if empty_chunks_count >= 100:
        #             break
        #     else:
        #         logging.error(f"Unexpected chunk format: {chunk}")
        #         yield '{"error":"Unexpected chunk format"}'

        #     await asyncio.sleep(0.1)

class LangChainController:
    def __init__(self, model):
        self.model = model

    async def handle_langchain_stream(self, image_bytes, langchain_prompt):
        image = None
        if image_bytes:
            image = ImageProcessor.load_image(image_bytes)

        langchain_msgs = [{'role': 'user', 'content': langchain_prompt}]
        logging.info(f"Langchain messages: {langchain_msgs}")

        # langchain_response = self.model.chat(
        #     image=image,
        #     msgs=langchain_msgs,
        #     sampling=True,
        #     temperature=0.1,
        #     # stream=True
        # )
        # Use torch.no_grad() to ensure no gradients are computed or stored
        with torch.no_grad():
            res = self.model.chat(
                image=image,
                msgs=langchain_msgs,
             
                sampling=True,
                temperature=0.1,
            )

        return res

        # end_of_stream_token = "<|eot_id|>"
        # op = "OpenAI."
        # empty_chunks_count = 0

        # for chunk in langchain_response:
        #     logging.info(f"Chunk received: {chunk}")

        #     if isinstance(chunk, str) and chunk != "":
        #         empty_chunks_count = 0
        #         if op in chunk:
        #             chunk = chunk.replace("OpenAI", "ReNoteAI")

        #         if end_of_stream_token in chunk:
        #             chunk = chunk.replace(end_of_stream_token, "")
        #             yield chunk
        #             break

        #         yield chunk
        #     elif chunk == "":
        #         empty_chunks_count += 1
        #         if empty_chunks_count >= 100:
        #             break
        #     else:
        #         logging.error(f"Unexpected chunk format: {chunk}")
        #         yield '{"error":"Unexpected chunk format"}'

        #     await asyncio.sleep(0.1)
# controllers.py

class OcrController:
    def __init__(self, model):
        self.model = model

    async def renote_ocr(self, image_bytes):

        
        image = None
        if image_bytes:
            image = ImageProcessor.load_image(image_bytes)

        ocr_prompt = "Please extract all the text from the provided image, regardless of the language it is written in. Ensure that the output is accurate and retains the original formatting as much as possible. Provide the text in a plain text format.and ignore all other explanations give only extracted text"
        langchain_msgs = [{'role': 'user', 'content': ocr_prompt}]
        logging.info(f"Langchain messages: {langchain_msgs}")

        # langchain_response = self.model.chat(
        #     image=image,
        #     msgs=langchain_msgs,
        #     sampling=True,
        #     temperature=0.1,
        #     # stream=True
        # )
        with torch.no_grad():
            res = self.model.chat(
                image=image,
                msgs=langchain_msgs,
             
                sampling=True,
                temperature=0.1,
            )
        return res

        
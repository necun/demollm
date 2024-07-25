# main.py
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException,Form
from fastapi.responses import JSONResponse
from models import AIModel
from controllers import TemplateController, LangChainController, OcrController
import logging
import nest_asyncio
from starlette.responses import JSONResponse
import asyncio
from exceptions_handlers import *
import json
# Function to clear CUDA cache
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
 
# Clear cache at the beginning
clear_cuda_cache()

# Disable gradient computation
torch.set_grad_enabled(False)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

access_token = "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"
try:
    model = AIModel('ReNoteTech/MiniCPM-Llama3-V-2_5-int4', 'ReNoteTech/MiniCPM-Llama3-V-2_5-int4', access_token)
    # Clear cache after model loading
    clear_cuda_cache()
except Exception as e:
    raise AIModelInitializationError(str(e))

template_controller = TemplateController(model)
langchain_controller = LangChainController(model)
ocr_controller = OcrController(model)

app = FastAPI()

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request, exc):
    return await validation_exception_handler(request, exc)

@app.exception_handler(AIModelInitializationError)
async def custom_ai_model_initialization_error_handler(request, exc):
    return await ai_model_initialization_error_handler(request, exc)

@app.exception_handler(TemplateProcessingError)
async def custom_template_processing_error_handler(request, exc):
    return await template_processing_error_handler(request, exc)

@app.exception_handler(LangChainProcessingError)
async def custom_langchain_processing_error_handler(request, exc):
    return await langchain_processing_error_handler(request, exc)

@app.exception_handler(OcrProcessingError)
async def custom_ocr_processing_error_handler(request, exc):
    return await ocr_processing_error_handler(request, exc)

@app.post("/templates_static/")
async def extract_template(template_image: UploadFile = File(...), template_id: str = Form(...)):
    try:
        image_bytes = await template_image.read()
        return await template_controller.handle_template_request(image_bytes, template_id)
    except Exception as e:
        logger.error(f"Template processing error: {str(e)}")
        raise TemplateProcessingError(str(e))

@app.post("/templates/")
async def templates_stream(template_image: UploadFile = File(...)):
    try:
        
        image_bytes = await template_image.read()
        res = await template_controller.handle_template_stream(image_bytes)
        response_dict = json.loads(res)
        # Clear cache after inference
        clear_cuda_cache()
        print(res)
        return {"response": response_dict}
    except Exception as e:
        logger.error(f"Template stream processing error: {str(e)}")
        raise TemplateProcessingError(str(e))

@app.post("/chatbot_streams")
async def langchain_stream(
    langchain_image: UploadFile = File(None),  # Make the file upload optional
    langchain_prompt: str = Form(...)
):
    try:
        image_bytes = None
        if langchain_image:
            image_bytes = await langchain_image.read()

        # res =  StreamingResponse(langchain_controller.handle_langchain_stream(image_bytes, langchain_prompt), media_type="text/plain")
        res= await langchain_controller.handle_langchain_stream(image_bytes, langchain_prompt)
        if "OpenAI" in res:
                    
                    res = res.replace("OpenAI", "ReNoteAI")
        response_dict = json.loads(res)

        print(res)
        # Clear cache after inference
        clear_cuda_cache()
        return JSONResponse(content={"response": response_dict, "status_code":200}, status_code=200)
    except Exception as e:
        logger.error(f"LangChain processing error: {str(e)}")
        raise LangChainProcessingError(str(e))
 

@app.post("/renote_ocr")
async def renote_ocr(
    renote_image: UploadFile = File(...)
):
    try:
        image_bytes = await renote_image.read()
        # res = StreamingResponse(ocr_controller.renote_ocr(image_bytes), media_type="text/plain")
        res= await ocr_controller.renote_ocr(image_bytes)

        print(res)
        # Clear cache after inference
        clear_cuda_cache()
        return JSONResponse(content={"response": res, "status_code":200}, status_code=200)
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        raise OcrProcessingError(str(e))

if __name__ == "__main__":
    import uvicorn
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

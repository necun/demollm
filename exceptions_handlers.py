# exception_handlers.py

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging
from utils import format_error_response

logger = logging.getLogger("uvicorn.error")

class AIModelInitializationError(Exception):
    pass

class TemplateProcessingError(Exception):
    pass

class LangChainProcessingError(Exception):
    pass


class OcrProcessingError(Exception):
    pass


async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(exc.status_code, exc.detail, "HTTPException", "An error occurred", "HTTPException", exc.status_code)
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=format_error_response(422, "Validation Error", "RequestValidationError", str(exc.errors()), "RequestValidationError", 422)
    )

async def ai_model_initialization_error_handler(request: Request, exc: AIModelInitializationError):
    logger.error(f"AI model initialization error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=format_error_response(500, "AI Model Initialization Error", "AIModelInitializationError", str(exc), "AIModelInitializationError", 500)
    )

async def template_processing_error_handler(request: Request, exc: TemplateProcessingError):
    logger.error(f"Template processing error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=format_error_response(500, "Template Processing Error", "TemplateProcessingError", str(exc), "TemplateProcessingError", 500)
    )

async def langchain_processing_error_handler(request: Request, exc: LangChainProcessingError):
    logger.error(f"LangChain processing error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=format_error_response(500, "LangChain Processing Error", "LangChainProcessingError", str(exc), "LangChainProcessingError", 500)
    )

async def ocr_processing_error_handler(request: Request, exc: OcrProcessingError):
    logger.error(f"OCR processing error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=format_error_response(500, "Ocr Processing Error", "OcrProcessingError", str(exc), "OcrProcessingError", 500)
    )
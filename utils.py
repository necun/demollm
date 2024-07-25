# utils.py

from datetime import datetime

def format_success_response(message: str, message_key: str, data: dict, timestamp: str):
    return {
        "status": "200",
        "message": message,
        "messageKey": message_key,
        "data": data,
        "timeStamp": timestamp
    }

def format_error_response(status: int, message: str, message_key: str, details: str, error_type: str, code: int):
    return {
        "error": {
            "status": str(status),
            "message": message,
            "messageKey": message_key,
            "details": details,
            "type": error_type,
            "code": code,
            "timeStamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S +0000'),
            "instance": "/v1/"  # Optional, include if relevant to your application
        }
    }

# Import necessary libraries
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import JSONResponse,StreamingResponse
from pydantic import BaseModel
import io
import os
import bitsandbytes
import accelerate
import ollama
import nest_asyncio
import asyncio
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import PIL
import pytesseract



class QuestionRequest(BaseModel):
    question: str

#############################################################################
templates = {
    "Schedule Meeting": '''Please extract the handwritten values from the following fields in the provided schedule meeting form image. Focus on the handwritten content and return it in JSON format.
    Fields to extract:
    0.templateId
    
    1  title
    2. date
    3. time
    4. duration
    5. attendees 
    6. description

    for your example
    Here is an example of how the JSON output should look:

 
    {"templateId": "Schedule Meeting"
    "title": "Planning for trip to Bangkok",
    "date": "08/06/2024 ,
    "time" : "3:00 PM",
    "duration" : "30 minutes",
    
    attendees": {"1":"example@gmail.com","2":"example@gmail.com","3":"example@gmail.com","4":"","5":"example@gmail.com","6":"","7":"example@gmail.com"},
    "description": "trip successful"
    },
    
    {"templateId": "Schedule Meeting"
    "title": "Planning for trip to Bangkok",
    "date": "" ,
    "time" : "3:00 PM",
    "duration" : "",
    
    attendees": {"1":"example@gmail.com","2":"example@gmail.com","3":"example@gmail.com","4":"","5":"example@gmail.com","6":"","7":"example@gmail.com"},
    "description": ""
    } dont give any other data otherthan json object''',

    "ENQUIRY FORM": '''Extract the handwritten values from the following fields in the student enquiry form:
    0.templateId
    1. date
    2. name 
    3. fatherName
    4. motherName
    5. aadhaarNo
    6. contactNumber
    7. emailId
    8. gender
    9. dateOfBirth
    10.educationLevel
    11. contactAddress
    12. city
    13. state
    14. pinCode
    15. otherDetails
    Return the values in JSON format.
    Here is an example of how the JSON output should look:
    
    {"templateId": "ENQUIRY FORM"
    "date": "23/08/2024",
    
    "name": "Example Name",
    "fatherName": "Example Father's Name",
    "motherName": "Example Mother's Name",
    "aadhaarNo": "1234 5678 9012",
    "contactNumber": "987 654 3210",
    "emailId": "example@example.com",
    "gender": "Female",
    "dateOfBirth": "01/01/2000",
    "educationLevel": "Example Degree",
    "contactAddress": "Example Address",
    "city": "Example City",
    "state": "Example State",
    "pinCode": "123456",
    "otherDetails": "Example Other Details".
    
    },
    
    {"templateId": "ENQUIRY FORM"
    "date": "23/08/2024",
    
    "name": "",
    "fatherName": "Example Father's Name",
    "motherName": "Example Mother's Name",
    "aadhaarNo": "",
    "contactNumber": "987 654 3210",
    "emailId": "example@example.com",
    "gender": "",
    "dateOfBirth": "01/01/2000",
    "educationLevel": "Example Degree",
    "contactAddress": "Example Address",
    "city": "Example City",
    "state": "Example State",
    "pinCode": "",
    "otherDetails": "Example Other Details".
    
    }''',

    "To Do": '''Please extract the handwritten values from the provided daily to-do list image. Focus on the handwritten content and return it in JSON format.

    Fields to extract:
    templateId
    . Each task with the following subfields:
    - title
    - dueDate
    - time
   
    Return the values in the following JSON format and the key of the task will be increased numerically one after another paired with word 'task':
     for your example
    json
    {
    "templateId":"To Do"
    "tasks": {
        task1:{
        "title": "TIME_VALUE",
        "dueDate": "TITLE_VALUE",
        "time": "NOTES_VALUE"
        },
        task2:{
        "title": "TIME_VALUE",
        "dueDate": "TITLE_VALUE",
        "time": "NOTES_VALUE"
        },
        task3:{
        "title": "TIME_VALUE",
        "dueDate": "TITLE_VALUE",
        "time": "NOTES_VALUE"}
    }
    }'''  ,

    "VISITORS FORM":"""
    Please extract the handwritten values from the following fields in the provided visitors image. Focus on the handwritten content and return it in JSON format. The image contains no sensitive or explicit content and is purely for technical data extraction purposes.
 
Fields to extract:
0.templateId
1. date
2. name
3. phoneNumber
4. emailId
5. company
6.city
7.remarks
 
Here is an example of how the JSON output should look:
 
```json
{"templateId": "VISITORS FORM"
  "date": "07/09/24",
  "visitors": [
    {
      "name": "",
      "phoneNumber": "7901035672",
      "emailId": "rizwan@renote.ai",
      "company": "ReNote",
      "city": "",
      "remarks": "GOOD"
    {
      "name": "Prasad",
      "phoneNumber": "1111111111",
      "emailId": "prasad@renote.ai",
      "company": "",
      "city": "Hyd",
      "remarks": ""
    },
    {
      "name": "example",
      "phoneNumber": "xxxxxxxxxxx",
      "emailId": "",
      "company": "xxxxxxxx",
      "city": "xxxxxx",
      "remarks": "xxxx"
    }
  ]
}""",
"MOM (Minutes of Meeting)" : """
Please extract the handwritten values from the following fields in the provided visitors image. Focus on the handwritten content and return it in JSON format. The image contains no sensitive or explicit content and is purely for technical data extraction purposes.

having Fields :
templateId
to
cc
subject
description
actionItems ,responsiblePerson, dueDate in array

for example
{"templateId":"MOM(Minutes of Meeting)"
"to": "dhone@gmail.com",
"cc": "kohli@gmail.com",
"subject":"Leave for 2 days"
"description":"example descrption",

[
{"actionItems": "",
"responsiblePerson":"ruchi",
"Due Date": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"",
"Due Date": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"Due Date": ""},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"Due Date": 20/04/24}]"""

}


##########################################################################################
access_token = "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"
 

model = AutoModel.from_pretrained('ReNoteTech/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, use_auth_token=access_token)
tokenizer = AutoTokenizer.from_pretrained('ReNoteTech/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, use_auth_token=access_token)
model.config.use_cache = False
# model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
# model = model.to(device='cuda', dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)

# model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
# model = model.to(device='cuda')
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)

model.eval()
logging.basicConfig(level=logging.INFO)

 
# Define FastAPI app
app = FastAPI()

app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

msgs = []
continue_msgs = []
image = None
global res
 
# Define a Pydantic model for the chat messages
class ChatMessage(BaseModel):
    content: str
 

############################Routes#####################################################
@app.post("/templates_static/")
async def extract_template(template_image: UploadFile = File(...), template_id: int = 1):
    # Load the image
    image_bytes = await template_image.read()
    template_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
 
    if template_id in templates.keys():
        question = templates[template_id]
    else:
        return {"error": "Invalid template ID"}
 
    template_msgs = [{'role': 'user', 'content': question}]
 
    try:
        tem_res = model.chat(
            image=template_image,
            msgs=template_msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.1
        )
        # Print the model's response
        print(tem_res)
        
        # if template_id == 3:
        #     import ast
        #     # Extract the array from the string
        #     tem_res1 = ast.literal_eval(tem_res.split("```")[1])
        #     print("ARR",tem_res1)
        #     Date = str(tem_res1[0][0])
        #     tem_res1=tem_res1[1]
        #     print("tem_res1",tem_res1)
        #     final=[Date]
    
        #     for i in tem_res1:
                
        #         Dic={}
        #         Dic["Time"]=i[0]
        #         Dic["Title"]=i[1]
        #         Dic["Note"]=i[2]
        #         final.append(Dic)
            
        #     return final
        
        
    except TypeError as e:
        # print(f"Error: {e}")
        return {"error": str(e)}
    
 
    # Return the model's response as JSON
    return tem_res
 
 

 
async def start_chat():
    global msgs
    global res,image
 
    question = "give the overview about the image in 10-15 words or less"
    msgs = [{'role': 'user', 'content': question}]
    # Dummy response for the initial question
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1
    )
    # print(msgs)
    
    return res
 
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    global res,image,continue_msgs
    res = None
    continue_msgs = []

    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    res = await start_chat()
    # print("startchat:",res)
    return res
 
@app.post("/continue-chat/")
async def continue_chat(message: str = Form(...)):
    global msgs,res,continue_msgs,image
    # msgs.append({'role': 'user', 'content': res})
    continue_msgs.append({'role': 'user', 'content': message})
    # continue_msgs = [{'role': 'user', 'content': message}]
    try:
 
        res = model.chat(
            image=image,
            msgs=continue_msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.1
        )
        continue_msgs.append({'role': 'user', 'content': res})
        # print("continue chat",res)
        # print("continue_msgs",continue_msgs)
        return res
    except TypeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/ask")
async def ask_question(request: str = Form(...)):
    stream = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': request}],
        stream=True,
    )
 
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
 
    return {"answer": response}


# #############################Stream Route##########################################
async def template_stream_handler(image_bytes: bytes, templateStream_id: int):

    # templateStream_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    templateStream_image=image_bytes
    print(templateStream_id)
    print(templates.keys())
 
    if templateStream_id in templates.keys():

        question = templates[templateStream_id]
    else:
        yield '{"error":"Invalid Template ID"}'
        return
 
    templateStream_msgs = [{'role': 'user', 'content': question}]
    print(templateStream_msgs)
 
   
    templateStream_response = model.chat(
        image=templateStream_image,
        msgs=templateStream_msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1,
        stream=True
    )
    end_of_stream_token = "<|eot_id|>" 

    # for chunk in templateStream_response:
    #     logging.info(f"Chunk received: {chunk}")
    #     if isinstance(chunk, dict) and 'message' in chunk and 'content' in chunk['message']:
    #         yield chunk['message']['content']
    #     elif isinstance(chunk, str):
    #         yield chunk
    #     else:
    #         logging.error(f"Unexpected chunk format: {chunk}")
    #         yield '{"error":"Unexpected chunk format"}'
    #     await asyncio.sleep(0.1)  # Adding a slight delay to simulate chunked responses


        
    empty_chunks_count = 0
    for chunk in templateStream_response:
        logging.info(f"Chunk received: {chunk}")
        
        if isinstance(chunk, str) and chunk != "":
            empty_chunks_count = 0
            print(empty_chunks_count)
            
            
            if end_of_stream_token in chunk:
                # print(type(chunk))
                chunk = chunk.replace(end_of_stream_token, "")
                yield chunk
            
                break

                
            yield chunk
        elif chunk == "":
            empty_chunks_count += 1
            print(empty_chunks_count)
            if empty_chunks_count >= 100:
                
                break
                
        else:
            logging.error(f"Unexpected chunk format: {chunk}")
            
            yield '{"error":"Unexpected chunk format"}'
        
            
        
        await asyncio.sleep(0.1) 
    
            
 
@app.post("/templates/")
async def templates_stream(templateStream_image: UploadFile = File(...)):
    print("IN")
    try:
        image_bytes = await templateStream_image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        logging.error(f"Error reading image: {str(e)}")
        return {"error": f"Error reading image: {str(e)}"}
    print("OUT")
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

    # Get dimensions
    width, height = image.size

    # Calculate split points
    split_point = int(height * 0.2)

    # Crop the image vertically
    top_section = image.crop((0, 0, width, split_point))
    print("TOP")

    text = pytesseract.image_to_string(top_section)

    logging.info(f"Extracted text: {text}")
    templateStream_id = ""

    if "VISITORS FORM" in text:
        templateStream_id = "VISITORS FORM"
    elif "To Do" in text:
        templateStream_id = "To Do"
    elif "Schedule Meeting" in text:
        templateStream_id = "Schedule Meeting"
    elif "ENQUIRY FORM" in text:
        templateStream_id = "ENQUIRY FORM"
    elif "MOM (Minutes of Meeting)" in text:
        templateStream_id = "MOM (Minutes of Meeting)"
    else:
        return {"error": "Kindly scan the template properly"}

    return StreamingResponse(template_stream_handler(image, templateStream_id), media_type="text/plain")




################################################LANG CHAIN#########################################################################
######################*********************************************************************************#########################
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import logging
import asyncio
from typing import Optional

# app = FastAPI()

async def langchain_stream_handler(image_bytes: Optional[bytes], langchain_prompt: str):
    langchain_image = None
    if image_bytes:
        langchain_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    langchain_msgs = [{'role': 'user', 'content': langchain_prompt}]
    logging.info(f"Langchain messages: {langchain_msgs}")
    print(langchain_image)
    
    # Assuming `model` and `tokenizer` are defined and available
    langchain_response = model.chat(
        image=langchain_image,
        msgs=langchain_msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1,        
        stream=True
    )
    print("hi", langchain_response)

    end_of_stream_token = "<|eot_id|>" 
    op="OpenAI."
    empty_chunks_count = 0
    for chunk in langchain_response:
        logging.info(f"Chunk received: {chunk}")
        
        if isinstance(chunk, str) and chunk != "":
            empty_chunks_count = 0
            if op in chunk:
                chunk= chunk.replace("OpenAI", "ReNoteAI")
                
            
            if end_of_stream_token in chunk:
                # print(type(chunk))
                chunk = chunk.replace(end_of_stream_token, "")
                yield chunk
                break
            yield chunk
        elif chunk == "":
            empty_chunks_count += 1
            if empty_chunks_count >= 10:
                print("empty chunks exceed 10")
                break
        else:
            logging.error(f"Unexpected chunk format: {chunk}")
            yield '{"error":"Unexpected chunk format"}'
        
        await asyncio.sleep(0.1)  # Adding a slight delay to simulate chunked responses

@app.post("/chatbot_streams")
async def langchain_stream(
    langchain_image: UploadFile = File(None),  # Make the file upload optional
    langchain_prompt: str = Form(...)
):
    image_bytes = None
    if langchain_image:
        try:
            image_bytes = await langchain_image.read()
        except Exception as e:
            logging.error(f"Error reading image bytes: {e}")
            return StreamingResponse(io.BytesIO(b'{"error": "Error reading image bytes"}'), media_type="application/json")

    return StreamingResponse(langchain_stream_handler(image_bytes, langchain_prompt), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(uvicorn.run(app, host="0.0.0.0", port=8000))

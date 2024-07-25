templates = {
    "Schedule Meeting": '''Please extract the handwritten values from the following fields in the provided schedule meeting form image. Focus on the handwritten content and return it in JSON format.
    Fields to extract:
    0.templateId
    
    1  title
    2. date
    3. time
    4. duration
    5. attendees (sometimes it may be side by side also detect it properly and perform accordingly)
    6. description

    for your example
    Here is an example of how the JSON output should look:

 
    {"templateId": "Schedule Meeting",
    "title": "Planning for trip to Bangkok",
    "date": "08/06/2024 ,
    "time" : "3:00 PM",
    "duration" : "30 minutes",
    
    attendees": ["example@gmail.com","example@gmail.com","example@gmail.com","","example@gmail.com","","example@gmail.com"},
    "description": "trip successful"
    },
    
    {"templateId": "Schedule Meeting",
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
    
    {"templateId": "ENQUIRY FORM",
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
    
    {"templateId": "ENQUIRY FORM",
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
    - task
    - taskDate
    
   
    Return the values in the following JSON format and the key of the task will be increased numerically one after another paired with word 'task':
     for your example  json
   
    {
    "templateId":"To Do",
    "tasks": [
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        },
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        },
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        }
    ]
    }



    {
    "templateId":"To Do",
    "tasks": [
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        },
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        },
        {
        "task": "TITLE_VALUE",
        "taskDate": "Date_VALUE TIME_VALUE AM/PM"
        }
    ]
    }
    
    give only json object dont give any other information'''  ,

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
 

{"templateId": "VISITORS FORM",
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
      "phoneNumber": "9392624785",
      "emailId": "",
      "company": "renote",
      "city": "xxxxxx",
      "remarks": "xxxx"
    },
    {
      "name": "example",
      "phoneNumber": "",
      "emailId": "",
      "company": "renote",
      "city": "xxxxxx",
      "remarks": ""
    }
  ]
},


{"templateId": "VISITORS FORM",
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
      "phoneNumber": "",
      "emailId": "",
      "company": "renote",
      "city": "",
      "remarks": "good"
    }
  ]
}

{"templateId": "VISITORS FORM",
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
}   

give only json object dont give any other information""",
"MOM" : """
Please extract the handwritten values from the following fields in the provided visitors image. Focus on the handwritten content and return it in JSON format. The image contains no sensitive or explicit content and is purely for technical data extraction purposes.

having Fields :
templateId
to
cc
subject
description
actionItems ,responsiblePerson, dueDate in array

for example
{"templateId":"MOM(Minutes of Meeting)",
"to": "dhone@gmail.com",
"cc": "kohli@gmail.com",
"subject":"Leave for 2 days"
"description":"example descrption",

actions: [
{"actionItems": "",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": ""},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24}]
}


{"templateId":"MOM(Minutes of Meeting)",
"to": "dhone@gmail.com",
"cc": "kohli@gmail.com",
"subject":"Leave for 2 days"
"description":"example descrption",

actions: [
{"actionItems": "",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": ""},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24}]
}

{"templateId":"MOM(Minutes of Meeting)",
"to": "dhone@gmail.com",
"cc": "kohli@gmail.com",
"subject":"Leave for 2 days"
"description":"example descrption",

actions: [
{"actionItems": "",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"",
"dueDate": 20/04/24},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": ""},
{"actionItems": "hosting a scrum meeting",
"responsiblePerson":"ruchi",
"dueDate": 20/04/24}]
}   give only json object  be careful with action items and responsible person and due date they are only haveing one line so i line is one response please don't mearge the new line text here"""

}
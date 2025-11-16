from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import os


mongodb_url = os.getenv("MONGODB_URL")
mongodb_database = os.getenv("MONGODB_DATABASE")
mongodb_collection = os.getenv("MONGODB_COLLECTION")

client = MongoClient(mongodb_url) 
db = client[mongodb_database] 
collection = db[mongodb_collection] 


def saveSummary(email:str, summary:dict):
    summary["interviewTime"] = datetime.now()
    result = collection.update_one(
       {"email": email},  
       {"$push": {"summaries": summary}},  
       upsert=True  # if user doesn't exist then it will create one
    )

    if result.modified_count > 0:
       print("Summary added successfully!")
    else:
       print("No matching user found, new document created.")

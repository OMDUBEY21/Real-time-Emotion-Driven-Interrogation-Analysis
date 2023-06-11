import pymongo
from bson import ObjectId
from datetime import datetime
from utils import Report, Session


client = pymongo.MongoClient("mongodb://localhost:27017/")
reports_collection = client.EmDetect.reports

def create_session(session: Session, oid: str):
    session_dict = session.dict()
    session_dict.update({'created_at': datetime.now()})
    reports_collection.update_one({"_id": ObjectId(oid)}, {'$push': {'sessions': session_dict}})

def create_report(report: Report):
    report_dict = report.dict()
    report_dict.update({'created_at': datetime.now()})
    report_id = reports_collection.insert_one(report_dict)
    print(type(report_id.inserted_id))
    return str(report_id.inserted_id)
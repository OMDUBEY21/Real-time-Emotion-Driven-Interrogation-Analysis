from pydantic import BaseModel

class Session(BaseModel):
    blink_stat: dict
    emotion_stat: dict
    truth_percent: float
    lie_percent: float
    neutral_percent: float

class Report(BaseModel):
    suspect_name: str 
    case_name: str
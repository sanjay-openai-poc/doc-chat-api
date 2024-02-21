from pydantic import BaseModel

class UrlContext(BaseModel):
    type: str
    value: str 
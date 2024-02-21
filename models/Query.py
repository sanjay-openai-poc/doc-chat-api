from pydantic import BaseModel


class Query(BaseModel):
    context_id: str
    query: str

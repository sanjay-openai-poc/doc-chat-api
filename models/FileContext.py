from pydantic import BaseModel

from typing import Annotated
from fastapi import  UploadFile, Form, File

class FileContext(BaseModel):
    type: Annotated[str, Form()]
    file: Annotated[UploadFile, File()]
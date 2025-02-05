#list of key values
# Define the Key-Value model
from typing import List
from pydantic import BaseModel


class KeyValue(BaseModel):
    key: str
    value: str


# Define the main model containing a list of key-value pairs
class KeyValueList(BaseModel):
    items: List[KeyValue]

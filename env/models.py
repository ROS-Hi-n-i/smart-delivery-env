from pydantic import BaseModel
from typing import List

class Package(BaseModel):
    id: int
    location: str
    delivered: bool
    priority: int   # 1 = normal, 2 = high priority

class AgentState(BaseModel):
    location: str
    carrying: List[int]

class EnvironmentState(BaseModel):
    agent: AgentState
    packages: List[Package]

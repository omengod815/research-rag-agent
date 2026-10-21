from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=5000)
    filename: str = Field(default="", max_length=255)


class AgentResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    approved: bool
